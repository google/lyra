// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "lyra/comfort_noise_generator.h"

#include <cmath>
#include <complex>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/random/random.h"
#include "absl/types/span.h"
#include "audio/dsp/mfcc/mel_filterbank.h"
#include "audio/dsp/number_util.h"
#include "audio/dsp/spectrogram/inverse_spectrogram.h"
#include "glog/logging.h"  // IWYU pragma: keep
#include "lyra/dsp_utils.h"
#include "lyra/log_mel_spectrogram_extractor_impl.h"

namespace chromemedia {
namespace codec {

std::unique_ptr<ComfortNoiseGenerator> ComfortNoiseGenerator::Create(
    int sample_rate_hz, int num_samples_per_hop, int window_length_samples,
    int num_mel_bins) {
  const int kFftSize = static_cast<int>(
      audio_dsp::NextPowerOfTwo(static_cast<unsigned>(window_length_samples)));
  const int kNumFftBins = kFftSize / 2 + 1;
  auto mel_filterbank = std::make_unique<audio_dsp::MelFilterbank>();
  if (!mel_filterbank->Initialize(
          kNumFftBins, static_cast<double>(sample_rate_hz), num_mel_bins,
          LogMelSpectrogramExtractorImpl::GetLowerFreqLimit(),
          LogMelSpectrogramExtractorImpl::GetUpperFreqLimit(sample_rate_hz))) {
    LOG(ERROR) << "Could not initialize mel filterbank.";
    return nullptr;
  }

  auto inverse_spectrogram = std::make_unique<audio_dsp::InverseSpectrogram>();
  if (!inverse_spectrogram->Initialize(kFftSize, num_samples_per_hop)) {
    LOG(ERROR) << "Could not initialize inverse spectrogram.";
    return nullptr;
  }

  return absl::WrapUnique(new ComfortNoiseGenerator(
      sample_rate_hz, num_samples_per_hop, num_mel_bins,
      std::move(mel_filterbank), std::move(inverse_spectrogram)));
}

ComfortNoiseGenerator::ComfortNoiseGenerator(
    int sample_rate_hz, int num_samples_per_hop, int num_mel_bins,
    std::unique_ptr<audio_dsp::MelFilterbank> mel_filterbank,
    std::unique_ptr<audio_dsp::InverseSpectrogram> inverse_spectrogram)
    : GenerativeModel(num_samples_per_hop, num_mel_bins),
      mel_filterbank_(std::move(mel_filterbank)),
      inverse_spectrogram_(std::move(inverse_spectrogram)),
      squared_magnitude_fft_(num_samples_per_hop),
      reconstructed_samples_(num_samples_per_hop) {}

bool ComfortNoiseGenerator::RunConditioning(
    const std::vector<float>& features) {
  FftFromFeatures(features);
  return InvertFft();
}

std::optional<std::vector<int16_t>> ComfortNoiseGenerator::RunModel(
    int num_samples) {
  return std::vector<int16_t>(
      reconstructed_samples_.begin() + next_sample_in_hop(),
      reconstructed_samples_.begin() + next_sample_in_hop() + num_samples);
}

void ComfortNoiseGenerator::FftFromFeatures(
    const std::vector<float>& log_mel_features) {
  std::vector<double> mel_features(log_mel_features.size());
  for (int i = 0; i < mel_features.size(); ++i) {
    mel_features.at(i) = static_cast<double>(
        std::exp(log_mel_features.at(i) *
                 LogMelSpectrogramExtractorImpl::GetNormalizationFactor()));
  }
  mel_filterbank_->EstimateInverse(mel_features, &squared_magnitude_fft_);
}

bool ComfortNoiseGenerator::InvertFft() {
  // Add random phase to squared-magnitude FFT to make it a complex FFT.
  // InverseSpectrogram class expects a 2D spectrogram, so one containing just
  // one slice is constructed.
  std::vector<std::vector<std::complex<double>>> random_phase_spectrogram(1);
  absl::BitGen gen;
  for (int i = 0; i < squared_magnitude_fft_.size(); ++i) {
    double magnitude = sqrt(squared_magnitude_fft_.at(i));
    double random_angle = absl::Uniform<double>(gen, 0, 2 * M_PI);
    random_phase_spectrogram[0].push_back(
        magnitude * std::exp(std::complex<double>(0.0, 1.0) * random_angle));
  }

  std::vector<double> temp_samples;
  if (!inverse_spectrogram_->Process(random_phase_spectrogram, &temp_samples)) {
    return false;
  }

  // Store samples in buffer to ensure continuity between samples.
  reconstructed_samples_ = ClipToInt16(absl::MakeConstSpan(temp_samples));
  return true;
}

}  // namespace codec
}  // namespace chromemedia
