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

#include "comfort_noise_generator.h"

#include <cmath>
#include <complex>
#include <cstdint>
#include <iterator>
#include <memory>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/random/random.h"
#include "absl/types/optional.h"
#include "audio/dsp/mfcc/mel_filterbank.h"
#include "audio/dsp/number_util.h"
#include "audio/dsp/spectrogram/inverse_spectrogram.h"
#include "dsp_util.h"
#include "glog/logging.h"
#include "log_mel_spectrogram_extractor_impl.h"

#ifdef BENCHMARK
#include "absl/time/clock.h"
#endif  // BENCHMARK

namespace chromemedia {
namespace codec {

std::unique_ptr<ComfortNoiseGenerator> ComfortNoiseGenerator::Create(
    int sample_rate_hz, int num_mel_bins, int window_length_samples,
    int hop_length_samples) {
  const int kFftSize = static_cast<int>(
      audio_dsp::NextPowerOfTwo(static_cast<unsigned>(window_length_samples)));
  const int kNumFftBins = kFftSize / 2 + 1;

  auto mel_filterbank = absl::make_unique<audio_dsp::MelFilterbank>();
  if (!mel_filterbank->Initialize(
          kNumFftBins, static_cast<double>(sample_rate_hz), num_mel_bins,
          LogMelSpectrogramExtractorImpl::GetLowerFreqLimit(),
          LogMelSpectrogramExtractorImpl::GetUpperFreqLimit(sample_rate_hz))) {
    LOG(ERROR) << "Could not initialize mel filterbank.";
    return nullptr;
  }

  auto inverse_spectrogram = absl::make_unique<audio_dsp::InverseSpectrogram>();
  if (!inverse_spectrogram->Initialize(kFftSize, hop_length_samples)) {
    LOG(ERROR) << "Could not initialize inverse spectrogram.";
    return nullptr;
  }

  return absl::WrapUnique(new ComfortNoiseGenerator(
      kNumFftBins, sample_rate_hz, num_mel_bins, hop_length_samples,
      std::move(mel_filterbank), std::move(inverse_spectrogram)));
}

ComfortNoiseGenerator::ComfortNoiseGenerator(
    int num_fft_bins, int sample_rate_hz, int num_mel_bins,
    int hop_length_samples,
    std::unique_ptr<audio_dsp::MelFilterbank> mel_filterbank,
    std::unique_ptr<audio_dsp::InverseSpectrogram> inverse_spectrogram)
    : mel_filterbank_(std::move(mel_filterbank)),
      inverse_spectrogram_(std::move(inverse_spectrogram)),
      num_fft_bins_(num_fft_bins),
      num_mel_bins_(num_mel_bins),
      hop_length_samples_(hop_length_samples) {}

void ComfortNoiseGenerator::AddFeatures(const std::vector<float>& features) {
  log_mel_features_ = features;
#ifdef BENCHMARK
  // No conditioning happens in the comfort noise generator.
  conditioning_timings_microsecs_.push_back(0);
#endif  // BENCHMARK
}

absl::optional<std::vector<int16_t>> ComfortNoiseGenerator::GenerateSamples(
    int num_samples) {
  if (num_samples > hop_length_samples_) {
    LOG(ERROR) << "Number of samples requested cannot be larger than the "
                  "hop length.";
    return absl::nullopt;
  }
  if (num_samples < 0) {
    LOG(ERROR)
        << "Number of samples requested must be greater than or equal to 0.";
    return absl::nullopt;
  }
  if (log_mel_features_.size() != num_mel_bins_) {
    LOG(ERROR) << "Size of features is " << log_mel_features_.size()
               << ", but should be " << num_mel_bins_ << ".";
    return absl::nullopt;
  }

#ifdef BENCHMARK
  const int64_t comfort_noise_generator_start_microsecs =
      absl::ToUnixMicros(absl::Now());
#endif  // BENCHMARK

  // Ensure there are enough samples in the buffer to return the requested
  // amount.
  if (num_samples > reconstructed_samples_.size()) {
    if (!FftFromFeatures()) return absl::nullopt;
    if (!InvertFft()) return absl::nullopt;
  }

  // Only return the number of samples requested and remove the returned samples
  // from the buffer.
  std::vector<int16_t> samples_to_return(
      reconstructed_samples_.begin(),
      reconstructed_samples_.begin() + num_samples);
  reconstructed_samples_.erase(reconstructed_samples_.begin(),
                               reconstructed_samples_.begin() + num_samples);

#ifdef BENCHMARK
  model_timings_microsecs_.push_back(absl::ToUnixMicros(absl::Now()) -
                                     comfort_noise_generator_start_microsecs);
#endif  // BENCHMARK

  return samples_to_return;
}

void ComfortNoiseGenerator::Reset() {
  log_mel_features_.clear();
  squared_magnitude_fft_.clear();
  reconstructed_samples_.clear();
}

bool ComfortNoiseGenerator::FftFromFeatures() {
  std::vector<double> mel_features(log_mel_features_.size());
  for (int i = 0; i < log_mel_features_.size(); ++i) {
    mel_features[i] = static_cast<double>(
        std::exp(log_mel_features_[i] *
                 LogMelSpectrogramExtractorImpl::GetNormalizationFactor()));
  }

  mel_filterbank_->EstimateInverse(mel_features, &squared_magnitude_fft_);

  if (squared_magnitude_fft_.size() != num_fft_bins_) {
    LOG(ERROR) << "Size of squared-magnitude FFT is "
               << squared_magnitude_fft_.size() << ", but should be "
               << num_fft_bins_ << ".";
    return false;
  }

  return true;
}

bool ComfortNoiseGenerator::InvertFft() {
  // Add random phase to squared-magnitude FFT to make it a complex FFT.
  // InverseSpectrogram class expects a 2D spectrogram, so one containing just
  // one slice is constructed.
  std::vector<std::vector<std::complex<double>>> random_phase_spectrogram(1);
  absl::BitGen gen;
  for (int i = 0; i < num_fft_bins_; ++i) {
    double magnitude = sqrt(squared_magnitude_fft_[i]);
    double random_angle = absl::Uniform<double>(gen, 0, 2 * M_PI);
    random_phase_spectrogram[0].push_back(
        magnitude * std::exp(std::complex<double>(0.0, 1.0) * random_angle));
  }

  std::vector<double> temp_samples;
  if (!inverse_spectrogram_->Process(random_phase_spectrogram, &temp_samples)) {
    return false;
  }

  if (temp_samples.size() != hop_length_samples_) {
    LOG(ERROR) << "Size of samples gotten from inverse FFT operation is "
               << temp_samples.size() << ", but should be "
               << hop_length_samples_ << ".";
    return false;
  }

  // Store samples in buffer to ensure continuity between samples.
  for (int i = 0; i < temp_samples.size(); ++i) {
    reconstructed_samples_.push_back(
        ClipToInt16(static_cast<float>(temp_samples[i])));
  }

  return true;
}

}  // namespace codec
}  // namespace chromemedia
