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

#include "lyra/log_mel_spectrogram_extractor_impl.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/types/span.h"
#include "audio/dsp/mfcc/mel_filterbank.h"
#include "audio/dsp/number_util.h"
#include "audio/dsp/spectrogram/spectrogram.h"
#include "glog/logging.h"  // IWYU pragma: keep

namespace chromemedia {
namespace codec {

namespace {

static constexpr float kNorm = 10.f;
static constexpr float kLogFloor = 500.f;
static constexpr double kLowerFreqLimit = 0.0;
static constexpr double kUpperFreqLimitFactor = 0.495;

}  // namespace

LogMelSpectrogramExtractorImpl::LogMelSpectrogramExtractorImpl(
    std::unique_ptr<audio_dsp::Spectrogram> spectrogram,
    std::unique_ptr<audio_dsp::MelFilterbank> mel_filterbank,
    int hop_length_samples)
    : spectrogram_(std::move(spectrogram)),
      mel_filterbank_(std::move(mel_filterbank)),
      hop_length_samples_(hop_length_samples),
      samples_(hop_length_samples, 0.0) {}

std::unique_ptr<LogMelSpectrogramExtractorImpl>
LogMelSpectrogramExtractorImpl::Create(int sample_rate_hz,
                                       int hop_length_samples,
                                       int window_length_samples,
                                       int num_mel_bins) {
  if (window_length_samples < hop_length_samples) {
    LOG(ERROR) << "Window length samples was " << window_length_samples
               << " but must be >= hop length samples which was "
               << hop_length_samples;
    return nullptr;
  }
  auto spectrogram = std::make_unique<audio_dsp::Spectrogram>();
  if (!spectrogram->Initialize(window_length_samples, hop_length_samples)) {
    LOG(ERROR) << "Could not initialize spectrogram for feature extraction.";
    return nullptr;
  }
  // To get a spectrogram for the first audio of hop length samples we first
  // calculate the spectrogram of an empty window to fill the internal queue.
  std::vector<std::vector<double>> unused_spectrogram_slices;
  if (!spectrogram->ComputeSpectrogram(
          std::vector<double>(window_length_samples, 0.0),
          &unused_spectrogram_slices)) {
    LOG(ERROR) << "Error calculating spectrogram of empty window.";
    return nullptr;
  }

  // Compute the next power of two for FFT size.
  const int kFftSize = static_cast<int>(
      audio_dsp::NextPowerOfTwo(static_cast<unsigned>(window_length_samples)));
  // Number of unique FFT bins.
  const int kFftBins = kFftSize / 2 + 1;
  auto mel_filterbank = std::make_unique<audio_dsp::MelFilterbank>();
  if (!mel_filterbank->Initialize(kFftBins, sample_rate_hz, num_mel_bins,
                                  kLowerFreqLimit,
                                  GetUpperFreqLimit(sample_rate_hz))) {
    LOG(ERROR) << "Could not initialize mel filterbank for feature extraction.";
    return nullptr;
  }

  return absl::WrapUnique(new LogMelSpectrogramExtractorImpl(
      std::move(spectrogram), std::move(mel_filterbank), hop_length_samples));
}

std::optional<std::vector<float>> LogMelSpectrogramExtractorImpl::Extract(
    const absl::Span<const int16_t> audio) {
  if (audio.size() != hop_length_samples_) {
    LOG(ERROR) << "Input audio should have " << hop_length_samples_
               << " samples but instead had " << audio.size() << ".";
    return std::nullopt;
  }

  std::copy(audio.begin(), audio.end(), samples_.begin());

  std::vector<std::vector<double>> spectrogram_slices;
  if (!spectrogram_->ComputeSpectrogram(samples_, &spectrogram_slices)) {
    LOG(ERROR) << "Could not compute spectrogram from audio.";
    return std::nullopt;
  }
  if (spectrogram_slices.size() != 1) {
    LOG(ERROR) << "Spectrogram had unexpected number of output features.";
    return std::nullopt;
  }

  std::vector<double> temp_features;
  mel_filterbank_->Compute(spectrogram_slices.at(0), &temp_features);
  std::vector<float> mel_features(temp_features.begin(), temp_features.end());
  // Compute the log, but disallow values below the floor, then
  // normalize the amplitude to avoid clipping in Wavenet.
  for (auto& val : mel_features) {
    val = std::log(std::max(val, kLogFloor)) / kNorm;
  }

  return mel_features;
}

double LogMelSpectrogramExtractorImpl::GetLowerFreqLimit() {
  return kLowerFreqLimit;
}

double LogMelSpectrogramExtractorImpl::GetUpperFreqLimit(int sample_rate_hz) {
  return kUpperFreqLimitFactor * sample_rate_hz;
}

float LogMelSpectrogramExtractorImpl::GetNormalizationFactor() { return kNorm; }

float LogMelSpectrogramExtractorImpl::GetSilenceValue() {
  return std::log(kLogFloor) / kNorm;
}

}  // namespace codec
}  // namespace chromemedia
