/*
 * Copyright 2021 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef LYRA_CODEC_LOG_MEL_SPECTROGRAM_EXTRACTOR_IMPL_H_
#define LYRA_CODEC_LOG_MEL_SPECTROGRAM_EXTRACTOR_IMPL_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "absl/types/span.h"
#include "audio/dsp/mfcc/mel_filterbank.h"
#include "audio/dsp/spectrogram/spectrogram.h"
#include "feature_extractor_interface.h"

namespace chromemedia {
namespace codec {

// This class extracts mel spectrogram features from audio samples.
class LogMelSpectrogramExtractorImpl : public FeatureExtractorInterface {
 public:
  // Returns a nullptr if creation fails.
  static std::unique_ptr<LogMelSpectrogramExtractorImpl> Create(
      int sample_rate_hz, int hop_length_samples, int window_length_samples,
      int num_mel_bins);

  ~LogMelSpectrogramExtractorImpl() override {}

  // Extracts the mel features from the audio. On failure returns a nullopt.
  // The size of |audio| must match the value of |hop_length_samples_|.
  // This assumes that audio samples are passed in order.
  std::optional<std::vector<float>> Extract(
      const absl::Span<const int16_t> audio) override;

  // Returns the lower frequency limit used to initialize the MelFilterbank
  // class.
  static double GetLowerFreqLimit();

  // Returns the upper frequency limit used to initialize the MelFilterbank
  // class.
  static double GetUpperFreqLimit(int sample_rate_hz);

  // Returns the normalization factor used to normalize the log of the mel
  // features.
  static float GetNormalizationFactor();

  // Returns minimum value, that represents silence.
  static float GetSilenceValue();

 private:
  LogMelSpectrogramExtractorImpl() = delete;
  LogMelSpectrogramExtractorImpl(
      std::unique_ptr<audio_dsp::Spectrogram> spectrogram,
      std::unique_ptr<audio_dsp::MelFilterbank> mel_filterbank,
      int hop_length_samples);

  const std::unique_ptr<audio_dsp::Spectrogram> spectrogram_;
  const std::unique_ptr<const audio_dsp::MelFilterbank> mel_filterbank_;
  const int hop_length_samples_;
  std::vector<double> samples_;
};

}  // namespace codec
}  // namespace chromemedia

#endif  // LYRA_CODEC_LOG_MEL_SPECTROGRAM_EXTRACTOR_IMPL_H_
