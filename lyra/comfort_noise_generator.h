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

#ifndef LYRA_COMFORT_NOISE_GENERATOR_H_
#define LYRA_COMFORT_NOISE_GENERATOR_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "audio/dsp/mfcc/mel_filterbank.h"
#include "audio/dsp/spectrogram/inverse_spectrogram.h"
#include "lyra/generative_model_interface.h"

namespace chromemedia {
namespace codec {

// This class generates comfort noise by estimating audio samples that
// correspond to the given features.
class ComfortNoiseGenerator : public GenerativeModel {
 public:
  // Returns a nullptr on failure.
  static std::unique_ptr<ComfortNoiseGenerator> Create(
      int sample_rate_hz, int num_samples_per_hop, int window_length_samples,
      int num_mel_bins);

  ~ComfortNoiseGenerator() override {}

 private:
  ComfortNoiseGenerator(
      int sample_rate_hz, int num_samples_per_hop, int num_mel_bins,
      std::unique_ptr<audio_dsp::MelFilterbank> mel_filterbank,
      std::unique_ptr<audio_dsp::InverseSpectrogram> inverse_spectrogram);

  bool RunConditioning(const std::vector<float>& features) override;

  std::optional<std::vector<int16_t>> RunModel(int num_samples) override;

  // Estimates the Squared-Magnitude FFT that corresponds to the Log Mel
  // features. Returns true if the estimation completed successfully and false
  // otherwise.
  void FftFromFeatures(const std::vector<float>& log_mel_features);

  // Produces time-domain inverse of a Squared-Magnitude FFT by adding a random
  // phase to each element. Returns true if the inversion completed successfully
  // and false otherwise.
  bool InvertFft();

  const std::unique_ptr<const audio_dsp::MelFilterbank> mel_filterbank_;
  const std::unique_ptr<audio_dsp::InverseSpectrogram> inverse_spectrogram_;

  std::vector<double> squared_magnitude_fft_;
  std::vector<int16_t> reconstructed_samples_;
};

}  // namespace codec
}  // namespace chromemedia

#endif  // LYRA_COMFORT_NOISE_GENERATOR_H_
