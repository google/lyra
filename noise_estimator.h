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

#ifndef LYRA_CODEC_NOISE_ESTIMATOR_H_
#define LYRA_CODEC_NOISE_ESTIMATOR_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/types/span.h"
#include "log_mel_spectrogram_extractor_impl.h"
#include "noise_estimator_interface.h"

namespace chromemedia {
namespace codec {

// This class estimates a noise vector from incoming packets from which a
// generative model can generate noise. The implementation is based on
// minimum statistics estimation in the log frequency domain.
class NoiseEstimator : public NoiseEstimatorInterface {
 public:
  static std::unique_ptr<NoiseEstimator> Create(int sample_rate_hz,
                                                int num_samples_per_hop,
                                                int num_samples_per_window,
                                                int num_features);

  // Buffers samples until a log mel spectrogram can be extracted.
  // If the latest log mel spectrogram extracted is different enough from the
  // current noise estimate, it updates the current noise estimate.
  // Cumulative length of |samples| may never straddle a multiple of the
  // number of samples per hop.
  // Returns true on success, false on failure.
  bool ReceiveSamples(const absl::Span<const int16_t> samples) override;

  // Returns the minimum noise statistic estimate from the last extracted
  // log mel spectrogram from |ReceiveSamples|.
  std::vector<float> noise_estimate() const override;

  // Returns whether the last log mel spectrogram extracted from
  // |ReceiveSamples| is noise.
  bool is_noise() const override;

 private:
  NoiseEstimator(int num_samples_per_hop, int num_hops_per_update,
                 int num_features, float max_smoothing,
                 float bound_decay_factor,
                 std::unique_ptr<LogMelSpectrogramExtractorImpl>
                     log_mel_spectrogram_extractor);

  NoiseEstimator() = delete;

  // Calculates and stores the minimum noise statistics given the current power
  // per frequency band and the previous state.
  void UpdateNoiseEstimate(const std::vector<float>& current_power_db);

  void ComputeBounds();

  // Identifies if |current_power_db| is similar to previously identified
  // noise.
  bool ComputeIsNoise(const std::vector<float>& current_power_db);

  void DecayBounds();

  const int num_samples_per_hop_;
  const int num_hops_per_update_;
  const float max_smoothing_;
  const float bound_decay_factor_;
  std::vector<float> smoothed_power_;
  std::vector<float> squared_smoothed_power_;
  std::vector<float> tmp_min_smoothed_power_;
  std::vector<float> noise_estimate_;
  std::vector<float> noise_bound_;
  std::vector<int16_t> past_samples_hop_;

  bool is_noise_;
  int num_hops_received_;
  int next_sample_in_hop_;

  std::unique_ptr<LogMelSpectrogramExtractorImpl>
      log_mel_spectrogram_extractor_;

  friend class NoiseEstimatorPeer;
};

}  // namespace codec
}  // namespace chromemedia

#endif  // LYRA_CODEC_NOISE_ESTIMATOR_H_
