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

#include <memory>
#include <vector>

#include "absl/types/optional.h"
#include "noise_estimator_interface.h"

namespace chromemedia {
namespace codec {

// This class estimates a noise vector from incoming packets which Wavenet can
// generate noise from. The implementation is based on minimum statistics
// estimation in the log frequency domain.
class NoiseEstimator : public NoiseEstimatorInterface {
 public:
  static std::unique_ptr<NoiseEstimator> Create(int num_features,
                                                float num_seconds_per_frame);

  // Calculates and stores the minimum noise statistics given the current power
  // per frequency band and the previous state.
  bool Update(const std::vector<float>& curr_power_db) override;

  // Returns the minimum noise statistic estimate.
  std::vector<float> NoiseEstimate() const override;

  // Identifies if current frame is similar to previously identified noise.
  // Returns a nullopt if the size of curr_power_db does not match
  // num_features_. Otherwise value is true if current frame is noise, false if
  // not.
  absl::optional<bool> IsSimilarNoise(
      const std::vector<float>& curr_power_db) override;

 private:
  NoiseEstimator(int num_features, int num_frames_per_update,
                 float max_smoothing, float bound_decay_factor);
  void ComputeBounds();

  const int num_features_;
  const int num_frames_per_update_;
  const float max_smoothing_;
  const float bound_decay_factor_;
  std::vector<float> smoothed_power_;
  std::vector<float> squared_smoothed_power_;
  std::vector<float> tmp_min_smoothed_power_;
  std::vector<float> noise_estimate_;
  std::vector<float> noise_bound_;
  int num_frames_received_ = 0;
};

}  // namespace codec
}  // namespace chromemedia

#endif  // LYRA_CODEC_NOISE_ESTIMATOR_H_
