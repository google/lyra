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

#ifndef LYRA_CODEC_NOISE_ESTIMATOR_INTERFACE_H_
#define LYRA_CODEC_NOISE_ESTIMATOR_INTERFACE_H_

#include <vector>

#include "absl/types/optional.h"  // IWYU pragma: keep

namespace chromemedia {
namespace codec {

// An interface to abstract the noise estimation implementation.
class NoiseEstimatorInterface {
 public:
  virtual ~NoiseEstimatorInterface() {}

  virtual std::vector<float> NoiseEstimate() const = 0;

  virtual bool Update(const std::vector<float>& curr_power_db) = 0;

  virtual absl::optional<bool> IsSimilarNoise(
      const std::vector<float>& curr_power_db) = 0;
};

}  // namespace codec
}  // namespace chromemedia

#endif  // LYRA_CODEC_NOISE_ESTIMATOR_INTERFACE_H_
