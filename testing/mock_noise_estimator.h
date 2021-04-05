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

#ifndef LYRA_CODEC_TESTING_MOCK_NOISE_ESTIMATOR_H_
#define LYRA_CODEC_TESTING_MOCK_NOISE_ESTIMATOR_H_

#include <vector>

#include "gmock/gmock.h"
#include "noise_estimator_interface.h"

namespace chromemedia {
namespace codec {

class MockNoiseEstimator : public NoiseEstimatorInterface {
 public:
  ~MockNoiseEstimator() override {}

  MOCK_METHOD(std::vector<float>, NoiseEstimate, (), (const, override));

  MOCK_METHOD(bool, Update, (const std::vector<float>&), (override));

  MOCK_METHOD(absl::optional<bool>, IsSimilarNoise, (const std::vector<float>&),
              (override));
};

}  // namespace codec
}  // namespace chromemedia

#endif  // LYRA_CODEC_TESTING_MOCK_NOISE_ESTIMATOR_H_
