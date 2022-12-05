/*
 * Copyright 2022 Google LLC
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

#ifndef LYRA_ZERO_FEATURE_ESTIMATOR_H_
#define LYRA_ZERO_FEATURE_ESTIMATOR_H_

#include <vector>

#include "absl/types/span.h"
#include "lyra/feature_estimator_interface.h"

namespace chromemedia {
namespace codec {

class ZeroFeatureEstimator : public FeatureEstimatorInterface {
 public:
  explicit ZeroFeatureEstimator(int num_features)
      : estimated_features_(num_features, 0.f) {}
  void Update(absl::Span<const float> features) override {
    // Do nothing.
  }

  std::vector<float> Estimate() const override { return estimated_features_; }

 private:
  ZeroFeatureEstimator() = delete;

  std::vector<float> estimated_features_;
};

}  // namespace codec
}  // namespace chromemedia

#endif  // LYRA_ZERO_FEATURE_ESTIMATOR_H_
