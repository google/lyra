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

#include "wavegru_model_impl.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

// Placeholder for get runfiles header.
#include "gtest/gtest.h"
#include "include/ghc/filesystem.hpp"
#include "lyra_config.h"

namespace chromemedia {
namespace codec {
namespace {

class WavegruModelImplTest : public testing::Test {
 protected:
  WavegruModelImplTest()
      : num_samples_per_hop_(GetNumSamplesPerHop(kInternalSampleRateHz)),
        model_(WavegruModelImpl::Create(
            num_samples_per_hop_, kNumFeatures, kNumFramesPerPacket, 0.0f,
            ghc::filesystem::current_path() / "wavegru")) {}
  const int num_samples_per_hop_;
  std::unique_ptr<WavegruModelImpl> model_;
};

TEST_F(WavegruModelImplTest, ModelExists) { EXPECT_NE(model_, nullptr); }

TEST_F(WavegruModelImplTest, RunModelExpectedOutputSize) {
  std::vector<float> features(kNumFeatures);

  model_->AddFeatures(features);
  auto samples_or = model_->GenerateSamples(num_samples_per_hop_);

  ASSERT_TRUE(samples_or.has_value());
  EXPECT_EQ(samples_or.value().size(),
            GetNumSamplesPerHop(kInternalSampleRateHz));
}

TEST_F(WavegruModelImplTest, AddFeaturesAndGenerateSamplesExpectedOutputSize) {
  std::vector<float> features(kNumFeatures);
  model_->AddFeatures(features);

  for (const int num_samples : {15, 103, 84}) {
    auto samples_or = model_->GenerateSamples(num_samples);
    ASSERT_TRUE(samples_or.has_value());
    EXPECT_EQ(samples_or.value().size(), num_samples);
  }
}

TEST_F(WavegruModelImplTest, GenerateMoreThanNumSamplesPerHopExpectDeath) {
  std::vector<float> features(kNumFeatures);
  model_->AddFeatures(features);

  // First generate number of samples per hop, which will use up all the added
  // features.
  auto samples_or = model_->GenerateSamples(num_samples_per_hop_);
  ASSERT_TRUE(samples_or.has_value());
  EXPECT_EQ(samples_or.value().size(), num_samples_per_hop_);

  // Subsequent calls to GenerateSamples() all fail.
  for (const int num_samples : {108, 57, 261}) {
    EXPECT_DEATH(samples_or = model_->GenerateSamples(num_samples), "");
  }
}

}  // namespace
}  // namespace codec
}  // namespace chromemedia
