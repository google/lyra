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

#include "lyra/lyra_gan_model.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

// Placeholder for get runfiles header.
#include "gtest/gtest.h"
#include "include/ghc/filesystem.hpp"
#include "lyra/lyra_config.h"

namespace chromemedia {
namespace codec {
namespace {

class LyraGanModelTest : public testing::Test {
 protected:
  LyraGanModelTest()
      : model_(LyraGanModel::Create(
            ghc::filesystem::current_path() / "lyra/model_coeffs",
            kNumFeatures)),
        features_(kNumFeatures) {}

  void ExpectValidSampleGeneration(int num_samples) {
    auto samples = model_->GenerateSamples(num_samples);
    ASSERT_TRUE(samples.has_value());
    EXPECT_EQ(samples.value().size(), num_samples);
  }

  std::unique_ptr<LyraGanModel> model_;
  std::vector<float> features_;
};

TEST_F(LyraGanModelTest, CreationFailsWithInvalidModelPath) {
  EXPECT_EQ(LyraGanModel::Create("invalid/model/path", features_.size()),
            nullptr);
}

TEST_F(LyraGanModelTest, CreationSucceedsWithValidModelPath) {
  EXPECT_NE(model_, nullptr);
}

TEST_F(LyraGanModelTest, NoSampleCanBeGeneratedBeforeAddingFeatures) {
  ASSERT_NE(model_, nullptr);

  EXPECT_FALSE(model_->GenerateSamples(1).has_value());
}

TEST_F(LyraGanModelTest, SamplesAreGeneratedUntilHopLimit) {
  ASSERT_NE(model_, nullptr);

  model_->AddFeatures(features_);

  const int first_request = 1;
  ExpectValidSampleGeneration(first_request);
  ExpectValidSampleGeneration(GetNumSamplesPerHop(kInternalSampleRateHz) -
                              first_request);
  EXPECT_FALSE(model_->GenerateSamples(1).has_value());
}

}  // namespace
}  // namespace codec
}  // namespace chromemedia
