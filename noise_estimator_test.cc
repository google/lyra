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

#include "noise_estimator.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

#include "absl/types/optional.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace chromemedia {
namespace codec {
namespace {

static constexpr float kNumSecondsPerFrame = 0.025f;
static const int kNumFramesPerSecond = std::round(1 / kNumSecondsPerFrame);
static constexpr float kMaxAbsError = 1e-1f;
static constexpr int kTestNumFeatures = 160;
static constexpr float kBoundHalfLifeSec = 1.f;
static constexpr int kNumSeconds = 4;

class NoiseEstimatorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    noise_estimator_ =
        NoiseEstimator::Create(kTestNumFeatures, kNumSecondsPerFrame);
    ASSERT_NE(noise_estimator_, nullptr);
  }

  // Sparsely populates a vector with uniform power values.
  std::vector<float> RandomSignal(const std::vector<float>& noise) {
    // Each frequency bin has a 1 in 10 probability of containing a uniform
    // signal.
    const int kSignalProbability = 10;
    const float kSignalPower = 1;
    std::vector<float> signal(noise);
    for (auto& sample : signal) {
      int signal_prob = rand_r(&seed_) % kSignalProbability;
      if (signal_prob == 0) {
        sample = kSignalPower;
      }
    }
    return signal;
  }

  // Adds a small amount of variability to the base noise.
  std::vector<float> RandomNoise(const std::vector<float>& base_noise) {
    const float kNoiseVariability = 1e-1f;
    std::vector<float> noise(base_noise);
    for (auto& sample : noise) {
      sample += -kNoiseVariability +
                static_cast<float>(rand_r(&seed_)) /
                    (static_cast<float>(RAND_MAX) / (2.f * kNoiseVariability));
    }
    return noise;
  }

  // Noise approximated by a line across frequency bins.
  std::vector<float> BaseNoise() {
    const float kNoiseLowerBound = -2.f;
    const float kNoiseUpperBound = -1.f;
    std::vector<float> noise(kTestNumFeatures);
    // Approximate the base noise with a line.
    float rise = (kNoiseLowerBound - kNoiseUpperBound) / kTestNumFeatures;
    for (int i = 0; i < noise.size(); i++) {
      noise.at(i) = rise * i + kNoiseUpperBound;
    }
    return noise;
  }

  std::unique_ptr<NoiseEstimator> noise_estimator_;
  uint seed_ = 1;
};

TEST_F(NoiseEstimatorTest, ThreeSecondsEstimate) {
  const std::vector<float> kBaseNoise = BaseNoise();
  std::vector<float> noise = RandomNoise(kBaseNoise);

  ASSERT_TRUE(noise_estimator_->Update(noise));
  // Do not add the uniform signal to the first frame.
  for (int i = 1; i < kNumSeconds * kNumFramesPerSecond; ++i) {
    noise = RandomNoise(kBaseNoise);
    std::vector<float> signal = RandomSignal(noise);
    ASSERT_TRUE(noise_estimator_->Update(signal));
  }

  EXPECT_THAT(noise_estimator_->NoiseEstimate(),
              testing::Pointwise(testing::FloatNear(kMaxAbsError), kBaseNoise));
}

TEST_F(NoiseEstimatorTest, NoiseIdentification) {
  const std::vector<float> kBaseNoise = BaseNoise();

  const std::vector<float> wrong_size_vector(kTestNumFeatures - 1);
  EXPECT_FALSE(noise_estimator_->IsSimilarNoise(wrong_size_vector).has_value());

  // Run noise through estimator first.
  for (int i = 1; i < kNumSeconds * kNumFramesPerSecond; ++i) {
    const std::vector<float> noise = RandomNoise(kBaseNoise);
    ASSERT_TRUE(noise_estimator_->Update(noise));
  }

  EXPECT_TRUE(
      noise_estimator_->IsSimilarNoise(RandomNoise(kBaseNoise)).value());
  EXPECT_FALSE(
      noise_estimator_->IsSimilarNoise(RandomSignal(RandomNoise(kBaseNoise)))
          .value());
}

TEST_F(NoiseEstimatorTest, BoundsDecay) {
  const std::vector<float> kBaseNoise = BaseNoise();

  // Run noise through estimator first.
  for (int i = 1; i < kNumSeconds * kNumFramesPerSecond; ++i) {
    const std::vector<float> noise = RandomNoise(kBaseNoise);
    ASSERT_TRUE(noise_estimator_->Update(noise));
  }

  const float kOneHalfLifeFrames = kBoundHalfLifeSec / kNumSecondsPerFrame;
  auto similar_noise = RandomNoise(kBaseNoise);
  EXPECT_TRUE(noise_estimator_->IsSimilarNoise(similar_noise).value());

  // Re-query for one half-life
  for (int i = 0; i < kOneHalfLifeFrames; ++i) {
    ASSERT_TRUE(noise_estimator_->IsSimilarNoise(similar_noise).has_value());
  }

  EXPECT_FALSE(noise_estimator_->IsSimilarNoise(similar_noise).value());
}

}  // namespace
}  // namespace codec
}  // namespace chromemedia
