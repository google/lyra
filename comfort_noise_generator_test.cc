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

#include "comfort_noise_generator.h"

#include <memory>
#include <vector>

#include "absl/types/optional.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace chromemedia {
namespace codec {
namespace {

using ::testing::Each;
using ::testing::Ne;
using ::testing::Optional;
using ::testing::SizeIs;

static constexpr int kTestSampleRate = 16000;
static constexpr int kTestNumFeatures = 3;
static constexpr int kTestWindowLengthSamples = 10;
static constexpr int kTestHopLengthSamples = 5;

TEST(ComfortNoiseGeneratorTest, NumSamplesRequestedOutOfBounds) {
  auto comfort_noise_generator = ComfortNoiseGenerator::Create(
      kTestSampleRate, kTestNumFeatures, kTestWindowLengthSamples,
      kTestHopLengthSamples);

  std::vector<float> features(kTestNumFeatures, 0.0);
  comfort_noise_generator->AddFeatures(features);
  EXPECT_EQ(comfort_noise_generator->GenerateSamples(kTestHopLengthSamples + 1),
            absl::nullopt);
  EXPECT_EQ(comfort_noise_generator->GenerateSamples(-1), absl::nullopt);
  // Confirm that a request for 0 samples returns an empty vector.
  EXPECT_THAT(comfort_noise_generator->GenerateSamples(0), Optional(SizeIs(0)));
}

TEST(ComfortNoiseGeneratorTest, SamplesGeneratedOnlyWithCorrectNumFeatures) {
  auto comfort_noise_generator = ComfortNoiseGenerator::Create(
      kTestSampleRate, kTestNumFeatures, kTestWindowLengthSamples,
      kTestHopLengthSamples);

  const int kNumRequestedSamples = kTestHopLengthSamples;

  std::vector<float> features;
  comfort_noise_generator->AddFeatures(features);
  EXPECT_EQ(comfort_noise_generator->GenerateSamples(kNumRequestedSamples),
            absl::nullopt);

  features.assign(kTestNumFeatures - 1, 1.0);
  comfort_noise_generator->AddFeatures(features);
  EXPECT_EQ(comfort_noise_generator->GenerateSamples(kNumRequestedSamples),
            absl::nullopt);

  features.assign(kTestNumFeatures + 1, 1.0);
  comfort_noise_generator->AddFeatures(features);
  EXPECT_EQ(comfort_noise_generator->GenerateSamples(kNumRequestedSamples),
            absl::nullopt);

  features.assign(kTestNumFeatures, 1.0);
  comfort_noise_generator->AddFeatures(features);
  EXPECT_NE(comfort_noise_generator->GenerateSamples(kNumRequestedSamples),
            absl::nullopt);
}

TEST(ComfortNoiseGeneratorTest, BufferGetsClearedCorrectly) {
  auto comfort_noise_generator = ComfortNoiseGenerator::Create(
      kTestSampleRate, kTestNumFeatures, kTestWindowLengthSamples,
      kTestHopLengthSamples);

  std::vector<float> features(kTestNumFeatures, 0.0);
  comfort_noise_generator->AddFeatures(features);
  ASSERT_NE(comfort_noise_generator->GenerateSamples(kTestHopLengthSamples),
            absl::nullopt);
  comfort_noise_generator->Reset();
  // A call to generate samples should fail after a reset.
  EXPECT_EQ(comfort_noise_generator->GenerateSamples(kTestHopLengthSamples),
            absl::nullopt);
}

TEST(ComfortNoiseGeneratorTest, BasicUseCaseSucceeds) {
  auto comfort_noise_generator = ComfortNoiseGenerator::Create(
      kTestSampleRate, kTestNumFeatures, kTestWindowLengthSamples,
      kTestHopLengthSamples);

  const int kNumRequestedSamples = kTestHopLengthSamples;

  // If all features are equal to 0 then all generated samples should be equal
  // to 0 as well.
  std::vector<float> features(kTestNumFeatures, 0.0);
  comfort_noise_generator->AddFeatures(features);
  auto generated_samples =
      comfort_noise_generator->GenerateSamples(kNumRequestedSamples);
  ASSERT_TRUE(generated_samples.has_value());
  EXPECT_THAT(generated_samples.value(), Each(0.0));

  // Test that adding features overwrites previous ones. It is only possible
  // that the newly produced samples equal the previously produced samples if
  // the features all stayed at 0.0.
  features.assign(kTestNumFeatures, 1.0);
  comfort_noise_generator->AddFeatures(features);
  auto previously_generated_samples = generated_samples;
  generated_samples =
      comfort_noise_generator->GenerateSamples(kNumRequestedSamples);
  ASSERT_TRUE(generated_samples.has_value());
  EXPECT_THAT(generated_samples.value(), Ne(previously_generated_samples));

  // Test that GenerateSamples() can be called multiple times in succession
  // without a call to AddFeatures() and produces different samples than the
  // previous call.
  const int kNumTimesToCall = 10;
  for (int i = 0; i < kNumTimesToCall; ++i) {
    previously_generated_samples = generated_samples;
    generated_samples =
        comfort_noise_generator->GenerateSamples(kNumRequestedSamples);
    ASSERT_TRUE(generated_samples.has_value());
    EXPECT_THAT(generated_samples.value(),
                Ne(previously_generated_samples.value()));
  }
}

}  // namespace
}  // namespace codec
}  // namespace chromemedia
