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

#include "lyra/comfort_noise_generator.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "lyra/dsp_utils.h"
#include "lyra/log_mel_spectrogram_extractor_impl.h"

namespace chromemedia {
namespace codec {
namespace {

using ::testing::Each;
using ::testing::Optional;
using ::testing::SizeIs;

static constexpr int kTestSampleRate = 16000;
static constexpr int kTestNumFeatures = 160;
static constexpr int kTestWindowLengthSamples = 640;
static constexpr int kTestHopLengthSamples = 320;

TEST(ComfortNoiseGeneratorTest, NumSamplesRequestedOutOfBounds) {
  auto comfort_noise_generator =
      ComfortNoiseGenerator::Create(kTestSampleRate, kTestHopLengthSamples,
                                    kTestWindowLengthSamples, kTestNumFeatures);

  std::vector<float> features(kTestNumFeatures, 0.0);
  comfort_noise_generator->AddFeatures(features);
  EXPECT_EQ(comfort_noise_generator->GenerateSamples(kTestHopLengthSamples + 1),
            std::nullopt);
  EXPECT_EQ(comfort_noise_generator->GenerateSamples(-1), std::nullopt);
  // Confirm that a request for 0 samples returns an empty vector.
  EXPECT_THAT(comfort_noise_generator->GenerateSamples(0), Optional(SizeIs(0)));
}

TEST(ComfortNoiseGeneratorTest, SamplesGeneratedOnlyWithCorrectNumFeatures) {
  auto comfort_noise_generator =
      ComfortNoiseGenerator::Create(kTestSampleRate, kTestHopLengthSamples,
                                    kTestWindowLengthSamples, kTestNumFeatures);

  const int kNumRequestedSamples = kTestHopLengthSamples;

  std::vector<float> features;
  comfort_noise_generator->AddFeatures(features);
  EXPECT_EQ(comfort_noise_generator->GenerateSamples(kNumRequestedSamples),
            std::nullopt);

  features.assign(kTestNumFeatures - 1, 1.0);
  comfort_noise_generator->AddFeatures(features);
  EXPECT_EQ(comfort_noise_generator->GenerateSamples(kNumRequestedSamples),
            std::nullopt);

  features.assign(kTestNumFeatures + 1, 1.0);
  comfort_noise_generator->AddFeatures(features);
  EXPECT_EQ(comfort_noise_generator->GenerateSamples(kNumRequestedSamples),
            std::nullopt);

  features.assign(kTestNumFeatures, 1.0);
  comfort_noise_generator->AddFeatures(features);
  EXPECT_NE(comfort_noise_generator->GenerateSamples(kNumRequestedSamples),
            std::nullopt);
}

TEST(ComfortNoiseGeneratorTest, BasicUseCaseSucceeds) {
  auto comfort_noise_generator =
      ComfortNoiseGenerator::Create(kTestSampleRate, kTestHopLengthSamples,
                                    kTestWindowLengthSamples, kTestNumFeatures);

  const int kNumRequestedSamples = kTestHopLengthSamples;

  // If features have no energy neither will the output samples.
  std::vector<float> features(kTestNumFeatures, 0.0);
  ASSERT_TRUE(comfort_noise_generator->AddFeatures(features));
  auto generated_samples =
      comfort_noise_generator->GenerateSamples(kNumRequestedSamples);
  ASSERT_TRUE(generated_samples.has_value());
  EXPECT_THAT(generated_samples.value(), Each(0.0));
}

TEST(ComfortNoiseGeneratorTest, GeneratedNoiseHasSimilarFeatures) {
  // Since log-mel-spectrogram extractors are stateful, it is necessary to
  // create separate ones for input and output.
  auto input_extractor = LogMelSpectrogramExtractorImpl::Create(
      kTestSampleRate, kTestHopLengthSamples, kTestWindowLengthSamples,
      kTestNumFeatures);
  auto output_extractor = LogMelSpectrogramExtractorImpl::Create(
      kTestSampleRate, kTestHopLengthSamples, kTestWindowLengthSamples,
      kTestNumFeatures);
  auto noise_generator =
      ComfortNoiseGenerator::Create(kTestSampleRate, kTestHopLengthSamples,
                                    kTestWindowLengthSamples, kTestNumFeatures);

  std::mt19937 gen(1);
  std::uniform_int_distribution<int16_t> prob(-10000, 10000);
  std::vector<int16_t> input_samples(kTestHopLengthSamples);
  for (int i = 0; i < kTestHopLengthSamples; ++i) {
    input_samples.at(i) = prob(gen);
  }

  std::vector<float> last_input_features;
  std::vector<float> last_output_features;
  const int kNumTimesToCall = 10;
  for (int i = 0; i < kNumTimesToCall; ++i) {
    auto input_features = input_extractor->Extract(input_samples);
    ASSERT_TRUE(input_features.has_value());
    last_input_features = input_features.value();
    noise_generator->AddFeatures(last_input_features);
    auto output_samples =
        noise_generator->GenerateSamples(kTestHopLengthSamples);
    ASSERT_TRUE(output_samples.has_value());
    auto output_features = output_extractor->Extract(output_samples.value());
    ASSERT_TRUE(output_features.has_value());
    last_output_features = output_features.value();
  }
  auto spectral_distance =
      LogSpectralDistance(last_input_features, last_output_features);
  ASSERT_TRUE(spectral_distance.has_value());
  EXPECT_LT(spectral_distance.value(), 0.7f);
}

}  // namespace
}  // namespace codec
}  // namespace chromemedia
