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

#include "residual_vector_quantizer.h"

#include <cmath>
#include <memory>
#include <optional>
#include <string>
#include <vector>

// Placeholder for get runfiles header.
#include "gtest/gtest.h"
#include "include/ghc/filesystem.hpp"
#include "log_mel_spectrogram_extractor_impl.h"
#include "lyra_config.h"

namespace chromemedia {
namespace codec {
namespace {

class ResidualVectorQuantizerTest : public testing::TestWithParam<int> {
 protected:
  ResidualVectorQuantizerTest()
      : num_quantized_bits_(GetParam()),
        quantizer_(ResidualVectorQuantizer::Create(
            ghc::filesystem::current_path() / "model_coeffs")),
        // These features correspond to silence run through the
        // SoundStreamEncoder using the SoundStreamEncoderTest.
        features_{
            5.18127,   0.156109,  -0.875549, 1.90394,   4.27785,   0.184078,
            2.03794,   0.895547,  6.61436,   3.61373,   1.84045,   2.34979,
            1.91443,   2.46864,   2.49996,   -0.78883,  2.04522,   -0.0539977,
            -0.206427, -0.856873, 1.56033,   1.48176,   1.82138,   0.900604,
            -0.10602,  -0.548707, 0.33733,   7.63183,   -0.199688, 6.35543,
            2.47549,   -0.854709, 0.0588712, -0.144105, 7.68603,   2.78211,
            1.89553,   1.46111,   1.60068,   -0.310399, 1.4651,    2.05484,
            0.460265,  1.88702,   -0.186116, 0.134471,  -0.304016, 0.924312,
            9.56944,   0.877297,  0.825455,  2.45036,   2.36505,   1.02132,
            2.03803,   0.308894,  -0.930119, 3.16624,   -0.743392, 0.137643,
            2.01814,   3.39578,   4.30634,   0.880378} {}

  float FeatureDistance(const std::vector<float>& decoded_features) {
    float distance = 0.0;
    float energy = 0.0;
    for (int i = 0; i < features_.size(); ++i) {
      float d = features_[i] - decoded_features[i];
      distance += d * d;
      energy += features_[i] * features_[i];
    }
    return std::sqrt(distance / energy);
  }

  const int num_quantized_bits_;
  std::unique_ptr<ResidualVectorQuantizer> quantizer_;
  std::vector<float> features_;
};

TEST_P(ResidualVectorQuantizerTest, CreationFailsWithInvalidModelPath) {
  EXPECT_EQ(ResidualVectorQuantizer::Create("invalid/model/path"), nullptr);
}

TEST_P(ResidualVectorQuantizerTest, CreationSucceedsWithValidModelPath) {
  EXPECT_NE(quantizer_, nullptr);
}

TEST_P(ResidualVectorQuantizerTest, QuantizationFailsWithTooManyBits) {
  constexpr int kTooManyBits = 185;
  EXPECT_FALSE(quantizer_->Quantize(features_, kTooManyBits).has_value());
}

TEST_P(ResidualVectorQuantizerTest, QuantizationFailsWithNonDivisibleBits) {
  constexpr int kNonDivisibleBits = 62;
  EXPECT_FALSE(quantizer_->Quantize(features_, kNonDivisibleBits).has_value());
}

TEST_P(ResidualVectorQuantizerTest, DecodingFailsWithTooManyBits) {
  constexpr int kTooManyBits = 185;
  std::string toolong_quantized(kTooManyBits, '0');
  EXPECT_FALSE(
      quantizer_->DecodeToLossyFeatures(toolong_quantized).has_value());
}

TEST_P(ResidualVectorQuantizerTest, DecodingFailsWithNonDivisibleBits) {
  constexpr int kNonDivisibleBits = 62;
  std::string nondivisible_quantized(kNonDivisibleBits, '0');
  EXPECT_FALSE(
      quantizer_->DecodeToLossyFeatures(nondivisible_quantized).has_value());
}

TEST_P(ResidualVectorQuantizerTest, EncodeDecodeResultsInSimilarFeatures) {
  auto quantized = quantizer_->Quantize(features_, num_quantized_bits_);
  ASSERT_TRUE(quantized.has_value());
  auto decoded_features = quantizer_->DecodeToLossyFeatures(quantized.value());
  ASSERT_TRUE(decoded_features.has_value());
  EXPECT_EQ(decoded_features.value().size(), features_.size());
  EXPECT_LT(FeatureDistance(decoded_features.value()), 1.11);
}

INSTANTIATE_TEST_SUITE_P(NumQuantizedBits, ResidualVectorQuantizerTest,
                         testing::ValuesIn(GetSupportedQuantizedBits()));

}  // namespace
}  // namespace codec
}  // namespace chromemedia
