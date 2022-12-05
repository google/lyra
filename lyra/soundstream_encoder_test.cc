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

#include "lyra/soundstream_encoder.h"

#include <cstdint>
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

class SoundStreamEncoderTest : public testing::Test {
 protected:
  SoundStreamEncoderTest()
      : encoder_(SoundStreamEncoder::Create(ghc::filesystem::current_path() /
                                            "lyra/model_coeffs")) {}

  std::unique_ptr<SoundStreamEncoder> encoder_;
};

TEST_F(SoundStreamEncoderTest, CreationFailsWithInvalidModelPath) {
  EXPECT_EQ(SoundStreamEncoder::Create("invalid/model/path"), nullptr);
}

TEST_F(SoundStreamEncoderTest, CreationSucceedsWithValidModelPath) {
  EXPECT_NE(encoder_, nullptr);
}

TEST_F(SoundStreamEncoderTest, ExtractingExpectedNumberOfFeatures) {
  EXPECT_NE(encoder_, nullptr);
  std::vector<int16_t> audio(GetNumSamplesPerHop(kInternalSampleRateHz), 0);
  auto features = encoder_->Extract(audio);
  ASSERT_TRUE(features.has_value());
  EXPECT_EQ(features.value().size(), kNumFeatures);
}

}  // namespace
}  // namespace codec
}  // namespace chromemedia
