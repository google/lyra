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

#include "lyra/lyra_config.h"

#include <fstream>
#include <sstream>
#include <string>

// Placeholder for get runfiles header.
#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "gtest/gtest.h"
#include "include/ghc/filesystem.hpp"
#include "lyra/lyra_config.pb.h"

#include "lyra/model_coeffs/_models.h"

namespace chromemedia {
namespace codec {
namespace {

class LyraConfigTest : public testing::Test {
 protected:
  LyraConfigTest() : test_models_(GetEmbeddedLyraModels()) {}

  void SetUp() override {
    // Create a uniqe sub-directory so tests do not interfere with each other.
    const testing::TestInfo* const test_info =
        testing::UnitTest::GetInstance()->current_test_info();
  }

  void DeleteFile(const ghc::filesystem::path& to_delete) {
    ghc::filesystem::permissions(to_delete, ghc::filesystem::perms::owner_write,
                                 ghc::filesystem::perm_options::add,
                                 error_code_);
    ASSERT_FALSE(error_code_) << error_code_.message();
    ghc::filesystem::remove(to_delete, error_code_);
    ASSERT_FALSE(error_code_) << error_code_.message();
  }

  LyraModels test_models_;
  std::error_code error_code_;
};

TEST(LyraConfig, TestGetVersionString) {
  const std::string& version = GetVersionString();

  // Parse the period-separated version components.
  std::istringstream version_stream(version);
  std::string major_string, minor_string, micro_string;
  ASSERT_TRUE(std::getline(version_stream, major_string, '.'));
  ASSERT_TRUE(std::getline(version_stream, minor_string, '.'));
  ASSERT_TRUE(std::getline(version_stream, micro_string, '.'));
  ASSERT_TRUE(version_stream.eof());

  EXPECT_EQ(std::stoi(major_string), kVersionMajor);
  EXPECT_EQ(std::stoi(minor_string), kVersionMinor);
  EXPECT_EQ(std::stoi(micro_string), kVersionMicro);
}

TEST_F(LyraConfigTest, GoodPacketSizeSupported) {
  for (int num_quantized_bits : GetSupportedQuantizedBits()) {
    EXPECT_EQ(PacketSizeToNumQuantizedBits(GetPacketSize(num_quantized_bits)),
              num_quantized_bits);
  }
}

TEST_F(LyraConfigTest, GoodBitrateSupported) {
  for (int num_quantized_bits : GetSupportedQuantizedBits()) {
    EXPECT_EQ(BitrateToNumQuantizedBits(GetBitrate(num_quantized_bits)),
              num_quantized_bits);
  }
}

TEST_F(LyraConfigTest, BadPacketSizeNotSupported) {
  EXPECT_LT(PacketSizeToNumQuantizedBits(0), 0);
}

TEST_F(LyraConfigTest, BadBitrateNotSupported) {
  EXPECT_LT(BitrateToNumQuantizedBits(0), 0);
}

TEST_F(LyraConfigTest, GoodParamsSupported) {
  EXPECT_TRUE(
      AreParamsSupported(kInternalSampleRateHz, kNumChannels, test_models_)
          .ok());
}

TEST_F(LyraConfigTest, BadParamsNotSupported) {
  EXPECT_FALSE(
      AreParamsSupported(/*sample_rate_hz=*/137, kNumChannels, test_models_)
          .ok());

  EXPECT_FALSE(AreParamsSupported(kInternalSampleRateHz, /*num_channels=*/-1,
                                  test_models_)
                   .ok());
}

}  // namespace
}  // namespace codec
}  // namespace chromemedia
