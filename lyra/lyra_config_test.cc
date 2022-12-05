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

namespace chromemedia {
namespace codec {
namespace {

class LyraConfigTest : public testing::Test {
 protected:
  LyraConfigTest()
      : source_model_path_(ghc::filesystem::current_path() /
                           "lyra/model_coeffs") {}

  void SetUp() override {
    // Create a uniqe sub-directory so tests do not interfere with each other.
    const testing::TestInfo* const test_info =
        testing::UnitTest::GetInstance()->current_test_info();
    test_model_path_ =
        ghc::filesystem::path(testing::TempDir()) / test_info->name();
    ghc::filesystem::create_directory(test_model_path_, error_code_);
    ASSERT_FALSE(error_code_) << error_code_.message();
    ghc::filesystem::permissions(
        test_model_path_, ghc::filesystem::perms::owner_write,
        ghc::filesystem::perm_options::add, error_code_);
    ASSERT_FALSE(error_code_) << error_code_.message();

    // Copy model files.
    ghc::filesystem::copy(source_model_path_, test_model_path_,
                          ghc::filesystem::copy_options::overwrite_existing |
                              ghc::filesystem::copy_options::recursive,
                          error_code_);
    ASSERT_FALSE(error_code_) << error_code_.message();
  }

  void DeleteFile(const ghc::filesystem::path& to_delete) {
    ghc::filesystem::permissions(to_delete, ghc::filesystem::perms::owner_write,
                                 ghc::filesystem::perm_options::add,
                                 error_code_);
    ASSERT_FALSE(error_code_) << error_code_.message();
    ghc::filesystem::remove(to_delete, error_code_);
    ASSERT_FALSE(error_code_) << error_code_.message();
  }

  // Folder containing files from |source_model_path_| with some modifications
  // to simulate various run-time conditions, e.g. missing some files, having
  // mismatched files.
  ghc::filesystem::path test_model_path_;
  std::error_code error_code_;

 private:
  const ghc::filesystem::path source_model_path_;
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
      AreParamsSupported(kInternalSampleRateHz, kNumChannels, test_model_path_)
          .ok());
}

TEST_F(LyraConfigTest, BadParamsNotSupported) {
  EXPECT_FALSE(
      AreParamsSupported(/*sample_rate_hz=*/137, kNumChannels, test_model_path_)
          .ok());

  EXPECT_FALSE(AreParamsSupported(kInternalSampleRateHz, /*num_channels=*/-1,
                                  test_model_path_)
                   .ok());
}

TEST_F(LyraConfigTest, MissingAssetNotSupported) {
  DeleteFile(test_model_path_ / "soundstream_encoder.tflite");
  EXPECT_FALSE(
      AreParamsSupported(kInternalSampleRateHz, kNumChannels, test_model_path_)
          .ok());
}

TEST_F(LyraConfigTest, MismatchedIdentifierNotSupported) {
  // Replace the lyra_condfig.binarypb with one that contains an identifier
  // that does not match |kVersionMinor|.
  const ghc::filesystem::path lyra_config_proto_path =
      test_model_path_ / "lyra_config.binarypb";
  DeleteFile(lyra_config_proto_path);
  third_party::lyra_codec::lyra::LyraConfig lyra_config_proto;
  lyra_config_proto.set_identifier(kVersionMinor + 100);
  std::ofstream output_proto(lyra_config_proto_path.string(),
                             std::ofstream::out | std::ofstream::binary);
  ASSERT_TRUE(output_proto.is_open());
  ASSERT_TRUE(lyra_config_proto.SerializeToOstream(&output_proto));
  output_proto.close();

  EXPECT_FALSE(
      AreParamsSupported(kInternalSampleRateHz, kNumChannels, test_model_path_)
          .ok());
}

}  // namespace
}  // namespace codec
}  // namespace chromemedia
