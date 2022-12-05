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

#include "lyra/cli_example/encoder_main_lib.h"

#include <string>
#include <system_error>  // NOLINT(build/c++11)

// Placeholder for get runfiles header.
// Placeholder for testing header.
#include "absl/flags/flag.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"
#include "include/ghc/filesystem.hpp"

namespace chromemedia {
namespace codec {
namespace {

static constexpr absl::string_view kWavFiles[] = {
    "sample1_8kHz", "sample1_16kHz", "sample1_32kHz", "sample1_48kHz"};

static constexpr absl::string_view kTestdataDir = "lyra/testdata";

class EncoderMainLibTest : public testing::Test {
 protected:
  EncoderMainLibTest()
      : output_dir_(ghc::filesystem::path(testing::TempDir()) / "output"),
        testdata_dir_(ghc::filesystem::current_path() / kTestdataDir),
        model_path_(ghc::filesystem::current_path() / "lyra/model_coeffs") {}

  void SetUp() override {
    std::error_code error_code;
    ghc::filesystem::create_directories(output_dir_, error_code);
    ASSERT_FALSE(error_code);
  }

  void TearDown() override {
    std::error_code error_code;
    ghc::filesystem::remove_all(output_dir_, error_code);
    ASSERT_FALSE(error_code);
  }

  const ghc::filesystem::path output_dir_;
  const ghc::filesystem::path testdata_dir_;
  const ghc::filesystem::path model_path_;
};

TEST_F(EncoderMainLibTest, WavFileNotFound) {
  const ghc::filesystem::path kNonExistentWav("should/not/exist.wav");
  const ghc::filesystem::path kOutputEncoded(output_dir_ / "exists.lyra");

  EXPECT_FALSE(EncodeFile(kNonExistentWav, kOutputEncoded, /*bitrate=*/3200,
                          /*enable_preprocessing=*/false,
                          /*enable_dtx=*/false, model_path_));

  std::error_code error_code;
  EXPECT_FALSE(ghc::filesystem::is_regular_file(kOutputEncoded, error_code));
}

TEST_F(EncoderMainLibTest, EncodeSingleWavFiles) {
  for (const auto wav_file : kWavFiles) {
    const auto kInputWavepath = (testdata_dir_ / wav_file).concat(".wav");
    const auto kOutputEncoded = (output_dir_ / wav_file).concat(".lyra");
    EXPECT_TRUE(EncodeFile(kInputWavepath, kOutputEncoded, /*bitrate=*/3200,
                           /*enable_preprocessing=*/false,
                           /*enable_dtx=*/false, model_path_));
  }
}

}  // namespace
}  // namespace codec
}  // namespace chromemedia
