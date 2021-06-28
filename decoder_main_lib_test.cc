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

#include "decoder_main_lib.h"

#include <cstdio>
#include <memory>
#include <string>
#include <system_error>  // NOLINT(build/c++11)
#include <tuple>

// Placeholder for get runfiles header.
#include "gmock/gmock.h"
// Placeholder for testing header.
#include "absl/flags/flag.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"
#include "include/ghc/filesystem.hpp"
#include "lyra_config.h"
#include "wav_util.h"

namespace chromemedia {
namespace codec {
namespace {

static constexpr absl::string_view kTestdataDir = "testdata";
static constexpr absl::string_view kExportedModelPath = "wavegru";

class DecoderMainLibTest : public testing::TestWithParam<int> {
 protected:
  DecoderMainLibTest()
      : output_dir_(ghc::filesystem::path(testing::TempDir()) / "output/"),
        testdata_dir_(ghc::filesystem::current_path() / kTestdataDir),
        model_path_(ghc::filesystem::current_path() / kExportedModelPath),
        sample_rate_hz_(GetParam()),
        num_samples_in_packet_(kNumFramesPerPacket *
                               GetNumSamplesPerHop(sample_rate_hz_)) {}

  void SetUp() override {
    std::error_code error_code;
    ghc::filesystem::create_directories(output_dir_, error_code);
    ASSERT_FALSE(error_code);
  }

  int NumSamplesInWavFile(const ghc::filesystem::path output_path) {
    absl::StatusOr<ReadWavResult> read_result =
        Read16BitWavFileToVector(output_path);
    if (read_result.ok()) {
      return read_result->samples.size() / read_result->num_channels;
    }
    return 0;
  }

  const ghc::filesystem::path output_dir_;
  const ghc::filesystem::path testdata_dir_;
  const ghc::filesystem::path model_path_;
  const int sample_rate_hz_;
  const int num_samples_in_packet_;
};

TEST_P(DecoderMainLibTest, NoEncodedPacket) {
  const std::string kInputBaseName = "no_encoded_frames";
  const auto input_filepath = testdata_dir_ / kInputBaseName;
  const auto output_filepath =
      output_dir_ / absl::StrCat(kInputBaseName, "_", GetParam(), ".wav");

  EXPECT_FALSE(DecodeFile(input_filepath, output_filepath, sample_rate_hz_,
                          /*packet_loss_rate=*/0.f,
                          /*average_burst_length=*/1.f, model_path_));
}

TEST_P(DecoderMainLibTest, OneEncodedPacket) {
  const std::string kInputBaseName = "one_encoded_frame_16khz";
  const auto input_filepath = testdata_dir_ / kInputBaseName;
  const auto output_filepath =
      output_dir_ / absl::StrCat(kInputBaseName, "_", GetParam(), ".wav");
  EXPECT_TRUE(DecodeFile(input_filepath, output_filepath, sample_rate_hz_,
                         /*packet_loss_rate=*/0.f,
                         /*average_burst_length=*/1.f, model_path_));

  EXPECT_EQ(NumSamplesInWavFile(output_filepath), num_samples_in_packet_);
}

TEST_P(DecoderMainLibTest, FileDoesNotExist) {
  const std::string kInputBaseName = "non_existent";
  const auto input_filepath = testdata_dir_ / kInputBaseName;
  const auto output_filepath =
      output_dir_ / absl::StrCat(kInputBaseName, "_", GetParam(), ".wav");

  EXPECT_FALSE(DecodeFile(input_filepath, output_filepath, sample_rate_hz_,
                          /*packet_loss_rate=*/0.f,
                          /*average_burst_length=*/1.f, model_path_));
}

// Tests an encoded features file with less than 1 frame's worth of data.
TEST_P(DecoderMainLibTest, IncompleteEncodedFrame) {
  const std::string kInputBaseName = "incomplete_encoded_frame";
  const auto input_filepath = testdata_dir_ / kInputBaseName;
  const auto output_filepath =
      output_dir_ / absl::StrCat(kInputBaseName, "_", GetParam(), ".wav");

  EXPECT_FALSE(DecodeFile(input_filepath, output_filepath, sample_rate_hz_,
                          /*packet_loss_rate=*/0.f,
                          /*average_burst_length=*/1.f, model_path_));
}

TEST_P(DecoderMainLibTest, TwoEncodedFramesWithPacketLoss) {
  const std::string kInputBaseName = "two_encoded_frames_16khz";
  const auto input_filepath = testdata_dir_ / (kInputBaseName + ".lyra");
  const auto output_filepath =
      output_dir_ / absl::StrCat(kInputBaseName, "_", GetParam(), ".wav");
  const int expected_num_samples = 2 * num_samples_in_packet_;

  EXPECT_TRUE(DecodeFile(input_filepath, output_filepath, sample_rate_hz_,
                         /*packet_loss_rate=*/0.5f,
                         /*average_burst_length=*/2.f, model_path_));
  EXPECT_EQ(NumSamplesInWavFile(output_filepath), expected_num_samples);

  EXPECT_TRUE(DecodeFile(input_filepath, output_filepath, sample_rate_hz_,
                         /*packet_loss_rate=*/0.9f,
                         /*average_burst_length=*/10.f, model_path_));
  EXPECT_EQ(NumSamplesInWavFile(output_filepath), expected_num_samples);
}

INSTANTIATE_TEST_SUITE_P(SampleRates, DecoderMainLibTest,
                         testing::ValuesIn(kSupportedSampleRates));

}  // namespace
}  // namespace codec
}  // namespace chromemedia
