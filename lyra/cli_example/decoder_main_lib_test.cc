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

#include "lyra/cli_example/decoder_main_lib.h"

#include <string>
#include <system_error>  // NOLINT(build/c++11)
#include <vector>

// Placeholder for get runfiles header.
#include "absl/flags/flag.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"
#include "include/ghc/filesystem.hpp"
#include "lyra/lyra_config.h"
#include "lyra/wav_utils.h"

namespace chromemedia {
namespace codec {
namespace {

static constexpr absl::string_view kTestdataDir = "lyra/testdata";
static constexpr absl::string_view kExportedModelPath = "lyra/model_coeffs";

class DecoderMainLibTest : public testing::TestWithParam<int> {
 protected:
  DecoderMainLibTest()
      : output_dir_(ghc::filesystem::path(testing::TempDir()) / "output/"),
        testdata_dir_(ghc::filesystem::current_path() / kTestdataDir),
        model_path_(ghc::filesystem::current_path() / kExportedModelPath),
        sample_rate_hz_(GetParam()),
        num_samples_in_packet_(GetNumSamplesPerHop(sample_rate_hz_)) {}

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

  void SetInputOutputPath(const absl::string_view input_base_name) {
    input_path_ = testdata_dir_ / absl::StrCat(input_base_name, ".lyra");
    output_path_ =
        output_dir_ / absl::StrCat(input_base_name, "_", GetParam(), ".wav");
  }

  const ghc::filesystem::path output_dir_;
  const ghc::filesystem::path testdata_dir_;
  const ghc::filesystem::path model_path_;
  ghc::filesystem::path input_path_;
  ghc::filesystem::path output_path_;
  const int sample_rate_hz_;
  const int num_samples_in_packet_;
};

TEST_P(DecoderMainLibTest, NoEncodedPacket) {
  SetInputOutputPath("no_encoded_packet");
  EXPECT_FALSE(DecodeFile(
      input_path_, output_path_, sample_rate_hz_,
      /*bitrate=*/3200, /*randomize_num_samples_requested=*/false,
      /*packet_loss_rate=*/0.f,
      /*average_burst_length=*/1.f, PacketLossPattern({}, {}), model_path_));
}

TEST_P(DecoderMainLibTest, OneEncodedPacket) {
  SetInputOutputPath("one_encoded_packet_16khz");
  EXPECT_TRUE(DecodeFile(
      input_path_, output_path_, sample_rate_hz_,
      /*bitrate=*/6000, /*randomize_num_samples_requested=*/false,
      /*packet_loss_rate=*/0.f,
      /*average_burst_length=*/1.f, PacketLossPattern({}, {}), model_path_));

  EXPECT_EQ(NumSamplesInWavFile(output_path_), num_samples_in_packet_);
}

TEST_P(DecoderMainLibTest, RandomizeSampleRequests) {
  SetInputOutputPath("one_encoded_packet_16khz");
  EXPECT_TRUE(DecodeFile(
      input_path_, output_path_, sample_rate_hz_,
      /*bitrate=*/6000, /*randomize_num_samples_requested=*/true,
      /*packet_loss_rate=*/0.f,
      /*average_burst_length=*/1.f, PacketLossPattern({}, {}), model_path_));

  EXPECT_EQ(NumSamplesInWavFile(output_path_), num_samples_in_packet_);
}

TEST_P(DecoderMainLibTest, FileDoesNotExist) {
  SetInputOutputPath("non_existent");
  EXPECT_FALSE(DecodeFile(
      input_path_, output_path_, sample_rate_hz_,
      /*bitrate=*/6000, /*randomize_num_samples_requested=*/false,
      /*packet_loss_rate=*/0.f,
      /*average_burst_length=*/1.f, PacketLossPattern({}, {}), model_path_));
}

// Tests an encoded features file with less than 1 packet's worth of data.
TEST_P(DecoderMainLibTest, IncompleteEncodedPacket) {
  SetInputOutputPath("incomplete_encoded_packet");

  EXPECT_FALSE(DecodeFile(
      input_path_, output_path_, sample_rate_hz_,
      /*bitrate=*/6000, /*randomize_num_samples_requested=*/false,
      /*packet_loss_rate=*/0.f,
      /*average_burst_length=*/1.f, PacketLossPattern({}, {}), model_path_));
}

TEST_P(DecoderMainLibTest, TwoEncodedPacketsWithPacketLoss) {
  SetInputOutputPath("two_encoded_packets_16khz");
  const int expected_num_samples = 2 * num_samples_in_packet_;

  EXPECT_TRUE(DecodeFile(
      input_path_, output_path_, sample_rate_hz_,
      /*bitrate=*/6000, /*randomize_num_samples_requested=*/false,
      /*packet_loss_rate=*/0.5f,
      /*average_burst_length=*/2.f, PacketLossPattern({}, {}), model_path_));
  EXPECT_EQ(NumSamplesInWavFile(output_path_), expected_num_samples);

  EXPECT_TRUE(DecodeFile(
      input_path_, output_path_, sample_rate_hz_,
      /*bitrate=*/6000, /*randomize_num_samples_requested=*/false,
      /*packet_loss_rate=*/0.9f,
      /*average_burst_length=*/10.f, PacketLossPattern({}, {}), model_path_));
  EXPECT_EQ(NumSamplesInWavFile(output_path_), expected_num_samples);
}

TEST_P(DecoderMainLibTest, TwoEncodedPacketsWithFixedPacketLoss) {
  SetInputOutputPath("two_encoded_packets_16khz");
  const int expected_num_samples = 2 * num_samples_in_packet_;

  EXPECT_TRUE(DecodeFile(
      input_path_, output_path_, sample_rate_hz_,
      /*bitrate=*/6000, /*randomize_num_samples_requested=*/false,
      /*packet_loss_rate=*/0.9f,
      /*average_burst_length=*/10.f, PacketLossPattern({1}, {0}), model_path_));
  EXPECT_EQ(NumSamplesInWavFile(output_path_), expected_num_samples);

  EXPECT_TRUE(DecodeFile(input_path_, output_path_, sample_rate_hz_,
                         /*bitrate=*/6000,
                         /*randomize_num_samples_requested=*/false,
                         /*packet_loss_rate=*/0.9f,
                         /*average_burst_length=*/10.f,
                         PacketLossPattern({0}, {100}), model_path_));
  EXPECT_EQ(NumSamplesInWavFile(output_path_), expected_num_samples);
}

INSTANTIATE_TEST_SUITE_P(SampleRates, DecoderMainLibTest,
                         testing::ValuesIn(kSupportedSampleRates));

}  // namespace
}  // namespace codec
}  // namespace chromemedia
