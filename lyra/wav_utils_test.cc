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

#include "lyra/wav_utils.h"

#include <cstdint>
#include <string>
#include <vector>

// Placeholder for get runfiles header.
// Placeholder for testing header.
#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "gtest/gtest.h"
#include "include/ghc/filesystem.hpp"

namespace chromemedia::codec {
namespace {

class WavUtilTest : public testing::Test {
 protected:
  // Note that string & is used because ghc:filesystem::path has a compilation
  // error on platforms where absl::string_view is not aliased to
  // std::string_view.
  absl::StatusOr<ReadWavResult> ReadWav(const std::string& file_name) {
    const ghc::filesystem::path wav_path =
        ghc::filesystem::current_path() / "lyra/testdata" / file_name;
    return Read16BitWavFileToVector(wav_path.string());
  }
};

TEST_F(WavUtilTest, NonExistentWav) {
  absl::StatusOr<ReadWavResult> read_result = ReadWav("/should/not/exist.wav");

  EXPECT_FALSE(read_result.ok());
}

TEST_F(WavUtilTest, InvalidWav) {
  absl::StatusOr<ReadWavResult> read_result = ReadWav("invalid.wav");

  EXPECT_FALSE(read_result.ok());
}

TEST_F(WavUtilTest, ValidWav) {
  absl::StatusOr<ReadWavResult> read_result = ReadWav("sample1_16kHz.wav");

  EXPECT_TRUE(read_result.ok());
  EXPECT_EQ(read_result->num_channels, 1);
  EXPECT_EQ(read_result->sample_rate_hz, 16000);
  EXPECT_GT(read_result->samples.size(), 0);
}

TEST_F(WavUtilTest, WriteSingleChannelSamples) {
  // Create some interesting samples.
  std::vector<int16_t> samples(28, 496);
  samples[0] = 6;
  samples[27] = 8128;

  // Write the samples to disk.
  ghc::filesystem::path output_path =
      ghc::filesystem::path(testing::TempDir()) / "output.wav";
  absl::Status result =
      Write16BitWavFileFromVector(output_path, 1, 16000, samples);
  EXPECT_TRUE(result.ok());

  // Read the samples back in.
  absl::StatusOr<ReadWavResult> read_result = ReadWav(output_path);

  EXPECT_TRUE(read_result.ok());
  EXPECT_EQ(read_result->num_channels, 1);
  EXPECT_EQ(read_result->sample_rate_hz, 16000);
  EXPECT_EQ(read_result->samples.size(), 28);
  EXPECT_EQ(read_result->samples[0], 6);
  EXPECT_EQ(read_result->samples[1], 496);
  EXPECT_EQ(read_result->samples[27], 8128);
}

TEST_F(WavUtilTest, WriteToBadPath) {
  std::vector<int16_t> samples;
  absl::Status result =
      Write16BitWavFileFromVector("/invalid/path/test", 1, 16000, samples);
  EXPECT_FALSE(result.ok());
}

}  // namespace
}  // namespace chromemedia::codec
