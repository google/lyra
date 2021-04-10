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

#include "lyra_config.h"

#include "gtest/gtest.h"

namespace chromemedia {
namespace codec {
namespace {

TEST(LyraConfigTest, TestGetVersionString) {
  const std::string& version = GetVersionString();

  // Parse the period-separated version components.
  std::istringstream version_stream(version);
  std::string majorStr, minorStr, microStr;
  EXPECT_TRUE(std::getline(version_stream, majorStr, '.'));
  EXPECT_TRUE(std::getline(version_stream, minorStr, '.'));
  EXPECT_TRUE(std::getline(version_stream, microStr, '.'));
  EXPECT_TRUE(version_stream.eof());

  EXPECT_EQ(std::stoi(majorStr), kVersionMajor);
  EXPECT_EQ(std::stoi(minorStr), kVersionMinor);
  EXPECT_EQ(std::stoi(microStr), kVersionMicro);
}

}  // namespace
}  // namespace codec
}  // namespace chromemedia
