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

#include "no_op_preprocessor.h"

#include <numeric>

#include "gtest/gtest.h"

namespace chromemedia {
namespace codec {
namespace {

TEST(NoOpPreprocessorTest, IntOutputIsCopy) {
  static constexpr int kNumSamples = 640;
  static constexpr int kSampleRateHz = 16000;
  std::vector<int16_t> input(kNumSamples);
  std::iota(input.begin(), input.end(), -100);

  NoOpPreprocessor no_op_preprocessor;
  std::vector<int16_t> output = no_op_preprocessor.Process(
      absl::MakeConstSpan(input.data(), input.size()), kSampleRateHz);
  ASSERT_EQ(input, output);
}

}  // namespace
}  // namespace codec
}  // namespace chromemedia
