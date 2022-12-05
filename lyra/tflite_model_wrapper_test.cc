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

#include "lyra/tflite_model_wrapper.h"

#include <algorithm>
#include <memory>
#include <string>

// Placeholder for get runfiles header.
#include "absl/types/span.h"
#include "gtest/gtest.h"
#include "include/ghc/filesystem.hpp"

namespace chromemedia {
namespace codec {
namespace {

TEST(TfLiteModelWrapperTest, CreateFailsWithInvalidModelFile) {
  EXPECT_EQ(TfLiteModelWrapper::Create("invalid/model/path", true, false),
            nullptr);
}

class TfLiteModelWrapperTest : public testing::TestWithParam<bool> {};

TEST_P(TfLiteModelWrapperTest, CreateSucceedsAndMethodsRun) {
  const bool int8_quantized = GetParam();
  auto model_wrapper = TfLiteModelWrapper::Create(
      ghc::filesystem::current_path() / "lyra/model_coeffs/lyragan.tflite",
      true, int8_quantized);
  ASSERT_NE(model_wrapper, nullptr);
  absl::Span<float> input = model_wrapper->get_input_tensor<float>(0);
  std::fill(input.begin(), input.end(), 0);
  EXPECT_TRUE(model_wrapper->Invoke());
  EXPECT_TRUE(model_wrapper->ResetVariableTensors());
}

INSTANTIATE_TEST_SUITE_P(Int8QuantizedOrNot, TfLiteModelWrapperTest,
                         testing::Bool());

}  // namespace
}  // namespace codec
}  // namespace chromemedia
