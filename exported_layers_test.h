/*
 * Copyright 2021 Google LLC
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

#ifndef LYRA_CODEC_EXPORTED_LAYERS_TEST_H_
#define LYRA_CODEC_EXPORTED_LAYERS_TEST_H_

#include <algorithm>
#include <string>
#include <vector>

// Placeholder for get runfiles header.
#include "absl/random/random.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "include/ghc/filesystem.hpp"
#include "layer_wrappers_lib.h"
#include "lyra_types.h"
#include "sparse_matmul/sparse_matmul.h"

namespace chromemedia {
namespace codec {

template <typename LayerTypes>
class ExportedLayersTest : public testing::Test {
 protected:
  ExportedLayersTest()
      : model_path_(ghc::filesystem::current_path() / "wavegru"),
        params_(LayerTypes::Params(model_path_.string())),
        spin_barrier_(1) {}

  std::vector<float> GenerateRandomInput(const int size,
                                         const int exponent_bits) {
    std::vector<float> input(size);
    const float scale = static_cast<float>(1 << exponent_bits);
    for (auto& element : input) {
      // Note: the biggest source of error comes from clipping, so we keep the
      // input range small to avoid it.
      element = absl::Uniform<float>(gen_, -0.01, 0.01) * scale;
    }
    return input;
  }

  const ghc::filesystem::path model_path_;
  LayerParams params_;
  csrblocksparse::SpinBarrier spin_barrier_;
  absl::BitGen gen_;
};

TYPED_TEST_SUITE_P(ExportedLayersTest);

TYPED_TEST_P(ExportedLayersTest, FixedPointResultMatchesFloat) {
  using FloatLayerType = typename TypeParam::FloatLayerType;
  using FixedLayerType = typename TypeParam::FixedLayerType;
  using FixedInputType = typename TypeParam::FixedLayerType::Input;
  using FixedOutputType = typename TypeParam::FixedLayerType::Output;

  // Run the float layer.
  auto float_layer = FloatLayerType::Create(this->params_);
  std::vector<float> input_float =
      this->GenerateRandomInput(float_layer->InputViewToUpdate().rows() *
                                    float_layer->InputViewToUpdate().cols(),
                                FixedInputType::kExponentBits);
  ASSERT_NE(float_layer, nullptr);

  const int output_rows = float_layer->rows();
  const int output_cols = this->params_.length;
  std::vector<float> output_float(output_rows * output_cols);
  std::copy(input_float.begin(), input_float.end(),
            float_layer->InputViewToUpdate().data());
  float_layer->Run(0, &this->spin_barrier_,
                   csrblocksparse::MutableVectorView<float>(
                       output_float.data(), output_rows, output_cols));

  // Run again using the fixed-point layer.
  auto fixed_layer = FixedLayerType::Create(this->params_);
  ASSERT_NE(fixed_layer, nullptr);
  std::transform(input_float.begin(), input_float.end(),
                 fixed_layer->InputViewToUpdate().data(),
                 [](float x) { return FixedInputType(x); });
  std::vector<FixedOutputType> output_fixed(output_rows * output_cols);
  fixed_layer->Run(0, &this->spin_barrier_,
                   csrblocksparse::MutableVectorView<FixedOutputType>(
                       output_fixed.data(), output_rows, output_cols));

  // Compare, with a tolerance adaptive to the precision of the fixed-point
  // representation.
  const float tolerance = (1 << FixedOutputType::kExponentBits) * 5e-4f;
  EXPECT_THAT(std::vector<float>(output_fixed.begin(), output_fixed.end()),
              testing::Pointwise(testing::FloatNear(tolerance), output_float));
}

REGISTER_TYPED_TEST_SUITE_P(ExportedLayersTest, FixedPointResultMatchesFloat);

}  // namespace codec
}  // namespace chromemedia

#endif  // LYRA_CODEC_EXPORTED_LAYERS_TEST_H_
