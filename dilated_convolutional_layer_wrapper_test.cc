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

#include "dilated_convolutional_layer_wrapper.h"

#include <algorithm>
#include <string>
#include <vector>

// placeholder for get runfiles header.
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "include/ghc/filesystem.hpp"
#include "layer_wrapper.h"
#include "layer_wrapper_test_common.h"
#include "sparse_inference_matrixvector.h"

namespace chromemedia {
namespace codec {

namespace {

static constexpr char kDilatedLayerPrefix[] = "lyra_conditioning_stack_2_";

template <typename ComputeType>
class DilatedConvolutionalLayerWrapperTest : public ::testing::Test {
 public:
  DilatedConvolutionalLayerWrapperTest()
      : testdata_dir_(ghc::filesystem::current_path() /
                      "testdata"),
        dilated_params_{.num_input_channels = kNumCondHidden,
                        .num_filters = kNumCondHidden,
                        .length = 1,
                        .kernel_size = kDilatedKernel,
                        .dilation = kDilation,
                        .stride = 1,
                        .relu = false,
                        .skip_connection = true,
                        .type = LayerType::kDilated,
                        .num_threads = kNumThreads,
                        .per_column_barrier = false,
                        .from =
                            LayerParams::FromDisk{
                                .path = this->testdata_dir_.string(),
                                .zipped = true,
                            },
                        .prefix = kDilatedLayerPrefix},
        spin_barrier_(kNumThreads) {}

 protected:
  using OutputType = typename LayerWrapperPeer<ComputeType>::OutputType;
  using RhsType = typename LayerWrapperPeer<ComputeType>::RhsType;

  const int kDilation = 4;
  const int kDilatedKernel = 2;
  const int kNumCondHidden = 8;
  const int kNumThreads = 1;
  const ghc::filesystem::path testdata_dir_;
  const LayerParams dilated_params_;

  csrblocksparse::SpinBarrier spin_barrier_;
  csrblocksparse::FatCacheAlignedVector<OutputType> output_buffer_;
};

using ComputeTypes = ::testing::Types<float, csrblocksparse::fixed16_type>;
TYPED_TEST_SUITE(DilatedConvolutionalLayerWrapperTest, ComputeTypes);

TYPED_TEST(DilatedConvolutionalLayerWrapperTest, CreateSucceeds) {
  EXPECT_NE(LayerWrapperPeer<TypeParam>::Create(this->dilated_params_),
            nullptr);
}

TYPED_TEST(DilatedConvolutionalLayerWrapperTest,
           CreateWithBadParamsReturnNullptr) {
  LayerParams invalid_file_params(this->dilated_params_);
  invalid_file_params.prefix = "does_not_exist";
  EXPECT_EQ(LayerWrapperPeer<TypeParam>::Create(invalid_file_params), nullptr);

  // |num_filters| should be equal to the number of rows of the loaded
  // weight matrix.
  LayerParams incompatible_layer_params(this->dilated_params_);
  incompatible_layer_params.num_filters += 1;
  EXPECT_EQ(LayerWrapperPeer<TypeParam>::Create(incompatible_layer_params),
            nullptr);

  // |stride| > 1 is not supported for dilated convolutional layers.
  LayerParams stride_2_params(this->dilated_params_);
  stride_2_params.stride = 2;
  EXPECT_EQ(LayerWrapperPeer<TypeParam>::Create(stride_2_params), nullptr);
}

TYPED_TEST(DilatedConvolutionalLayerWrapperTest,
           ResetCyclesThroughInputBuffer) {
  auto layer = LayerWrapperPeer<TypeParam>::Create(this->dilated_params_);

  // Populate the input buffer with |kDilation| columns of numbers and
  // then call Reset() for each block.
  using RhsType = typename LayerWrapperPeer<TypeParam>::RhsType;
  for (int i = 0; i < this->kDilation; ++i) {
    auto input_view = layer->InputViewToUpdate();
    std::fill_n(input_view.data(), input_view.rows() * input_view.cols(),
                static_cast<RhsType>(static_cast<float>(i)));
    layer->Reset(0, &this->spin_barrier_);
  }

  // Verify that the input buffer looks like this:
  // | v0 | v1 | v2 | v3 |
  // | x  | x  | x  | x  |
  // for |kernel_size| = 2 and |dilation| = 4, where "v0" means a
  // column of |num_input_channels| elements with all 0s, "v1" with all 1s,
  // and so on, and "x" means values that we do not care.
  const auto input_buffer = layer->input_buffer();
  const int num_elements = this->dilated_params_.num_input_channels;
  for (int i = 0; i < this->dilated_params_.dilation; ++i) {
    std::vector<float> block_i(num_elements);
    for (int j = 0; j < num_elements; ++j) {
      block_i[j] =
          static_cast<float>(input_buffer[i * input_buffer.rows() + j]);
    }
    EXPECT_THAT(block_i, testing::Each(testing::FloatEq(i)));
  }
}

// For dilated convolutional layers, an output vector at time t is formed by
// a weighted sum of input vectors at t, t - |dilation|, ...,
// t - |kernel_size| * |dilation|.
//
// The input to the matrix multiplication is |kernel_size|  input vectors
// stacked vertically (a total of |kernel_size| * |num_input_channels| rows),
// each spacing |dilation| apart in time. The input buffer should keep
// |dilation| columns of these "stacked" input vectors around and cycle through
// these columns. In summary, the input buffer is of shape
// [|kernel_size| * |num_input_channels|, |dilation|].
//
// The matrix multiplication takes one column of the input buffer, and then
// map it to |num_filters| rows. So
// input shape: [|kernel_size| * |num_input_channels|, 1]
// output shape:  [|num_filters|, 1]
// weight matrix shape: [|num_filters|, |kernel_size| * |num_input_channels|].
//
// Currently we only support dilated convolutional layers with |stride| == 1.
TYPED_TEST(DilatedConvolutionalLayerWrapperTest, LayerLoadSucceeds) {
  const LayerParams params = this->dilated_params_;
  auto layer = LayerWrapperPeer<TypeParam>::Create(params);
  EXPECT_GE(layer->bytes(), 0);

  // Verify that the weight matrix's shape is
  // [|num_filters|, |kernel_size| * |num_input_channels|].
  EXPECT_EQ(layer->rows(), params.num_filters);
  EXPECT_EQ(layer->cols(), params.kernel_size * params.num_input_channels);

  // Verify that the input buffer is of shape
  // [|kernel_size| * |num_input_channels|, |dilation|]
  auto input_buffer = layer->input_buffer();
  EXPECT_EQ(input_buffer.rows(),
            params.kernel_size * params.num_input_channels);
  EXPECT_EQ(input_buffer.cols(), params.dilation);
}

TYPED_TEST(DilatedConvolutionalLayerWrapperTest,
           LayerLoadDynamicDimensionsSucceeds) {
  LayerParams dynamic_params = this->dilated_params_;

  // Setting to zeros means to dynamically decide the dimensions.
  dynamic_params.num_input_channels = 0;
  dynamic_params.num_filters = 0;
  auto layer = LayerWrapperPeer<TypeParam>::Create(dynamic_params);
  EXPECT_GE(layer->bytes(), 0);

  // Verify that the weight matrix's shape is
  // [|num_filters|, |kernel_size| * |num_input_channels|] of the original
  // non-zero params.
  const auto params = this->dilated_params_;
  EXPECT_EQ(layer->rows(), params.num_filters);
  EXPECT_EQ(layer->cols(), params.kernel_size * params.num_input_channels);

  // Verify that the input buffer is of shape
  // [|kernel_size| * |num_input_channels|, |dilation|] of the original
  // non-zero params.
  auto input_buffer = layer->input_buffer();
  EXPECT_EQ(input_buffer.rows(),
            params.kernel_size * params.num_input_channels);
  EXPECT_EQ(input_buffer.cols(), params.dilation);
}

TYPED_TEST(DilatedConvolutionalLayerWrapperTest,
           LayerCreateWithConstantSucceeds) {
  auto params = this->dilated_params_;
  params.from = LayerParams::FromConstant{
      .value = 0.5f,
      .sparsity = -1.0f,
  };
  auto layer = LayerWrapperPeer<TypeParam>::Create(params);
  EXPECT_NE(layer, nullptr);
  EXPECT_GE(layer->bytes(), 0);

  // Verify that the weight matrix's shape is
  // [|num_filters|, |kernel_size| * |num_input_channels|].
  EXPECT_EQ(layer->rows(), params.num_filters);
  EXPECT_EQ(layer->cols(), params.kernel_size * params.num_input_channels);

  // Verify that the input buffer is of shape
  // [|kernel_size| * |num_input_channels|, |dilation|]
  auto input_buffer = layer->input_buffer();
  EXPECT_EQ(input_buffer.rows(),
            params.kernel_size * params.num_input_channels);
  EXPECT_EQ(input_buffer.cols(), params.dilation);
}

TYPED_TEST(DilatedConvolutionalLayerWrapperTest, LayerRuns) {
  const LayerParams params = this->dilated_params_;
  auto layer = LayerWrapperPeer<TypeParam>::Create(params);

  // Only a part of the input buffer corresponding to time t is updated each
  // time, i.e. the bottom |num_input_channels| rows out of the total
  // |kernel_size| * |num_input_channels| rows of the current column.
  auto output_view = PrepareInputOutput(
      /*expected_input_rows=*/params.num_input_channels,
      /*expected_input_cols=*/1,
      /*expected_output_rows=*/params.num_filters,
      /*expected_output_cols=*/1, 1.0f, layer->InputViewToUpdate(),
      &this->output_buffer_);

  // Check that Run() writes some non-zero results to the output buffer.
  layer->Run(0, &this->spin_barrier_, output_view);
  EXPECT_THAT(std::vector<float>(
                  this->output_buffer_.data(),
                  this->output_buffer_.data() + this->output_buffer_.size()),
              testing::Contains(testing::Ne(0.0f)));
}

// Test that the difference of the results between running with
// |skip_connection| = true vs |skip_connection| = false is the added input.
TYPED_TEST(DilatedConvolutionalLayerWrapperTest, SkipConnectionAddInput) {
  const float kInputValue = 2.0f;
  const auto params = this->dilated_params_;

  // Run through a layer with a skip connection.
  auto layer_skip = LayerWrapperPeer<TypeParam>::Create(params);
  auto output_view_skip = PrepareInputOutput(
      /*expected_input_rows=*/params.num_input_channels,
      /*expected_input_cols=*/1,
      /*expected_output_rows=*/params.num_filters,
      /*expected_output_cols=*/1, kInputValue, layer_skip->InputViewToUpdate(),
      &this->output_buffer_);
  layer_skip->Run(0, &this->spin_barrier_, output_view_skip);

  // Save the result.
  std::vector<float> saved_output_skip(
      output_view_skip.data(),
      output_view_skip.data() + output_view_skip.rows());

  // Run through another layer without a skip connection, using the same
  // input.
  LayerParams params_no_skip(params);
  params_no_skip.skip_connection = false;
  auto layer_no_skip = LayerWrapperPeer<TypeParam>::Create(params_no_skip);
  auto output_view_no_skip = PrepareInputOutput(
      /*expected_input_rows=*/params.num_input_channels,
      /*expected_input_cols=*/1,
      /*expected_output_rows=*/params.num_filters,
      /*expected_output_cols=*/1, kInputValue,
      layer_no_skip->InputViewToUpdate(), &this->output_buffer_);
  layer_no_skip->Run(0, &this->spin_barrier_, output_view_no_skip);

  // Verify that the element-wise difference between the two outputs is just the
  // input value.
  std::vector<float> difference(output_view_no_skip.rows());
  for (size_t i = 0; i < difference.size(); ++i) {
    difference[i] =
        saved_output_skip[i] - static_cast<float>(output_view_no_skip[i]);
  }
  EXPECT_THAT(difference, testing::Each(testing::FloatEq(kInputValue)));
}

TYPED_TEST(DilatedConvolutionalLayerWrapperTest,
           MultipleThreadsYieldSameResults) {
  auto params = this->dilated_params_;
  VerifyMultipleThreadsYeldSameResults<LayerWrapperPeer<TypeParam>>(
      /*iterations=*/8, /*threads_to_test=*/{1, 2, 4}, params,
      /*expected_input_rows=*/params.num_input_channels,
      /*expected_input_cols=*/1,
      /*expected_output_rows=*/params.num_filters,
      /*expected_output_cols=*/1);

  // Turn on per-column barrier and the result should still be the same.
  params.per_column_barrier = true;
  VerifyMultipleThreadsYeldSameResults<LayerWrapperPeer<TypeParam>>(
      /*iterations=*/8, /*threads_to_test=*/{1, 2, 4}, params,
      /*expected_input_rows=*/params.num_input_channels,
      /*expected_input_cols=*/1,
      /*expected_output_rows=*/params.num_filters,
      /*expected_output_cols=*/1);
}

TYPED_TEST(DilatedConvolutionalLayerWrapperTest, NumericalResults) {
  const LayerParams params{.num_input_channels = 3,
                           .num_filters = 2,
                           .length = 1,
                           .kernel_size = 2,
                           .dilation = 2,
                           .stride = 1,
                           .relu = false,
                           .skip_connection = false,
                           .type = LayerType::kDilated,
                           .num_threads = 1,
                           .per_column_barrier = false,
                           .from =
                               LayerParams::FromDisk{
                                   .path = this->testdata_dir_.string(),
                                   .zipped = true,
                               },
                           .prefix = "test_dilated_"};
  auto layer = LayerWrapperPeer<TypeParam>::Create(params);

  using RhsType = typename LayerWrapperPeer<TypeParam>::RhsType;
  const std::vector<std::vector<RhsType>> inputs = {
      {RhsType(1.0f), RhsType(2.0f), RhsType(3.0f)},
      {RhsType(4.0f), RhsType(5.0f), RhsType(6.0f)},
      {RhsType(7.0f), RhsType(8.0f), RhsType(9.0f)},
      {RhsType(10.0f), RhsType(11.0f), RhsType(12.0f)}};

  using OutputType = typename LayerWrapperPeer<TypeParam>::OutputType;
  const std::vector<std::vector<OutputType>> expected_outputs = {
      {OutputType(30.0f), OutputType(60.0f)},
      {OutputType(48.0f), OutputType(96.0f)},
  };

  csrblocksparse::FatCacheAlignedVector<OutputType> output_buffer(2, 1);
  for (int i = 0; i < 4; ++i) {
    std::copy(inputs[i].begin(), inputs[i].end(),
              layer->InputViewToUpdate().data());
    layer->Run(0, &this->spin_barrier_,
               csrblocksparse::MutableVectorView<OutputType>(&output_buffer));

    // We only care about the last two outputs.
    if (i >= 2) {
      // Convert to float because fixed points do not have comparison operators.
      const std::vector<float> actual_output_float(
          output_buffer.data(), output_buffer.data() + output_buffer.size());
      const std::vector<float> expected_output_float(
          expected_outputs[i - 2].begin(), expected_outputs[i - 2].end());
      EXPECT_THAT(
          actual_output_float,
          testing::Pointwise(testing::FloatEq(), expected_output_float));
    }
  }
}

}  // namespace
}  // namespace codec
}  // namespace chromemedia
