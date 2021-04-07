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

#include "transpose_convolutional_layer_wrapper.h"

#include <algorithm>
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

static constexpr char kTransposeLayerPrefix[] = "lyra_transpose_2_";

template <typename ComputeType>
class TransposeConvolutionalLayerWrapperTest : public ::testing::Test {
 public:
  TransposeConvolutionalLayerWrapperTest()
      : testdata_dir_(ghc::filesystem::current_path() /
                      "testdata"),
        transpose_params_{.num_input_channels = kNumCondHidden,
                          .num_filters = kNumCondHidden,
                          .length = kTransposeLength,
                          .kernel_size = kTransposeStride,
                          .dilation = 1,
                          .stride = kTransposeStride,
                          .relu = true,
                          .skip_connection = false,
                          .type = LayerType::kTranspose,
                          .num_threads = kNumThreads,
                          .per_column_barrier = false,
                          .from =
                              LayerParams::FromDisk{
                                  .path = this->testdata_dir_.string(),
                                  .zipped = true,
                              },
                          .prefix = kTransposeLayerPrefix},
        spin_barrier_(kNumThreads) {}

 protected:
  using OutputType = typename LayerWrapperPeer<ComputeType>::OutputType;
  using RhsType = typename LayerWrapperPeer<ComputeType>::RhsType;

  const int kNumCondHidden = 8;
  const int kTransposeLength = 4;
  const int kTransposeStride = 2;
  const int kNumThreads = 1;
  const ghc::filesystem::path testdata_dir_;
  const LayerParams transpose_params_;

  csrblocksparse::SpinBarrier spin_barrier_;
  csrblocksparse::FatCacheAlignedVector<OutputType> output_buffer_;
};

using ComputeTypes = ::testing::Types<float, csrblocksparse::fixed16_type>;
TYPED_TEST_SUITE(TransposeConvolutionalLayerWrapperTest, ComputeTypes);

TYPED_TEST(TransposeConvolutionalLayerWrapperTest, CreateSucceeds) {
  EXPECT_NE(LayerWrapperPeer<TypeParam>::Create(this->transpose_params_),
            nullptr);
}

TYPED_TEST(TransposeConvolutionalLayerWrapperTest,
           CreateWithBadParamsReturnNullptr) {
  // Skip connections are not supported for transpose layers.
  LayerParams skip_connection_params(this->transpose_params_);
  skip_connection_params.skip_connection = true;
  EXPECT_EQ(LayerWrapperPeer<TypeParam>::Create(skip_connection_params),
            nullptr);

  // |kernel_size| must be equal to |stride| for transpose layers.
  LayerParams kernel_size_stride_different_params(this->transpose_params_);
  kernel_size_stride_different_params.stride =
      kernel_size_stride_different_params.kernel_size + 1;
  EXPECT_EQ(
      LayerWrapperPeer<TypeParam>::Create(kernel_size_stride_different_params),
      nullptr);
}

TYPED_TEST(TransposeConvolutionalLayerWrapperTest, ResetDoesNothing) {
  auto params = this->transpose_params_;
  auto layer = LayerWrapperPeer<TypeParam>::Create(params);

  auto input_view_to_update_1 = layer->InputViewToUpdate();

  // The input view to update should be the whole input buffer.
  EXPECT_EQ(input_view_to_update_1.data(), layer->input_buffer().data());
  EXPECT_EQ(input_view_to_update_1.rows(), layer->input_buffer().rows());
  EXPECT_EQ(input_view_to_update_1.cols(), layer->input_buffer().cols());

  // After reset, the input view to update still points to the same address.
  layer->Reset(0, &this->spin_barrier_);
  auto input_view_to_update_2 = layer->InputViewToUpdate();
  EXPECT_EQ(input_view_to_update_1.data(), input_view_to_update_2.data());
}

// For transpose convolutional layers, the goal is to "upsample" the time
// dimension (|length|) by |stride|, so the final output shape should be
// [|num_filters|, |stride| * |length|].
//
// This is achieved by:
// 1. Map input of shape [|num_input_channels|, |length|] to output of shape
//    [|kernel_size| * |num_filters|, |length|]. So the weight matrix is of
//    shape [|kernel_size| * |num_filters|, |num_input_channels|].
// 2. Combine the result of the matrix multiplication to form a matrix of shape
//    [|stride| * |num_filters, |length|],
// 3. Reshape to the final output shape [|num_filters|, |stride| * |length|].
//
// But currently we only support |kernel_size| == |stride|, so step 2 is
// skipped.
TYPED_TEST(TransposeConvolutionalLayerWrapperTest, LayerLoadSucceeds) {
  const LayerParams params = this->transpose_params_;
  auto layer = LayerWrapperPeer<TypeParam>::Create(params);
  EXPECT_GE(layer->bytes(), 0);

  // Verify that the weight matrix's shape is
  // [|kernel_size| * |num_filters|, |num_input_channels|].
  EXPECT_EQ(layer->rows(), params.kernel_size * params.num_filters);
  EXPECT_EQ(layer->cols(), params.num_input_channels);

  // Verify that the input buffer is of shape [|num_input_channels|, |length|].
  auto input_buffer = layer->input_buffer();
  EXPECT_EQ(input_buffer.rows(), params.num_input_channels);
  EXPECT_EQ(input_buffer.cols(), params.length);
}

TYPED_TEST(TransposeConvolutionalLayerWrapperTest,
           LayerLoadDynamicDimensionsSucceeds) {
  LayerParams dynamic_params = this->transpose_params_;
  // Setting to zeros means to dynamically decide the dimensions.
  dynamic_params.num_input_channels = 0;
  dynamic_params.num_filters = 0;
  auto layer = LayerWrapperPeer<TypeParam>::Create(dynamic_params);
  EXPECT_GE(layer->bytes(), 0);

  // Verify that the weight matrix's shape is
  // [|kernel_size| * |num_filters|, |num_input_channels|] of the original
  // non-zero params.
  const auto params = this->transpose_params_;
  EXPECT_EQ(layer->rows(), params.kernel_size * params.num_filters);
  EXPECT_EQ(layer->cols(), params.num_input_channels);

  // Verify that the input buffer is of shape [|num_input_channels|, |length|]
  // of the original non-zero params.
  auto input_buffer = layer->input_buffer();
  EXPECT_EQ(input_buffer.rows(), params.num_input_channels);
  EXPECT_EQ(input_buffer.cols(), params.length);
}

TYPED_TEST(TransposeConvolutionalLayerWrapperTest,
           LayerCreateWithConstantSucceeds) {
  auto params = this->transpose_params_;
  params.from = LayerParams::FromConstant{
      .value = 0.5f,
      .sparsity = -1.0f,
  };
  auto layer = LayerWrapperPeer<TypeParam>::Create(params);
  EXPECT_GE(layer->bytes(), 0);

  // Verify that the weight matrix's shape is
  // [|kernel_size| * |num_filters|, |num_input_channels|].
  EXPECT_EQ(layer->rows(), params.kernel_size * params.num_filters);
  EXPECT_EQ(layer->cols(), params.num_input_channels);

  // Verify that the input buffer is of shape [|num_input_channels|, |length|].
  auto input_buffer = layer->input_buffer();
  EXPECT_EQ(input_buffer.rows(), params.num_input_channels);
  EXPECT_EQ(input_buffer.cols(), params.length);
}

TYPED_TEST(TransposeConvolutionalLayerWrapperTest, TransposeLayerRuns) {
  const LayerParams params = this->transpose_params_;
  auto layer = LayerWrapperPeer<TypeParam>::Create(params);
  auto output_view = PrepareInputOutput(
      /*expected_input_rows=*/params.num_input_channels,
      /*expected_input_cols=*/params.length,
      /*expected_output_rows=*/params.kernel_size * params.num_filters,
      /*expected_output_cols=*/params.length, 1.0f, layer->InputViewToUpdate(),
      &this->output_buffer_);

  // Check that Run() writes some non-zero results to the output buffer.
  layer->Run(0, &this->spin_barrier_, output_view);
  EXPECT_THAT(std::vector<float>(
                  this->output_buffer_.data(),
                  this->output_buffer_.data() + this->output_buffer_.size()),
              testing::Contains(testing::Ne(0.0f)));
}

TYPED_TEST(TransposeConvolutionalLayerWrapperTest,
           MultipleThreadsYieldSameResults) {
  auto params = this->transpose_params_;
  VerifyMultipleThreadsYeldSameResults<LayerWrapperPeer<TypeParam>>(
      /*iterations=*/8, /*threads_to_test=*/{1, 2, 4}, params,
      /*expected_input_rows=*/params.num_input_channels,
      /*expected_input_cols=*/params.length,
      /*expected_output_rows=*/params.kernel_size * params.num_filters,
      /*expected_output_cols=*/params.length);

  // Turn on per-column barrier and the result should still be the same.
  params.per_column_barrier = true;
  VerifyMultipleThreadsYeldSameResults<LayerWrapperPeer<TypeParam>>(
      /*iterations=*/8, /*threads_to_test=*/{1, 2, 4}, params,
      /*expected_input_rows=*/params.num_input_channels,
      /*expected_input_cols=*/params.length,
      /*expected_output_rows=*/params.kernel_size * params.num_filters,
      /*expected_output_cols=*/params.length);
}

TYPED_TEST(TransposeConvolutionalLayerWrapperTest, NumericalResults) {
  const LayerParams params{.num_input_channels = 3,
                           .num_filters = 2,
                           .length = 4,
                           .kernel_size = 2,
                           .dilation = 1,
                           .stride = 2,
                           .relu = false,
                           .skip_connection = false,
                           .type = LayerType::kTranspose,
                           .num_threads = 1,
                           .per_column_barrier = false,
                           .from =
                               LayerParams::FromDisk{
                                   .path = this->testdata_dir_.string(),
                                   .zipped = true,
                               },
                           .prefix = "test_transpose_"};
  auto layer = LayerWrapperPeer<TypeParam>::Create(params);

  using RhsType = typename LayerWrapperPeer<TypeParam>::RhsType;
  const std::vector<std::vector<RhsType>> inputs = {
      {RhsType(1.0f), RhsType(2.0f), RhsType(3.0f)},
      {RhsType(4.0f), RhsType(5.0f), RhsType(6.0f)},
      {RhsType(7.0f), RhsType(8.0f), RhsType(9.0f)},
      {RhsType(10.0f), RhsType(11.0f), RhsType(12.0f)}};

  using OutputType = typename LayerWrapperPeer<TypeParam>::OutputType;
  const std::vector<std::vector<OutputType>> expected_outputs = {
      {OutputType(6.0f), OutputType(12.0f)},
      {OutputType(18.0f), OutputType(24.0f)},
      {OutputType(15.0f), OutputType(30.0f)},
      {OutputType(45.0f), OutputType(60.0f)},
      {OutputType(24.0f), OutputType(48.0f)},
      {OutputType(72.0f), OutputType(96.0f)},
      {OutputType(33.0f), OutputType(66.0f)},
      {OutputType(99.0f), OutputType(132.0f)},
  };

  // Load inputs corresponding to all 4 time steps at once to the layer's
  // input buffer.
  for (int i = 0; i < 4; ++i) {
    std::copy(inputs[i].begin(), inputs[i].end(),
              layer->InputViewToUpdate().data() +
                  i * layer->InputViewToUpdate().rows());
  }

  csrblocksparse::FatCacheAlignedVector<OutputType> output_buffer(2, 8);
  layer->Run(0, &this->spin_barrier_,
             csrblocksparse::MutableVectorView<OutputType>(&output_buffer));

  // Verify outputs corresponding to the resulting 8 time steps.
  for (int i = 0; i < 8; ++i) {
    auto output_col = output_buffer.slice(i);
    const std::vector<float> actual_output_float(
        output_col.data(), output_col.data() + output_buffer.rows());
    const std::vector<float> expected_output_float(expected_outputs[i].begin(),
                                                   expected_outputs[i].end());
    EXPECT_THAT(actual_output_float,
                testing::Pointwise(testing::FloatEq(), expected_output_float));
  }
}

}  // namespace
}  // namespace codec
}  // namespace chromemedia
