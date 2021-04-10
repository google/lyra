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

#include "conv1d_layer_wrapper.h"

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

static constexpr char kConv1DLayerPrefix[] = "lyra_conv1d_";

template <typename ComputeType>
class Conv1DLayerWrapperTest : public ::testing::Test {
 public:
  Conv1DLayerWrapperTest()
      : testdata_dir_(ghc::filesystem::current_path() /
                      "testdata"),
        conv1d_params_{.num_input_channels = kFeatureDepth,
                       .num_filters = kNumCondHidden,
                       .length = 1,
                       .kernel_size = kConv1DKernel,
                       .dilation = 1,
                       .stride = 1,
                       .relu = false,
                       .skip_connection = false,
                       .type = LayerType::kConv1D,
                       .num_threads = kNumThreads,
                       .per_column_barrier = false,
                       .from =
                           LayerParams::FromDisk{
                               .path = testdata_dir_.string(),
                               .zipped = true,
                           },
                       .prefix = kConv1DLayerPrefix},
        spin_barrier_(kNumThreads) {}

 protected:
  using OutputType = typename LayerWrapperPeer<ComputeType>::OutputType;
  using RhsType = typename LayerWrapperPeer<ComputeType>::RhsType;

  const int kFeatureDepth = 3;
  const int kConv1DKernel = 3;
  const int kNumCondHidden = 8;
  const int kNumThreads = 1;
  const ghc::filesystem::path testdata_dir_;
  const LayerParams conv1d_params_;

  csrblocksparse::SpinBarrier spin_barrier_;
  csrblocksparse::FatCacheAlignedVector<OutputType> output_buffer_;
};

using ComputeTypes = ::testing::Types<float, csrblocksparse::fixed16_type>;
TYPED_TEST_SUITE(Conv1DLayerWrapperTest, ComputeTypes);

TYPED_TEST(Conv1DLayerWrapperTest, CreateSucceeds) {
  EXPECT_NE(LayerWrapperPeer<TypeParam>::Create(this->conv1d_params_), nullptr);
}

TYPED_TEST(Conv1DLayerWrapperTest, CreateWithBadParamsReturnNullptr) {
  // Skip connections are not supported.
  LayerParams skip_connection_params(this->conv1d_params_);
  skip_connection_params.skip_connection = true;
  EXPECT_EQ(LayerWrapperPeer<TypeParam>::Create(skip_connection_params),
            nullptr);

  // |dilation| != 1 is not supported.
  LayerParams bad_dilation_params(this->conv1d_params_);
  bad_dilation_params.dilation = 2;
  EXPECT_EQ(LayerWrapperPeer<TypeParam>::Create(bad_dilation_params), nullptr);
}

TYPED_TEST(Conv1DLayerWrapperTest, ResetShiftsInputBufferUp) {
  auto params = this->conv1d_params_;

  // Set up the input buffer expectations.
  std::vector<std::vector<float>> expected_input_buffer(3);

  // For |kernel_size| = 3 and |num_input_channels| = 3, the input buffer has
  // 9 elements total.
  // For |stride| = 1, each input loading pushes in
  // |stride * num_input_channels| = 3 numbers, so after 3 loadings the input
  // buffer should look like this:
  //  | 0 |
  //  | 0 |
  //  | 0 |
  //  | 1 |
  //  | 1 |
  //  | 1 |
  //  | 2 |
  //  | 2 |
  //  | 2 |
  // I.e. input 0, input 1, input 2 are all in the buffer.
  expected_input_buffer[0] = {0, 0, 0, 1, 1, 1, 2, 2, 2};

  // For |stride| = 2, each loading pushes in 6 numbers, and the input buffer
  // should look like this in the end:
  //  | 1 |
  //  | 1 |
  //  | 1 |
  //  | 2 |
  //  | 2 |
  //  | 2 |
  //  | 2 |
  //  | 2 |
  //  | 2 |
  // I.e. input 0 has been pushed out, so have the first 3 elements of input 1.
  expected_input_buffer[1] = {1, 1, 1, 2, 2, 2, 2, 2, 2};

  // For |stride| = 3, each loading pushes in 9 numbers, and the input buffer
  // should look like this in the end:
  //  | 2 |
  //  | 2 |
  //  | 2 |
  //  | 2 |
  //  | 2 |
  //  | 2 |
  //  | 2 |
  //  | 2 |
  //  | 2 |
  // I.e. input 0 and input 1 have all been pushed out, leaving only input 2.
  expected_input_buffer[2] = {2, 2, 2, 2, 2, 2, 2, 2, 2};

  using RhsType = typename LayerWrapperPeer<TypeParam>::RhsType;
  for (int stride = 1; stride <= 3; ++stride) {
    params.stride = stride;
    auto layer = LayerWrapperPeer<TypeParam>::Create(params);

    // Load input 3 times (each time loading |stride * num_input_channels|
    // elements) and calling Reset() in between.
    auto input_view = layer->InputViewToUpdate();
    const int input_view_size = input_view.rows() * input_view.cols();
    std::fill_n(input_view.data(), input_view_size, static_cast<RhsType>(0.0f));
    layer->Reset(0, &this->spin_barrier_);
    std::fill_n(input_view.data(), input_view_size, static_cast<RhsType>(1.0f));
    layer->Reset(0, &this->spin_barrier_);
    std::fill_n(input_view.data(), input_view_size, static_cast<RhsType>(2.0f));

    // Verify the input buffer is as expected.
    EXPECT_THAT(std::vector<float>(layer->input_buffer().data(),
                                   layer->input_buffer().data() + 9),
                testing::ElementsAreArray(expected_input_buffer[stride - 1]));
  }
}

TYPED_TEST(Conv1DLayerWrapperTest, LayerLoadSucceeds) {
  const LayerParams params = this->conv1d_params_;
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

TYPED_TEST(Conv1DLayerWrapperTest, LayerLoadDynamicDimensionsSucceeds) {
  LayerParams dynamic_params = this->conv1d_params_;

  // Setting to zeros means to dynamically decide the dimensions.
  dynamic_params.num_input_channels = 0;
  dynamic_params.num_filters = 0;
  auto layer = LayerWrapperPeer<TypeParam>::Create(dynamic_params);
  EXPECT_GE(layer->bytes(), 0);

  // Verify that the weight matrix's shape is
  // [|num_filters|, |kernel_size| * |num_input_channels|] of the original
  // non-zero params.
  const auto params = this->conv1d_params_;
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

TYPED_TEST(Conv1DLayerWrapperTest, LayerCreateWithConstantSucceeds) {
  auto params = this->conv1d_params_;
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
}

TYPED_TEST(Conv1DLayerWrapperTest, LayerRuns) {
  const LayerParams params = this->conv1d_params_;
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

TYPED_TEST(Conv1DLayerWrapperTest, MultipleThreadsYieldSameResults) {
  auto params = this->conv1d_params_;
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

TYPED_TEST(Conv1DLayerWrapperTest, NumericalResults) {
  const LayerParams params{.num_input_channels = 3,
                           .num_filters = 2,
                           .length = 1,
                           .kernel_size = 2,
                           .dilation = 1,
                           .stride = 2,
                           .relu = false,
                           .skip_connection = false,
                           .type = LayerType::kConv1D,
                           .num_threads = 1,
                           .per_column_barrier = false,
                           .from =
                               LayerParams::FromDisk{
                                   .path = this->testdata_dir_.string(),
                                   .zipped = true,
                               },
                           .prefix = "test_conv1d_"};
  auto layer = LayerWrapperPeer<TypeParam>::Create(params);

  using RhsType = typename LayerWrapperPeer<TypeParam>::RhsType;
  const std::vector<std::vector<RhsType>> inputs = {
      {RhsType(1.0f), RhsType(2.0f), RhsType(3.0f), RhsType(4.0f),
       RhsType(5.0f), RhsType(6.0f)},
      {RhsType(7.0f), RhsType(8.0f), RhsType(9.0f), RhsType(10.0f),
       RhsType(11.0f), RhsType(12.0f)}};

  using OutputType = typename LayerWrapperPeer<TypeParam>::OutputType;
  const std::vector<std::vector<OutputType>> expected_outputs = {
      {OutputType(21.0f), OutputType(42.0f)},
      {OutputType(57.0f), OutputType(114.0f)},
  };

  csrblocksparse::FatCacheAlignedVector<OutputType> output_buffer(2, 1);
  for (int i = 0; i < 2; ++i) {
    std::copy(inputs[i].begin(), inputs[i].end(),
              layer->InputViewToUpdate().data());
    layer->Run(0, &this->spin_barrier_,
               csrblocksparse::MutableVectorView<OutputType>(&output_buffer));

    // Convert to float because fixed points do not have comparison operators.
    const std::vector<float> actual_output_float(
        output_buffer.data(), output_buffer.data() + output_buffer.size());
    const std::vector<float> expected_output_float(expected_outputs[i].begin(),
                                                   expected_outputs[i].end());
    EXPECT_THAT(actual_output_float,
                testing::Pointwise(testing::FloatEq(), expected_output_float));
  }
}

}  // namespace
}  // namespace codec
}  // namespace chromemedia
