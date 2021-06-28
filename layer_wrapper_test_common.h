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

#ifndef LYRA_CODEC_LAYER_WRAPPER_TEST_COMMON_H_
#define LYRA_CODEC_LAYER_WRAPPER_TEST_COMMON_H_

#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "layer_wrappers_lib.h"
#include "sparse_matmul/sparse_matmul.h"

namespace chromemedia {
namespace codec {

// Use a peer to access the input buffer of a layer wrapper, so that we can test
// the effect of Reset(). The layer wrapper's type will be determined using the
// passed-in |LayerWrapperTypeTemplate| and four types for weight, input,
// output and disk-weight inferred from |WeightTypeKind|.
template <typename WeightTypeKind,
          template <typename, typename, typename, typename>
          class LayerWrapperTypeTemplate = LayerWrapper>
class LayerWrapperPeer {
 public:
  static constexpr int kWeightExponentBits = 6;
  static constexpr int kRhsExponentBits = 6;
  static constexpr bool kUseFixedPoint =
      std::is_same<WeightTypeKind, csrblocksparse::fixed16_type>::value;
  using WeightType =
      typename std::conditional<kUseFixedPoint,
                                csrblocksparse::fixed16<kWeightExponentBits>,
                                WeightTypeKind>::type;
  using RhsType = typename std::conditional<
      kUseFixedPoint, csrblocksparse::fixed16<kRhsExponentBits>, float>::type;
  using OutputType = RhsType;
  using DiskWeightType =
      typename std::conditional<kUseFixedPoint, csrblocksparse::fixed16_type,
                                float>::type;
  using LayerWrapperType =
      LayerWrapperTypeTemplate<WeightType, RhsType, OutputType, DiskWeightType>;

  static std::unique_ptr<
      LayerWrapperPeer<WeightTypeKind, LayerWrapperTypeTemplate>>
  Create(const LayerParams& params) {
    auto layer_wrapper = LayerWrapperType::Create(params);
    if (layer_wrapper == nullptr) {
      return nullptr;
    }
    return absl::WrapUnique(
        new LayerWrapperPeer<WeightTypeKind, LayerWrapperTypeTemplate>(
            std::move(layer_wrapper)));
  }

  void Run(int tid, csrblocksparse::SpinBarrier* spin_barrier,
           csrblocksparse::MutableVectorView<OutputType> output_view) {
    layer_wrapper_->Run(tid, spin_barrier, output_view);
  }

  csrblocksparse::MutableVectorView<RhsType> InputViewToUpdate() {
    return layer_wrapper_->InputViewToUpdate();
  }

  int bytes() { return layer_wrapper_->bytes(); }
  int rows() { return layer_wrapper_->rows(); }
  int cols() { return layer_wrapper_->cols(); }

  // Protected in LayerWrapper.
  void Reset(int tid, csrblocksparse::SpinBarrier* spin_barrier) {
    layer_wrapper_->Reset(tid, spin_barrier);
  }
  const csrblocksparse::FatCacheAlignedVector<RhsType>& input_buffer() {
    return layer_wrapper_->input_buffer_;
  }

 protected:
  explicit LayerWrapperPeer(std::unique_ptr<LayerWrapperType> layer_wrapper)
      : layer_wrapper_(std::move(layer_wrapper)) {}

  std::unique_ptr<LayerWrapperType> layer_wrapper_;
};

template <typename OutputType, typename RhsType>
csrblocksparse::MutableVectorView<OutputType> PrepareInputOutput(
    int expected_input_rows, int expected_input_cols, int expected_output_rows,
    int expected_output_cols, float input_value,
    csrblocksparse::MutableVectorView<RhsType> input_view,
    csrblocksparse::FatCacheAlignedVector<OutputType>* output_buffer) {
  // Check shapes of the space to load input.
  EXPECT_EQ(input_view.rows(), expected_input_rows);
  EXPECT_EQ(input_view.cols(), expected_input_cols);

  // Load input values.
  std::fill_n(input_view.data(), input_view.rows() * input_view.cols(),
              static_cast<RhsType>(input_value));

  // Space to store output.
  *output_buffer = csrblocksparse::FatCacheAlignedVector<OutputType>(
      expected_output_rows, expected_output_cols);
  return csrblocksparse::MutableVectorView<OutputType>(output_buffer);
}

template <typename LayerType>
void VerifyMultipleThreadsYeldSameResults(
    int iterations, const std::vector<int>& threads_to_test, LayerParams params,
    int expected_input_rows, int expected_input_cols, int expected_output_rows,
    int expected_output_cols) {
  csrblocksparse::FatCacheAlignedVector<typename LayerType::OutputType>
      output_buffer;

  // Run layers with different number of threads for specified iterations.
  // Compare the first result with every other ones.
  std::vector<std::vector<float>> saved_output_first(iterations);
  for (const int num_threads : threads_to_test) {
    params.num_threads = num_threads;
    auto layer = LayerType::Create(params);
    for (int i = 0; i < iterations; ++i) {
      auto output_view =
          PrepareInputOutput(expected_input_rows, expected_input_cols,
                             expected_output_rows, expected_output_cols,
                             /*input_value=*/static_cast<float>(i),
                             layer->InputViewToUpdate(), &output_buffer);
      auto f = [&](csrblocksparse::SpinBarrier* spin_barrier, int tid) {
        layer->Run(tid, spin_barrier, output_view);
      };
      csrblocksparse::LaunchOnThreadsWithBarrier(params.num_threads, f);
      std::vector<float> saved_output(output_view.data(),
                                      output_view.data() + output_view.rows());

      // Save the first result and compare others against it.
      if (num_threads == threads_to_test[0]) {
        saved_output_first[i] = saved_output;
      } else {
        EXPECT_THAT(saved_output, testing::Pointwise(testing::FloatEq(),
                                                     saved_output_first[i]));
      }
    }
  }
}

}  // namespace codec
}  // namespace chromemedia

#endif  // LYRA_CODEC_LAYER_WRAPPER_TEST_COMMON_H_
