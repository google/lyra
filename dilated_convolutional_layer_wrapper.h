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

#ifndef LYRA_CODEC_DILATED_CONVOLUTIONAL_LAYER_WRAPPER_H_
#define LYRA_CODEC_DILATED_CONVOLUTIONAL_LAYER_WRAPPER_H_

#include <algorithm>
#include <memory>
#include <string>
#include <utility>

#include "glog/logging.h"
#include "absl/memory/memory.h"
#include "layer_wrapper.h"
#include "sparse_inference_matrixvector.h"

namespace chromemedia {
namespace codec {

// Class that wraps the data and logic of dilated convolutional layers.
template <typename WeightType, typename RhsType, typename OutputType,
          typename DiskWeightType>
class DilatedConvolutionalLayerWrapper
    : public LayerWrapper<WeightType, RhsType, OutputType, DiskWeightType> {
 public:
  using Super = LayerWrapper<WeightType, RhsType, OutputType, DiskWeightType>;

  static std::unique_ptr<DilatedConvolutionalLayerWrapper<
      WeightType, RhsType, OutputType, DiskWeightType>>
  Create(const LayerParams& params) {
    const std::string layer_prompt = "|" + params.prefix + "| layer: ";

    // TODO(b/161015017): Support more general stride and kernel size
    // combinations.
    if (params.stride != 1) {
      LOG(ERROR) << layer_prompt
                 << "Dilated convolutional layer with |stride| != 1"
                 << "is not supported.";
      return nullptr;
    }
    if (params.length != 1) {
      LOG(ERROR) << layer_prompt
                 << "Dilated convolutional layer with |length| != 1"
                 << "is not supported.";
      return nullptr;
    }

    auto layer = Super::LoadAndCheckLayer(
        params.from, params.prefix, layer_prompt, params.num_filters,
        params.kernel_size * params.num_input_channels, params.num_threads);
    if (layer == nullptr) {
      return nullptr;
    }
    const int input_buffer_rows = layer->cols();
    const int num_input_channels = input_buffer_rows / params.kernel_size;
    const int output_rows = layer->rows();

    if (params.skip_connection && num_input_channels != output_rows) {
      LOG(ERROR) << layer_prompt
                 << "Skip connection can only be performed if the input and "
                 << "output have the same dimensions: "
                 << params.num_input_channels << " vs " << output_rows;
      return nullptr;
    }

    return absl::WrapUnique(
        new DilatedConvolutionalLayerWrapper<WeightType, RhsType, OutputType,
                                             DiskWeightType>(
            num_input_channels, output_rows, input_buffer_rows, params.dilation,
            params.relu, params.per_column_barrier, params.skip_connection,
            params.num_threads, std::move(layer)));
  }

  // Runs the layer as a matrix multiplication and a bias-add, optionally
  // adding a skip connection.
  void Run(int tid, csrblocksparse::SpinBarrier* spin_barrier,
           csrblocksparse::MutableVectorView<OutputType> output_view) override {
    // If |skip_connection| is true, the input to is first saved first and
    // Relu'd, then SpMM_bias is applied with no Relu. Then the saved input
    // is added.
    //
    //            |
    //          Input ---------
    //            |           |
    //           Relu         |
    //            |           |
    //        SpMM_bias       |
    //            |           |
    //           Add ----------
    //            |
    //          Output
    if (skip_connection_) {
      auto prev_layer_output_view = InputViewToUpdate();
      SaveSkipConnectionInput(tid, spin_barrier, prev_layer_output_view);
      Relu(tid, spin_barrier, &prev_layer_output_view);
    }

    // Select a part of the |input_buffer_| as the input to the matrix
    // multiplication.
    csrblocksparse::VectorView<RhsType> input_view(InputColumnStart(),
                                                   this->input_buffer_rows_, 1);

    this->layer_->SpMM_bias(input_view, &output_view, this->relu_, tid,
                            this->per_column_barrier_ ? spin_barrier : nullptr);
    spin_barrier->barrier();

    if (skip_connection_) {
      AddSkipConnection(tid, spin_barrier, &output_view);
    }

    Reset(tid, spin_barrier);
  }

  // The part of |input_buffer_| updated by the previous layer corresponding to
  // the current step (out of all past values). It is the bottom
  // |num_input_channels_| rows of the current column.
  csrblocksparse::MutableVectorView<RhsType> InputViewToUpdate() override {
    return csrblocksparse::MutableVectorView<RhsType>(
        InputColumnStart() + this->input_buffer_rows_ -
            this->num_input_channels_,
        this->num_input_channels_, 1);
  }

  int PrepareForThreads(int num_threads) override {
    num_elements_per_thread_ = this->output_rows_ / num_threads;
    return this->layer_->PrepareForThreads(num_threads);
  }

 private:
  DilatedConvolutionalLayerWrapper() = delete;
  explicit DilatedConvolutionalLayerWrapper(
      int num_input_channels, int output_rows, int input_buffer_rows,
      int input_buffer_cols, bool relu, bool per_column_barrier,
      bool skip_connection, int num_threads,
      std::unique_ptr<csrblocksparse::SparseLinearLayer<WeightType, RhsType>>
          layer)
      : Super(num_input_channels, output_rows, /*length=*/1, input_buffer_rows,
              input_buffer_cols, relu, per_column_barrier, std::move(layer)),
        skip_connection_(skip_connection),
        num_resets_(0),
        num_elements_per_thread_(output_rows / num_threads),
        skip_connection_buffer_(output_rows, 1) {}

  // For dilated convolutional layers, the matrix multiplication needs inputs
  // from t, t - |dilation|, ..., t - |kernel_size| * |dilation|.
  // In a layer where |kernel_size| = 2 and |dilation| = 4, the memory layout of
  // |input_buffer_| is stacks of |kernel_size| vectors, each having
  // |num_input_channels| elements, spanning |dilation| columns.
  //
  //  | v0 | v1 | v2 | v3 |  \                          //
  //  |----|----|----|----|   --> |kernel_size| stacks  //
  //  | v4 | v5 | v6 | v7 |  /                          //
  //  <--  |dilation |  -->                             //
  //
  // where v0 is the input vector at t = 0, v1 is at t = 1, and so on.
  // Then for example at the beginning of t = 4, we will have access to what we
  // need: v4 and v0, which are stacked as column 0 in the input buffer:
  //
  //  col 0                                             //
  //    |                                               //
  //  | v0 | v1 | v2 | v3 |                             //
  //  |----|----|----|----|                             //
  //  | v4 |    |    |    |                             //
  //
  // After calling Run(), we need to shift v4 up (for future reuse at t = 8),
  // and also advance the column read head to column 1. So that at the
  // beginning of t = 5, the buffer looks like this:
  //
  //       col 1                                        //
  //         |                                          //
  //  | v4 | v1 | v2 | v3 |                             //
  //  |----|----|----|----|                             //
  //  |    | v5 |    |    |                             //
  //
  // This "shifting and advancing" is done by the Reset() function.
  void Reset(int tid, csrblocksparse::SpinBarrier* spin_barrier) override {
    if (tid == 0) {
      // Shift the current column up by |num_input_channels_| elements.
      auto shift_to = InputColumnStart();
      std::move(shift_to + this->num_input_channels_,
                shift_to + this->input_buffer_rows_, shift_to);

      // Perform the modulo operation on |num_resets_| to prevent
      // overflow.
      num_resets_ = (num_resets_ + 1) % this->input_buffer_cols_;
    }
    spin_barrier->barrier();
  }

  // Points to the current column of |input_buffer_| depending on the current
  // |num_resets_|.
  RhsType* InputColumnStart() {
    return this->input_buffer_.slice(num_resets_ % this->input_buffer_cols_)
        .data();
  }

  // TODO(b/163000746): Make skip connection a decorator.
  void SaveSkipConnectionInput(int tid,
                               csrblocksparse::SpinBarrier* spin_barrier,
                               const csrblocksparse::MutableVectorView<RhsType>&
                                   prev_layer_output_view) {
    if (tid == 0) {
      std::copy(prev_layer_output_view.data(),
                prev_layer_output_view.data() + this->output_rows_,
                skip_connection_buffer_.data());
    }
    spin_barrier->barrier();
  }

  // TODO(b/163000746): Make skip connection a decorator.
  // TODO(b/123254413): SIMD-optimize the Skip connection.
  // Element wise addition of the contents of |skip_connection_buffer_| to
  // |dilated_conv_output|.
  void AddSkipConnection(
      int tid, csrblocksparse::SpinBarrier* spin_barrier,
      csrblocksparse::MutableVectorView<OutputType>* output_view) {
    const int tid_offset = tid * num_elements_per_thread_;
    for (int i = tid_offset; i < tid_offset + num_elements_per_thread_; ++i) {
      const float sum = static_cast<float>((*output_view)[i]) +
                        static_cast<float>(skip_connection_buffer_[i]);
      (*output_view)[i] = static_cast<OutputType>(sum);
    }
    spin_barrier->barrier();
  }

  // TODO(b/123254413): SIMD-optimize the Relu layer.
  void Relu(
      int tid, csrblocksparse::SpinBarrier* spin_barrier,
      csrblocksparse::MutableVectorView<RhsType>* prev_layer_output_buffer) {
    const int tid_offset = tid * num_elements_per_thread_;
    for (int i = tid_offset; i < tid_offset + num_elements_per_thread_; ++i) {
      (*prev_layer_output_buffer)[i] = static_cast<RhsType>(
          std::max(static_cast<float>((*prev_layer_output_buffer)[i]), 0.0f));
    }
    spin_barrier->barrier();
  }

  // Whether to add a skip connection from the input to the output. The
  // input will also go through Relu before the matrix multiplication.
  const bool skip_connection_;

  // Keep track of which part of the buffer to use next.
  int num_resets_;

  // Used in splitting work of AddSkipConnection() and Relu() among threads.
  // May change between runs.
  int num_elements_per_thread_;

  csrblocksparse::FatCacheAlignedVector<RhsType> skip_connection_buffer_;
};

}  // namespace codec
}  // namespace chromemedia

#endif  // LYRA_CODEC_DILATED_CONVOLUTIONAL_LAYER_WRAPPER_H_
