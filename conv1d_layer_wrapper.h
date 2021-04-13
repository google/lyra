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

#ifndef LYRA_CODEC_CONV1D_LAYER_WRAPPER_H_
#define LYRA_CODEC_CONV1D_LAYER_WRAPPER_H_

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

// Class that wraps the data and logic of conv1d layers.
template <typename WeightType, typename RhsType, typename OutputType,
          typename DiskWeightType>
class Conv1DLayerWrapper
    : public LayerWrapper<WeightType, RhsType, OutputType, DiskWeightType> {
 public:
  using Super = LayerWrapper<WeightType, RhsType, OutputType, DiskWeightType>;

  static std::unique_ptr<
      Conv1DLayerWrapper<WeightType, RhsType, OutputType, DiskWeightType>>
  Create(const LayerParams& params) {
    const std::string layer_prompt = "|" + params.prefix + "| layer: ";

    // TODO(b/161015017): Support more general stride and kernel size
    // combinations.
    if (params.skip_connection) {
      LOG(ERROR) << layer_prompt
                 << "Conv1D Layer does not support skip connections.";
      if (params.stride == 1) {
        LOG(WARNING) << layer_prompt
                     << "Use DilatedConvolutionalLayerWrapper with |dilation| "
                     << "= 1 and |stride| = 1 to allow skip connections.";
      }
      return nullptr;
    }
    if (params.dilation != 1) {
      LOG(ERROR) << layer_prompt
                 << "Use DilatedConvolutionalLayerWrapper instead by setting "
                 << "|params.type| to |kDilated|.";
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

    return absl::WrapUnique(
        new Conv1DLayerWrapper<WeightType, RhsType, OutputType, DiskWeightType>(
            num_input_channels, output_rows, params.length, input_buffer_rows,
            params.stride, params.relu, params.per_column_barrier,
            std::move(layer)));
  }

  void Run(int tid, csrblocksparse::SpinBarrier* spin_barrier,
           csrblocksparse::MutableVectorView<OutputType> output_view) override {
    this->layer_->SpMM_bias(
        csrblocksparse::VectorView<RhsType>(this->input_buffer_), &output_view,
        this->relu_, tid, this->per_column_barrier_ ? spin_barrier : nullptr);
    spin_barrier->barrier();
    Reset(tid, spin_barrier);
  }

  // The part of |input_buffer_| updated by the previous layer corresponding to
  // time = t (out of all past values). It is the bottom
  // |num_inputs_to_update_| rows of the current column.
  csrblocksparse::MutableVectorView<RhsType> InputViewToUpdate() override {
    return csrblocksparse::MutableVectorView<RhsType>(
        this->input_buffer_.data() + this->input_buffer_rows_ -
            this->num_inputs_to_update_,
        this->num_inputs_to_update_, this->length_, this->input_buffer_rows_);
  }

 private:
  Conv1DLayerWrapper() = delete;
  explicit Conv1DLayerWrapper(
      int num_input_channels, int output_rows, int length,
      int input_buffer_rows, int stride, bool relu, bool per_column_barrier,
      std::unique_ptr<csrblocksparse::SparseLinearLayer<WeightType, RhsType>>
          layer)
      : Super(num_input_channels, output_rows, length, input_buffer_rows,
              length, relu, per_column_barrier, std::move(layer)),
        num_inputs_to_update_(
            std::min(stride * num_input_channels, input_buffer_rows)) {}

  // For Conv1D layers, |stride| > 1 is supported. Every time
  // |stride| * |num_input_channels| elements are pushed in from the bottom.
  //
  // For example, for a layer of |kernel_size| = 3 and |stride| = 1, the memory
  // layout of |input_buffer_| after loading v0 and calling Reset() should look
  // like this (v0 is the input vector at t = 0, v1 is at t = 1, and so on):
  //
  //  |    | \                           //
  //  |----|  \                          //
  //  | v0 |   --> |kernel_size| stacks  //
  //  |----|  /                          //
  //  |    | /                           //
  //
  // Leaving enough space to load v1. Then after loading v1, it should be:
  //
  //  |    |                             //
  //  |----|                             //
  //  | v0 |                             //
  //  |----|                             //
  //  | v1 |                             //
  //
  // On the other hand, If |stride| = 2, then after the first loading, which
  // pushes in v0 and v1. Calling Reset() would move them two stacks up (and
  // pushing v0 out), and the buffer looks like this:
  //
  //  | v1 |                             //
  //  |----|                             //
  //  |    | \                           //
  //  |----|  --> |stride| spaces        //
  //  |    | /                           //
  //
  // Leaving enough space to load the next 2 inputs. After the next loading,
  // the buffer should look like this:
  //
  //  | v1 |                             //
  //  |----|                             //
  //  | v2 |                             //
  //  |----|                             //
  //  | v3 |                             //
  //
  void Reset(int tid, csrblocksparse::SpinBarrier* spin_barrier) override {
    // If |num_inputs_to_update_| == |input_buffer_rows_| it means that
    // the whole buffer is overwritten evey time, so there is no need to move
    // elements.
    if (this->num_inputs_to_update_ < this->input_buffer_rows_) {
      if (tid == 0) {
        // Shift the current column up by |num_inputs_to_update_| elements.
        std::move(this->input_buffer_.data() + this->num_inputs_to_update_,
                  this->input_buffer_.data() + this->input_buffer_rows_,
                  this->input_buffer_.data());
      }
      spin_barrier->barrier();
    }
  }

  // Number of input elements to update after each matrix multiplication. Equal
  // to the minimum between |input_buffer_rows_| and
  // |num_input_channels_| * stride (not stored).
  const int num_inputs_to_update_;
};

}  // namespace codec
}  // namespace chromemedia

#endif  // LYRA_CODEC_CONV1D_LAYER_WRAPPER_H_
