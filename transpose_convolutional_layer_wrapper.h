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

#ifndef LYRA_CODEC_TRANSPOSE_CONVOLUTIONAL_LAYER_WRAPPER_H_
#define LYRA_CODEC_TRANSPOSE_CONVOLUTIONAL_LAYER_WRAPPER_H_

#include <memory>
#include <string>
#include <utility>

#include "glog/logging.h"
#include "absl/memory/memory.h"
#include "layer_wrapper.h"
#include "sparse_inference_matrixvector.h"

namespace chromemedia {
namespace codec {

// Class that wraps the data and logic of transpose convolutional layers.
template <typename WeightType, typename RhsType, typename OutputType,
          typename DiskWeightType>
class TransposeConvolutionalLayerWrapper
    : public LayerWrapper<WeightType, RhsType, OutputType, DiskWeightType> {
 public:
  using Super = LayerWrapper<WeightType, RhsType, OutputType, DiskWeightType>;

  static std::unique_ptr<TransposeConvolutionalLayerWrapper<
      WeightType, RhsType, OutputType, DiskWeightType>>
  Create(const LayerParams& params) {
    const std::string layer_prompt = "|" + params.prefix + "| layer: ";

    // TODO(b/161015017): Support more general stride and kernel size
    // combinations.
    if (params.kernel_size != params.stride) {
      LOG(ERROR) << layer_prompt
                 << "Transpose convolutional layer with |kernel_size| != "
                 << "|stride| is not supported.";
      return nullptr;
    }
    if (params.dilation != 1) {
      LOG(ERROR) << layer_prompt
                 << "Transpose convolutional layer with |dilation| != 1"
                 << "is not supported.";
      return nullptr;
    }
    if (params.skip_connection) {
      LOG(ERROR) << layer_prompt
                 << "Transpose convolutional layer does not support "
                 << "skip connections.";
      return nullptr;
    }

    auto layer =
        Super::LoadAndCheckLayer(params.from, params.prefix, layer_prompt,
                                 params.kernel_size * params.num_filters,
                                 params.num_input_channels, params.num_threads);
    if (layer == nullptr) {
      return nullptr;
    }
    const int input_buffer_rows = layer->cols();
    const int num_input_channels = input_buffer_rows;
    const int output_rows = layer->rows();

    return absl::WrapUnique(
        new TransposeConvolutionalLayerWrapper<WeightType, RhsType, OutputType,
                                               DiskWeightType>(
            num_input_channels, output_rows, params.length, input_buffer_rows,
            params.dilation * params.length, params.relu,
            params.per_column_barrier, std::move(layer)));
  }

  void Run(int tid, csrblocksparse::SpinBarrier* spin_barrier,
           csrblocksparse::MutableVectorView<OutputType> output_view) override {
    // Reshape the |output_view| to be compatible with the matrix
    // multiplication.
    output_view.reshape(this->output_rows_, this->length_);
    this->layer_->SpMM_bias(
        csrblocksparse::VectorView<RhsType>(this->input_buffer_), &output_view,
        this->relu_, tid, this->per_column_barrier_ ? spin_barrier : nullptr);
    spin_barrier->barrier();
    Reset(tid, spin_barrier);
  }

  // Transpose layers update the whole buffer every time.
  csrblocksparse::MutableVectorView<RhsType> InputViewToUpdate() override {
    return csrblocksparse::MutableVectorView<RhsType>(&this->input_buffer_);
  }

 private:
  TransposeConvolutionalLayerWrapper() = delete;
  explicit TransposeConvolutionalLayerWrapper(
      int num_input_channels, int output_rows, int length,
      int input_buffer_rows, int input_buffer_cols, bool relu,
      bool per_column_barrier,
      std::unique_ptr<csrblocksparse::SparseLinearLayer<WeightType, RhsType>>
          layer)
      : Super(num_input_channels, output_rows, length, input_buffer_rows,
              input_buffer_cols, relu, per_column_barrier, std::move(layer)) {}

  // The whole buffer is overwritten every time so no need to move anything.
  void Reset(int tid, csrblocksparse::SpinBarrier* spin_barrier) override {}
};

}  // namespace codec
}  // namespace chromemedia

#endif  // LYRA_CODEC_TRANSPOSE_CONVOLUTIONAL_LAYER_WRAPPER_H_
