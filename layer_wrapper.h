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

#ifndef LYRA_CODEC_LAYER_WRAPPER_H_
#define LYRA_CODEC_LAYER_WRAPPER_H_

#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "dsp_util.h"
#include "glog/logging.h"
#include "layer_wrapper_interface.h"
#include "sparse_matmul/sparse_matmul.h"

namespace chromemedia {
namespace codec {

// Forward declarations.
template <typename WeightType, typename RhsType, typename OutputType,
          typename DiskWeightType>
class TransposeConvolutionalLayerWrapper;
template <typename WeightType, typename RhsType, typename OutputType,
          typename DiskWeightType>
class DilatedConvolutionalLayerWrapper;
template <typename WeightType, typename RhsType, typename OutputType,
          typename DiskWeightType>
class Conv1DLayerWrapper;

// Class that wraps the data and logic of sparse linear layers.
template <typename WeightType, typename RhsType, typename OutputType,
          typename DiskWeightType>
class LayerWrapper : public LayerWrapperInterface<WeightType, RhsType,
                                                  OutputType, DiskWeightType> {
 public:
  using Input = RhsType;
  using Output = OutputType;

  // Factory method to decide which subclass to create.
  static std::unique_ptr<
      LayerWrapper<WeightType, RhsType, OutputType, DiskWeightType>>
  Create(const LayerParams& params) {
    if (params.type == LayerType::kTranspose) {
      return TransposeConvolutionalLayerWrapper<WeightType, RhsType, OutputType,
                                                DiskWeightType>::Create(params);
    } else if (params.type == LayerType::kDilated) {
      return DilatedConvolutionalLayerWrapper<WeightType, RhsType, OutputType,
                                              DiskWeightType>::Create(params);
    } else if (params.type == LayerType::kConv1D) {
      return Conv1DLayerWrapper<WeightType, RhsType, OutputType,
                                DiskWeightType>::Create(params);
    } else {
      LOG(ERROR) << "Unrecognized type";
      return nullptr;
    }
  }

  // Convenient method used in all subclass creation methods.
  static std::unique_ptr<csrblocksparse::SparseLinearLayer<WeightType, RhsType>>
  LoadAndCheckLayer(
      const std::variant<LayerParams::FromDisk, LayerParams::FromConstant> from,
      const std::string& prefix, const std::string& layer_prompt,
      int expected_rows, int expected_cols, int num_threads) {
    auto layer = absl::make_unique<
        csrblocksparse::SparseLinearLayer<WeightType, RhsType>>();
    if (std::holds_alternative<LayerParams::FromDisk>(from)) {
      const auto from_disk = std::get<LayerParams::FromDisk>(from);
      auto LoadLayer =
          csrblocksparse::LoadSparseLayer<WeightType, RhsType, DiskWeightType>;
      if (!LoadLayer(prefix, from_disk.zipped, layer.get(), from_disk.path)
               .ok()) {
        LOG(ERROR) << layer_prompt << " loading failed.";
        return nullptr;
      }
    } else {
      const auto from_constant = std::get<LayerParams::FromConstant>(from);
      *layer = csrblocksparse::CreateConstantLayer<WeightType, RhsType>(
          expected_rows, expected_cols, from_constant.sparsity,
          from_constant.value);
    }
    LOG(INFO) << layer_prompt << " Shape: [" << layer->rows() << ", "
              << layer->cols() << "]."
              << " Sparsity: " << layer->sparsity();

    // Dimension checks for the loaded layer.
    if ((expected_rows > 0 && layer->rows() != expected_rows) ||
        (expected_cols > 0 && layer->cols() != expected_cols)) {
      LOG(ERROR) << layer_prompt << "Incompatible layer shape: expecting "
                 << "[ " << expected_rows << ", " << expected_cols << "], "
                 << "but is [" << layer->rows() << ", " << layer->cols()
                 << "].";
      return nullptr;
    }

    if (layer->PrepareForThreads(num_threads) != num_threads) {
      LOG(ERROR) << layer_prompt << "Could not prepare for " << num_threads
                 << " threads.";
      return nullptr;
    }
    return layer;
  }

  virtual ~LayerWrapper() {}

  // Runs the layer as a matrix multiplication and a bias-add, optionally
  virtual void Run(
      int tid, csrblocksparse::SpinBarrier* spin_barrier,
      csrblocksparse::MutableVectorView<OutputType> output_view) = 0;

  // The part of |input_buffer_| updated by the previous layer.
  virtual csrblocksparse::MutableVectorView<RhsType> InputViewToUpdate() = 0;

  virtual int PrepareForThreads(int num_threads) {
    return layer_->PrepareForThreads(num_threads);
  }

  virtual int bytes() { return layer_->bytes(); }

  virtual int rows() { return layer_->rows(); }

  virtual int cols() { return layer_->cols(); }

 protected:
  LayerWrapper() = delete;
  explicit LayerWrapper(
      int num_input_channels, int output_rows, int length,
      int input_buffer_rows, int input_buffer_cols, bool relu,
      bool per_column_barrier,
      std::unique_ptr<csrblocksparse::SparseLinearLayer<WeightType, RhsType>>
          layer)
      : num_input_channels_(num_input_channels),
        output_rows_(output_rows),
        length_(length),
        input_buffer_rows_(input_buffer_rows),
        input_buffer_cols_(input_buffer_cols),
        relu_(relu),
        per_column_barrier_(per_column_barrier),
        layer_(std::move(layer)),
        input_buffer_(input_buffer_rows_, input_buffer_cols_) {
    input_buffer_.FillZero();
  }

  // Perform necessary memory shifting after each Run().
  virtual void Reset(int tid, csrblocksparse::SpinBarrier* spin_barrier) = 0;

  // Dimensions of matrices participating in y = Wx + b,
  //   y: |output_rows_| rows and |length_| columns.
  //   W: |output_rows_| rows and |input_buffer_rows_| columns.
  //   x: |input_buffer_rows_| rows and |length_| columns.
  //   b: |output_rows_| rows broadcasted to |length_| columns.
  //
  // Number of input channels. This is also the number of rows (out of
  // |input_buffer_rows_| rows) in |input_buffer_| updated by the previous
  // layer.
  const int num_input_channels_;

  // Number of output filters. This is the number of rows of the result of
  // the matrix multiplication.
  const int output_rows_;

  // Number of columns (out of |input_buffer_cols_|) participating in the
  // matrix multiplication.
  const int length_;

  // Number of rows of the input matrix of the multiplication. Equals to
  // |num_input_channels_| * kernel size (not stored).
  const int input_buffer_rows_;

  // Number of columns of the input buffer. Because a layer may use past
  // results as inputs (e.g. a dilated causal convolutional layer), the buffer
  // needs to store more columns than those actually participate in the matrix
  // multiplication.
  const int input_buffer_cols_;

  // Whether to perform Relu after the matrix multiplication.
  const bool relu_;

  // Whether to synchronize among threads after each column of matrix
  // multiplication is done.
  const bool per_column_barrier_;

  std::unique_ptr<csrblocksparse::SparseLinearLayer<WeightType, RhsType>>
      layer_;
  csrblocksparse::FatCacheAlignedVector<RhsType> input_buffer_;

  template <typename WeightTypeKindPeer,
            template <typename, typename, typename, typename>
            class LayerWrapperTypeTemplate>
  friend class LayerWrapperPeer;
};

// Provide operator<< for unique_ptr to allow using this with CHECK()
// macros.  This is required in gcc/libstdc++ at -std=gnu++17, but
// clang/libstdc++ at the same -std=gnu++17 provides this operator.
// Note that this will not be necessary in c++20, where the operator
// is defined for unique_ptr in the standard:
// https://en.cppreference.com/w/cpp/memory/unique_ptr/operator_ltlt
template <typename WeightType, typename RhsType, typename OutputType,
          typename DiskWeightType>
std::ostream& operator<<(
    std::ostream& out_stream,
    const std::unique_ptr<LayerWrapper<WeightType, RhsType, OutputType,
                                       DiskWeightType>>& layer_ptr) {
  return out_stream << layer_ptr.get();
}

}  // namespace codec
}  // namespace chromemedia

#endif  // LYRA_CODEC_LAYER_WRAPPER_H_
