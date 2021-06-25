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

#ifndef LYRA_CODEC_LAYER_WRAPPER_INTERFACE_H_
#define LYRA_CODEC_LAYER_WRAPPER_INTERFACE_H_

#include <string>
#include <variant>

#include "sparse_matmul/sparse_matmul.h"

namespace chromemedia {
namespace codec {

enum class LayerType { kConv1D, kDilated, kTranspose };

// Parameters to construct a LayerWrapper object.
struct LayerParams {
  // Parameters of the convolution operation.
  // Some examples of how to set the parameters for transpose and dilated
  // convolutional layers are in the test.

  // Number of input channels and output filters for this layer. Set to 0 to be
  // dynamically decided by the weights matrix loaded from disk (no dimensions
  // check).
  int num_input_channels = 0;
  int num_filters = 0;
  int length = 1;
  int kernel_size = 1;
  int dilation = 1;
  int num_blocks = 0;

  // Currently |stride| > 1 is only supported for non-dilated layers
  // (|dilation| == 1). For transpose convolutional layers it should be equal
  // to |kernel_size|.
  // TODO(b/161015017): Support more general stride and kernel size
  // combinations.
  int stride = 1;

  // Whether to apply Relu AFTER the matrix multiplication.
  bool relu = false;

  // Whether there a skip connection is applied. If true, then |num_filters|
  // and |num_input_channels| must be the same, so that the input and output
  // have the same shape and can be added.
  bool skip_connection = false;

  LayerType type = LayerType::kConv1D;
  int num_threads = 1;
  bool per_column_barrier = false;

  // Where the layer get its values. Either from disk or from/ a specified
  // constant.
  struct FromDisk {
    // Path to load the weights and biases from disk.
    std::string path = "";

    // Whether the layer's weights and biases are stored in zipped files.
    bool zipped = true;
  };
  struct FromConstant {
    // The constant value to fill the weight matrix.
    float value = 1.0f;

    // Desired sparsity, achieved by probabilistically masking out elements.
    // Sparsity < 0.0 means to create a fully dense layer.
    float sparsity = -1.0f;
  };
  std::variant<FromDisk, FromConstant> from = FromDisk();

  std::string prefix = "";
};

// Abstract class for layer wrappers.
template <typename WeightType, typename RhsType, typename OutputType,
          typename DiskWeightType>
class LayerWrapperInterface {
 public:
  virtual ~LayerWrapperInterface() {}

  // Runs the layer as a matrix multiplication and a bias-add, optionally
  virtual void Run(
      int tid, csrblocksparse::SpinBarrier* spin_barrier,
      csrblocksparse::MutableVectorView<OutputType> output_view) = 0;

  // The part of |input_buffer_| updated by the previous layer.
  virtual csrblocksparse::MutableVectorView<RhsType> InputViewToUpdate() = 0;

  virtual int PrepareForThreads(int num_threads) = 0;

  virtual int bytes() = 0;

  virtual int rows() = 0;

  virtual int cols() = 0;
};

}  // namespace codec
}  // namespace chromemedia

#endif  // LYRA_CODEC_LAYER_WRAPPER_INTERFACE_H_
