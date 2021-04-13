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

#ifndef LYRA_CODEC_CAUSAL_CONVOLUTIONAL_CONDITIONING_H_
#define LYRA_CODEC_CAUSAL_CONVOLUTIONAL_CONDITIONING_H_

#include <memory>
#include <string>

#include "glog/logging.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "dsp_util.h"
#include "layer_wrappers_lib.h"
#include "lyra_types.h"
#include "sparse_inference_matrixvector.h"

namespace chromemedia {
namespace codec {

// Computes conditioning using a convolutional network.
template <typename Types>
class CausalConvolutionalConditioning {
 public:
  using DiskWeightType = typename Types::DiskWeightType;
  using Conv1DWeightType = typename Types::Conv1DWeightType;
  using Conv1DRhsType = typename Types::Conv1DRhsType;
  using CondStack0WeightType = typename Types::CondStack0WeightType;
  using CondStack0RhsType = typename Types::CondStack0RhsType;
  using CondStack1WeightType = typename Types::CondStack1WeightType;
  using CondStack1RhsType = typename Types::CondStack1RhsType;
  using CondStack2WeightType = typename Types::CondStack2WeightType;
  using CondStack2RhsType = typename Types::CondStack2RhsType;
  using Transpose0WeightType = typename Types::Transpose0WeightType;
  using Transpose0RhsType = typename Types::Transpose0RhsType;
  using Transpose1WeightType = typename Types::Transpose1WeightType;
  using Transpose1RhsType = typename Types::Transpose1RhsType;
  using Transpose2WeightType = typename Types::Transpose2WeightType;
  using Transpose2RhsType = typename Types::Transpose2RhsType;
  using ConvCondWeightType = typename Types::ConvCondWeightType;
  using ConvCondRhsType = typename Types::ConvCondRhsType;
  using ConvCondOutputType = typename Types::ConvCondOutputType;
  using ConvToGatesWeightType = typename Types::ConvToGatesWeightType;
  using ConvToGatesRhsType = typename Types::ConvToGatesRhsType;
  using ConvToGatesOutType = typename Types::ConvToGatesOutType;

  using Conv1DLayerType = LayerWrapper<Conv1DWeightType, Conv1DRhsType,
                                       CondStack0RhsType, DiskWeightType>;
  using CondStack0LayerType =
      LayerWrapper<CondStack0WeightType, CondStack0RhsType, CondStack1RhsType,
                   DiskWeightType>;
  using CondStack1LayerType =
      LayerWrapper<CondStack1WeightType, CondStack1RhsType, CondStack2RhsType,
                   DiskWeightType>;
  using CondStack2LayerType =
      LayerWrapper<CondStack2WeightType, CondStack2RhsType, Transpose0RhsType,
                   DiskWeightType>;
  using Transpose0LayerType =
      LayerWrapper<Transpose0WeightType, Transpose0RhsType, Transpose1RhsType,
                   DiskWeightType>;
  using Transpose1LayerType =
      LayerWrapper<Transpose1WeightType, Transpose1RhsType, Transpose2RhsType,
                   DiskWeightType>;
  using Transpose2LayerType =
      LayerWrapper<Transpose2WeightType, Transpose2RhsType, ConvCondRhsType,
                   DiskWeightType>;
  using ConvCondLayerType = LayerWrapper<ConvCondWeightType, ConvCondRhsType,
                                         ConvCondOutputType, DiskWeightType>;
  using ConvToGatesLayerType =
      LayerWrapper<ConvToGatesWeightType, ConvToGatesRhsType,
                   ConvToGatesOutType, DiskWeightType>;

  using InputType = typename Types::Conv1DRhsType;
  using OutputType = typename Types::OutputType;

  // |num_threads| must be less than or equal to |num_cond_hiddens|.
  // |num_cond_hiddens| are the number of hidden states in the conditioning
  // stack, while |num_hiddens| are the ones in the main RNN stack. This is
  // needed because the last layers are actually mapping the output of the
  // conditioning stack into the RNN space.
  // |num_samples_per_hop| must be greater than 0.
  CausalConvolutionalConditioning(int feature_depth, int num_cond_hiddens,
                                  int num_hiddens, int num_samples_per_hop,
                                  int num_frames_per_packet, int num_threads,
                                  const std::string& path,
                                  const std::string& prefix)
      : feature_depth_(feature_depth),
        num_hiddens_(num_hiddens),
        num_cond_hiddens_(num_cond_hiddens),
        num_samples_per_hop_(num_samples_per_hop),
        num_frames_per_packet_(num_frames_per_packet),
        num_threads_(num_threads),
        path_(path),
        prefix_(prefix),
        num_precomputed_frames_(0),
        spin_barrier_(num_threads_) {
    // Crash ok.
    CHECK_LE(num_threads_, num_cond_hiddens)
        << "Number of threads must be <= the number of hidden layers "
           "but were "
        << num_threads_ << " and " << num_cond_hiddens_;
    CHECK_GT(num_threads_, 0) << "Number of threads must be > 0.";
    CHECK_GT(num_samples_per_hop_, 0)
        << "Number of samples per hop must be > 0.";
    CHECK_GT(num_frames_per_packet_, 0)
        << "Number of frames per packet must be > 0.";

    CreateLayers();
    PrepareOutput();
  }

  ~CausalConvolutionalConditioning() {}

  // Return the conditioning vector corresponding to |step| in sample domain.
  absl::Span<OutputType> AtStep(int step) {
    const int samples_per_cond_output =
        num_samples_per_hop_ / kCondUpsamplingRatio;
    const int conditioning_column =
        (step % (num_frames_per_packet_ * num_samples_per_hop_)) /
        samples_per_cond_output;
    const int num_output_elements = 3 * num_hiddens_;
    return absl::Span<OutputType>(
        conditioning_.data() + conditioning_column * num_output_elements,
        num_output_elements);
  }

  void Precompute(const csrblocksparse::FatCacheAlignedVector<float>& input,
                  int num_threads) {
    CHECK_EQ(input.cols(), kCondInputNumTimesteps);
    CHECK_EQ(feature_depth_, input.rows());
    InsertNewInput(input);

    auto f = [this](csrblocksparse::SpinBarrier* barrier, int tid) {
      ComputeFunction(barrier, tid);
    };
    LaunchOnThreadsWithBarrier(num_threads_, f);
  }

  int num_samples() const {
    return num_precomputed_frames_ * num_samples_per_hop_;
  }

 private:
  // TODO(b/161825447): Allow more general layer connections.
  static constexpr int kConv1DKernel = 3;
  static constexpr int kDilatedKernel = 2;
  static constexpr int kDilation[] = {1, 2, 4};
  static constexpr int kTranspose[] = {1, 2, 4};

  // This is the upsampling ratio per transpose layer.
  static constexpr int kTransposeStride = 2;
  static constexpr int kCondInputNumTimesteps = 1;
  static constexpr int kCondUpsamplingRatio = 8;

  static LayerParams Conv1DParams(int feature_depth, int num_cond_hiddens,
                                  int num_threads,
                                  const std::string& model_path,
                                  const std::string& prefix) {
    return LayerParams{.num_input_channels = feature_depth,
                       .num_filters = num_cond_hiddens,
                       .length = 1,
                       .kernel_size = kConv1DKernel,
                       .dilation = 1,
                       .stride = 1,
                       .relu = false,
                       .skip_connection = false,
                       .type = LayerType::kConv1D,
                       .num_threads = num_threads,
                       .per_column_barrier = false,
                       .from =
                           LayerParams::FromDisk{
                               .path = model_path,
                               .zipped = true,
                           },
                       .prefix = prefix + "_conv1d_"};
  }

  // All dilated layers differ only in their input buffer's number of
  // columns, which depends on the dilations (1, 2, or 4). They also
  // have skip connections.
  static LayerParams DilatedParams(int num_cond_hiddens, int level,
                                   int num_threads,
                                   const std::string& model_path,
                                   const std::string& prefix) {
    return LayerParams{
        .num_input_channels = num_cond_hiddens,
        .num_filters = num_cond_hiddens,
        .length = 1,
        .kernel_size = kDilatedKernel,
        .dilation = kDilation[level],
        .stride = 1,
        .relu = false,
        .skip_connection = true,
        .type = LayerType::kDilated,
        .num_threads = num_threads,
        .per_column_barrier = false,
        .from = LayerParams::FromDisk{.path = model_path, .zipped = true},
        .prefix = prefix + absl::StrFormat("_conditioning_stack_%d_", level)};
  }

  // Transpose layers operate on increasingly larger matrices, with number of
  // columns (length) being 1, 2, and 4. Each layer also projects the number
  // of rows from |num_cond_hiddens_| to
  // |kTransposeStride = 2| * |num_cond_hiddens_|.
  // They also have Relu activations after the multiplication.
  static LayerParams TransposeParams(int num_cond_hiddens, int level,
                                     int num_threads,
                                     const std::string& model_path,
                                     const std::string& prefix) {
    return LayerParams{
        .num_input_channels = num_cond_hiddens,
        .num_filters = num_cond_hiddens,
        .length = kTranspose[level],
        .kernel_size = kTransposeStride,
        .dilation = 1,
        .stride = kTransposeStride,
        .relu = true,
        .skip_connection = false,
        .type = LayerType::kTranspose,
        .num_threads = num_threads,
        .per_column_barrier = false,
        .from =
            LayerParams::FromDisk{
                .path = model_path,
                .zipped = true,
            },
        .prefix = prefix + absl::StrFormat("_transpose_%d_", level)};
  }

  // Projection layers project the result from |num_cond_hiddens_| to
  // |num_hiddens_| and to |3 * num_hiddens_| rows successively.
  static LayerParams ConvCondParams(int num_cond_hiddens, int num_hiddens,
                                    int num_threads,
                                    const std::string& model_path,
                                    const std::string& prefix) {
    return LayerParams{.num_input_channels = num_cond_hiddens,
                       .num_filters = num_hiddens,
                       .length = kCondUpsamplingRatio,
                       .kernel_size = 1,
                       .dilation = 1,
                       .stride = 1,
                       .relu = false,
                       .skip_connection = false,
                       .type = LayerType::kConv1D,
                       .num_threads = num_threads,
                       .per_column_barrier = false,
                       .from =
                           LayerParams::FromDisk{
                               .path = model_path,
                               .zipped = true,
                           },
                       .prefix = prefix + "_conv_cond_"};
  }

  static LayerParams ConvToGatesParams(int num_hiddens, int num_threads,
                                       const std::string& model_path,
                                       const std::string& prefix) {
    return LayerParams{.num_input_channels = num_hiddens,
                       .num_filters = 3 * num_hiddens,
                       .length = kCondUpsamplingRatio,
                       .kernel_size = 1,
                       .dilation = 1,
                       .stride = 1,
                       .relu = false,
                       .skip_connection = false,
                       .type = LayerType::kConv1D,
                       .num_threads = num_threads,
                       .per_column_barrier = false,
                       .from =
                           LayerParams::FromDisk{
                               .path = model_path,
                               .zipped = true,
                           },
                       .prefix = prefix + "_conv_to_gates_"};
  }

  void CreateLayers() {
    // TODO(b/161822329): Put these layers in a container.
    const LayerParams conv1d_params = Conv1DParams(
        feature_depth_, num_cond_hiddens_, num_threads_, path_, prefix_);
    conv1d_layer_ = Conv1DLayerType::Create(conv1d_params);
    CHECK_NE(conv1d_layer_, nullptr);

    const LayerParams dilated_params_0 =
        DilatedParams(num_cond_hiddens_, 0, num_threads_, path_, prefix_);
    dilated_conv_layer_0_ = CondStack0LayerType::Create(dilated_params_0);
    CHECK_NE(dilated_conv_layer_0_, nullptr);

    const LayerParams dilated_params_1 =
        DilatedParams(num_cond_hiddens_, 1, num_threads_, path_, prefix_);
    dilated_conv_layer_1_ = CondStack1LayerType::Create(dilated_params_1);
    CHECK_NE(dilated_conv_layer_1_, nullptr);

    const LayerParams dilated_params_2 =
        DilatedParams(num_cond_hiddens_, 2, num_threads_, path_, prefix_);
    dilated_conv_layer_2_ = CondStack2LayerType::Create(dilated_params_2);
    CHECK_NE(dilated_conv_layer_2_, nullptr);

    const LayerParams transpose_params_0 =
        TransposeParams(num_cond_hiddens_, 0, num_threads_, path_, prefix_);
    transpose_conv_layer_0_ = Transpose0LayerType::Create(transpose_params_0);
    CHECK_NE(transpose_conv_layer_0_, nullptr);

    const LayerParams transpose_params_1 =
        TransposeParams(num_cond_hiddens_, 1, num_threads_, path_, prefix_);
    transpose_conv_layer_1_ = Transpose1LayerType::Create(transpose_params_1);
    CHECK_NE(transpose_conv_layer_1_, nullptr);

    const LayerParams transpose_params_2 =
        TransposeParams(num_cond_hiddens_, 2, num_threads_, path_, prefix_);
    transpose_conv_layer_2_ = Transpose2LayerType::Create(transpose_params_2);
    CHECK_NE(transpose_conv_layer_2_, nullptr);

    const LayerParams conv_cond_params = ConvCondParams(
        num_cond_hiddens_, num_hiddens_, num_threads_, path_, prefix_);
    conv_cond_layer_ = ConvCondLayerType::Create(conv_cond_params);
    CHECK_NE(conv_cond_layer_, nullptr);

    const LayerParams conv_to_gates_params =
        ConvToGatesParams(num_hiddens_, num_threads_, path_, prefix_);
    conv_to_gates_layer_ = ConvToGatesLayerType::Create(conv_to_gates_params);
    CHECK_NE(conv_to_gates_layer_, nullptr);
  }

  void PrepareOutput() {
    conv_cond_out_ = csrblocksparse::FatCacheAlignedVector<ConvCondOutputType>(
        num_hiddens_, kCondUpsamplingRatio);
    conv_to_gates_out_ =
        csrblocksparse::FatCacheAlignedVector<ConvToGatesOutType>(
            3 * num_hiddens_, kCondUpsamplingRatio);
    conv_to_gates_out_.FillZero();
    conditioning_ = csrblocksparse::FatCacheAlignedVector<OutputType>(
        conv_to_gates_out_.rows(),
        num_frames_per_packet_ * conv_to_gates_out_.cols());
    conditioning_.FillZero();
  }

  void InsertNewInput(
      const csrblocksparse::FatCacheAlignedVector<float>& input) {
    // This conversion might not always be necessary, will
    // optimize to copy if RhsType == InputType.
    const csrblocksparse::FatCacheAlignedVector<InputType> input_rhs_type(
        input);

    // Copy the values of |input| to the input buffer of the first layer.
    std::copy(input_rhs_type.data(),
              input_rhs_type.data() + input_rhs_type.size(),
              conv1d_layer_->InputViewToUpdate().data());
  }

  void RunLayers(csrblocksparse::SpinBarrier* spin_barrier, int tid) {
    conv1d_layer_->Run(tid, spin_barrier,
                       dilated_conv_layer_0_->InputViewToUpdate());

    // Dilated layers.
    dilated_conv_layer_0_->Run(tid, spin_barrier,
                               dilated_conv_layer_1_->InputViewToUpdate());
    dilated_conv_layer_1_->Run(tid, spin_barrier,
                               dilated_conv_layer_2_->InputViewToUpdate());
    dilated_conv_layer_2_->Run(tid, spin_barrier,
                               transpose_conv_layer_0_->InputViewToUpdate());

    // Transpose layers.
    transpose_conv_layer_0_->Run(tid, spin_barrier,
                                 transpose_conv_layer_1_->InputViewToUpdate());
    transpose_conv_layer_1_->Run(tid, spin_barrier,
                                 transpose_conv_layer_2_->InputViewToUpdate());
    transpose_conv_layer_2_->Run(tid, spin_barrier,
                                 conv_cond_layer_->InputViewToUpdate());

    // Projection layers.
    conv_cond_layer_->Run(
        tid, spin_barrier,
        csrblocksparse::MutableVectorView<ConvCondOutputType>(&conv_cond_out_));

    if (tid == 0) {
      CastVector(0, conv_cond_out_.size(), conv_cond_out_.data(),
                 conv_to_gates_layer_->InputViewToUpdate().data());
    }
    spin_barrier->barrier();

    conv_to_gates_layer_->Run(
        tid, spin_barrier,
        csrblocksparse::MutableVectorView<ConvToGatesOutType>(
            &conv_to_gates_out_));
  }

  void CopyToOutput(csrblocksparse::SpinBarrier* spin_barrier, int tid) {
    // Convert the output to the input type  of the GRU gate in lyra_wavegru.h.
    if (tid == 0) {
      // Shift the content of |conditioning_| if necessary.
      if (num_precomputed_frames_ == num_frames_per_packet_) {
        std::copy(conditioning_.data() + conv_to_gates_out_.size(),
                  conditioning_.data() + conditioning_.size(),
                  conditioning_.data());
      }

      num_precomputed_frames_ =
          std::min(num_precomputed_frames_ + 1, num_frames_per_packet_);
      auto destination_start =
          conditioning_.data() +
          (num_precomputed_frames_ - 1) * conv_to_gates_out_.size();
      CastVector(0, conv_to_gates_out_.size(), conv_to_gates_out_.data(),
                 destination_start);
    }
    spin_barrier->barrier();
  }

  void ComputeFunction(csrblocksparse::SpinBarrier* spin_barrier, int tid) {
    RunLayers(spin_barrier, tid);
    CopyToOutput(spin_barrier, tid);
  }

  const int feature_depth_;  // E.g. the number of mel bins.
  const int num_hiddens_;
  const int num_cond_hiddens_;
  const int num_samples_per_hop_;
  const int num_frames_per_packet_;
  const int num_threads_;
  const std::string path_;
  const std::string prefix_;

  int num_precomputed_frames_;
  csrblocksparse::SpinBarrier spin_barrier_;

  std::unique_ptr<Conv1DLayerType> conv1d_layer_;
  std::unique_ptr<CondStack0LayerType> dilated_conv_layer_0_;
  std::unique_ptr<CondStack1LayerType> dilated_conv_layer_1_;
  std::unique_ptr<CondStack2LayerType> dilated_conv_layer_2_;
  std::unique_ptr<Transpose0LayerType> transpose_conv_layer_0_;
  std::unique_ptr<Transpose1LayerType> transpose_conv_layer_1_;
  std::unique_ptr<Transpose2LayerType> transpose_conv_layer_2_;

  // Wavegru Projection Layers.
  std::unique_ptr<ConvCondLayerType> conv_cond_layer_;
  std::unique_ptr<ConvToGatesLayerType> conv_to_gates_layer_;

  // Buffers before and after |conv_to_gates_layer_|.
  csrblocksparse::FatCacheAlignedVector<ConvCondOutputType> conv_cond_out_;
  csrblocksparse::FatCacheAlignedVector<ConvToGatesOutType> conv_to_gates_out_;

  // Stores |num_frames_per_packet_| frames worth of conditioning output.
  csrblocksparse::FatCacheAlignedVector<OutputType> conditioning_;

  template <typename WeightTypeKindPeer>
  friend class CausalConvolutionalConditioningPeer;
};

}  // namespace codec
}  // namespace chromemedia

#endif  // LYRA_CODEC_CAUSAL_CONVOLUTIONAL_CONDITIONING_H_
