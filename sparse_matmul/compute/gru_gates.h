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

#ifndef LYRA_CODEC_SPARSE_MATMUL_COMPUTE_GRU_GATES_H_
#define LYRA_CODEC_SPARSE_MATMUL_COMPUTE_GRU_GATES_H_

#include <cstdint>
#include <vector>

// IWYU pragma: begin_exports
#include "sparse_matmul/compute/ar_inputs.h"
#include "sparse_matmul/compute/gru_gates_arm.h"
#include "sparse_matmul/compute/gru_gates_avx_fixed.h"
#include "sparse_matmul/compute/gru_gates_generic.h"
#include "sparse_matmul/compute/matmul.h"
#include "sparse_matmul/numerics/fixed_types.h"
#include "sparse_matmul/numerics/type_utils.h"
#include "sparse_matmul/vector/cache_aligned_vector.h"
// IWYU pragma: end_exports

namespace csrblocksparse {

// The master template is really a catch-all for the unimplemented cases to
// run the generics.
template <typename GRUStateType, typename InputType, typename SampleType = void>
class GruGates : public MatmulBase {
 public:
  using SampleWeightType = float;
  static constexpr int kSIMDWidth = kGenericSIMDWidth;

  // Generic GRU function covers all uses for WaveRNN-like architectures and
  // conditioning.
  // Controlled by template parameters thus:
  // - |kInputsMode| == |k0ARInputs|: There are no autoregressive inputs so
  //   |ar_sample0|, |ar_sample1|, |ar_sample2|, |ar_01_weights|,
  //   |ar_2_weights| are ignored.
  // - |kInputsMode| == |k2ARInputs|: |ar_sample0|, |ar_sample1| are multiplied
  //   by |ar_01_weights| and added to the (conditioning) input.
  // - |kInputsMode| == |k3ARInputs|: |ar_sample2| is multiplied by
  //   |ar_2_weights| and added to the other two |ar_inputs| (and added to the
  //   conditioning input).
  // - If |kSplitGates| is true: The |*gru_recurrent_other_ptr| is secondary
  //   recurrent input that must be added to |*gru_recurrent_ptr|.
  // - |num_replicas| determines the number of duplicates of the output to be
  //   written, separated by |replica_stride|.
  // - |start|, |end| are |rows| in [0, |state_size|] to be processed by this
  //   thread.
  //
  // Previous state is read from |*gru_state_ptr| and the new state is written
  // to *(|gru_state_ptr| + i * |replica_stride| for i in [0, |num_replicas|)).
  template <ARInputsMode kInputsMode = ARInputsMode::k2ARInputs,
            bool kSplitGates = false>
  void GruWithARInput(int start, int end, int state_size,
                      const InputType* gru_recurrent_ptr,
                      const InputType* input_ptr, GRUStateType* gru_state_ptr,
                      const SampleType* ar_sample0 = nullptr,
                      const SampleType* ar_sample1 = nullptr,
                      const SampleWeightType* ar_01_weights = nullptr,
                      int num_replicas = 1, int replica_stride = 0,
                      const SampleType* ar_sample2 = nullptr,
                      const SampleWeightType* ar_2_weights = nullptr,
                      const InputType* gru_recurrent_other_ptr = nullptr) {
    CHECK_EQ(num_replicas, 1) << "Generic code should always have 1 replica";
    GoThroughGates<GRUStateType, InputType, SampleWeightType, SampleType,
                   kInputsMode, kSplitGates>(
        start, end, ar_01_weights, gru_recurrent_ptr, gru_recurrent_other_ptr,
        input_ptr, gru_state_ptr, ar_2_weights, state_size, ar_sample0,
        ar_sample1, ar_sample2);
  }

  // No AR inputs, no split gates, no batching, no replicated outputs.
  // TODO(b/188702959): Redirect conditioning GRU here, removing code from
  // gru_layer.h.
  // Copy to specializations.
  void PlainGru(int start, int end, int state_size,
                const InputType* gru_recurrent_ptr, const InputType* input_ptr,
                GRUStateType* gru_state_ptr) {
    GruWithARInput<ARInputsMode::k0ARInputs>(
        start, end, state_size, gru_recurrent_ptr, input_ptr, gru_state_ptr);
  }
};

#if defined __ARM_NEON || defined __aarch64__
// Partial specialization for float.
template <>
class GruGates<float, float, float> : public MatmulBase {
 public:
  static constexpr int kSIMDWidth = kNeonSIMDWidth;

  // Generic GRU function covers all uses for WaveRNN-like architectures and
  // conditioning.
  template <ARInputsMode kInputsMode = ARInputsMode::k2ARInputs,
            bool kSplitGates = false>
  void GruWithARInput(int start, int end, int state_size,
                      const float* gru_recurrent_data, const float* input_data,
                      float* gru_state_data, const float* ar_sample0 = nullptr,
                      const float* ar_sample1 = nullptr,
                      const float* ar_01_weights = nullptr,
                      int num_replicas = 1, int replica_stride = 0,
                      const float* ar_sample2 = nullptr,
                      const float* ar_2_weights = nullptr,
                      const float* gru_recurrent_other_data = nullptr) {
    DCHECK_EQ(num_replicas, 1) << "ARM code should always have 1 replica";
    GoThroughGatesFloat<kInputsMode, kSplitGates>(
        start, end, ar_01_weights, gru_recurrent_data, gru_recurrent_other_data,
        input_data, gru_state_data, ar_2_weights, state_size, ar_sample0,
        ar_sample1, ar_sample2);
  }
};
#endif  // defined __ARM_NEON || defined __aarch64__

// Partial specialization for fixed types. The sample weights are always float
// whatever the fixed type of the other weights.
template <int kGRUStateBits, int kInputBits, int kSampleBits>
class GruGates<fixed16<kGRUStateBits>, fixed32<kInputBits>,
               fixed16<kSampleBits>> : public MatmulBase {
 public:
#if defined __ARM_NEON || defined __aarch64__
  static constexpr int kSIMDWidth = kNeonSIMDWidth;
#elif defined __AVX2__
  static constexpr int kSIMDWidth = kAVX2SIMDWidth * 2;
#else   // Generic case.
  static constexpr int kSIMDWidth = kGenericSIMDWidth;
#endif  // __ARM_NEON || defined __aarch64__ / __AVX2__

  using GRUStateType = fixed16<kGRUStateBits>;
  using InputType = fixed32<kInputBits>;
  using SampleType = fixed16<kSampleBits>;
  using SampleWeightType = float;
  static constexpr int kInputMantissaBits = InputType::kMantissaBits;
  static constexpr int kSampleMantissaBits = SampleType::kMantissaBits;
  static constexpr int kStateMantissaBits = GRUStateType::kMantissaBits;
  // Generic GRU function covers all uses for WaveRNN-like architectures and
  // conditioning.
  template <ARInputsMode kInputsMode = ARInputsMode::k2ARInputs,
            bool kSplitGates = false>
  void GruWithARInput(int start, int end, int state_size,
                      const InputType* gru_recurrent_data,
                      const InputType* input_data, GRUStateType* gru_state_data,
                      const SampleType* ar_sample0 = nullptr,
                      const SampleType* ar_sample1 = nullptr,
                      const SampleWeightType* ar_01_weights = nullptr,
                      int num_replicas = 1, int replica_stride = 0,
                      const SampleType* ar_sample2 = nullptr,
                      const SampleWeightType* ar_2_weights = nullptr,
                      const InputType* gru_recurrent_other_data = nullptr) {
#if defined __ARM_NEON || defined __aarch64__ || defined __AVX2__
    const int32_t* gru_recurrent_ptr =
        reinterpret_cast<const int32_t*>(gru_recurrent_data);
    const int32_t* gru_recurrent_other_ptr =
        reinterpret_cast<const int32_t*>(gru_recurrent_other_data);
    const int32_t* input_ptr = reinterpret_cast<const int32_t*>(input_data);
    int16_t* gru_state_ptr = reinterpret_cast<int16_t*>(gru_state_data);
#if defined __AVX2__
    // The samples are fixed16, but we scale them up here and convert to float
    // so that the product with the QR weights is always on the same scale as
    // InputType, so we don't have to do any more scaling inside.
    const float sample_factor = static_cast<float>(1 << kInputMantissaBits);
#else
    const float sample_factor = 1.0f;
#endif
    // AR sample 0 and 1 are packed into a pair because the QR weights are
    // formatted with the weights interleaved for sample 0 and 1.
    std::pair<float, float> ar_sample01;
    float ar_sample2_float = 0.0f;
    if (kInputsMode == ARInputsMode::k2ARInputs ||
        kInputsMode == ARInputsMode::k3ARInputs) {
      ar_sample01 = {static_cast<float>(*ar_sample0) * sample_factor,
                     static_cast<float>(*ar_sample1) * sample_factor};
      if (kInputsMode == ARInputsMode::k3ARInputs) {
        ar_sample2_float = static_cast<float>(*ar_sample2) * sample_factor;
      }
    }
#if defined __AVX2__
    CHECK(using_avx2_) << "Compiled for AVX2, but cpu flag not set!";
    GruGatesAVXFixed<kInputMantissaBits, kStateMantissaBits, kInputsMode,
                     kSplitGates>(
        start, end, state_size, gru_recurrent_ptr, input_ptr, &ar_sample01,
        ar_01_weights, num_replicas, replica_stride, &ar_sample2_float,
        ar_2_weights, gru_recurrent_other_ptr, gru_state_ptr);
#else   // ARM.
    DCHECK_EQ(num_replicas, 1) << "ARM code should always have 1 replica";
    GoThroughGatesFixed<GRUStateType, InputType, kInputsMode, kSplitGates>(
        start, end, ar_01_weights, gru_recurrent_ptr, gru_recurrent_other_ptr,
        input_ptr, gru_state_ptr, ar_2_weights, state_size, &ar_sample01,
        &ar_sample2_float);
#endif  // __AVX2__ / ARM.
#else   // Generic case.
    CHECK_EQ(num_replicas, 1) << "Generic code should always have 1 replica";
    GoThroughGates<GRUStateType, InputType, SampleWeightType, SampleType,
                   kInputsMode, kSplitGates>(
        start, end, ar_01_weights, gru_recurrent_data, gru_recurrent_other_data,
        input_data, gru_state_data, ar_2_weights, state_size, ar_sample0,
        ar_sample1, ar_sample2);
#endif  // __ARM_NEON || defined __aarch64__ / __AVX2__
  }
};

}  // namespace csrblocksparse

#endif  // LYRA_CODEC_SPARSE_MATMUL_COMPUTE_GRU_GATES_H_
