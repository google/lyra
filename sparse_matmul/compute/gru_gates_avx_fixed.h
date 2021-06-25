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

#ifndef LYRA_CODEC_SPARSE_MATMUL_COMPUTE_GRU_GATES_AVX_FIXED_H_
#define LYRA_CODEC_SPARSE_MATMUL_COMPUTE_GRU_GATES_AVX_FIXED_H_

#include <cstdint>
#if defined __AVX2__
#include <immintrin.h>
#endif
#include <vector>

#include "sparse_matmul/compute/ar_inputs.h"
#include "sparse_matmul/numerics/fast_transcendentals.h"

namespace csrblocksparse {

#if defined __AVX2__

constexpr int kAVX2SIMDWidth = 8;

// Loads 8x fixed32 from |ptr0| and adds to |input|.
// If |kTwoInputs|, also loads from |ptr1| and adds that as well.
// Returns the 2 or 3-way sum.
template <bool kTwoInputs>
inline __m256i LoadAndAddFixed32(const int32_t* ptr0, const int32_t* ptr1,
                                 const __m256i& input) {
  __m256i data0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(ptr0));
  if (kTwoInputs) {
    __m256i data1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(ptr1));
    data0 = _mm256_add_epi32(data0, data1);
  }
  return _mm256_add_epi32(data0, input);
}

// Loads 8x fixed32 from ptr0.
// If |kTwoInputs|, also loads from |ptr1| and adds.
// Multiplies the loaded values by the factor and adds to |input|, which also
// is converted to float.
// Returns the sum.
template <bool kTwoInputs>
inline __m256 LoadMultiplyAddToFloat(const int32_t* ptr0, const int32_t* ptr1,
                                     const __m256& float_factor,
                                     const __m256& input) {
  __m256i data0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(ptr0));
  if (kTwoInputs) {
    __m256i data1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(ptr1));
    data0 = _mm256_add_epi32(data0, data1);
  }
  __m256 float_result = _mm256_cvtepi32_ps(data0);
  float_result = _mm256_mul_ps(float_result, float_factor);
  return _mm256_add_ps(float_result, input);
}

// Loads 16x float in 2x 8x registers from |ptr0_1| and multiplies by
// |input_pairs|, likewise formatted as 8x floats, alternating between the two
// AR inputs and sums each pair of results, making 8x float results.
// If |kThreeInputs|, also loads 8x float from |ptr2| and multiplies by
// |third_input|, which must be formatted as 8x float. The second product is
// added to the previous result.
// Returns the sum added to |accumulator|.
template <bool kThreeInputs>
inline __m256 MultiplyAddFloat(const __m256& input_pairs,
                               const __m256& third_input, const float* ptr0_1,
                               const float* ptr2, const __m256& accumulator) {
  __m256 data_pair0 = _mm256_load_ps(ptr0_1);
  __m256 data_pair1 = _mm256_load_ps(ptr0_1 + 8);
  data_pair0 = _mm256_mul_ps(data_pair0, input_pairs);
  data_pair1 = _mm256_mul_ps(data_pair1, input_pairs);
  data_pair0 = _mm256_hadd_ps(data_pair0, data_pair1);
  // Swap the middle 2 64 bit pairs to correct the hadd result.
  data_pair0 = _mm256_permute4x64_pd(data_pair0, 0xd8);
  if (kThreeInputs) {
    // Load 256 bits (8 x float) of data, then multiply-accumulate.
    data_pair1 = _mm256_load_ps(ptr2);
    data_pair1 = _mm256_mul_ps(data_pair1, third_input);
    data_pair0 = _mm256_add_ps(data_pair0, data_pair1);
  }
  // Add conditioning.
  return _mm256_add_ps(data_pair0, accumulator);
}

// Processes the tanh and the final combination, returns the new GRU state.
template <int kInputMantissaBits, int kStateMantissaBits, bool kSplitGates>
inline __m256i GRUComputeState(const __m256& cell0, const __m256& cell1,
                               const __m256& reset0, const __m256& reset1,
                               const __m256& update0, const __m256& update1,
                               const int32_t* gate_ptr,
                               const int32_t* gate_other_ptr,
                               const void* gru_h_ptr) {
  // Multiply the cell gru output and the reset.
  __m256 float_gru0 = LoadMultiplyAddToFloat<kSplitGates>(
      gate_ptr, gate_other_ptr, reset0, cell0);
  __m256 float_gru1 = LoadMultiplyAddToFloat<kSplitGates>(
      gate_ptr + kAVX2SIMDWidth, gate_other_ptr + kAVX2SIMDWidth, reset1,
      cell1);
  // Compute tanh on the result.
  __m256 hbar0, hbar1;
  float_tanh_float<kInputMantissaBits, TM_ORDER4_FLOAT>(float_gru0, float_gru1,
                                                        hbar0, hbar1);
  // Load the 16-bit previous gru state and update.
  __m256i gru = _mm256_load_si256(reinterpret_cast<__m256i const*>(gru_h_ptr));
  __m256 state_factor =
      _mm256_set1_ps(1.0f / (static_cast<float>(1 << kStateMantissaBits)));
  float_gru0 =
      _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(gru)));
  float_gru1 = _mm256_cvtepi32_ps(
      _mm256_cvtepi16_epi32(_mm256_extractf128_si256(gru, 1)));
  float_gru0 = _mm256_mul_ps(float_gru0, state_factor);
  float_gru1 = _mm256_mul_ps(float_gru1, state_factor);
  float_gru0 = _mm256_sub_ps(float_gru0, hbar0);
  float_gru1 = _mm256_sub_ps(float_gru1, hbar1);
  float_gru0 = _mm256_mul_ps(float_gru0, update0);
  float_gru1 = _mm256_mul_ps(float_gru1, update1);
  state_factor = _mm256_set1_ps(static_cast<float>(1 << kStateMantissaBits));
  float_gru0 = _mm256_add_ps(float_gru0, hbar0);
  float_gru1 = _mm256_add_ps(float_gru1, hbar1);
  float_gru0 = _mm256_mul_ps(float_gru0, state_factor);
  float_gru1 = _mm256_mul_ps(float_gru1, state_factor);
  return PackFloatsToFixed16(float_gru0, float_gru1);
}

// According to |kInputsMode|, processes 0, 2 or 3 autoregressive inputs and
// combines with |input| and |gates*|.
// With 2 AR inputs, loads 8x pairs of float from |pair_weights| and multiplies
// by |paired_ar|, likewise formatted as 8x float, but scaled such that the
// product with pair_weights is on the same scale as |*input| and |*gates0|,
// and sums each pair result, making 8x float results.
// If 3 AR inputs, also loads 8x float from |third_weights| and multiplies by
// |third_ar|, which must be formatted as 8x scaled floats. The second product
// is added to the previous result.
// Inputs, 8x fixed32 are loaded from |input|, and added to the total.
// Finally 8x fixed32 from |gates0| (and |gates1| if |kTwoGates|) are added as
// well.
// Returns the total sum as a float, but on the scale of |*input|.
template <bool kTwoGates, ARInputsMode kInputsMode>
inline __m256i GruInput32ToFloat(const __m256& paired_ar,
                                 const __m256& third_ar,
                                 const float* pair_weights,
                                 const float* third_weights,
                                 const int32_t* gates0, const int32_t* gates1,
                                 const int32_t* input) {
  __m256i data32 = _mm256_load_si256(reinterpret_cast<__m256i const*>(input));
  data32 = LoadAndAddFixed32<kTwoGates>(gates0, gates1, data32);
  __m256 float_data = _mm256_cvtepi32_ps(data32);
  if (kInputsMode != ARInputsMode::k0ARInputs) {
    float_data = MultiplyAddFloat<kInputsMode == ARInputsMode::k3ARInputs>(
        paired_ar, third_ar, pair_weights, third_weights, float_data);
  }
  return float_data;
}

// Generic GRU gates function controlled by template parameters thus:
// - |kInputBits|: the mantissa bits in |*input_ptr|, |*gru_recurrent_ptr|.
// - |kStateBits|: the mantissa_bits in |*gru_state_ptr|.
// - |kInputsMode == |k0ARInputs|: There are no autoregressive inputs so
//   |ar_sample, |ar_sample1|, |ar_sample2|, |ar_01_weights|, |ar_2_weights| are
//   ignored.
// - |kInputsMode| == |k2ARInputs|: |ar_sample0|, |ar_sample1| are multiplied by
//   |ar_01_weights| and added to the (conditioning) input.
// - |kInputsMode| == |k3ARInputs|: |ar_sample2| is multiplied by |ar_2_weights|
//   and added to the other two AR inputs (and added to the conditioning input).
// - |kReplicas| determines the number of duplicates of the output to be
//   written, separated by |replica_stride|. If zero, then the number of
//   replicas is variable and taken from the |replicas| argument.
// - If |kSplitGates| is true: The |*gru_recurrent_other_ptr| is secondary
//   recurrent input that must be added to |*gru_recurrent_ptr|.
// - |start|, |end| are |rows| in [0, |state_size|] to be processed by this
//   thread.
//
// Previous state is read from |*gru_state_ptr| and the new state is written to
// *(|gru_state_ptr| + i * |replica_stride| for i in [0, |kReplicas|]).
template <int kInputBits, int kStateBits,
          ARInputsMode kInputsMode = ARInputsMode::k0ARInputs,
          int kReplicas = 1, bool kSplitGates = false>
inline void GruGatesTemplate(
    int start, int end, int state_size, int replicas, int replica_stride,
    const int32_t* gru_recurrent_ptr, const int32_t* input_ptr,
    const std::pair<float, float>* ar_sample01, const float* ar_01_weights,
    const float* ar_sample2, const float* ar_2_weights,
    const int32_t* gru_recurrent_other_ptr, int16_t* gru_state_ptr) {
  constexpr int kQRIncrement = kAVX2SIMDWidth;
  // Increment all the pointers to save on pointer arithmetic in the loop.
  input_ptr += start;
  gru_state_ptr += start;
  gru_recurrent_ptr += start;
  if (kSplitGates) gru_recurrent_other_ptr += start;
  __m256 ar_2_inputs, ar_3rd_input;
  if (kInputsMode != ARInputsMode::k0ARInputs) {
    ar_01_weights += 2 * start;
    ar_2_inputs = _mm256_castsi256_ps(
        _mm256_set1_epi64x(*reinterpret_cast<const int64_t*>(ar_sample01)));
    if (kInputsMode == ARInputsMode::k3ARInputs) {
      ar_2_weights += start;
      ar_3rd_input = _mm256_set1_ps(*ar_sample2);
    } else {
      ar_3rd_input = {};
    }
  } else {
    ar_2_inputs = {};
    ar_3rd_input = {};
  }
  // The transcendentals handle 2x registers of data at once, so we have to do
  // everything in duplicate.
  for (int i = start; i < end; i += kQRIncrement * 2) {
    // Load 8 pairs of fixed16s for each of reset, update and cell.
    __m256 reset0 = GruInput32ToFloat<kSplitGates, kInputsMode>(
        ar_2_inputs, ar_3rd_input, ar_01_weights, ar_2_weights,
        gru_recurrent_ptr, gru_recurrent_other_ptr, input_ptr);
    __m256 reset1 = GruInput32ToFloat<kSplitGates, kInputsMode>(
        ar_2_inputs, ar_3rd_input, ar_01_weights + 2 * kQRIncrement,
        ar_2_weights + kQRIncrement, gru_recurrent_ptr + kAVX2SIMDWidth,
        gru_recurrent_other_ptr + kAVX2SIMDWidth, input_ptr + kAVX2SIMDWidth);
    float_sigmoid_float<kInputBits>(reset0, reset1);
    __m256 update0 = GruInput32ToFloat<kSplitGates, kInputsMode>(
        ar_2_inputs, ar_3rd_input, ar_01_weights + 2 * state_size,
        ar_2_weights + state_size, gru_recurrent_ptr + state_size,
        gru_recurrent_other_ptr + state_size, input_ptr + state_size);
    __m256 update1 = GruInput32ToFloat<kSplitGates, kInputsMode>(
        ar_2_inputs, ar_3rd_input,
        ar_01_weights + 2 * state_size + 2 * kQRIncrement,
        ar_2_weights + state_size + kQRIncrement,
        gru_recurrent_ptr + state_size + kAVX2SIMDWidth,
        gru_recurrent_other_ptr + state_size + kAVX2SIMDWidth,
        input_ptr + state_size + kAVX2SIMDWidth);
    float_sigmoid_float<kInputBits>(update0, update1);
    __m256 cell0 = _mm256_cvtepi32_ps(_mm256_load_si256(
        reinterpret_cast<__m256i const*>(input_ptr + 2 * state_size)));
    __m256 cell1 =
        _mm256_cvtepi32_ps(_mm256_load_si256(reinterpret_cast<__m256i const*>(
            input_ptr + 2 * state_size + kAVX2SIMDWidth)));
    if (kInputsMode != ARInputsMode::k0ARInputs) {
      cell0 = MultiplyAddFloat<kInputsMode == ARInputsMode::k3ARInputs>(
          ar_2_inputs, ar_3rd_input, ar_01_weights + 4 * state_size,
          ar_2_weights + 2 * state_size, cell0);
      cell1 = MultiplyAddFloat<kInputsMode == ARInputsMode::k3ARInputs>(
          ar_2_inputs, ar_3rd_input,
          ar_01_weights + 4 * state_size + 2 * kQRIncrement,
          ar_2_weights + 2 * state_size + kQRIncrement, cell1);
    }
    __m256i gru_state = GRUComputeState<kInputBits, kStateBits, kSplitGates>(
        cell0, cell1, reset0, reset1, update0, update1,
        gru_recurrent_ptr + 2 * state_size,
        gru_recurrent_other_ptr + 2 * state_size, gru_state_ptr);
    if (kReplicas > 0) {
      // With |kReplicas| a template parameter, the compiler will unroll the
      // loop.
      for (int j = 0; j < kReplicas; ++j) {
        _mm256_store_si256(
            reinterpret_cast<__m256i*>(gru_state_ptr + j * replica_stride),
            gru_state);
      }
    } else {
      // This loop will not unroll as replicas is variable.
      for (int j = 0; j < replicas; ++j) {
        _mm256_store_si256(
            reinterpret_cast<__m256i*>(gru_state_ptr + j * replica_stride),
            gru_state);
      }
    }
    // Increment all the pointers.
    input_ptr += 2 * kAVX2SIMDWidth;
    gru_state_ptr += 2 * kAVX2SIMDWidth;
    gru_recurrent_ptr += 2 * kAVX2SIMDWidth;
    if (kSplitGates) gru_recurrent_other_ptr += 2 * kAVX2SIMDWidth;
    if (kInputsMode != ARInputsMode::k0ARInputs) {
      ar_01_weights += 4 * kQRIncrement;
      if (kInputsMode == ARInputsMode::k3ARInputs)
        ar_2_weights += 2 * kQRIncrement;
    }
  }
}

// Dispatches calls to the GruGatesTemplate function above converting the
// replicas variable argument to a template parameter to allow the compiler to
// unroll the write loop.
// |ar_sample01| packs sample 0 and 1 into a pair because the QR weights are
// formatted with the weights interleaved for sample 0 and 1. The two samples
// represent coarse and fine for WaveRNN.
template <int kInputBits, int kStateBits,
          ARInputsMode kInputsMode = ARInputsMode::k2ARInputs,
          bool kSplitGates = false>
inline void GruGatesAVXFixed(
    int start, int end, int state_size, const int32_t* gru_recurrent_ptr,
    const int32_t* input_ptr, const std::pair<float, float>* ar_sample01,
    const float* ar_01_weights, int num_replicas, int replica_stride,
    const float* ar_sample2, const float* ar_2_weights,
    const int32_t* gru_recurrent_other_ptr, int16_t* gru_state_ptr) {
  // Convert the number of replicas from a variable to a template parameter
  // with a switch. This enables the compiler to unroll the loop for
  // the write, making it faster for common numbers of threads.
  switch (num_replicas) {
    case 1:
      GruGatesTemplate<kInputBits, kStateBits, kInputsMode, /*kReplicas=*/1,
                       kSplitGates>(
          start, end, state_size, num_replicas, replica_stride,
          gru_recurrent_ptr, input_ptr, ar_sample01, ar_01_weights, ar_sample2,
          ar_2_weights, gru_recurrent_other_ptr, gru_state_ptr);
      break;
    case 2:
      GruGatesTemplate<kInputBits, kStateBits, kInputsMode, /*kReplicas=*/2,
                       kSplitGates>(
          start, end, state_size, num_replicas, replica_stride,
          gru_recurrent_ptr, input_ptr, ar_sample01, ar_01_weights, ar_sample2,
          ar_2_weights, gru_recurrent_other_ptr, gru_state_ptr);
      break;
    case 4:
      GruGatesTemplate<kInputBits, kStateBits, kInputsMode, /*kReplicas=*/4,
                       kSplitGates>(
          start, end, state_size, num_replicas, replica_stride,
          gru_recurrent_ptr, input_ptr, ar_sample01, ar_01_weights, ar_sample2,
          ar_2_weights, gru_recurrent_other_ptr, gru_state_ptr);
      break;
    case 6:
      GruGatesTemplate<kInputBits, kStateBits, kInputsMode, /*kReplicas=*/6,
                       kSplitGates>(
          start, end, state_size, num_replicas, replica_stride,
          gru_recurrent_ptr, input_ptr, ar_sample01, ar_01_weights, ar_sample2,
          ar_2_weights, gru_recurrent_other_ptr, gru_state_ptr);
      break;
    default:
      // Zero |kReplicas| tells the function to use the |num_replicas| variable.
      GruGatesTemplate<kInputBits, kStateBits, kInputsMode, /*kReplicas=*/0,
                       kSplitGates>(
          start, end, state_size, num_replicas, replica_stride,
          gru_recurrent_ptr, input_ptr, ar_sample01, ar_01_weights, ar_sample2,
          ar_2_weights, gru_recurrent_other_ptr, gru_state_ptr);
  }
}

#endif  // __AVX2__

}  // namespace csrblocksparse

#endif  // LYRA_CODEC_SPARSE_MATMUL_COMPUTE_GRU_GATES_AVX_FIXED_H_
