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

#include "sparse_matmul/compute/matmul_fixed_avx2.h"

#include <cstdint>

#if defined __AVX__
#include <immintrin.h>
#endif

#include "sparse_matmul/compute/matmul.h"

namespace csrblocksparse {
namespace detail {

#if defined __AVX2__
// In-line function computes and returns the result of one row (of blocks) as
// 4x int32_t. |weights_ptr| is a non-const reference so it can easily be
// interpreted as belonging to the caller.
inline __m256i ComputeRowResults(const __m128i& bias128, const int16_t* rhs,
                                 const int16_t* rhs_indices, int nnz,
                                 int16_t const*& weights_ptr) {
  // Expand bias to 64 bits in a 256 bit register [0 z 1 z 2 z 3 z], where z is
  // Zero and 0-3 are the 4x32 bit bias values.
  __m256i sum = _mm256_cvtepu32_epi64(bias128);

  for (int c = 0; c < nnz; ++c) {
    int rhs_index = rhs_indices[c];
    // Load all 16 weights.
    __m256i weights =
        _mm256_load_si256(reinterpret_cast<__m256i const*>(weights_ptr));
    // Get the 4x int16_t into the bottom of |rhs_64|.
    __m128i rhs_64 = _mm_loadl_epi64(
        reinterpret_cast<__m128i const*>(rhs + rhs_index * kBlockSize));
    // Broadcast the rhs, pretending that each is a 64-bit unit:
    // [0123 0123 0123 0123].
    __m256i rhs_value = _mm256_broadcastq_epi64(rhs_64);
    weights_ptr += 16;
    sum = _mm256_add_epi32(sum, _mm256_madd_epi16(weights, rhs_value));
  }
  // Horizontally add the results. We have 1 register that contains results
  // [0 0 1 1 2 2 3 3], but hadd (and almost no other AVX instruction) will not
  // cross lanes, so we end up with [0 1 0 1 2 3 2 3]
  sum = _mm256_hadd_epi32(sum, sum);
  // Permutes the middle two pairs to get the answers together.
  return _mm256_permute4x64_epi64(sum, 0xd8);
}

// Template that allows any fixed combination of OutType and replicas, plus
// variable |relu|, |shift_out|. Note that |kReplicas| is a template arg as
// well as a function arg so we can hard-code a limited amount of unrolling.
template <typename OutType, int kReplicas>
void MatVec4x4FixedAVX2Template(const int16_t* weights_ptr, const int16_t* rhs,
                                const int32_t* bias, const int32_t* nnz_per_row,
                                const int16_t* rhs_indices, int start_row,
                                int end_row, bool relu, int shift_out,
                                int replicas, int stride, OutType* output) {
  int rounding_addon = shift_out > 0 ? (1 << (shift_out - 1)) : 0;
  __m256i rounding = _mm256_set1_epi32(rounding_addon);
  __m256i zero = relu ? _mm256_setzero_si256() : _mm256_set1_epi32(kint32min);
  for (int row_block = start_row; row_block < end_row; ++row_block) {
    // Load 4 biases [0 1 2 3].
    __m128i bias128 = _mm_load_si128(reinterpret_cast<__m128i const*>(bias));
    bias += kBlockSize;
    int nnz = nnz_per_row[row_block];
    __m256i sum =
        ComputeRowResults(bias128, rhs, rhs_indices, nnz, weights_ptr);
    rhs_indices += nnz;
    // Shift right with rounding to get the right number of mantissa bits.
    sum = _mm256_add_epi32(sum, rounding);
    sum = _mm256_srai_epi32(sum, shift_out);
    // Now sum contains [res0, res1, res2, res3, res0, res1, res2, res3]
    sum = _mm256_max_epi32(sum, zero);
    if (sizeof(OutType) == 2) {
      // Clip to 16 bit range (with saturation) and pack in the bottom 64
      // bits. The 64 bit result is replicated across the whole 256 bit
      // register. [0123 0123 0123 0123]
      sum = _mm256_packs_epi32(sum, sum);
      int64_t result = _mm256_extract_epi64(sum, 0);
      *reinterpret_cast<int64_t*>(output) = result;
      if (kReplicas > 1) {
        *reinterpret_cast<int64_t*>(output + stride) = result;
        if (kReplicas > 2) {
          for (int r = 2; r < replicas; ++r) {
            *reinterpret_cast<int64_t*>(output + r * stride) = result;
          }
        }
      }
    } else {
      // Save the lower 128 bits (4x int32_t).
      __m128i result = _mm256_extractf128_si256(sum, 0);
      _mm_store_si128(reinterpret_cast<__m128i*>(output), result);
      if (kReplicas > 1) {
        _mm_store_si128(reinterpret_cast<__m128i*>(output + stride), result);
        if (kReplicas > 2) {
          for (int r = 2; r < replicas; ++r) {
            _mm_store_si128(reinterpret_cast<__m128i*>(output + r * stride),
                            result);
          }
        }
      }
    }
    output += kBlockSize;
  }
}

// Version that covers all possible combinations of the variable conditions:
// |relu|, |shift_out|, |replicas|, with int16_t |output|.
void MatVec4x4FixedAVX2(const int16_t* weights_ptr, const int16_t* rhs,
                        const int32_t* bias, const int32_t* nnz_per_row,
                        const int16_t* rhs_indices, int start_row, int end_row,
                        bool relu, int shift_out, int replicas, int stride,
                        int16_t* output) {
  if (replicas <= 1) {
    MatVec4x4FixedAVX2Template<int16_t, 1>(weights_ptr, rhs, bias, nnz_per_row,
                                           rhs_indices, start_row, end_row,
                                           relu, shift_out, 1, stride, output);
  } else if (replicas == 2) {
    MatVec4x4FixedAVX2Template<int16_t, 2>(weights_ptr, rhs, bias, nnz_per_row,
                                           rhs_indices, start_row, end_row,
                                           relu, shift_out, 2, stride, output);
  } else {
    MatVec4x4FixedAVX2Template<int16_t, 3>(
        weights_ptr, rhs, bias, nnz_per_row, rhs_indices, start_row, end_row,
        relu, shift_out, replicas, stride, output);
  }
}

// Version that covers all possible combinations of the variable conditions:
// |relu|, |shift_out|, |replicas|, with int32_t |output|.
void MatVec4x4FixedAVX2(const int16_t* weights_ptr, const int16_t* rhs,
                        const int32_t* bias, const int32_t* nnz_per_row,
                        const int16_t* rhs_indices, int start_row, int end_row,
                        bool relu, int shift_out, int replicas, int stride,
                        int32_t* output) {
  if (replicas <= 1) {
    MatVec4x4FixedAVX2Template<int32_t, 1>(weights_ptr, rhs, bias, nnz_per_row,
                                           rhs_indices, start_row, end_row,
                                           relu, shift_out, 1, stride, output);
  } else if (replicas == 2) {
    MatVec4x4FixedAVX2Template<int32_t, 2>(weights_ptr, rhs, bias, nnz_per_row,
                                           rhs_indices, start_row, end_row,
                                           relu, shift_out, 2, stride, output);
  } else {
    MatVec4x4FixedAVX2Template<int32_t, 3>(
        weights_ptr, rhs, bias, nnz_per_row, rhs_indices, start_row, end_row,
        relu, shift_out, replicas, stride, output);
  }
}

// In-line function computes and returns the result of one row (of blocks) as
// 8x int32_t. weights_ptr is a non-const reference so it can easily be
// interpreted as belonging to the caller.
inline __m256i Compute8RowResults(const __m256i& bias256, const int16_t* rhs,
                                  const int16_t* rhs_indices, int nnz,
                                  int16_t const*& weights_ptr) {
  // Expand bias to 64 bits in a 256 bit register [0 z 1 z 2 z 3 z], where z is
  // Zero and 0-3 are the 4x32 bit bias values from 128 bit half of the input.
  __m256i sum1 = _mm256_cvtepu32_epi64(_mm256_castsi256_si128(bias256));
  // Plus 4 more in another sum register from the upper 128 bit half.
  __m256i sum2 = _mm256_cvtepu32_epi64(_mm256_extractf128_si256(bias256, 1));

  for (int c = 0; c < nnz; ++c) {
    int rhs_index = rhs_indices[c];
    // Load all 16 weights.
    __m256i weights =
        _mm256_load_si256(reinterpret_cast<__m256i const*>(weights_ptr));
    // Get the 4x int16_t into the bottom of |rhs_64|.
    __m128i rhs_64 = _mm_loadl_epi64(
        reinterpret_cast<__m128i const*>(rhs + rhs_index * kBlockSize));
    // Broadcast the rhs, pretending that each is a 64-bit unit:
    // [0123 0123 0123 0123].
    __m256i rhs_value = _mm256_broadcastq_epi64(rhs_64);
    weights_ptr += 16;
    sum1 = _mm256_add_epi32(sum1, _mm256_madd_epi16(weights, rhs_value));
    // Same again for the other 4 results, re-using the same rhs value.
    weights = _mm256_load_si256(reinterpret_cast<__m256i const*>(weights_ptr));
    weights_ptr += 16;
    sum2 = _mm256_add_epi32(sum2, _mm256_madd_epi16(weights, rhs_value));
  }
  // Horizontally add the results. We have 2 registers that contain results
  // [0 0 1 1 2 2 3 3], and [4 4 5 5 6 6 7 7] but hadd (and almost no other AVX
  // instruction) will not cross lanes, so we end up with [0 1 4 5 2 3 6 7]
  sum1 = _mm256_hadd_epi32(sum1, sum2);
  // Permutes the middle two pairs to get the answers in the right order.
  return _mm256_permute4x64_epi64(sum1, 0xd8);
}

// Version that covers the main conditions used with 8x4:
// |relu|, |shift_out|, with int32_t |output|.
void MatVec8x4FixedAVX2(const int16_t* weights_ptr, const int16_t* rhs,
                        const int32_t* bias, const int32_t* nnz_per_row,
                        const int16_t* rhs_indices, int start_row, int end_row,
                        bool relu, int shift_out, int32_t* output) {
  int rounding_addon = shift_out > 0 ? (1 << (shift_out - 1)) : 0;
  __m256i rounding = _mm256_set1_epi32(rounding_addon);
  __m256i zero = relu ? _mm256_setzero_si256() : _mm256_set1_epi32(kint32min);
  for (int row_block = start_row; row_block < end_row; ++row_block) {
    // Load 4 biases [0 1 2 3 4 5 6 7].
    __m256i bias256 = _mm256_load_si256(reinterpret_cast<__m256i const*>(bias));
    bias += kBlockSize * 2;
    int nnz = nnz_per_row[row_block];
    __m256i sum =
        Compute8RowResults(bias256, rhs, rhs_indices, nnz, weights_ptr);
    rhs_indices += nnz;
    // Shift right with rounding to get the right number of mantissa bits.
    sum = _mm256_add_epi32(sum, rounding);
    sum = _mm256_srai_epi32(sum, shift_out);
    // Now sum contains [res0, res1, res2, res3, res0, res1, res2, res3]
    sum = _mm256_max_epi32(sum, zero);
    // Save the all 256 bits (8x int32_t).
    _mm256_store_si256(reinterpret_cast<__m256i*>(output), sum);
    output += kBlockSize * 2;
  }
}

#endif

}  // namespace detail
}  // namespace csrblocksparse
