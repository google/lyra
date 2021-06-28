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

#ifndef LYRA_CODEC_SPARSE_MATMUL_COMPUTE_KERNELS_AVX_H_
#define LYRA_CODEC_SPARSE_MATMUL_COMPUTE_KERNELS_AVX_H_

#if defined __AVX__
#include <immintrin.h>

#include <algorithm>
#include <type_traits>
// TODO(b/188702959): Remove fast_transcendentals with GRU refactor.
#include "sparse_matmul/numerics/fast_transcendentals.h"
#include "sparse_matmul/numerics/fixed_types.h"
#include "sparse_matmul/numerics/float16_types.h"
#include "sparse_matmul/numerics/type_utils.h"

namespace csrblocksparse {
namespace detail {

template <typename WeightType, typename RhsType, typename OutType>
struct IsAllowableFloatTypes
    : std::integral_constant<bool, std::is_same<WeightType, float>::value &&
                                       std::is_same<RhsType, float>::value &&
                                       std::is_same<OutType, float>::value> {};

#if defined __AVX2__
// 16-bit inputs, 32-bit output exponent matches sum of input exponents
// OR
// 16-bit inputs, 16-bit output - will shift to match exponent
template <typename WeightType, typename RhsType, typename OutType>
struct IsAllowableFixedTypes
    : std::integral_constant<bool, (IsFixed16Type<WeightType>::value &&
                                    IsFixed16Type<RhsType>::value) &&
                                       (IsFixed32Type<OutType>::value ||
                                        IsFixed16Type<OutType>::value)> {};

template <typename WeightType, typename RhsType, typename OutType>
struct ShouldEnableGenericKernel
    : std::integral_constant<
          bool,
          !IsAllowableFloatTypes<WeightType, RhsType, OutType>::value &&
              !IsAllowableFixedTypes<WeightType, RhsType, OutType>::value> {};

template <typename Type>
struct IsAddableFixedTypes
    : std::integral_constant<bool, IsFixed32Type<Type>::value ||
                                       IsFixed16Type<Type>::value> {};
template <typename Type>
struct ShouldEnableGenericAdd
    : std::integral_constant<bool, !IsAddableFixedTypes<Type>::value> {};

#else   // No AVX2.

template <typename WeightType, typename RhsType, typename OutType>
struct ShouldEnableGenericKernel
    : std::integral_constant<
          bool, !IsAllowableFloatTypes<WeightType, RhsType, OutType>::value> {};

template <typename Type>
struct ShouldEnableGenericAdd : std::true_type {};
#endif  // __AVX2__

template <typename WeightType, typename RhsType, typename OutType>
struct ShouldEnableGenericSpMV_4x4
    : ShouldEnableGenericKernel<WeightType, RhsType, OutType> {};
template <typename WeightType, typename RhsType, typename OutType>
struct ShouldEnableGenericSpMM5_4x4
    : ShouldEnableGenericKernel<WeightType, RhsType, OutType> {};
template <typename WeightType, typename RhsType, typename OutType>
struct ShouldEnableGenericSpMV_1x1 : std::true_type {};
template <typename WeightType, typename RhsType, typename OutType>
struct ShouldEnableGenericSpMM5_1x1 : std::true_type {};

// The computational routines do NO error checking for speed.  It is assumed
// that this has been handled by CSRBlockSparseMatrix.

// In-line function to extract results from a pair of registers and store in
// memory. Note that the non-const references are registers, and are modified
// by this function!
inline void Extract4Results(bool relu, __m256& sum1, __m256& sum2,
                            float** out_ptr) {
  // Horizontally add the results. We have 2 registers, |sum1| and |sum2| that
  // each contain 2 sets of 4 values that need to be added.
  sum1 = _mm256_hadd_ps(sum1, sum2);
  sum1 = _mm256_hadd_ps(sum1, sum1);
  // Now |sum1| contains [|res0|, |res2|, |res0|, |res2|, |res1|, |res3|,
  // |res1|, |res3|]
  if (relu) {
    sum1 = _mm256_max_ps(sum1, _mm256_setzero_ps());
  }
  // It is really hard in AVX to cross the 128 bit 'lanes' and this is the
  // *only* way to do it.
  // Get the top half of |sum1| in to bottom of |sum2|.
  sum2 = _mm256_permute2f128_ps(sum1, sum1, 1);
  // Interleave the values between the two registers.
  sum1 = _mm256_unpacklo_ps(sum1, sum2);
  // Save the lower 128 bits (4 floats).
  __m128 result = _mm256_extractf128_ps(sum1, 0);
  _mm_store_ps(*out_ptr, result);
  *out_ptr += 4;
}

// Performs the calculation y = A * x + b where A is a sparse matrix with a 4x4
// blocked pattern, x is a vector and b is vector. Weights are stored for this
// routine by making each 4x4 block contiguous. Blocks are ordered in standard
// row-major format. column indices are converted to deltas and then multiplied
// by 2 to convert to bytes, so that the value can be used directly to offset
// the pointer into the rhs vector.
//
// NOTE: The bias is expected to have be multiplied by .25f prior to calling
// this function.  This is automatically taken care of in SparseLinearLayer.
// The bias is reconstructed through horizontal additions, leads to a small
// speedup by reducing latencies at the end of the loop.
template <typename WeightType, typename RhsType, typename OutType>
typename std::enable_if<std::is_same<WeightType, float>::value &&
                        std::is_same<RhsType, float>::value &&
                        std::is_same<OutType, float>::value>::type
SpMV_4x4(const WeightType* weights_ptr, const int16_t* col_deltas_bytes,
         const int32_t* nnz_per_row, const RhsType* rhs_ptr,
         const typename TypeOfProduct<WeightType, RhsType>::type* bias_ptr,
         OutType* out_ptr, int64_t assigned_rows,
         int64_t rows /* only used in SpMM variants */,
         int64_t cols /* only used in SpMM variants */, int relu) {
  for (int reduced_row = 0; reduced_row < assigned_rows; ++reduced_row) {
    // Broadcast the biases by 4 to undo the division by 4 in the input biases.
    __m256 sum1 = _mm256_set_m128(_mm_broadcast_ss(bias_ptr + 1),
                                  _mm_broadcast_ss(bias_ptr));
    bias_ptr += 2;
    __m256 sum2 = _mm256_set_m128(_mm_broadcast_ss(bias_ptr + 1),
                                  _mm_broadcast_ss(bias_ptr));
    bias_ptr += 2;

    int reduced_col_count = *nnz_per_row++;
    for (int c = 0; c < reduced_col_count; ++c) {
      int col_delta = *col_deltas_bytes++ / sizeof(RhsType);
      rhs_ptr += col_delta;
      // Multiply this 4x4 block.
      __m256 rhs =
          _mm256_broadcast_ps(reinterpret_cast<const __m128*>(rhs_ptr));
      __m256 weights1 = _mm256_load_ps(weights_ptr);
      weights_ptr += 8;
      sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(weights1, rhs));
      __m256 weights2 = _mm256_load_ps(weights_ptr);
      weights_ptr += 8;
      sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(weights2, rhs));
    }
    Extract4Results(relu, sum1, sum2, &out_ptr);
  }
}

// Performs the calculation y = A * x + b where A is a sparse matrix with a 4x4
// blocked pattern, x is a fat vector with 5 columns and b is vector. b is
// broadcast. Weights are stored for this routine by making each 4x4 block
// contiguous. Blocks are ordered in standard row-major format. column indices
// are converted to deltas and then multiplied by 2 to convert to bytes, so
// that the value can be used directly to offset the pointer into the rhs
// vector.
//
// NOTE: The bias is expected to have be multiplied by .25f prior to calling
// this function.  This is automatically taken care of in SparseLinearLayer.
// The bias is reconstructed through horizontal additions, leads to a small
// speedup by reducing latencies at the end of the loop.
template <typename WeightType, typename RhsType, typename OutType>
typename std::enable_if<std::is_same<WeightType, float>::value &&
                        std::is_same<RhsType, float>::value &&
                        std::is_same<OutType, float>::value>::type
SpMM5_4x4(const WeightType* weights_ptr, const int16_t* col_deltas_bytes,
          const int32_t* nnz_per_row, const RhsType* rhs_ptr,
          const typename TypeOfProduct<WeightType, RhsType>::type* bias_ptr,
          OutType* out_ptr, int64_t assigned_rows, int64_t rows, int64_t cols,
          int relu) {
  const RhsType* rhs_ptrs[5];
  for (int i = 0; i < 5; ++i) rhs_ptrs[i] = rhs_ptr + i * cols;

  OutType* out_ptrs[5];
  for (int i = 0; i < 5; ++i) out_ptrs[i] = out_ptr + i * rows;

  for (int reduced_row = 0; reduced_row < assigned_rows; ++reduced_row) {
    // We will acumulate the results in 10 registers, |sum1_0| to |sum2_4|.
    // Broadcast the biases by 4 to undo the division by 4 in the input biases.
    __m256 sum1_0 = _mm256_set_m128(_mm_broadcast_ss(bias_ptr + 1),
                                    _mm_broadcast_ss(bias_ptr));
    bias_ptr += 2;
    __m256 sum2_0 = _mm256_set_m128(_mm_broadcast_ss(bias_ptr + 1),
                                    _mm_broadcast_ss(bias_ptr));
    bias_ptr += 2;
    __m256 sum1_1 = sum1_0;
    __m256 sum2_1 = sum2_0;
    __m256 sum1_2 = sum1_0;
    __m256 sum2_2 = sum2_0;
    __m256 sum1_3 = sum1_0;
    __m256 sum2_3 = sum2_0;
    __m256 sum1_4 = sum1_0;
    __m256 sum2_4 = sum2_0;

    int reduced_col_count = *nnz_per_row++;
    for (int c = 0; c < reduced_col_count; ++c) {
      int col_delta = *col_deltas_bytes++ / sizeof(RhsType);
      for (int k = 0; k < 5; ++k) rhs_ptrs[k] += col_delta;

      // Multiply this 4x4 block.
      __m256 rhs =
          _mm256_broadcast_ps(reinterpret_cast<const __m128*>(rhs_ptrs[0]));
      __m256 weights1 = _mm256_load_ps(weights_ptr);
      weights_ptr += 8;
      sum1_0 = _mm256_add_ps(sum1_0, _mm256_mul_ps(weights1, rhs));
      __m256 weights2 = _mm256_load_ps(weights_ptr);
      weights_ptr += 8;
      sum2_0 = _mm256_add_ps(sum2_0, _mm256_mul_ps(weights2, rhs));
      rhs = _mm256_broadcast_ps(reinterpret_cast<const __m128*>(rhs_ptrs[1]));
      sum1_1 = _mm256_add_ps(sum1_1, _mm256_mul_ps(weights1, rhs));
      sum2_1 = _mm256_add_ps(sum2_1, _mm256_mul_ps(weights2, rhs));
      rhs = _mm256_broadcast_ps(reinterpret_cast<const __m128*>(rhs_ptrs[2]));
      sum1_2 = _mm256_add_ps(sum1_2, _mm256_mul_ps(weights1, rhs));
      sum2_2 = _mm256_add_ps(sum2_2, _mm256_mul_ps(weights2, rhs));
      rhs = _mm256_broadcast_ps(reinterpret_cast<const __m128*>(rhs_ptrs[3]));
      sum1_3 = _mm256_add_ps(sum1_3, _mm256_mul_ps(weights1, rhs));
      sum2_3 = _mm256_add_ps(sum2_3, _mm256_mul_ps(weights2, rhs));
      rhs = _mm256_broadcast_ps(reinterpret_cast<const __m128*>(rhs_ptrs[4]));
      sum1_4 = _mm256_add_ps(sum1_4, _mm256_mul_ps(weights1, rhs));
      sum2_4 = _mm256_add_ps(sum2_4, _mm256_mul_ps(weights2, rhs));
    }

    Extract4Results(relu, sum1_0, sum2_0, &out_ptrs[0]);
    Extract4Results(relu, sum1_1, sum2_1, &out_ptrs[1]);
    Extract4Results(relu, sum1_2, sum2_2, &out_ptrs[2]);
    Extract4Results(relu, sum1_3, sum2_3, &out_ptrs[3]);
    Extract4Results(relu, sum1_4, sum2_4, &out_ptrs[4]);
  }
}

#ifdef __AVX2__

// In-line function to finish the computation of the result as 4x int32 in
// |sum|.
inline void Compute4Results(bool relu, int kShiftAmount, __m256i& sum) {
  // Horizontally add the results. We have 1 register that contains results
  // [0 0 1 1 2 2 3 3], but hadd (and almost no other AVX instruction) will not
  // cross lanes, so we end up with [0 1 0 1 2 3 2 3]
  sum = _mm256_hadd_epi32(sum, sum);
  // Permutes the middle two pairs to get the answers together.
  sum = _mm256_permute4x64_epi64(sum, 0xd8);
  if (kShiftAmount > 0) {
    // Shift right with rounding to get the right number of mantissa bits.
    __m256i rounding = _mm256_set1_epi32(1 << (kShiftAmount - 1));
    sum = _mm256_add_epi32(sum, rounding);
    sum = _mm256_srai_epi32(sum, kShiftAmount);
  }
  // Now |sum| contains [|res0|, |res1|, |res2|, |res3|, |res0|, |res1|,
  // |res2|, |res3|]
  if (relu) {
    sum = _mm256_max_epi32(sum, _mm256_setzero_si256());
  }
}

// In-line function to extract the 4x int32 results from |sum| to memory.
// Non-const reference for |sum| as it is a register.
inline void Extract4xint32(bool relu, int kShiftAmount, __m256i& sum,
                           int32_t** out_ptr) {
  Compute4Results(relu, kShiftAmount, sum);
  // Save the lower 128 bits (4x int32).
  __m128i result = _mm256_extractf128_si256(sum, 0);
  _mm_store_si128(reinterpret_cast<__m128i*>(*out_ptr), result);
  *out_ptr += 4;
}

// In-line function to extract the 4x int32 results from sum to 4x int16 in
// memory.
// Non-const reference for |sum| as it is a register.
inline void Extract4xint16(bool relu, int kShiftAmount, __m256i& sum,
                           int16_t** out_ptr) {
  Compute4Results(relu, kShiftAmount, sum);
  // Clip to 16 bit range (with saturation) and pack in the bottom 64 bits.
  // Converts the lower 4x int32 in bottom 128 bits to 4x int16 in bottom 64
  // bits, replicated in the next 64 bits.
  sum = _mm256_packs_epi32(sum, sum);
  // Save 4x int 16 from the bottom 64 bits.
  *reinterpret_cast<int64_t*>(*out_ptr) = _mm256_extract_epi64(sum, 0);
  *out_ptr += 4;
}

// Performs the calculation y = A * x + b where A is a sparse matrix with a 4x4
// blocked pattern, x is a vector and b is vector. Weights are stored for this
// routine by making each 4x4 block contiguous. Blocks are ordered in standard
// row-major format. column indices are converted to deltas and then multiplied
// by 2 to convert to bytes, so that the value can be used directly to offset
// the pointer into the rhs vector.
//
// NOTE: The bias is expected to have be multiplied by .25f prior to calling
// this function.  This is automatically taken care of in  SparseLinearLayer.
// The bias is reconstructed through horizontal additions, leads to a small
// speedup by reducing latencies at the end of the loop.
template <typename WeightType, typename RhsType, typename OutType>
typename std::enable_if<
    IsFixed16Type<WeightType>::value && IsFixed16Type<RhsType>::value &&
    (IsFixed32Type<OutType>::value || IsFixed16Type<OutType>::value)>::type
SpMV_4x4(const WeightType* weights_ptr, const int16_t* col_deltas_bytes,
         const int32_t* nnz_per_row, const RhsType* rhs_ptr,
         const typename TypeOfProduct<WeightType, RhsType>::type* bias_ptr,
         OutType* out_ptr, int64_t assigned_rows,
         int64_t rows /* only used in SpMM variants */,
         int64_t cols /* only used in SpMM variants */, int relu) {
  constexpr int kShiftAmount =
      TypeOfProduct<WeightType, RhsType>::type::kMantissaBits -
      OutType::kMantissaBits;
  static_assert(kShiftAmount >= 0,
                "Result must have fewer mantissa bits than product");
  for (int reduced_row = 0; reduced_row < assigned_rows; ++reduced_row) {
    // Load the biases duplicated into a 256 bit register [0 1 2 3 0 1 2 3].
    __m128i bias = _mm_load_si128(reinterpret_cast<__m128i const*>(bias_ptr));
    __m256i biases = _mm256_set_m128i(bias, bias);
    bias_ptr += 4;
    // Swap the top two pairs: [0 1 2 3 2 3 0 1]
    // TODO(b/188702959): consider |_mm256_permutevar8x32|, and set the index
    // register outside the row loop.
    biases = _mm256_permute4x64_epi64(biases, 0xb4);
    // Duplicate the low pairs in each lane: [0 0 1 1 2 2 3 3].
    biases = _mm256_unpacklo_epi32(biases, biases);
    // Double the results to make up for the division by 4.
    // TODO(b/188702959): consider moving this to where the biases are computed.
    __m256i sum = _mm256_add_epi32(biases, biases);

    // TODO(b/188702959): People don't like the old-fashioned, close-to-the-
    // metal notation of *|nnz_per_row|++, so measure the effect of putting the
    // increment in the for loop.
    int reduced_col_count = *nnz_per_row;
    ++nnz_per_row;
    for (int c = 0; c < reduced_col_count; ++c) {
      int col_delta = *col_deltas_bytes++ / sizeof(RhsType);
      rhs_ptr += col_delta;
      // Multiply this 4x4 block.
      // Get the 4x int16 into the bottom of rhs_64.
      __m128i rhs_64 =
          _mm_loadl_epi64(reinterpret_cast<__m128i const*>(rhs_ptr));
      // Load all 16 weights.
      __m256i weights =
          _mm256_load_si256(reinterpret_cast<__m256i const*>(weights_ptr));
      // Broadcast the rhs, pretending that each is a 64-bit unit:
      // [0123 0123 0123 0123].
      __m256i rhs = _mm256_broadcastq_epi64(rhs_64);
      weights_ptr += 16;
      // |_mm256_madd_epi16| does 16x16x16=16x32 bit multiply and horizontally
      // adds adjacent pairs to make 8x32 bit results. Add these to the sum.
      sum = _mm256_add_epi32(sum, _mm256_madd_epi16(weights, rhs));
    }
    static_assert(
        IsFixed16Type<OutType>::value || IsFixed32Type<OutType>::value,
        "AVX2 kernel only supports fixed16 and fixed32 types");
    // The only significant difference between fixed16 and fixed32 is the size
    // of the storage unit. The registers have to be repacked accordingly.
    if (IsFixed32Type<OutType>::value) {
      Extract4xint32(relu, kShiftAmount, sum,
                     reinterpret_cast<int32_t**>(&out_ptr));
    } else {
      Extract4xint16(relu, kShiftAmount, sum,
                     reinterpret_cast<int16_t**>(&out_ptr));
    }
  }
}

// Performs the calculation y = A * x + b where A is a sparse matrix with a 4x4
// blocked pattern, x is a fat vector with 5 columns and b is vector. b is
// broadcast. Weights are stored for this routine by making each 4x4 block
// contiguous. Blocks are ordered in standard row-major format. column indices
// are converted to deltas and then multiplied by 2 to convert to bytes, so
// that the value can be used directly to offset the pointer into the rhs
// vector.
//
// NOTE: The bias is expected to have be multiplied by .25f prior to calling
// this function.  This is automatically taken care of in SparseLinearLayer.
// The bias is reconstructed through horizontal additions, leads to a small
// speedup by reducing latencies at the end of the loop.
template <typename WeightType, typename RhsType, typename OutType>
typename std::enable_if<
    IsFixed16Type<WeightType>::value && IsFixed16Type<RhsType>::value &&
    (IsFixed32Type<OutType>::value || IsFixed16Type<OutType>::value)>::type
SpMM5_4x4(const WeightType* weights_ptr, const int16_t* col_deltas_bytes,
          const int32_t* nnz_per_row, const RhsType* rhs_ptr,
          const typename TypeOfProduct<WeightType, RhsType>::type* bias_ptr,
          OutType* out_ptr, int64_t assigned_rows, int64_t rows, int64_t cols,
          int relu) {
  constexpr int kShiftAmount =
      TypeOfProduct<WeightType, RhsType>::type::kMantissaBits -
      OutType::kMantissaBits;
  static_assert(kShiftAmount >= 0,
                "Result must have fewer mantissa bits than product");
  const RhsType* rhs_ptrs[5];
  for (int i = 0; i < 5; ++i) rhs_ptrs[i] = rhs_ptr + i * cols;

  OutType* out_ptrs[5];
  for (int i = 0; i < 5; ++i) out_ptrs[i] = out_ptr + i * rows;

  for (int reduced_row = 0; reduced_row < assigned_rows; ++reduced_row) {
    // We will acumulate the results in 5 registers, sum_0 to sum_4.
    // Load the biases duplicated into a 256 bit register [0 1 2 3 0 1 2 3].
    __m128i bias = _mm_load_si128(reinterpret_cast<__m128i const*>(bias_ptr));
    __m256i biases = _mm256_set_m128i(bias, bias);
    bias_ptr += 4;
    // Swap the top two pairs: [0 1 2 3 2 3 0 1]
    biases = _mm256_permute4x64_epi64(biases, 0xb4);
    // Duplicate the low pairs in each lane: [0 0 1 1 2 2 3 3].
    biases = _mm256_unpacklo_epi32(biases, biases);
    // Double the results to make up for the division by 4.
    __m256i sum_0 = _mm256_add_epi32(biases, biases);
    __m256i sum_1 = sum_0;
    __m256i sum_2 = sum_0;
    __m256i sum_3 = sum_0;
    __m256i sum_4 = sum_0;

    int reduced_col_count = *nnz_per_row;
    ++nnz_per_row;
    for (int c = 0; c < reduced_col_count; ++c) {
      int col_delta = *col_deltas_bytes++ / sizeof(RhsType);
      for (int k = 0; k < 5; ++k) rhs_ptrs[k] += col_delta;
      // Multiply this 4x4 block.
      // Get the 4x int16 into the bottom of |rhs_64|.
      __m128i rhs_64 =
          _mm_loadl_epi64(reinterpret_cast<__m128i const*>(rhs_ptrs[0]));
      // Load all 16 weights.
      __m256i weights =
          _mm256_load_si256(reinterpret_cast<__m256i const*>(weights_ptr));
      // Broadcast the rhs, pretending that each is a 64-bit unit:
      // [0123 0123 0123 0123].
      __m256i rhs = _mm256_broadcastq_epi64(rhs_64);
      weights_ptr += 16;
      // |_mm256_madd_epi16| does 16x16x16=16x32 bit multiply and horizontally
      // adds adjacent pairs to make 8x32 bit results. Add these to the sum.
      sum_0 = _mm256_add_epi32(sum_0, _mm256_madd_epi16(weights, rhs));
      rhs_64 = _mm_loadl_epi64(reinterpret_cast<__m128i const*>(rhs_ptrs[1]));
      rhs = _mm256_broadcastq_epi64(rhs_64);
      sum_1 = _mm256_add_epi32(sum_1, _mm256_madd_epi16(weights, rhs));
      rhs_64 = _mm_loadl_epi64(reinterpret_cast<__m128i const*>(rhs_ptrs[2]));
      rhs = _mm256_broadcastq_epi64(rhs_64);
      sum_2 = _mm256_add_epi32(sum_2, _mm256_madd_epi16(weights, rhs));
      rhs_64 = _mm_loadl_epi64(reinterpret_cast<__m128i const*>(rhs_ptrs[3]));
      rhs = _mm256_broadcastq_epi64(rhs_64);
      sum_3 = _mm256_add_epi32(sum_3, _mm256_madd_epi16(weights, rhs));
      rhs_64 = _mm_loadl_epi64(reinterpret_cast<__m128i const*>(rhs_ptrs[4]));
      rhs = _mm256_broadcastq_epi64(rhs_64);
      sum_4 = _mm256_add_epi32(sum_4, _mm256_madd_epi16(weights, rhs));
    }
    static_assert(
        IsFixed16Type<OutType>::value || IsFixed32Type<OutType>::value,
        "AVX2 kernel only supports fixed16 and fixed32 types");
    // The only significant difference between fixed16 and fixed32 is the size
    // of the storage unit. The registers have to be repacked accordingly.
    if (IsFixed32Type<OutType>::value) {
      Extract4xint32(relu, kShiftAmount, sum_0,
                     reinterpret_cast<int32_t**>(&out_ptrs[0]));
      Extract4xint32(relu, kShiftAmount, sum_1,
                     reinterpret_cast<int32_t**>(&out_ptrs[1]));
      Extract4xint32(relu, kShiftAmount, sum_2,
                     reinterpret_cast<int32_t**>(&out_ptrs[2]));
      Extract4xint32(relu, kShiftAmount, sum_3,
                     reinterpret_cast<int32_t**>(&out_ptrs[3]));
      Extract4xint32(relu, kShiftAmount, sum_4,
                     reinterpret_cast<int32_t**>(&out_ptrs[4]));
    } else {
      Extract4xint16(relu, kShiftAmount, sum_0,
                     reinterpret_cast<int16_t**>(&out_ptrs[0]));
      Extract4xint16(relu, kShiftAmount, sum_1,
                     reinterpret_cast<int16_t**>(&out_ptrs[1]));
      Extract4xint16(relu, kShiftAmount, sum_2,
                     reinterpret_cast<int16_t**>(&out_ptrs[2]));
      Extract4xint16(relu, kShiftAmount, sum_3,
                     reinterpret_cast<int16_t**>(&out_ptrs[3]));
      Extract4xint16(relu, kShiftAmount, sum_4,
                     reinterpret_cast<int16_t**>(&out_ptrs[4]));
    }
  }
}

// Processes one GRU gate input with sigmoid.
template <int InputMantissaBits, int StateMantissaBits, bool SplitGates>
inline __m256i GRUGateSigmoid(const void* gate_ptr, const void* gate_other_ptr,
                              const __m256i& input,
                              const int32_t* sigmoid_table) {
  __m256i gate = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(gate_ptr));
  if (SplitGates) {
    __m256i other =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(gate_other_ptr));
    gate = _mm256_add_epi32(gate, other);
  }
  gate = _mm256_add_epi32(gate, input);
  // Compute sigmoids on reset and update.
  return csrblocksparse::fixed32_sigmoid_fixed16<InputMantissaBits,
                                                 StateMantissaBits>(
      sigmoid_table, gate);
}

// Processes the tanh and the final combination, returning the new GRU state.
template <int InputMantissaBits, int StateMantissaBits, bool SplitGates = false>
inline __m256i GRUGateState(const __m256i& cell, const __m256i& reset,
                            const __m256i& update,
                            const __m256i& rounding_offset,
                            const void* gate_ptr, const void* gate_other_ptr,
                            const void* gru_h_ptr, const int32_t* tanh_table) {
  // Multiply the cell GRU output and the reset. There is a slight danger of
  // loss of precision here, so use 32x32=64 bit and shift back after.
  __m256i gru = _mm256_loadu_si256(reinterpret_cast<__m256i const*>(gate_ptr));
  if (SplitGates) {
    __m256i other_gru =
        _mm256_loadu_si256(reinterpret_cast<__m256i const*>(gate_other_ptr));
    gru = _mm256_add_epi32(gru, other_gru);
  }
  // This only computes the products of the low-order 32 bits of each pair.
  __m256i gru_lo = _mm256_mul_epi32(gru, reset);
  // Swap odd and even 32-bit units and do it again to get the high products.
  gru = _mm256_shuffle_epi32(gru, 0xb1);
  __m256i gru_hi = _mm256_mul_epi32(gru, _mm256_shuffle_epi32(reset, 0xb1));
  // Now shift right to compensate for the multiply and re-interleave the
  // 32-bit results.
  // NOTE: There is no shift right arithmetic for 64 bit values until AVX512!
  // Fortunately it doesn't matter, as the results are being truncated to 32
  // bits and we aren't shifting right by more than 32 bits here.
  gru_lo = _mm256_srli_epi64(gru_lo, StateMantissaBits);
  // The upper results are shifted LEFT, so we can use blend to recombine in
  // a single instruction.
  gru_hi = _mm256_slli_epi64(gru_hi, 32 - StateMantissaBits);
  // Recombine the 32 bit results from lo and hi, alternating.
  gru = _mm256_blend_epi32(gru_lo, gru_hi, 0xaa);
  gru = _mm256_add_epi32(cell, gru);
  // Compute tanh on the result. Although this instantly discards a bunch of
  // bits, there were only 7 surplus bits for the multiply, which isn't enough
  // to do it as 16x16=32.
  __m256i hbar =
      csrblocksparse::fixed32_tanh_fixed16<InputMantissaBits,
                                           StateMantissaBits>(tanh_table, gru);
  // Load the 16-bit previous GRU state and sign-extend to 32 bits.
  gru = _mm256_cvtepi16_epi32(
      _mm_load_si128(reinterpret_cast<__m128i const*>(gru_h_ptr)));
  gru = _mm256_sub_epi32(gru, hbar);
  // Since |gru| is 16 bit sign-extended to 32, and |update| is the output of
  // sigmoid, it is always contained within 16 bits and never negative, we can
  // use |madd_epi16| to do 16x16=32 multiply with horizontal adding as the
  // addend will always be zero, and this is twice as fast as full blown
  // 32x32=32. The only possible problem is if the subtract above caused
  // overflow.
  gru = _mm256_madd_epi16(gru, update);
  // Renormalize to fixed16. This time rounding is critical, as this is the
  // output GRU state.
  gru = _mm256_add_epi32(gru, rounding_offset);
  gru = _mm256_srai_epi32(gru, StateMantissaBits);
  return _mm256_add_epi32(gru, hbar);
}

template <typename Type>
typename std::enable_if<IsFixed32Type<Type>::value>::type SumVectors(
    int start, int end, const Type* add1, const Type* add2, Type* result) {
  constexpr int kSIMDWidth = 8;
  for (int i = start; i < end; i += kSIMDWidth) {
    __m256i data1 =
        _mm256_load_si256(reinterpret_cast<__m256i const*>(add1 + i));
    __m256i data2 =
        _mm256_load_si256(reinterpret_cast<__m256i const*>(add2 + i));
    data1 = _mm256_add_epi32(data1, data2);
    _mm256_store_si256(reinterpret_cast<__m256i*>(result + i), data1);
  }
}

template <typename Type>
typename std::enable_if<IsFixed16Type<Type>::value>::type SumVectors(
    int start, int end, const Type* add1, const Type* add2, Type* result) {
  constexpr int kSIMDWidth = 16;
  for (int i = start; i < end; i += kSIMDWidth) {
    __m256i data1 =
        _mm256_load_si256(reinterpret_cast<__m256i const*>(add1 + i));
    __m256i data2 =
        _mm256_load_si256(reinterpret_cast<__m256i const*>(add2 + i));
    data1 = _mm256_add_epi16(data1, data2);
    _mm256_store_si256(reinterpret_cast<__m256i*>(result + i), data1);
  }
}

#endif  // __AVX2__

}  // namespace detail
}  // namespace csrblocksparse

#undef LABEL_COL_LOOP
#undef LABEL_ROW_LOOP
#undef LABEL_SKIP_COL_LOOP
#undef LABEL_TOP_LOOP

#endif  // __AVX__

#endif  // LYRA_CODEC_SPARSE_MATMUL_COMPUTE_KERNELS_AVX_H_
