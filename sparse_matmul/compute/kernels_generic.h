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

#ifndef LYRA_CODEC_SPARSE_MATMUL_COMPUTE_KERNELS_GENERIC_H_
#define LYRA_CODEC_SPARSE_MATMUL_COMPUTE_KERNELS_GENERIC_H_

#include <algorithm>
#include <type_traits>

#include "sparse_matmul/numerics/fixed_types.h"
#include "sparse_matmul/numerics/float16_types.h"
#include "sparse_matmul/numerics/type_utils.h"

// Separate out the assembly kernels for readability. Eventually this will
// become an ifdef switch on the architecture type.
#if defined __aarch64__
#include "sparse_matmul/compute/kernels_arm.h"
#elif defined __AVX__
#include "sparse_matmul/compute/kernels_avx.h"
#else   // defined __AVX__
// If there is no architecture-specific implementation, then always use generic.
template <typename WeightType, typename RhsType, typename OutType>
struct ShouldEnableGenericSpMV_4x4 : std::true_type {};
template <typename WeightType, typename RhsType, typename OutType>
struct ShouldEnableGenericSpMM5_4x4 : std::true_type {};
template <typename WeightType, typename RhsType, typename OutType>
struct ShouldEnableGenericSpMV_1x1 : std::true_type {};
template <typename WeightType, typename RhsType, typename OutType>
struct ShouldEnableGenericSpMM5_1x1 : std::true_type {};
template <typename Type>
struct ShouldEnableGenericAdd : std::true_type {};
#endif  // defined __arch64__

namespace csrblocksparse {
namespace detail {

// The computational routines do NO error checking for speed.  It is assumed
// that this has been handled by CSRBlockSparseMatrix.

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
typename std::enable_if<
    ShouldEnableGenericSpMV_4x4<WeightType, RhsType, OutType>::value>::type
SpMV_4x4(const WeightType* weights_ptr, const int16_t* col_deltas_bytes,
         const int32_t* nnz_per_row, const RhsType* rhs_ptr,
         const typename TypeOfProduct<WeightType, RhsType>::type* bias_ptr,
         OutType* out_ptr, int64_t assigned_rows,
         int64_t rows /* only used in SpMM variants */,
         int64_t cols /* only used in SpMM variants */, int relu) {
  for (int reduced_row = 0; reduced_row < assigned_rows; ++reduced_row) {
    float accumulators[4];
    // Undo the divion by the happens for the assembly version.
    for (int i = 0; i < 4; ++i)
      accumulators[i] = 4.f * static_cast<float>(*bias_ptr++);

    int reduced_col_count = *nnz_per_row++;
    for (int c = 0; c < reduced_col_count; ++c) {
      int col_delta = *col_deltas_bytes++ / sizeof(RhsType);
      rhs_ptr += col_delta;

      // Multiply this 4x4 block.
      for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
          accumulators[i] += static_cast<float>(*weights_ptr++) *
                             static_cast<float>(rhs_ptr[j]);
        }
      }
    }

    for (int i = 0; i < 4; ++i)
      *out_ptr++ = static_cast<OutType>(relu ? std::max(accumulators[i], 0.f)
                                             : accumulators[i]);
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
    ShouldEnableGenericSpMM5_4x4<WeightType, RhsType, OutType>::value>::type
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
    float accumulators[4][5];
    // Undo the divion by the happens for the assembly version.
    for (int i = 0; i < 4; ++i) {
      for (int k = 0; k < 5; ++k) {
        accumulators[i][k] = 4.f * static_cast<float>(*bias_ptr);
      }
      ++bias_ptr;
    }

    int reduced_col_count = *nnz_per_row++;
    for (int c = 0; c < reduced_col_count; ++c) {
      int col_delta = *col_deltas_bytes++ / sizeof(RhsType);
      for (int k = 0; k < 5; ++k) rhs_ptrs[k] += col_delta;

      // multiply this 4x4 block
      for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
          for (int k = 0; k < 5; ++k) {
            accumulators[i][k] += static_cast<float>(*weights_ptr) *
                                  static_cast<float>(rhs_ptrs[k][j]);
          }
          weights_ptr++;
        }
      }
    }

    for (int k = 0; k < 5; ++k) {
      for (int i = 0; i < 4; ++i) {
        out_ptrs[k][0] = static_cast<OutType>(
            relu ? std::max(accumulators[i][k], 0.f) : accumulators[i][k]);
        out_ptrs[k]++;
      }
    }
  }
}

// Performs the calculation y = A * x + b where A is a sparse matrix with
// a 1x1 blocked pattern (ie unstructured), x is a
// vector and b is vector.
// Weights are stored for this routine in standard CSR format.  Each row must
// have a multiple of 8 columns.
// column indices are converted to deltas and then multiplied by 2 to convert
// to bytes, so that the value can be used directly to offset the pointer
// into the rhs vector.
// NOTE: The bias is expected to have be multiplied by .25f prior to calling
// this function.  This is automatically taken care of in SparseLinearLayer.
// The bias is reconstructed through horizontal additions, leads to a small
// speedup by reducing latencies at the end of the loop.
template <typename WeightType, typename RhsType, typename OutType>
typename std::enable_if<
    ShouldEnableGenericSpMV_1x1<WeightType, RhsType, OutType>::value>::type
SpMV_1x1(const WeightType* weights_ptr, const int16_t* col_deltas_bytes,
         const int32_t* nnz_per_row, const RhsType* rhs_ptr,
         const typename TypeOfProduct<WeightType, RhsType>::type* bias_ptr,
         OutType* out_ptr, int64_t assigned_rows,
         int64_t rows /* only used in SpMM variants */,
         int64_t cols /* only used in SpMM variants */, int relu) {
  for (int row = 0; row < assigned_rows; ++row) {
    // Undo the divion by the happens for the assembly version.
    float accumulator = 4.f * static_cast<float>(*bias_ptr++);

    int col_count = *nnz_per_row++;
    for (int c = 0; c < col_count; ++c) {
      int col_delta = *col_deltas_bytes++ / sizeof(RhsType);
      rhs_ptr += col_delta;

      accumulator +=
          static_cast<float>(*weights_ptr++) * static_cast<float>(*rhs_ptr);
    }

    *out_ptr++ =
        static_cast<OutType>(relu ? std::max(accumulator, 0.f) : accumulator);
  }
}

// Performs the calculation y = A * x + b where A is a sparse matrix with
// a 1x1 blocked pattern (ie unstructured), x is a
// vector and b is vector.
// Weights are stored for this routine in standard CSR format.  Each row must
// have a multiple of 8 columns.
// column indices are converted to deltas and then multiplied by 2 to convert
// to bytes, so that the value can be used directly to offset the pointer
// into the rhs vector.
// NOTE: The bias is expected to have be multiplied by .25f prior to calling
// this function.  This is automatically taken care of in SparseLinearLayer.
// The bias is reconstructed through horizontal additions, leads to a small
// speedup by reducing latencies at the end of the loop.
template <typename WeightType, typename RhsType, typename OutType>
typename std::enable_if<
    ShouldEnableGenericSpMM5_1x1<WeightType, RhsType, OutType>::value>::type
SpMM5_1x1(const WeightType* weights_ptr, const int16_t* col_deltas_bytes,
          const int32_t* nnz_per_row, const RhsType* rhs_ptr,
          const typename TypeOfProduct<WeightType, RhsType>::type* bias_ptr,
          OutType* out_ptr, int64_t assigned_rows, int64_t rows, int64_t cols,
          int relu) {
  const RhsType* rhs_ptrs[5];
  for (int i = 0; i < 5; ++i) rhs_ptrs[i] = rhs_ptr + i * cols;

  OutType* out_ptrs[5];
  for (int i = 0; i < 5; ++i) out_ptrs[i] = out_ptr + i * rows;

  for (int row = 0; row < assigned_rows; ++row) {
    // Undo the divion by the happens for the assembly version.
    float accumulator[5];
    for (int i = 0; i < 5; ++i)
      accumulator[i] = 4.f * static_cast<float>(*bias_ptr);

    ++bias_ptr;

    int col_count = *nnz_per_row++;
    for (int c = 0; c < col_count; ++c) {
      int col_delta = *col_deltas_bytes++ / sizeof(RhsType);
      for (int i = 0; i < 5; ++i) {
        rhs_ptrs[i] += col_delta;
        accumulator[i] += static_cast<float>(*weights_ptr) *
                          static_cast<float>(rhs_ptrs[i][0]);
      }
      weights_ptr++;
    }

    for (int i = 0; i < 5; ++i) {
      out_ptrs[i][0] = static_cast<OutType>(relu ? std::max(accumulator[i], 0.f)
                                                 : accumulator[i]);
      out_ptrs[i]++;
    }
  }
}

template <typename Type>
typename std::enable_if<ShouldEnableGenericAdd<Type>::value>::type SumVectors(
    int start, int end, const Type* add1, const Type* add2, Type* result) {
  LOG_FIRST_N(WARNING, 1) << "SumVectors: using generic kernel!";
  for (int i = start; i < end; ++i) {
    Type sum = static_cast<Type>(static_cast<float>(add1[i]) +
                                 static_cast<float>(add2[i]));
    result[i] = sum;
  }
}

}  // namespace detail
}  // namespace csrblocksparse

#undef LABEL_COL_LOOP
#undef LABEL_ROW_LOOP
#undef LABEL_SKIP_COL_LOOP
#undef LABEL_TOP_LOOP

#endif  // LYRA_CODEC_SPARSE_MATMUL_COMPUTE_KERNELS_GENERIC_H_
