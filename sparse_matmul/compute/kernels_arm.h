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

#ifndef LYRA_CODEC_SPARSE_MATMUL_COMPUTE_KERNELS_ARM_H_
#define LYRA_CODEC_SPARSE_MATMUL_COMPUTE_KERNELS_ARM_H_

#if defined __aarch64__

#include <arm_neon.h>

#include <type_traits>

#include "sparse_matmul/numerics/fixed_types.h"
#include "sparse_matmul/numerics/float16_types.h"
#include "sparse_matmul/numerics/type_utils.h"

#define LABEL_COL_LOOP "1"
#define LABEL_ROW_LOOP "2"
#define LABEL_SKIP_COL_LOOP "3"
#define LABEL_TOP_LOOP "4"

namespace csrblocksparse {
namespace detail {

template <typename T>
struct IsFloatOrBfloat
    : std::integral_constant<bool, std::is_same<T, float>::value ||
                                       std::is_same<T, bfloat16>::value> {};

template <typename WeightType, typename RhsType, typename OutType>
struct IsAllowableFloatTypes
    : std::integral_constant<bool, IsFloatOrBfloat<WeightType>::value &&
                                       std::is_same<RhsType, float>::value &&
                                       std::is_same<OutType, float>::value> {};

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
template <typename Type>
struct IsAddableFixedTypes
    : std::integral_constant<bool, IsFixed32Type<Type>::value ||
                                       IsFixed16Type<Type>::value> {};
template <typename Type>
struct ShouldEnableGenericAdd
    : std::integral_constant<bool, !IsAddableFixedTypes<Type>::value> {};

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
typename std::enable_if<std::is_same<WeightType, bfloat16>::value &&
                        std::is_same<RhsType, float>::value &&
                        std::is_same<OutType, float>::value>::type
SpMV_4x4(const bfloat16* weights_ptr, const int16_t* col_deltas_bytes,
         const int32_t* nnz_per_row, const float* rhs_ptr,
         const float* bias_ptr, float* out_ptr, int64_t assigned_rows,
         int64_t rows /* only used in SpMM variants */,
         int64_t cols /* only used in SpMM variants */, int relu) {
  /*   This instrinsic version exists for reference, note that in the
       intrinsic version col_deltas_bytes should NOT actually be in bytes,
       but rather elements.  Intrinsics are 25-35% slower than the
       assembly version.

       for (int r = 0; r < rows; r += 4) {
        int reduced_col_count = nnz_per_row[r / 4];
        float32x4_t accum0 = vdupq_n_f32(bias_ptr + r);
        float32x4_t accum1 = vdupq_n_f32(bias_ptr + r + 1);
        float32x4_t accum2 = vdupq_n_f32(bias_ptr + r + 2);
        float32x4_t accum3 = vdupq_n_f32(bias_ptr + r + 3);
        for (int c = 0; c < reduced_col_count; ++c) {
          int32_t offset = *col_deltas_bytes; col_deltas_bytes++;
          rhs_ptr += offset;
          float32x4_t rhs = vld1q_f32(rhs_ptr);

          uint16x4_t lhs0_int = vld1_u16(weights_ptr); weights_ptr += 4;
          uint16x4_t lhs1_int = vld1_u16(weights_ptr); weights_ptr += 4;
          uint16x4_t lhs2_int = vld1_u16(weights_ptr); weights_ptr += 4;
          uint16x4_t lhs3_int = vld1_u16(weights_ptr); weights_ptr += 4;

          float32x4_t lhs0 = vreinterpretq_f32_u32(vshll_n_u16(lhs0_int, 16));
          float32x4_t lhs1 = vreinterpretq_f32_u32(vshll_n_u16(lhs1_int, 16));
          float32x4_t lhs2 = vreinterpretq_f32_u32(vshll_n_u16(lhs2_int, 16));
          float32x4_t lhs3 = vreinterpretq_f32_u32(vshll_n_u16(lhs3_int, 16));

          accum0 = vmlaq_f32(accum0, lhs0, rhs);
          accum1 = vmlaq_f32(accum1, lhs1, rhs);
          accum2 = vmlaq_f32(accum2, lhs2, rhs);
          accum3 = vmlaq_f32(accum3, lhs3, rhs);
        }

        float32x4_t reduce0 = vpaddq_f32(accum0, accum1);
        float32x4_t reduce1 = vpaddq_f32(accum2, accum3);
        float32x4_t reduce2 = vpaddq_f32(reduce0, reduce1);
        vst1q_f32(out_ptr + r, reduce2);
      } */

  // If the relu is handled in the routine with a comparison and vbit (insert
  // if true), or by branching, then it is slightly, but noticeably slower
  // ~5%, the outer branch avoids that penalty.
  if (relu) {
    asm(
        // Load the first two column deltas.
        "ldrsh x7, [%[col_deltas_bytes]], #2\n"
        "ldrsh x8, [%[col_deltas_bytes]], #2\n"
        // ld1 doesn't support pre-index, so we do the first addition here.
        "add %[rhs_ptr], %[rhs_ptr], x7\n"

        "movi v25.4s, #0\n"

        LABEL_ROW_LOOP
        ":\n"

        // Load the bias.
        "ld1 {v27.4s}, [%[bias_ptr]], #16\n"

        // Zero out local accumulators.
        "dup v28.4s, v27.s[0]\n"  // accum_0 = 0
        "dup v29.4s, v27.s[1]\n"  // accum_1 = 0
        "dup v30.4s, v27.s[2]\n"  // accum_2 = 0
        "dup v31.4s, v27.s[3]\n"  // accum_3 = 0

        // Update the stopping condition for this set of rows.
        "ldr w6, [%[nnz_per_row]], #4\n"
        "cmp w6, #0\n"
        // Skip the body if there isn't anything in this row.
        "beq " LABEL_SKIP_COL_LOOP "f\n"

        LABEL_COL_LOOP
        ":\n"
        // Load 1 Rhs vectors of size 1x4 each.
        "ld1 {v0.4s}, [%[rhs_ptr]], x8\n"

        // Start this load now, which we won't need until the end of the loop.
        "ldrsh x8, [%[col_deltas_bytes]], #2\n"

        // Load 16 Lhs cells corresponding to a 4x4 block.
        "ld1 {v2.8h, v3.8h}, [%[weights_ptr]], #32\n"

        // Convert bfloat16 -> float32.
        "shll  v4.4s, v2.4h, #16\n"
        "shll2 v5.4s, v2.8h, #16\n"
        "shll  v6.4s, v3.4h, #16\n"
        "shll2 v7.4s, v3.8h, #16\n"

        // Multiply-accumulate.
        "fmla v28.4s, v4.4s, v0.4s\n"
        "fmla v29.4s, v5.4s, v0.4s\n"
        "fmla v30.4s, v6.4s, v0.4s\n"
        "fmla v31.4s, v7.4s, v0.4s\n"

        // Loop. Decrement loop index.
        "subs w6, w6, #1\n"  // decrement (reduced) columns left
        "bne " LABEL_COL_LOOP "b\n"

        LABEL_SKIP_COL_LOOP
        ":\n"

        // Horizontally add accumulators and store result
        "faddp v28.4s, v28.4s, v29.4s\n"
        "faddp v30.4s, v30.4s, v31.4s\n"
        "faddp v28.4s, v28.4s, v30.4s\n"

        // Do relu if requested.
        "fmax v28.4s, v28.4s, v25.4s\n"

        // Store accumulators.
        "st1 {v28.4s}, [%[out_ptr]], #16\n"

        // Decrement rows remaining.
        "subs %[assigned_rows], %[assigned_rows], #1\n"
        "bne " LABEL_ROW_LOOP "b\n"

        // clang-format off
        :  // outputs
        [out_ptr] "+r"(out_ptr),
        [weights_ptr] "+r"(weights_ptr),
        [col_deltas_bytes] "+r"(col_deltas_bytes),
        [bias_ptr] "+r"(bias_ptr),
        [nnz_per_row] "+r"(nnz_per_row),
        [assigned_rows] "+r"(assigned_rows),
        [rhs_ptr] "+r"(rhs_ptr)
        :  // inputs
        :  // clobbers
        "cc", "memory", "x6", "x7", "x8", "v0", "v1", "v2", "v3", "v4", "v5",
        "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
        "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
        "v26", "v27", "v28", "v29", "v30", "v31");
    // clang-format on
  } else {
    asm(
        // Load the first two column deltas.
        "ldrsh x7, [%[col_deltas_bytes]], #2\n"
        "ldrsh x8, [%[col_deltas_bytes]], #2\n"
        // ld1 doesn't support pre-index, so we do the first addition here.
        "add %[rhs_ptr], %[rhs_ptr], x7\n"

        LABEL_ROW_LOOP
        ":\n"

        // Load the bias.
        "ld1 {v27.4s}, [%[bias_ptr]], #16\n"

        // Zero out local accumulators.
        "dup v28.4s, v27.s[0]\n"  // accum_0 = 0
        "dup v29.4s, v27.s[1]\n"  // accum_1 = 0
        "dup v30.4s, v27.s[2]\n"  // accum_2 = 0
        "dup v31.4s, v27.s[3]\n"  // accum_3 = 0

        // Update the stopping condition for this set of rows.
        "ldr w6, [%[nnz_per_row]], #4\n"
        "cmp w6, #0\n"
        // Skip the body if there isn't anything in this row.
        "beq " LABEL_SKIP_COL_LOOP "f\n"

        LABEL_COL_LOOP
        ":\n"
        // Load 1 Rhs vectors of size 1x4 each
        "ld1 {v0.4s}, [%[rhs_ptr]], x8\n"

        // Start this load now, which we won't need until the end of the loop.
        "ldrsh x8, [%[col_deltas_bytes]], #2\n"

        // Load 16 Lhs cells corresponding to a 4x4 block.
        "ld1 {v2.8h, v3.8h}, [%[weights_ptr]], #32\n"

        // Convert bfloat16 -> float32.
        "shll  v4.4s, v2.4h, #16\n"
        "shll2 v5.4s, v2.8h, #16\n"
        "shll  v6.4s, v3.4h, #16\n"
        "shll2 v7.4s, v3.8h, #16\n"

        // Multiply-accumulate.
        "fmla v28.4s, v4.4s, v0.4s\n"
        "fmla v29.4s, v5.4s, v0.4s\n"
        "fmla v30.4s, v6.4s, v0.4s\n"
        "fmla v31.4s, v7.4s, v0.4s\n"

        // Loop. Decrement loop index.
        "subs w6, w6, #1\n"  // decrement (reduced) columns left
        "bne " LABEL_COL_LOOP "b\n"

        LABEL_SKIP_COL_LOOP
        ":\n"

        // Horizontally add accumulators and store result.
        "faddp v28.4s, v28.4s, v29.4s\n"
        "faddp v30.4s, v30.4s, v31.4s\n"
        "faddp v28.4s, v28.4s, v30.4s\n"

        // Store accumulators.
        "st1 {v28.4s}, [%[out_ptr]], #16\n"

        // Decrement rows remaining.
        "subs %[assigned_rows], %[assigned_rows], #1\n"
        "bne " LABEL_ROW_LOOP "b\n"

        // clang-format off
        :  // outputs
        [out_ptr] "+r"(out_ptr),
        [weights_ptr] "+r"(weights_ptr),
        [col_deltas_bytes] "+r"(col_deltas_bytes),
        [bias_ptr] "+r"(bias_ptr),
        [nnz_per_row] "+r"(nnz_per_row),
        [assigned_rows] "+r"(assigned_rows),
        [rhs_ptr] "+r"(rhs_ptr)
        :  // inputs
        :  // clobbers
        "cc", "memory", "x6", "x7", "x8", "v0", "v1", "v2", "v3", "v4", "v5",
        "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
        "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
        "v26", "v27", "v28", "v29", "v30", "v31");
    // clang-format on
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
typename std::enable_if<std::is_same<WeightType, bfloat16>::value &&
                        std::is_same<RhsType, float>::value &&
                        std::is_same<OutType, float>::value>::type
SpMM5_4x4(const bfloat16* weights_ptr, const int16_t* col_deltas_bytes,
          const int32_t* nnz_per_row, const float* rhs_ptr,
          const float* bias_ptr, float* out_ptr, int64_t assigned_rows,
          int64_t rows, int64_t cols, int relu) {
  /*   This instrinsic version exists for reference, note that in the
       intrinsic version col_deltas_bytes should NOT actually be in bytes,
       but rather elements.  Intrinsics are 25-35% slower than the
       assembly version.

       for (int r = 0; r < rows; r += 4) {
        int reduced_col_count = nnz_per_row[r / 4];
        float32x4_t accum0 = vdupq_n_f32(bias_ptr + r);
        float32x4_t accum1 = vdupq_n_f32(bias_ptr + r + 1);
        float32x4_t accum2 = vdupq_n_f32(bias_ptr + r + 2);
        float32x4_t accum3 = vdupq_n_f32(bias_ptr + r + 3);
        float32x4_t accum4 = vdupq_n_f32(bias_ptr + r);
        float32x4_t accum5 = vdupq_n_f32(bias_ptr + r + 1);
        float32x4_t accum6 = vdupq_n_f32(bias_ptr + r + 2);
        float32x4_t accum7 = vdupq_n_f32(bias_ptr + r + 3);
        ...
        for (int c = 0; c < reduced_col_count; ++c) {
          int32_t offset = *col_deltas_bytes; col_deltas_bytes++;
          rhs_ptr += offset;
          float32x4_t rhs = vld1q_f32(rhs_ptr);
          float32x4_t rhs2 = vld1q_f32(rhs2_ptr);
          float32x4_t rhs3 = vld1q_f32(rhs3_ptr);
          float32x4_t rhs4 = vld1q_f32(rhs4_ptr);
          float32x4_t rhs5 = vld1q_f32(rhs5_ptr);

          uint16x4_t lhs0_int = vld1_u16(weights_ptr); weights_ptr += 4;
          uint16x4_t lhs1_int = vld1_u16(weights_ptr); weights_ptr += 4;
          uint16x4_t lhs2_int = vld1_u16(weights_ptr); weights_ptr += 4;
          uint16x4_t lhs3_int = vld1_u16(weights_ptr); weights_ptr += 4;

          float32x4_t lhs0 = vreinterpretq_f32_u32(vshll_n_u16(lhs0_int, 16));
          float32x4_t lhs1 = vreinterpretq_f32_u32(vshll_n_u16(lhs1_int, 16));
          float32x4_t lhs2 = vreinterpretq_f32_u32(vshll_n_u16(lhs2_int, 16));
          float32x4_t lhs3 = vreinterpretq_f32_u32(vshll_n_u16(lhs3_int, 16));

          accum0 = vmlaq_f32(accum0, lhs0, rhs);
          accum1 = vmlaq_f32(accum1, lhs1, rhs);
          accum2 = vmlaq_f32(accum2, lhs2, rhs);
          accum3 = vmlaq_f32(accum3, lhs3, rhs);
          accum4 = vmlaq_f32(accum0, lhs0, rhs2);
          accum5 = vmlaq_f32(accum1, lhs1, rhs2);
          accum6 = vmlaq_f32(accum2, lhs2, rhs2);
          accum7 = vmlaq_f32(accum3, lhs3, rhs2);
          ...
        }

        float32x4_t reduce0 = vpaddq_f32(accum0, accum1);
        float32x4_t reduce1 = vpaddq_f32(accum2, accum3);
        float32x4_t reduce2 = vpaddq_f32(reduce0, reduce1);
        vst1q_f32(out_ptr + r, reduce2);

        float32x4_t reduce0 = vpaddq_f32(accum4, accum5);
        float32x4_t reduce1 = vpaddq_f32(accum6, accum7);
        float32x4_t reduce2 = vpaddq_f32(reduce0, reduce1);
        vst1q_f32(out2_ptr + r, reduce2);

        ...
      } */

  // If the relu is handled in the routine with a comparison and vbit (insert
  // if true), or by branching, then it is slightly, but noticeably slower
  // ~5%, the outer branch avoids that penalty.
  //
  // Pointers to the columns.
  const float* rhs2_ptr = rhs_ptr + cols;
  float* out2_ptr = out_ptr + rows;
  const float* rhs3_ptr = rhs_ptr + 2 * cols;
  float* out3_ptr = out_ptr + 2 * rows;
  const float* rhs4_ptr = rhs_ptr + 3 * cols;
  float* out4_ptr = out_ptr + 3 * rows;
  const float* rhs5_ptr = rhs_ptr + 4 * cols;
  float* out5_ptr = out_ptr + 4 * rows;
  if (relu) {
    asm(
        // Load the first two column deltas.
        "ldrsh x7, [%[col_deltas_bytes]], #2\n"
        "ldrsh x8, [%[col_deltas_bytes]], #2\n"
        // ld1 doesn't support pre-index, so we do the first addition here.
        "add %[rhs_ptr], %[rhs_ptr], x7\n"
        "add %[rhs2_ptr], %[rhs2_ptr], x7\n"
        "add %[rhs3_ptr], %[rhs3_ptr], x7\n"
        "add %[rhs4_ptr], %[rhs4_ptr], x7\n"
        "add %[rhs5_ptr], %[rhs5_ptr], x7\n"

        LABEL_ROW_LOOP
        ":\n"

        // Load the bias.
        "ld1 {v27.4s}, [%[bias_ptr]], #16\n"

        // Zero out local accumulators.
        "dup v28.4s, v27.s[0]\n"  // for 1st column
        "dup v29.4s, v27.s[1]\n"  // for 1st column
        "dup v30.4s, v27.s[2]\n"  // for 1st column
        "dup v31.4s, v27.s[3]\n"  // for 1st column
        "dup v23.4s, v27.s[0]\n"  // for 2nd column
        "dup v24.4s, v27.s[1]\n"  // for 2nd column
        "dup v25.4s, v27.s[2]\n"  // for 2nd column
        "dup v26.4s, v27.s[3]\n"  // for 2nd column
        "dup v19.4s, v27.s[0]\n"  // for 3rd column
        "dup v20.4s, v27.s[1]\n"  // for 3rd column
        "dup v21.4s, v27.s[2]\n"  // for 3rd column
        "dup v22.4s, v27.s[3]\n"  // for 3rd column
        "dup v15.4s, v27.s[0]\n"  // for 4th column
        "dup v16.4s, v27.s[1]\n"  // for 4th column
        "dup v17.4s, v27.s[2]\n"  // for 4th column
        "dup v18.4s, v27.s[3]\n"  // for 4th column
        "dup v11.4s, v27.s[0]\n"  // for 5th column
        "dup v12.4s, v27.s[1]\n"  // for 5th column
        "dup v13.4s, v27.s[2]\n"  // for 5th column
        "dup v14.4s, v27.s[3]\n"  // for 5th column

        // Update the stopping condition for this set of rows.
        "ldr w6, [%[nnz_per_row]], #4\n"
        "cmp w6, #0\n"
        // Skip the body if there isn't anything in this row.
        "beq " LABEL_SKIP_COL_LOOP "f\n"

        LABEL_COL_LOOP
        ":\n"
        // Load 1 Rhs vectors of size 1x4 each.
        "ld1 {v0.4s}, [%[rhs_ptr]], x8\n"
        "ld1 {v1.4s}, [%[rhs2_ptr]], x8\n"
        "ld1 {v8.4s}, [%[rhs3_ptr]], x8\n"
        "ld1 {v9.4s}, [%[rhs4_ptr]], x8\n"
        "ld1 {v10.4s}, [%[rhs5_ptr]], x8\n"

        // Start this load now, which we won't need until the end of the loop.
        "ldrsh x8, [%[col_deltas_bytes]], #2\n"

        // Load 16 Lhs cells corresponding to a 4x4 block.
        "ld1 {v2.8h, v3.8h}, [%[weights_ptr]], #32\n"

        // Convert bfloat16 -> float32.
        "shll  v4.4s, v2.4h, #16\n"
        "shll2 v5.4s, v2.8h, #16\n"
        "shll  v6.4s, v3.4h, #16\n"
        "shll2 v7.4s, v3.8h, #16\n"

        // Multiply-accumulate.
        "fmla v28.4s, v4.4s, v0.4s\n"   // for 1st column
        "fmla v29.4s, v5.4s, v0.4s\n"   // for 1st column
        "fmla v30.4s, v6.4s, v0.4s\n"   // for 1st column
        "fmla v31.4s, v7.4s, v0.4s\n"   // for 1st column
        "fmla v23.4s, v4.4s, v1.4s\n"   // for 2nd column
        "fmla v24.4s, v5.4s, v1.4s\n"   // for 2nd column
        "fmla v25.4s, v6.4s, v1.4s\n"   // for 2nd column
        "fmla v26.4s, v7.4s, v1.4s\n"   // for 2nd column
        "fmla v19.4s, v4.4s, v8.4s\n"   // for 3rd column
        "fmla v20.4s, v5.4s, v8.4s\n"   // for 3rd column
        "fmla v21.4s, v6.4s, v8.4s\n"   // for 3rd column
        "fmla v22.4s, v7.4s, v8.4s\n"   // for 3rd column
        "fmla v15.4s, v4.4s, v9.4s\n"   // for 4th column
        "fmla v16.4s, v5.4s, v9.4s\n"   // for 4th column
        "fmla v17.4s, v6.4s, v9.4s\n"   // for 4th column
        "fmla v18.4s, v7.4s, v9.4s\n"   // for 4th column
        "fmla v11.4s, v4.4s, v10.4s\n"  // for 5th column
        "fmla v12.4s, v5.4s, v10.4s\n"  // for 5th column
        "fmla v13.4s, v6.4s, v10.4s\n"  // for 5th column
        "fmla v14.4s, v7.4s, v10.4s\n"  // for 5th column

        // Loop. Decrement loop index.
        "subs w6, w6, #1\n"  // decrement (reduced) columns left
        "bne " LABEL_COL_LOOP "b\n"

        LABEL_SKIP_COL_LOOP
        ":\n"

        "movi v0.4s, #0\n"
        "faddp v28.4s, v28.4s, v29.4s\n"  // 1st column
        "faddp v23.4s, v23.4s, v24.4s\n"  // 2nd column
        "faddp v19.4s, v19.4s, v20.4s\n"  // 3rd column
        "faddp v15.4s, v15.4s, v16.4s\n"  // 4th column
        "faddp v11.4s, v11.4s, v12.4s\n"  // 5th column

        "faddp v30.4s, v30.4s, v31.4s\n"  // 1st column
        "faddp v25.4s, v25.4s, v26.4s\n"  // 2nd column
        "faddp v21.4s, v21.4s, v22.4s\n"  // 3rd column
        "faddp v17.4s, v17.4s, v18.4s\n"  // 4th column
        "faddp v13.4s, v13.4s, v14.4s\n"  // 5th column

        "faddp v28.4s, v28.4s, v30.4s\n"  // 1st column
        "faddp v23.4s, v23.4s, v25.4s\n"  // 2nd column
        "faddp v19.4s, v19.4s, v21.4s\n"  // 3rd column
        "faddp v15.4s, v15.4s, v17.4s\n"  // 4th column
        "faddp v11.4s, v11.4s, v13.4s\n"  // 5th column

        // Do relu as requested.
        "fmax v28.4s, v28.4s, v0.4s\n"
        "fmax v23.4s, v23.4s, v0.4s\n"
        "fmax v19.4s, v19.4s, v0.4s\n"
        "fmax v15.4s, v15.4s, v0.4s\n"
        "fmax v11.4s, v11.4s, v0.4s\n"

        // Store accumulators.
        "st1 {v28.4s}, [%[out_ptr]], #16\n"
        "st1 {v23.4s}, [%[out2_ptr]], #16\n"
        "st1 {v19.4s}, [%[out3_ptr]], #16\n"
        "st1 {v15.4s}, [%[out4_ptr]], #16\n"
        "st1 {v11.4s}, [%[out5_ptr]], #16\n"

        // Decrement rows remaining.
        "subs %[assigned_rows], %[assigned_rows], #1\n"
        "bne " LABEL_ROW_LOOP "b\n"

        // clang-format off
        :  // outputs
        [out_ptr] "+r"(out_ptr),
        [out2_ptr] "+r"(out2_ptr),
        [out3_ptr] "+r"(out3_ptr),
        [out4_ptr] "+r"(out4_ptr),
        [out5_ptr] "+r"(out5_ptr),
        [weights_ptr] "+r"(weights_ptr),
        [col_deltas_bytes] "+r"(col_deltas_bytes),
        [bias_ptr] "+r"(bias_ptr),
        [nnz_per_row] "+r"(nnz_per_row),
        [assigned_rows] "+r"(assigned_rows),
        [rhs_ptr] "+r"(rhs_ptr),
        [rhs2_ptr] "+r"(rhs2_ptr),
        [rhs3_ptr] "+r"(rhs3_ptr),
        [rhs4_ptr] "+r"(rhs4_ptr),
        [rhs5_ptr] "+r"(rhs5_ptr)
        :  // inputs
        :  // clobbers
        "cc", "memory", "x6", "x7", "x8", "v0", "v1", "v2", "v3", "v4", "v5",
        "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
        "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
        "v26", "v27", "v28", "v29", "v30", "v31");
    // clang-format on
  } else {
    asm(
        // Load the first two column deltas.
        "ldrsh x7, [%[col_deltas_bytes]], #2\n"
        "ldrsh x8, [%[col_deltas_bytes]], #2\n"
        // ld1 doesn't support pre-index, so we do the first addition here.
        "add %[rhs_ptr], %[rhs_ptr], x7\n"
        "add %[rhs2_ptr], %[rhs2_ptr], x7\n"
        "add %[rhs3_ptr], %[rhs3_ptr], x7\n"
        "add %[rhs4_ptr], %[rhs4_ptr], x7\n"
        "add %[rhs5_ptr], %[rhs5_ptr], x7\n"

        LABEL_ROW_LOOP
        ":\n"

        // Load the bias.
        "ld1 {v27.4s}, [%[bias_ptr]], #16\n"

        // Zero out local accumulators.
        "dup v28.4s, v27.s[0]\n"  // for 1st column
        "dup v29.4s, v27.s[1]\n"  // for 1st column
        "dup v30.4s, v27.s[2]\n"  // for 1st column
        "dup v31.4s, v27.s[3]\n"  // for 1st column
        "dup v23.4s, v27.s[0]\n"  // for 2nd column
        "dup v24.4s, v27.s[1]\n"  // for 2nd column
        "dup v25.4s, v27.s[2]\n"  // for 2nd column
        "dup v26.4s, v27.s[3]\n"  // for 2nd column
        "dup v19.4s, v27.s[0]\n"  // for 3rd column
        "dup v20.4s, v27.s[1]\n"  // for 3rd column
        "dup v21.4s, v27.s[2]\n"  // for 3rd column
        "dup v22.4s, v27.s[3]\n"  // for 3rd column
        "dup v15.4s, v27.s[0]\n"  // for 4th column
        "dup v16.4s, v27.s[1]\n"  // for 4th column
        "dup v17.4s, v27.s[2]\n"  // for 4th column
        "dup v18.4s, v27.s[3]\n"  // for 4th column
        "dup v11.4s, v27.s[0]\n"  // for 5th column
        "dup v12.4s, v27.s[1]\n"  // for 5th column
        "dup v13.4s, v27.s[2]\n"  // for 5th column
        "dup v14.4s, v27.s[3]\n"  // for 5th column

        // Update the stopping condition for this set of rows.
        "ldr w6, [%[nnz_per_row]], #4\n"
        "cmp w6, #0\n"
        // Skip the body if there isn't anything in this row.
        "beq " LABEL_SKIP_COL_LOOP "f\n"

        LABEL_COL_LOOP
        ":\n"
        // Load 1 Rhs vectors of size 1x4 each.
        "ld1 {v0.4s}, [%[rhs_ptr]], x8\n"
        "ld1 {v1.4s}, [%[rhs2_ptr]], x8\n"
        "ld1 {v8.4s}, [%[rhs3_ptr]], x8\n"
        "ld1 {v9.4s}, [%[rhs4_ptr]], x8\n"
        "ld1 {v10.4s}, [%[rhs5_ptr]], x8\n"

        // Start this load now, which we won't need until the end of the loop.
        "ldrsh x8, [%[col_deltas_bytes]], #2\n"

        // Load 16 Lhs cells corresponding to a 4x4 block.
        "ld1 {v2.8h, v3.8h}, [%[weights_ptr]], #32\n"

        // Convert bfloat16 -> float32.
        "shll  v4.4s, v2.4h, #16\n"
        "shll2 v5.4s, v2.8h, #16\n"
        "shll  v6.4s, v3.4h, #16\n"
        "shll2 v7.4s, v3.8h, #16\n"

        // Multiply-accumulate.
        "fmla v28.4s, v4.4s, v0.4s\n"   // for 1st column
        "fmla v29.4s, v5.4s, v0.4s\n"   // for 1st column
        "fmla v30.4s, v6.4s, v0.4s\n"   // for 1st column
        "fmla v31.4s, v7.4s, v0.4s\n"   // for 1st column
        "fmla v23.4s, v4.4s, v1.4s\n"   // for 2nd column
        "fmla v24.4s, v5.4s, v1.4s\n"   // for 2nd column
        "fmla v25.4s, v6.4s, v1.4s\n"   // for 2nd column
        "fmla v26.4s, v7.4s, v1.4s\n"   // for 2nd column
        "fmla v19.4s, v4.4s, v8.4s\n"   // for 3rd column
        "fmla v20.4s, v5.4s, v8.4s\n"   // for 3rd column
        "fmla v21.4s, v6.4s, v8.4s\n"   // for 3rd column
        "fmla v22.4s, v7.4s, v8.4s\n"   // for 3rd column
        "fmla v15.4s, v4.4s, v9.4s\n"   // for 4th column
        "fmla v16.4s, v5.4s, v9.4s\n"   // for 4th column
        "fmla v17.4s, v6.4s, v9.4s\n"   // for 4th column
        "fmla v18.4s, v7.4s, v9.4s\n"   // for 4th column
        "fmla v11.4s, v4.4s, v10.4s\n"  // for 5th column
        "fmla v12.4s, v5.4s, v10.4s\n"  // for 5th column
        "fmla v13.4s, v6.4s, v10.4s\n"  // for 5th column
        "fmla v14.4s, v7.4s, v10.4s\n"  // for 5th column

        // Loop. Decrement loop index.
        "subs w6, w6, #1\n"  // decrement (reduced) columns left
        "bne " LABEL_COL_LOOP "b\n"

        LABEL_SKIP_COL_LOOP
        ":\n"

        // Horizontally add accumulators and store result.
        "faddp v28.4s, v28.4s, v29.4s\n"  // 1st column
        "faddp v23.4s, v23.4s, v24.4s\n"  // 2nd column
        "faddp v19.4s, v19.4s, v20.4s\n"  // 3rd column
        "faddp v15.4s, v15.4s, v16.4s\n"  // 4th column
        "faddp v11.4s, v11.4s, v12.4s\n"  // 5th column

        "faddp v30.4s, v30.4s, v31.4s\n"  // 1st column
        "faddp v25.4s, v25.4s, v26.4s\n"  // 2nd column
        "faddp v21.4s, v21.4s, v22.4s\n"  // 3rd column
        "faddp v17.4s, v17.4s, v18.4s\n"  // 4th column
        "faddp v13.4s, v13.4s, v14.4s\n"  // 5th column

        "faddp v28.4s, v28.4s, v30.4s\n"  // 1st column
        "faddp v23.4s, v23.4s, v25.4s\n"  // 2nd column
        "faddp v19.4s, v19.4s, v21.4s\n"  // 3rd column
        "faddp v15.4s, v15.4s, v17.4s\n"  // 4th column
        "faddp v11.4s, v11.4s, v13.4s\n"  // 5th column

        // Store accumulators.
        "st1 {v28.4s}, [%[out_ptr]], #16\n"
        "st1 {v23.4s}, [%[out2_ptr]], #16\n"
        "st1 {v19.4s}, [%[out3_ptr]], #16\n"
        "st1 {v15.4s}, [%[out4_ptr]], #16\n"
        "st1 {v11.4s}, [%[out5_ptr]], #16\n"

        // Decrement rows remaining.
        "subs %[assigned_rows], %[assigned_rows], #1\n"
        "bne " LABEL_ROW_LOOP "b\n"

        // clang-format off
        :  // outputs
        [out_ptr] "+r"(out_ptr),
        [out2_ptr] "+r"(out2_ptr),
        [out3_ptr] "+r"(out3_ptr),
        [out4_ptr] "+r"(out4_ptr),
        [out5_ptr] "+r"(out5_ptr),
        [weights_ptr] "+r"(weights_ptr),
        [col_deltas_bytes] "+r"(col_deltas_bytes),
        [bias_ptr] "+r"(bias_ptr),
        [nnz_per_row] "+r"(nnz_per_row),
        [assigned_rows] "+r"(assigned_rows),
        [rhs_ptr] "+r"(rhs_ptr),
        [rhs2_ptr] "+r"(rhs2_ptr),
        [rhs3_ptr] "+r"(rhs3_ptr),
        [rhs4_ptr] "+r"(rhs4_ptr),
        [rhs5_ptr] "+r"(rhs5_ptr)
        :  // inputs
        :  // clobbers
        "cc", "memory", "x6", "x7", "x8", "v0", "v1", "v2", "v3", "v4", "v5",
        "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
        "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
        "v26", "v27", "v28", "v29", "v30", "v31");
    // clang-format on
  }
}

// float implementations below the line.

template <typename WeightType, typename RhsType, typename OutType>
typename std::enable_if<std::is_same<WeightType, float>::value &&
                        std::is_same<RhsType, float>::value &&
                        std::is_same<OutType, float>::value>::type
SpMV_4x4(const float* weights_ptr, const int16_t* col_deltas_bytes,
         const int32_t* nnz_per_row, const float* rhs_ptr,
         const float* bias_ptr, float* out_ptr, int64_t assigned_rows,
         int64_t rows /* only used in SpMM variants */,
         int64_t cols /* only used in SpMM variants */, int relu) {
  /*   This instrinsic version exists for reference, note that in the
       intrinsic version col_deltas_bytes should NOT actually be in bytes,
       but rather elements.  Intrinsics are 25-35% slower than the
       assembly version.

       for (int r = 0; r < rows; r += 4) {
        int reduced_col_count = nnz_per_row[r / 4];
        float32x4_t accum0 = vdupq_n_f32(bias_ptr + r);
        float32x4_t accum1 = vdupq_n_f32(bias_ptr + r + 1);
        float32x4_t accum2 = vdupq_n_f32(bias_ptr + r + 2);
        float32x4_t accum3 = vdupq_n_f32(bias_ptr + r + 3);
        for (int c = 0; c < reduced_col_count; ++c) {
          int32_t offset = *col_deltas_bytes; col_deltas_bytes++;
          rhs_ptr += offset;
          float32x4_t rhs = vld1q_f32(rhs_ptr);

          uint16x4_t lhs0_int = vld1_u16(weights_ptr); weights_ptr += 4;
          uint16x4_t lhs1_int = vld1_u16(weights_ptr); weights_ptr += 4;
          uint16x4_t lhs2_int = vld1_u16(weights_ptr); weights_ptr += 4;
          uint16x4_t lhs3_int = vld1_u16(weights_ptr); weights_ptr += 4;

          float32x4_t lhs0 = vreinterpretq_f32_u32(vshll_n_u16(lhs0_int, 16));
          float32x4_t lhs1 = vreinterpretq_f32_u32(vshll_n_u16(lhs1_int, 16));
          float32x4_t lhs2 = vreinterpretq_f32_u32(vshll_n_u16(lhs2_int, 16));
          float32x4_t lhs3 = vreinterpretq_f32_u32(vshll_n_u16(lhs3_int, 16));

          accum0 = vmlaq_f32(accum0, lhs0, rhs);
          accum1 = vmlaq_f32(accum1, lhs1, rhs);
          accum2 = vmlaq_f32(accum2, lhs2, rhs);
          accum3 = vmlaq_f32(accum3, lhs3, rhs);
        }

        float32x4_t reduce0 = vpaddq_f32(accum0, accum1);
        float32x4_t reduce1 = vpaddq_f32(accum2, accum3);
        float32x4_t reduce2 = vpaddq_f32(reduce0, reduce1);
        vst1q_f32(out_ptr + r, reduce2);
      } */

  // If the relu is handled in the routine with a comparison and vbit (insert
  // if true), or by branching, then it is slightly, but noticeably slower
  // ~5%, the outer branch avoids that penalty.
  if (relu) {
    asm(
        // Load the first two column deltas.
        "ldrsh x7, [%[col_deltas_bytes]], #2\n"
        "ldrsh x8, [%[col_deltas_bytes]], #2\n"
        // ld1 doesn't support pre-index, so we do the first addition here.
        "add %[rhs_ptr], %[rhs_ptr], x7\n"

        "movi v25.4s, #0\n"

        LABEL_ROW_LOOP
        ":\n"

        // Load the bias.
        "ld1 {v27.4s}, [%[bias_ptr]], #16\n"

        // Zero out local accumulators.
        "dup v28.4s, v27.s[0]\n"  // accum_0 = 0
        "dup v29.4s, v27.s[1]\n"  // accum_1 = 0
        "dup v30.4s, v27.s[2]\n"  // accum_2 = 0
        "dup v31.4s, v27.s[3]\n"  // accum_3 = 0

        // Update the stopping condition for this set of rows.
        "ldr w6, [%[nnz_per_row]], #4\n"
        "cmp w6, #0\n"
        // Skip the body if there isn't anything in this row.
        "beq " LABEL_SKIP_COL_LOOP "f\n"

        LABEL_COL_LOOP
        ":\n"
        // Load 1 Rhs vectors of size 1x4 each.
        "ld1 {v0.4s}, [%[rhs_ptr]], x8\n"

        // Start this load now, which we won't need until the end of the loop.
        "ldrsh x8, [%[col_deltas_bytes]], #2\n"

        // Load 16 Lhs cells corresponding to a 4x4 block.
        "ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [%[weights_ptr]], #64\n"

        // Multiply-accumulate.
        "fmla v28.4s, v4.4s, v0.4s\n"
        "fmla v29.4s, v5.4s, v0.4s\n"
        "fmla v30.4s, v6.4s, v0.4s\n"
        "fmla v31.4s, v7.4s, v0.4s\n"

        // Loop. Decrement loop index.
        "subs w6, w6, #1\n"  // decrement (reduced) columns left
        "bne " LABEL_COL_LOOP "b\n"

        LABEL_SKIP_COL_LOOP
        ":\n"

        // Horizontally add accumulators and store result.
        "faddp v28.4s, v28.4s, v29.4s\n"
        "faddp v30.4s, v30.4s, v31.4s\n"
        "faddp v28.4s, v28.4s, v30.4s\n"

        // Do relu as requested.
        "fmax v28.4s, v28.4s, v25.4s\n"

        // Store accumulators.
        "st1 {v28.4s}, [%[out_ptr]], #16\n"

        // Decrement rows remaining.
        "subs %[assigned_rows], %[assigned_rows], #1\n"
        "bne " LABEL_ROW_LOOP "b\n"

        // clang-format off
        :  // outputs
        [out_ptr] "+r"(out_ptr),
        [weights_ptr] "+r"(weights_ptr),
        [col_deltas_bytes] "+r"(col_deltas_bytes),
        [bias_ptr] "+r"(bias_ptr),
        [nnz_per_row] "+r"(nnz_per_row),
        [assigned_rows] "+r"(assigned_rows),
        [rhs_ptr] "+r"(rhs_ptr)
        :  // inputs
        :  // clobbers
        "cc", "memory", "x6", "x7", "x8", "v0", "v1", "v2", "v3", "v4", "v5",
        "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
        "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
        "v26", "v27", "v28", "v29", "v30", "v31");
    // clang-format on
  } else {
    asm(
        // Load the first two column deltas.
        "ldrsh x7, [%[col_deltas_bytes]], #2\n"
        "ldrsh x8, [%[col_deltas_bytes]], #2\n"
        // ld1 doesn't support pre-index, so we do the first addition here.
        "add %[rhs_ptr], %[rhs_ptr], x7\n"

        LABEL_ROW_LOOP
        ":\n"

        // Load the bias.
        "ld1 {v27.4s}, [%[bias_ptr]], #16\n"

        // Zero out local accumulators.
        "dup v28.4s, v27.s[0]\n"  // accum_0 = 0
        "dup v29.4s, v27.s[1]\n"  // accum_1 = 0
        "dup v30.4s, v27.s[2]\n"  // accum_2 = 0
        "dup v31.4s, v27.s[3]\n"  // accum_3 = 0

        // Update the stopping condition for this set of rows.
        "ldr w6, [%[nnz_per_row]], #4\n"
        "cmp w6, #0\n"
        // Skip the body if there isn't anything in this row.
        "beq " LABEL_SKIP_COL_LOOP "f\n"

        LABEL_COL_LOOP
        ":\n"
        // Load 1 Rhs vectors of size 1x4 each.
        "ld1 {v0.4s}, [%[rhs_ptr]], x8\n"

        // Start this load now, which we won't need until the end of the loop.
        "ldrsh x8, [%[col_deltas_bytes]], #2\n"

        // Load 16 Lhs cells corresponding to a 4x4 block.
        "ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [%[weights_ptr]], #64\n"

        // Multiply-accumulate.
        "fmla v28.4s, v4.4s, v0.4s\n"
        "fmla v29.4s, v5.4s, v0.4s\n"
        "fmla v30.4s, v6.4s, v0.4s\n"
        "fmla v31.4s, v7.4s, v0.4s\n"

        // Loop. Decrement loop index.
        "subs w6, w6, #1\n"  // decrement (reduced) columns left
        "bne " LABEL_COL_LOOP "b\n"

        LABEL_SKIP_COL_LOOP
        ":\n"

        // Horizontally add accumulators and store result.
        "faddp v28.4s, v28.4s, v29.4s\n"
        "faddp v30.4s, v30.4s, v31.4s\n"
        "faddp v28.4s, v28.4s, v30.4s\n"

        // Store accumulators.
        "st1 {v28.4s}, [%[out_ptr]], #16\n"

        // Decrement rows remaining.
        "subs %[assigned_rows], %[assigned_rows], #1\n"
        "bne " LABEL_ROW_LOOP "b\n"

        // clang-format off
        :  // outputs
        [out_ptr] "+r"(out_ptr),
        [weights_ptr] "+r"(weights_ptr),
        [col_deltas_bytes] "+r"(col_deltas_bytes),
        [bias_ptr] "+r"(bias_ptr),
        [nnz_per_row] "+r"(nnz_per_row),
        [assigned_rows] "+r"(assigned_rows),
        [rhs_ptr] "+r"(rhs_ptr)
        :  // inputs
        :  // clobbers
        "cc", "memory", "x6", "x7", "x8", "v0", "v1", "v2", "v3", "v4", "v5",
        "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
        "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
        "v26", "v27", "v28", "v29", "v30", "v31");
    // clang-format on
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
// this function.  This is automatically taken care of in sparse_linear_layer.
// The bias is reconstructed through horizontal additions, leads to a small
// speedup by reducing latencies at the end of the loop.
template <typename WeightType, typename RhsType, typename OutType>
typename std::enable_if<std::is_same<WeightType, float>::value &&
                        std::is_same<RhsType, float>::value &&
                        std::is_same<OutType, float>::value>::type
SpMM5_4x4(const float* weights_ptr, const int16_t* col_deltas_bytes,
          const int32_t* nnz_per_row, const float* rhs_ptr,
          const float* bias_ptr, float* out_ptr, int64_t assigned_rows,
          int64_t rows, int64_t cols, int relu) {
  /*   This instrinsic version exists for reference, note that in the
       intrinsic version col_deltas_bytes should NOT actually be in bytes,
       but rather elements.  Intrinsics are 25-35% slower than the
       assembly version.

       for (int r = 0; r < rows; r += 4) {
        int reduced_col_count = nnz_per_row[r / 4];
        float32x4_t accum0 = vdupq_n_f32(bias_ptr + r);
        float32x4_t accum1 = vdupq_n_f32(bias_ptr + r + 1);
        float32x4_t accum2 = vdupq_n_f32(bias_ptr + r + 2);
        float32x4_t accum3 = vdupq_n_f32(bias_ptr + r + 3);
        float32x4_t accum4 = vdupq_n_f32(bias_ptr + r);
        float32x4_t accum5 = vdupq_n_f32(bias_ptr + r + 1);
        float32x4_t accum6 = vdupq_n_f32(bias_ptr + r + 2);
        float32x4_t accum7 = vdupq_n_f32(bias_ptr + r + 3);
        ...
        for (int c = 0; c < reduced_col_count; ++c) {
          int32_t offset = *col_deltas_bytes; col_deltas_bytes++;
          rhs_ptr += offset;
          float32x4_t rhs = vld1q_f32(rhs_ptr);
          float32x4_t rhs2 = vld1q_f32(rhs2_ptr);
          float32x4_t rhs3 = vld1q_f32(rhs3_ptr);
          float32x4_t rhs4 = vld1q_f32(rhs4_ptr);
          float32x4_t rhs5 = vld1q_f32(rhs5_ptr);

          uint16x4_t lhs0_int = vld1_u16(weights_ptr); weights_ptr += 4;
          uint16x4_t lhs1_int = vld1_u16(weights_ptr); weights_ptr += 4;
          uint16x4_t lhs2_int = vld1_u16(weights_ptr); weights_ptr += 4;
          uint16x4_t lhs3_int = vld1_u16(weights_ptr); weights_ptr += 4;

          float32x4_t lhs0 = vreinterpretq_f32_u32(vshll_n_u16(lhs0_int, 16));
          float32x4_t lhs1 = vreinterpretq_f32_u32(vshll_n_u16(lhs1_int, 16));
          float32x4_t lhs2 = vreinterpretq_f32_u32(vshll_n_u16(lhs2_int, 16));
          float32x4_t lhs3 = vreinterpretq_f32_u32(vshll_n_u16(lhs3_int, 16));

          accum0 = vmlaq_f32(accum0, lhs0, rhs);
          accum1 = vmlaq_f32(accum1, lhs1, rhs);
          accum2 = vmlaq_f32(accum2, lhs2, rhs);
          accum3 = vmlaq_f32(accum3, lhs3, rhs);
          accum4 = vmlaq_f32(accum0, lhs0, rhs2);
          accum5 = vmlaq_f32(accum1, lhs1, rhs2);
          accum6 = vmlaq_f32(accum2, lhs2, rhs2);
          accum7 = vmlaq_f32(accum3, lhs3, rhs2);
          ...
        }

        float32x4_t reduce0 = vpaddq_f32(accum0, accum1);
        float32x4_t reduce1 = vpaddq_f32(accum2, accum3);
        float32x4_t reduce2 = vpaddq_f32(reduce0, reduce1);
        vst1q_f32(out_ptr + r, reduce2);

        float32x4_t reduce0 = vpaddq_f32(accum4, accum5);
        float32x4_t reduce1 = vpaddq_f32(accum6, accum7);
        float32x4_t reduce2 = vpaddq_f32(reduce0, reduce1);
        vst1q_f32(out2_ptr + r, reduce2);

        ...
      } */

  // If the relu is handled in the routine with a comparison and vbit (insert
  // if true), or by branching, then it is slightly, but noticeably slower
  // ~5%, the outer branch avoids that penalty.
  //
  // Pointers to the columns.
  const float* rhs2_ptr = rhs_ptr + cols;
  float* out2_ptr = out_ptr + rows;
  const float* rhs3_ptr = rhs_ptr + 2 * cols;
  float* out3_ptr = out_ptr + 2 * rows;
  const float* rhs4_ptr = rhs_ptr + 3 * cols;
  float* out4_ptr = out_ptr + 3 * rows;
  const float* rhs5_ptr = rhs_ptr + 4 * cols;
  float* out5_ptr = out_ptr + 4 * rows;
  if (relu) {
    asm(
        // Load the first two column deltas.
        "ldrsh x7, [%[col_deltas_bytes]], #2\n"
        "ldrsh x8, [%[col_deltas_bytes]], #2\n"
        // ld1 doesn't support pre-index, so we do the first addition here.
        "add %[rhs_ptr], %[rhs_ptr], x7\n"
        "add %[rhs2_ptr], %[rhs2_ptr], x7\n"
        "add %[rhs3_ptr], %[rhs3_ptr], x7\n"
        "add %[rhs4_ptr], %[rhs4_ptr], x7\n"
        "add %[rhs5_ptr], %[rhs5_ptr], x7\n"

        LABEL_ROW_LOOP
        ":\n"
        // Load the bias.
        "ld1 {v27.4s}, [%[bias_ptr]], #16\n"

        // Zero out local accumulators.
        "dup v28.4s, v27.s[0]\n"  // for 1st column
        "dup v29.4s, v27.s[1]\n"  // for 1st column
        "dup v30.4s, v27.s[2]\n"  // for 1st column
        "dup v31.4s, v27.s[3]\n"  // for 1st column
        "dup v23.4s, v27.s[0]\n"  // for 2nd column
        "dup v24.4s, v27.s[1]\n"  // for 2nd column
        "dup v25.4s, v27.s[2]\n"  // for 2nd column
        "dup v26.4s, v27.s[3]\n"  // for 2nd column
        "dup v19.4s, v27.s[0]\n"  // for 3rd column
        "dup v20.4s, v27.s[1]\n"  // for 3rd column
        "dup v21.4s, v27.s[2]\n"  // for 3rd column
        "dup v22.4s, v27.s[3]\n"  // for 3rd column
        "dup v15.4s, v27.s[0]\n"  // for 4th column
        "dup v16.4s, v27.s[1]\n"  // for 4th column
        "dup v17.4s, v27.s[2]\n"  // for 4th column
        "dup v18.4s, v27.s[3]\n"  // for 4th column
        "dup v11.4s, v27.s[0]\n"  // for 5th column
        "dup v12.4s, v27.s[1]\n"  // for 5th column
        "dup v13.4s, v27.s[2]\n"  // for 5th column
        "dup v14.4s, v27.s[3]\n"  // for 5th column

        // Update the stopping condition for this set of rows.
        "ldr w6, [%[nnz_per_row]], #4\n"
        "cmp w6, #0\n"
        // Skip the body if there isn't anything in this row.
        "beq " LABEL_SKIP_COL_LOOP "f\n"

        LABEL_COL_LOOP
        ":\n"
        // Load 1 Rhs vectors of size 1x4 each.
        "ld1 {v0.4s}, [%[rhs_ptr]], x8\n"
        "ld1 {v1.4s}, [%[rhs2_ptr]], x8\n"
        "ld1 {v8.4s}, [%[rhs3_ptr]], x8\n"
        "ld1 {v9.4s}, [%[rhs4_ptr]], x8\n"
        "ld1 {v10.4s}, [%[rhs5_ptr]], x8\n"

        // Start this load now, which we won't need until the end of the loop.
        "ldrsh x8, [%[col_deltas_bytes]], #2\n"

        // Load 16 Lhs cells corresponding to a 4x4 block.
        "ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [%[weights_ptr]], #64\n"

        // Multiply-accumulate.
        "fmla v28.4s, v4.4s, v0.4s\n"   // for 1st column
        "fmla v29.4s, v5.4s, v0.4s\n"   // for 1st column
        "fmla v30.4s, v6.4s, v0.4s\n"   // for 1st column
        "fmla v31.4s, v7.4s, v0.4s\n"   // for 1st column
        "fmla v23.4s, v4.4s, v1.4s\n"   // for 2nd column
        "fmla v24.4s, v5.4s, v1.4s\n"   // for 2nd column
        "fmla v25.4s, v6.4s, v1.4s\n"   // for 2nd column
        "fmla v26.4s, v7.4s, v1.4s\n"   // for 2nd column
        "fmla v19.4s, v4.4s, v8.4s\n"   // for 3rd column
        "fmla v20.4s, v5.4s, v8.4s\n"   // for 3rd column
        "fmla v21.4s, v6.4s, v8.4s\n"   // for 3rd column
        "fmla v22.4s, v7.4s, v8.4s\n"   // for 3rd column
        "fmla v15.4s, v4.4s, v9.4s\n"   // for 4th column
        "fmla v16.4s, v5.4s, v9.4s\n"   // for 4th column
        "fmla v17.4s, v6.4s, v9.4s\n"   // for 4th column
        "fmla v18.4s, v7.4s, v9.4s\n"   // for 4th column
        "fmla v11.4s, v4.4s, v10.4s\n"  // for 5th column
        "fmla v12.4s, v5.4s, v10.4s\n"  // for 5th column
        "fmla v13.4s, v6.4s, v10.4s\n"  // for 5th column
        "fmla v14.4s, v7.4s, v10.4s\n"  // for 5th column

        // Loop. Decrement loop index.
        "subs w6, w6, #1\n"  // decrement (reduced) columns left
        "bne " LABEL_COL_LOOP "b\n"

        LABEL_SKIP_COL_LOOP
        ":\n"

        "movi v0.4s, #0\n"
        "faddp v28.4s, v28.4s, v29.4s\n"  // 1st column
        "faddp v23.4s, v23.4s, v24.4s\n"  // 2nd column
        "faddp v19.4s, v19.4s, v20.4s\n"  // 3rd column
        "faddp v15.4s, v15.4s, v16.4s\n"  // 4th column
        "faddp v11.4s, v11.4s, v12.4s\n"  // 5th column

        "faddp v30.4s, v30.4s, v31.4s\n"  // 1st column
        "faddp v25.4s, v25.4s, v26.4s\n"  // 2nd column
        "faddp v21.4s, v21.4s, v22.4s\n"  // 3rd column
        "faddp v17.4s, v17.4s, v18.4s\n"  // 4th column
        "faddp v13.4s, v13.4s, v14.4s\n"  // 5th column

        "faddp v28.4s, v28.4s, v30.4s\n"  // 1st column
        "faddp v23.4s, v23.4s, v25.4s\n"  // 2nd column
        "faddp v19.4s, v19.4s, v21.4s\n"  // 3rd column
        "faddp v15.4s, v15.4s, v17.4s\n"  // 4th column
        "faddp v11.4s, v11.4s, v13.4s\n"  // 5th column

        // Do relu as requested.
        "fmax v28.4s, v28.4s, v0.4s\n"
        "fmax v23.4s, v23.4s, v0.4s\n"
        "fmax v19.4s, v19.4s, v0.4s\n"
        "fmax v15.4s, v15.4s, v0.4s\n"
        "fmax v11.4s, v11.4s, v0.4s\n"

        // Store accumulators.
        "st1 {v28.4s}, [%[out_ptr]], #16\n"
        "st1 {v23.4s}, [%[out2_ptr]], #16\n"
        "st1 {v19.4s}, [%[out3_ptr]], #16\n"
        "st1 {v15.4s}, [%[out4_ptr]], #16\n"
        "st1 {v11.4s}, [%[out5_ptr]], #16\n"

        // Decrement rows remaining.
        "subs %[assigned_rows], %[assigned_rows], #1\n"
        "bne " LABEL_ROW_LOOP "b\n"

        // clang-format off
        :  // outputs
        [out_ptr] "+r"(out_ptr),
        [out2_ptr] "+r"(out2_ptr),
        [out3_ptr] "+r"(out3_ptr),
        [out4_ptr] "+r"(out4_ptr),
        [out5_ptr] "+r"(out5_ptr),
        [weights_ptr] "+r"(weights_ptr),
        [col_deltas_bytes] "+r"(col_deltas_bytes),
        [bias_ptr] "+r"(bias_ptr),
        [nnz_per_row] "+r"(nnz_per_row),
        [assigned_rows] "+r"(assigned_rows),
        [rhs_ptr] "+r"(rhs_ptr),
        [rhs2_ptr] "+r"(rhs2_ptr),
        [rhs3_ptr] "+r"(rhs3_ptr),
        [rhs4_ptr] "+r"(rhs4_ptr),
        [rhs5_ptr] "+r"(rhs5_ptr)
        :  // inputs
        :  // clobbers
        "cc", "memory", "x6", "x7", "x8", "v0", "v1", "v2", "v3", "v4", "v5",
        "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
        "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
        "v26", "v27", "v28", "v29", "v30", "v31");
    // clang-format on
  } else {
    asm(
        // Load the first two column deltas.
        "ldrsh x7, [%[col_deltas_bytes]], #2\n"
        "ldrsh x8, [%[col_deltas_bytes]], #2\n"
        // ld1 doesn't support pre-index, so we do the first addition here.
        "add %[rhs_ptr], %[rhs_ptr], x7\n"
        "add %[rhs2_ptr], %[rhs2_ptr], x7\n"
        "add %[rhs3_ptr], %[rhs3_ptr], x7\n"
        "add %[rhs4_ptr], %[rhs4_ptr], x7\n"
        "add %[rhs5_ptr], %[rhs5_ptr], x7\n"

        LABEL_ROW_LOOP
        ":\n"

        // Load the bias.
        "ld1 {v27.4s}, [%[bias_ptr]], #16\n"

        // Zero out local accumulators.
        "dup v28.4s, v27.s[0]\n"  // for 1st column
        "dup v29.4s, v27.s[1]\n"  // for 1st column
        "dup v30.4s, v27.s[2]\n"  // for 1st column
        "dup v31.4s, v27.s[3]\n"  // for 1st column
        "dup v23.4s, v27.s[0]\n"  // for 2nd column
        "dup v24.4s, v27.s[1]\n"  // for 2nd column
        "dup v25.4s, v27.s[2]\n"  // for 2nd column
        "dup v26.4s, v27.s[3]\n"  // for 2nd column
        "dup v19.4s, v27.s[0]\n"  // for 3rd column
        "dup v20.4s, v27.s[1]\n"  // for 3rd column
        "dup v21.4s, v27.s[2]\n"  // for 3rd column
        "dup v22.4s, v27.s[3]\n"  // for 3rd column
        "dup v15.4s, v27.s[0]\n"  // for 4th column
        "dup v16.4s, v27.s[1]\n"  // for 4th column
        "dup v17.4s, v27.s[2]\n"  // for 4th column
        "dup v18.4s, v27.s[3]\n"  // for 4th column
        "dup v11.4s, v27.s[0]\n"  // for 5th column
        "dup v12.4s, v27.s[1]\n"  // for 5th column
        "dup v13.4s, v27.s[2]\n"  // for 5th column
        "dup v14.4s, v27.s[3]\n"  // for 5th column

        // Update the stopping condition for this set of rows.
        "ldr w6, [%[nnz_per_row]], #4\n"
        "cmp w6, #0\n"
        // Skip the body if there isn't anything in this row.
        "beq " LABEL_SKIP_COL_LOOP "f\n"

        LABEL_COL_LOOP
        ":\n"
        // Load 1 Rhs vectors of size 1x4 each.
        "ld1 {v0.4s}, [%[rhs_ptr]], x8\n"
        "ld1 {v1.4s}, [%[rhs2_ptr]], x8\n"
        "ld1 {v8.4s}, [%[rhs3_ptr]], x8\n"
        "ld1 {v9.4s}, [%[rhs4_ptr]], x8\n"
        "ld1 {v10.4s}, [%[rhs5_ptr]], x8\n"

        // Start this load now, which we won't need until the end of the loop.
        "ldrsh x8, [%[col_deltas_bytes]], #2\n"

        // Load 16 Lhs cells corresponding to a 4x4 block.
        "ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [%[weights_ptr]], #64\n"

        // Multiply-accumulate.
        "fmla v28.4s, v4.4s, v0.4s\n"   // for 1st column
        "fmla v29.4s, v5.4s, v0.4s\n"   // for 1st column
        "fmla v30.4s, v6.4s, v0.4s\n"   // for 1st column
        "fmla v31.4s, v7.4s, v0.4s\n"   // for 1st column
        "fmla v23.4s, v4.4s, v1.4s\n"   // for 2nd column
        "fmla v24.4s, v5.4s, v1.4s\n"   // for 2nd column
        "fmla v25.4s, v6.4s, v1.4s\n"   // for 2nd column
        "fmla v26.4s, v7.4s, v1.4s\n"   // for 2nd column
        "fmla v19.4s, v4.4s, v8.4s\n"   // for 3rd column
        "fmla v20.4s, v5.4s, v8.4s\n"   // for 3rd column
        "fmla v21.4s, v6.4s, v8.4s\n"   // for 3rd column
        "fmla v22.4s, v7.4s, v8.4s\n"   // for 3rd column
        "fmla v15.4s, v4.4s, v9.4s\n"   // for 4th column
        "fmla v16.4s, v5.4s, v9.4s\n"   // for 4th column
        "fmla v17.4s, v6.4s, v9.4s\n"   // for 4th column
        "fmla v18.4s, v7.4s, v9.4s\n"   // for 4th column
        "fmla v11.4s, v4.4s, v10.4s\n"  // for 5th column
        "fmla v12.4s, v5.4s, v10.4s\n"  // for 5th column
        "fmla v13.4s, v6.4s, v10.4s\n"  // for 5th column
        "fmla v14.4s, v7.4s, v10.4s\n"  // for 5th column

        // Loop. Decrement loop index.
        "subs w6, w6, #1\n"  // decrement (reduced) columns left
        "bne " LABEL_COL_LOOP "b\n"

        LABEL_SKIP_COL_LOOP
        ":\n"

        // Horizontally add accumulators and store result.
        "faddp v28.4s, v28.4s, v29.4s\n"  // 1st column
        "faddp v23.4s, v23.4s, v24.4s\n"  // 2nd column
        "faddp v19.4s, v19.4s, v20.4s\n"  // 3rd column
        "faddp v15.4s, v15.4s, v16.4s\n"  // 4th column
        "faddp v11.4s, v11.4s, v12.4s\n"  // 5th column

        "faddp v30.4s, v30.4s, v31.4s\n"  // 1st column
        "faddp v25.4s, v25.4s, v26.4s\n"  // 2nd column
        "faddp v21.4s, v21.4s, v22.4s\n"  // 3rd column
        "faddp v17.4s, v17.4s, v18.4s\n"  // 4th column
        "faddp v13.4s, v13.4s, v14.4s\n"  // 5th column

        "faddp v28.4s, v28.4s, v30.4s\n"  // 1st column
        "faddp v23.4s, v23.4s, v25.4s\n"  // 2nd column
        "faddp v19.4s, v19.4s, v21.4s\n"  // 3rd column
        "faddp v15.4s, v15.4s, v17.4s\n"  // 4th column
        "faddp v11.4s, v11.4s, v13.4s\n"  // 5th column

        // Store accumulators.
        "st1 {v28.4s}, [%[out_ptr]], #16\n"
        "st1 {v23.4s}, [%[out2_ptr]], #16\n"
        "st1 {v19.4s}, [%[out3_ptr]], #16\n"
        "st1 {v15.4s}, [%[out4_ptr]], #16\n"
        "st1 {v11.4s}, [%[out5_ptr]], #16\n"

        // Decrement rows remaining.
        "subs %[assigned_rows], %[assigned_rows], #1\n"
        "bne " LABEL_ROW_LOOP "b\n"

        // clang-format off
        :  // outputs
        [out_ptr] "+r"(out_ptr),
        [out2_ptr] "+r"(out2_ptr),
        [out3_ptr] "+r"(out3_ptr),
        [out4_ptr] "+r"(out4_ptr),
        [out5_ptr] "+r"(out5_ptr),
        [weights_ptr] "+r"(weights_ptr),
        [col_deltas_bytes] "+r"(col_deltas_bytes),
        [bias_ptr] "+r"(bias_ptr),
        [nnz_per_row] "+r"(nnz_per_row),
        [assigned_rows] "+r"(assigned_rows),
        [rhs_ptr] "+r"(rhs_ptr),
        [rhs2_ptr] "+r"(rhs2_ptr),
        [rhs3_ptr] "+r"(rhs3_ptr),
        [rhs4_ptr] "+r"(rhs4_ptr),
        [rhs5_ptr] "+r"(rhs5_ptr)
        :  // inputs
        :  // clobbers
        "cc", "memory", "x6", "x7", "x8", "v0", "v1", "v2", "v3", "v4", "v5",
        "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
        "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
        "v26", "v27", "v28", "v29", "v30", "v31");
    // clang-format on
  }
}

// Note that the number of exponent bits in the output must exactly match
// the sum of the input and rhs types.
template <typename WeightType, typename RhsType, typename OutType>
typename std::enable_if<
    IsFixed16Type<WeightType>::value && IsFixed16Type<RhsType>::value &&
    std::is_same<OutType, typename TypeOfProduct<WeightType,
                                                 RhsType>::type>::value>::type
SpMV_4x4(const WeightType* weights_ptr, const int16_t* col_deltas_bytes,
         const int32_t* nnz_per_row, const RhsType* rhs_ptr,
         const typename TypeOfProduct<WeightType, RhsType>::type* bias_ptr,
         OutType* out_ptr, int64_t assigned_rows,
         int64_t rows /* only used in SpMM variants */,
         int64_t cols /* only used in SpMM variants */, int relu) {
  if (relu) {
    asm(
        // Load the first two column deltas.
        "ldrsh x7, [%[col_deltas_bytes]], #2\n"
        "ldrsh x8, [%[col_deltas_bytes]], #2\n"
        // ld1 doesn't support pre-index, so we do the first addition here.
        "add %[rhs_ptr], %[rhs_ptr], x7\n"

        "movi v25.4s, #0\n"

        LABEL_ROW_LOOP
        ":\n"

        // Load the bias.
        "ld1 {v27.4s}, [%[bias_ptr]], #16\n"

        // Zero out local accumulators.
        "dup v28.4s, v27.s[0]\n"  // accum_0 = 0
        "dup v29.4s, v27.s[1]\n"  // accum_1 = 0
        "dup v30.4s, v27.s[2]\n"  // accum_2 = 0
        "dup v31.4s, v27.s[3]\n"  // accum_3 = 0

        // Update the stopping condition for this set of rows.
        "ldr w6, [%[nnz_per_row]], #4\n"
        "cmp w6, #0\n"
        // Skip the body if there isn't anything in this row.
        "beq " LABEL_SKIP_COL_LOOP "f\n"

        LABEL_COL_LOOP
        ":\n"
        // Load 1 Rhs vectors of size 1x4 each.
        "ld1 {v0.4h}, [%[rhs_ptr]], x8\n"
        // Duplicate the lower half into the upper half.
        "mov v0.d[1], v0.d[0]\n"

        // Start this load now, which we won't need until the end of the loop.
        "ldrsh x8, [%[col_deltas_bytes]], #2\n"

        // Load 16 Lhs cells corresponding to a 4x4 block.
        "ld1 {v2.8h, v3.8h}, [%[weights_ptr]], #32\n"

        // Multiply-accumulate.
        "smlal v28.4s, v2.4h, v0.4h\n"
        "smlal2 v29.4s, v2.8h, v0.8h\n"
        "smlal v30.4s, v3.4h, v0.4h\n"
        "smlal2 v31.4s, v3.8h, v0.8h\n"

        // Loop. Decrement loop index.
        "subs w6, w6, #1\n"  // decrement (reduced) columns left
        "bne " LABEL_COL_LOOP "b\n"

        LABEL_SKIP_COL_LOOP
        ":\n"

        // Horizontally add accumulators and store result.
        "addp v28.4s, v28.4s, v29.4s\n"
        "addp v30.4s, v30.4s, v31.4s\n"
        "addp v28.4s, v28.4s, v30.4s\n"

        // Do relu if requested.
        "smax v28.4s, v28.4s, v25.4s\n"

        // Store accumulators.
        "st1 {v28.4s}, [%[out_ptr]], #16\n"

        // Decrement rows remaining.
        "subs %[assigned_rows], %[assigned_rows], #1\n"
        "bne " LABEL_ROW_LOOP "b\n"

        // clang-format off
        :  // outputs
        [out_ptr] "+r"(out_ptr), [weights_ptr] "+r"(weights_ptr),
        [col_deltas_bytes] "+r"(col_deltas_bytes), [bias_ptr] "+r"(bias_ptr),
        [nnz_per_row] "+r"(nnz_per_row), [assigned_rows] "+r"(assigned_rows),
        [rhs_ptr] "+r"(rhs_ptr)
        :  // inputs
        :  // clobbers
        "cc", "memory", "x6", "x7", "x8", "v0", "v1", "v2", "v3", "v4", "v5",
        "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
        "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
        "v26", "v27", "v28", "v29", "v30", "v31");
    // clang-format on
  } else {
    asm(
        // Load the first two column deltas.
        "ldrsh x7, [%[col_deltas_bytes]], #2\n"
        "ldrsh x8, [%[col_deltas_bytes]], #2\n"
        // ld1 doesn't support pre-index, so we do the first addition here.
        "add %[rhs_ptr], %[rhs_ptr], x7\n"

        "movi v25.4s, #0\n"

        LABEL_ROW_LOOP
        ":\n"

        // Load the bias.
        "ld1 {v27.4s}, [%[bias_ptr]], #16\n"

        // Zero out local accumulators.
        "dup v28.4s, v27.s[0]\n"  // accum_0 = 0
        "dup v29.4s, v27.s[1]\n"  // accum_1 = 0
        "dup v30.4s, v27.s[2]\n"  // accum_2 = 0
        "dup v31.4s, v27.s[3]\n"  // accum_3 = 0

        // Update the stopping condition for this set of rows.
        "ldr w6, [%[nnz_per_row]], #4\n"
        "cmp w6, #0\n"
        // Skip the body if there isn't anything in this row.
        "beq " LABEL_SKIP_COL_LOOP "f\n"

        LABEL_COL_LOOP
        ":\n"
        // Load 1 Rhs vectors of size 1x4 each.
        "ld1 {v0.4h}, [%[rhs_ptr]], x8\n"
        // Duplicate the lower half into the upper half.
        "mov v0.d[1], v0.d[0]\n"

        // Start this load now, which we won't need until the end of the loop.
        "ldrsh x8, [%[col_deltas_bytes]], #2\n"

        // Load 16 Lhs cells corresponding to a 4x4 block.
        "ld1 {v2.8h, v3.8h}, [%[weights_ptr]], #32\n"

        // Multiply-accumulate.
        "smlal v28.4s, v2.4h, v0.4h\n"
        "smlal2 v29.4s, v2.8h, v0.8h\n"
        "smlal v30.4s, v3.4h, v0.4h\n"
        "smlal2 v31.4s, v3.8h, v0.8h\n"

        // Loop. Decrement loop index.
        "subs w6, w6, #1\n"  // decrement (reduced) columns left
        "bne " LABEL_COL_LOOP "b\n"

        LABEL_SKIP_COL_LOOP
        ":\n"

        // Horizontally add accumulators and store result.
        "addp v28.4s, v28.4s, v29.4s\n"
        "addp v30.4s, v30.4s, v31.4s\n"
        "addp v28.4s, v28.4s, v30.4s\n"

        // Store accumulators.
        "st1 {v28.4s}, [%[out_ptr]], #16\n"

        // Decrement rows remaining.
        "subs %[assigned_rows], %[assigned_rows], #1\n"
        "bne " LABEL_ROW_LOOP "b\n"

        // clang-format off
        :  // outputs
        [out_ptr] "+r"(out_ptr), [weights_ptr] "+r"(weights_ptr),
        [col_deltas_bytes] "+r"(col_deltas_bytes), [bias_ptr] "+r"(bias_ptr),
        [nnz_per_row] "+r"(nnz_per_row), [assigned_rows] "+r"(assigned_rows),
        [rhs_ptr] "+r"(rhs_ptr)
        :  // inputs
        :  // clobbers
        "cc", "memory", "x6", "x7", "x8", "v0", "v1", "v2", "v3", "v4", "v5",
        "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
        "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
        "v26", "v27", "v28", "v29", "v30", "v31");
    // clang-format on
  }
}

// Note that the number of exponent bits in the output must exactly match
// the sum of the input and rhs types.
template <typename WeightType, typename RhsType, typename OutType>
typename std::enable_if<
    IsFixed16Type<WeightType>::value && IsFixed16Type<RhsType>::value &&
    std::is_same<OutType, typename TypeOfProduct<WeightType,
                                                 RhsType>::type>::value>::type
SpMM5_4x4(const WeightType* weights_ptr, const int16_t* col_deltas_bytes,
          const int32_t* nnz_per_row, const RhsType* rhs_ptr,
          const typename TypeOfProduct<WeightType, RhsType>::type* bias_ptr,
          OutType* out_ptr, int64_t assigned_rows, int64_t rows, int64_t cols,
          int relu) {
  // Pointers to the columns.
  const RhsType* rhs2_ptr = rhs_ptr + cols;
  OutType* out2_ptr = out_ptr + rows;
  const RhsType* rhs3_ptr = rhs_ptr + 2 * cols;
  OutType* out3_ptr = out_ptr + 2 * rows;
  const RhsType* rhs4_ptr = rhs_ptr + 3 * cols;
  OutType* out4_ptr = out_ptr + 3 * rows;
  const RhsType* rhs5_ptr = rhs_ptr + 4 * cols;
  OutType* out5_ptr = out_ptr + 4 * rows;
  if (relu) {
    asm(
        // Load the first two column deltas.
        "ldrsh x7, [%[col_deltas_bytes]], #2\n"
        "ldrsh x8, [%[col_deltas_bytes]], #2\n"
        // ld1 doesn't support pre-index, so we do the first addition here.
        "add %[rhs_ptr], %[rhs_ptr], x7\n"
        "add %[rhs2_ptr], %[rhs2_ptr], x7\n"
        "add %[rhs3_ptr], %[rhs3_ptr], x7\n"
        "add %[rhs4_ptr], %[rhs4_ptr], x7\n"
        "add %[rhs5_ptr], %[rhs5_ptr], x7\n"

        LABEL_ROW_LOOP
        ":\n"

        // Load the bias.
        "ld1 {v27.4s}, [%[bias_ptr]], #16\n"

        // Zero out local accumulators.
        "dup v28.4s, v27.s[0]\n"  // for 1st column
        "dup v29.4s, v27.s[1]\n"  // for 1st column
        "dup v30.4s, v27.s[2]\n"  // for 1st column
        "dup v31.4s, v27.s[3]\n"  // for 1st column
        "dup v23.4s, v27.s[0]\n"  // for 2nd column
        "dup v24.4s, v27.s[1]\n"  // for 2nd column
        "dup v25.4s, v27.s[2]\n"  // for 2nd column
        "dup v26.4s, v27.s[3]\n"  // for 2nd column
        "dup v19.4s, v27.s[0]\n"  // for 3rd column
        "dup v20.4s, v27.s[1]\n"  // for 3rd column
        "dup v21.4s, v27.s[2]\n"  // for 3rd column
        "dup v22.4s, v27.s[3]\n"  // for 3rd column
        "dup v15.4s, v27.s[0]\n"  // for 4th column
        "dup v16.4s, v27.s[1]\n"  // for 4th column
        "dup v17.4s, v27.s[2]\n"  // for 4th column
        "dup v18.4s, v27.s[3]\n"  // for 4th column
        "dup v11.4s, v27.s[0]\n"  // for 5th column
        "dup v12.4s, v27.s[1]\n"  // for 5th column
        "dup v13.4s, v27.s[2]\n"  // for 5th column
        "dup v14.4s, v27.s[3]\n"  // for 5th column

        // Update the stopping condition for this set of rows.
        "ldr w6, [%[nnz_per_row]], #4\n"
        "cmp w6, #0\n"
        // Skip the body if there isn't anything in this row.
        "beq " LABEL_SKIP_COL_LOOP "f\n"

        LABEL_COL_LOOP
        ":\n"
        // Load 1 Rhs vectors of size 1x4 each and duplicate into upper half.
        "ld1 {v0.4h}, [%[rhs_ptr]], x8\n"
        "mov v0.d[1], v0.d[0]\n"
        "ld1 {v1.4h}, [%[rhs2_ptr]], x8\n"
        "mov v1.d[1], v1.d[0]\n"
        "ld1 {v8.4h}, [%[rhs3_ptr]], x8\n"
        "mov v8.d[1], v8.d[0]\n"
        "ld1 {v9.4h}, [%[rhs4_ptr]], x8\n"
        "mov v9.d[1], v9.d[0]\n"
        "ld1 {v10.4h}, [%[rhs5_ptr]], x8\n"
        "mov v10.d[1], v10.d[0]\n"

        // Start this load now, which we won't need until the end of the loop.
        "ldrsh x8, [%[col_deltas_bytes]], #2\n"

        // Load 16 Lhs cells corresponding to a 4x4 block.
        "ld1 {v2.8h, v3.8h}, [%[weights_ptr]], #32\n"

        // Multiply-accumulate.
        "smlal v28.4s, v2.4h, v0.4h\n"    // for 1st column
        "smlal2 v29.4s, v2.8h, v0.8h\n"   // for 1st column
        "smlal v30.4s, v3.4h, v0.4h\n"    // for 1st column
        "smlal2 v31.4s, v3.8h, v0.8h\n"   // for 1st columh
        "smlal v23.4s, v2.4h, v1.4h\n"    // for 2nd column
        "smlal2 v24.4s, v2.8h, v1.8h\n"   // for 2nd column
        "smlal v25.4s, v3.4h, v1.4h\n"    // for 2nd column
        "smlal2 v26.4s, v3.8h, v1.8h\n"   // for 2nd column
        "smlal v19.4s, v2.4h, v8.4h\n"    // for 3rd column
        "smlal2 v20.4s, v2.8h, v8.8h\n"   // for 3rd column
        "smlal v21.4s, v3.4h, v8.4h\n"    // for 3rd column
        "smlal2 v22.4s, v3.8h, v8.8h\n"   // for 3rd column
        "smlal v15.4s, v2.4h, v9.4h\n"    // for 4th column
        "smlal2 v16.4s, v2.8h, v9.8h\n"   // for 4th column
        "smlal v17.4s, v3.4h, v9.4h\n"    // for 4th column
        "smlal2 v18.4s, v3.8h, v9.8h\n"   // for 4th column
        "smlal v11.4s, v2.4h, v10.4h\n"   // for 5th column
        "smlal2 v12.4s, v2.8h, v10.8h\n"  // for 5th column
        "smlal v13.4s, v3.4h, v10.4h\n"   // for 5th column
        "smlal2 v14.4s, v3.8h, v10.8h\n"  // for 5th column

        // Loop. Decrement loop index.
        "subs w6, w6, #1\n"  // decrement (reduced) columns left
        "bne " LABEL_COL_LOOP "b\n"

        LABEL_SKIP_COL_LOOP
        ":\n"

        "movi v0.4s, #0\n"
        "addp v28.4s, v28.4s, v29.4s\n"  // 1st column
        "addp v23.4s, v23.4s, v24.4s\n"  // 2nd column
        "addp v19.4s, v19.4s, v20.4s\n"  // 3rd column
        "addp v15.4s, v15.4s, v16.4s\n"  // 4th column
        "addp v11.4s, v11.4s, v12.4s\n"  // 5th column

        "addp v30.4s, v30.4s, v31.4s\n"  // 1st column
        "addp v25.4s, v25.4s, v26.4s\n"  // 2nd column
        "addp v21.4s, v21.4s, v22.4s\n"  // 3rd column
        "addp v17.4s, v17.4s, v18.4s\n"  // 4th column
        "addp v13.4s, v13.4s, v14.4s\n"  // 5th column

        "addp v28.4s, v28.4s, v30.4s\n"  // 1st column
        "addp v23.4s, v23.4s, v25.4s\n"  // 2nd column
        "addp v19.4s, v19.4s, v21.4s\n"  // 3rd column
        "addp v15.4s, v15.4s, v17.4s\n"  // 4th column
        "addp v11.4s, v11.4s, v13.4s\n"  // 5th column

        // Do relu as requested.
        "smax v28.4s, v28.4s, v0.4s\n"
        "smax v23.4s, v23.4s, v0.4s\n"
        "smax v19.4s, v19.4s, v0.4s\n"
        "smax v15.4s, v15.4s, v0.4s\n"
        "smax v11.4s, v11.4s, v0.4s\n"

        // Store accumulators.
        "st1 {v28.4s}, [%[out_ptr]], #16\n"
        "st1 {v23.4s}, [%[out2_ptr]], #16\n"
        "st1 {v19.4s}, [%[out3_ptr]], #16\n"
        "st1 {v15.4s}, [%[out4_ptr]], #16\n"
        "st1 {v11.4s}, [%[out5_ptr]], #16\n"

        // Decrement rows remaining.
        "subs %[assigned_rows], %[assigned_rows], #1\n"
        "bne " LABEL_ROW_LOOP "b\n"

        // clang-format off
        :  // outputs
        [out_ptr] "+r"(out_ptr), [out2_ptr] "+r"(out2_ptr),
        [out3_ptr] "+r"(out3_ptr), [out4_ptr] "+r"(out4_ptr),
        [out5_ptr] "+r"(out5_ptr), [weights_ptr] "+r"(weights_ptr),
        [col_deltas_bytes] "+r"(col_deltas_bytes), [bias_ptr] "+r"(bias_ptr),
        [nnz_per_row] "+r"(nnz_per_row), [assigned_rows] "+r"(assigned_rows),
        [rhs_ptr] "+r"(rhs_ptr), [rhs2_ptr] "+r"(rhs2_ptr),
        [rhs3_ptr] "+r"(rhs3_ptr), [rhs4_ptr] "+r"(rhs4_ptr),
        [rhs5_ptr] "+r"(rhs5_ptr)
        :  // inputs
        :  // clobbers
        "cc", "memory", "x6", "x7", "x8", "v0", "v1", "v2", "v3", "v4", "v5",
        "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
        "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
        "v26", "v27", "v28", "v29", "v30", "v31");
    // clang-format on
  } else {
    asm(
        // Load the first two column deltas.
        "ldrsh x7, [%[col_deltas_bytes]], #2\n"
        "ldrsh x8, [%[col_deltas_bytes]], #2\n"
        // ld1 doesn't support pre-index, so we do the first addition here.
        "add %[rhs_ptr], %[rhs_ptr], x7\n"
        "add %[rhs2_ptr], %[rhs2_ptr], x7\n"
        "add %[rhs3_ptr], %[rhs3_ptr], x7\n"
        "add %[rhs4_ptr], %[rhs4_ptr], x7\n"
        "add %[rhs5_ptr], %[rhs5_ptr], x7\n"

        LABEL_ROW_LOOP
        ":\n"

        // Load the bias.
        "ld1 {v27.4s}, [%[bias_ptr]], #16\n"

        // Zero out local accumulators.
        "dup v28.4s, v27.s[0]\n"  // for 1st column
        "dup v29.4s, v27.s[1]\n"  // for 1st column
        "dup v30.4s, v27.s[2]\n"  // for 1st column
        "dup v31.4s, v27.s[3]\n"  // for 1st column
        "dup v23.4s, v27.s[0]\n"  // for 2nd column
        "dup v24.4s, v27.s[1]\n"  // for 2nd column
        "dup v25.4s, v27.s[2]\n"  // for 2nd column
        "dup v26.4s, v27.s[3]\n"  // for 2nd column
        "dup v19.4s, v27.s[0]\n"  // for 3rd column
        "dup v20.4s, v27.s[1]\n"  // for 3rd column
        "dup v21.4s, v27.s[2]\n"  // for 3rd column
        "dup v22.4s, v27.s[3]\n"  // for 3rd column
        "dup v15.4s, v27.s[0]\n"  // for 4th column
        "dup v16.4s, v27.s[1]\n"  // for 4th column
        "dup v17.4s, v27.s[2]\n"  // for 4th column
        "dup v18.4s, v27.s[3]\n"  // for 4th column
        "dup v11.4s, v27.s[0]\n"  // for 5th column
        "dup v12.4s, v27.s[1]\n"  // for 5th column
        "dup v13.4s, v27.s[2]\n"  // for 5th column
        "dup v14.4s, v27.s[3]\n"  // for 5th column

        // Update the stopping condition for this set of rows.
        "ldr w6, [%[nnz_per_row]], #4\n"
        "cmp w6, #0\n"
        // Skip the body if there isn't anything in this row.
        "beq " LABEL_SKIP_COL_LOOP "f\n"

        LABEL_COL_LOOP
        ":\n"
        // Load 1 Rhs vectors of size 1x4 each and duplicate into upper half.
        "ld1 {v0.4h}, [%[rhs_ptr]], x8\n"
        "mov v0.d[1], v0.d[0]\n"
        "ld1 {v1.4h}, [%[rhs2_ptr]], x8\n"
        "mov v1.d[1], v1.d[0]\n"
        "ld1 {v8.4h}, [%[rhs3_ptr]], x8\n"
        "mov v8.d[1], v8.d[0]\n"
        "ld1 {v9.4h}, [%[rhs4_ptr]], x8\n"
        "mov v9.d[1], v9.d[0]\n"
        "ld1 {v10.4h}, [%[rhs5_ptr]], x8\n"
        "mov v10.d[1], v10.d[0]\n"

        // Start this load now, which we won't need until the end of the loop.
        "ldrsh x8, [%[col_deltas_bytes]], #2\n"

        // Load 16 Lhs cells corresponding to a 4x4 block.
        "ld1 {v2.8h, v3.8h}, [%[weights_ptr]], #32\n"

        // Multiply-accumulate.
        "smlal v28.4s, v2.4h, v0.4h\n"    // for 1st column
        "smlal2 v29.4s, v2.8h, v0.8h\n"   // for 1st column
        "smlal v30.4s, v3.4h, v0.4h\n"    // for 1st column
        "smlal2 v31.4s, v3.8h, v0.8h\n"   // for 1st columh
        "smlal v23.4s, v2.4h, v1.4h\n"    // for 2nd column
        "smlal2 v24.4s, v2.8h, v1.8h\n"   // for 2nd column
        "smlal v25.4s, v3.4h, v1.4h\n"    // for 2nd column
        "smlal2 v26.4s, v3.8h, v1.8h\n"   // for 2nd column
        "smlal v19.4s, v2.4h, v8.4h\n"    // for 3rd column
        "smlal2 v20.4s, v2.8h, v8.8h\n"   // for 3rd column
        "smlal v21.4s, v3.4h, v8.4h\n"    // for 3rd column
        "smlal2 v22.4s, v3.8h, v8.8h\n"   // for 3rd column
        "smlal v15.4s, v2.4h, v9.4h\n"    // for 4th column
        "smlal2 v16.4s, v2.8h, v9.8h\n"   // for 4th column
        "smlal v17.4s, v3.4h, v9.4h\n"    // for 4th column
        "smlal2 v18.4s, v3.8h, v9.8h\n"   // for 4th column
        "smlal v11.4s, v2.4h, v10.4h\n"   // for 5th column
        "smlal2 v12.4s, v2.8h, v10.8h\n"  // for 5th column
        "smlal v13.4s, v3.4h, v10.4h\n"   // for 5th column
        "smlal2 v14.4s, v3.8h, v10.8h\n"  // for 5th column

        // Loop. Decrement loop index.
        "subs w6, w6, #1\n"  // decrement (reduced) columns left
        "bne " LABEL_COL_LOOP "b\n"

        LABEL_SKIP_COL_LOOP
        ":\n"

        "addp v28.4s, v28.4s, v29.4s\n"  // 1st column
        "addp v23.4s, v23.4s, v24.4s\n"  // 2nd column
        "addp v19.4s, v19.4s, v20.4s\n"  // 3rd column
        "addp v15.4s, v15.4s, v16.4s\n"  // 4th column
        "addp v11.4s, v11.4s, v12.4s\n"  // 5th column

        "addp v30.4s, v30.4s, v31.4s\n"  // 1st column
        "addp v25.4s, v25.4s, v26.4s\n"  // 2nd column
        "addp v21.4s, v21.4s, v22.4s\n"  // 3rd column
        "addp v17.4s, v17.4s, v18.4s\n"  // 4th column
        "addp v13.4s, v13.4s, v14.4s\n"  // 5th column

        "addp v28.4s, v28.4s, v30.4s\n"  // 1st column
        "addp v23.4s, v23.4s, v25.4s\n"  // 2nd column
        "addp v19.4s, v19.4s, v21.4s\n"  // 3rd column
        "addp v15.4s, v15.4s, v17.4s\n"  // 4th column
        "addp v11.4s, v11.4s, v13.4s\n"  // 5th column

        // Store accumulators.
        "st1 {v28.4s}, [%[out_ptr]], #16\n"
        "st1 {v23.4s}, [%[out2_ptr]], #16\n"
        "st1 {v19.4s}, [%[out3_ptr]], #16\n"
        "st1 {v15.4s}, [%[out4_ptr]], #16\n"
        "st1 {v11.4s}, [%[out5_ptr]], #16\n"

        // Decrement rows remaining.
        "subs %[assigned_rows], %[assigned_rows], #1\n"
        "bne " LABEL_ROW_LOOP "b\n"

        // clang-format off
        :  // outputs
        [out_ptr] "+r"(out_ptr), [out2_ptr] "+r"(out2_ptr),
        [out3_ptr] "+r"(out3_ptr), [out4_ptr] "+r"(out4_ptr),
        [out5_ptr] "+r"(out5_ptr), [weights_ptr] "+r"(weights_ptr),
        [col_deltas_bytes] "+r"(col_deltas_bytes), [bias_ptr] "+r"(bias_ptr),
        [nnz_per_row] "+r"(nnz_per_row), [assigned_rows] "+r"(assigned_rows),
        [rhs_ptr] "+r"(rhs_ptr), [rhs2_ptr] "+r"(rhs2_ptr),
        [rhs3_ptr] "+r"(rhs3_ptr), [rhs4_ptr] "+r"(rhs4_ptr),
        [rhs5_ptr] "+r"(rhs5_ptr)
        :  // inputs
        :  // clobbers
        "cc", "memory", "x6", "x7", "x8", "v0", "v1", "v2", "v3", "v4", "v5",
        "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
        "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
        "v26", "v27", "v28", "v29", "v30", "v31");
    // clang-format on
  }
}

// Note that the number of exponent bits in the bias must exactly match
// the sum of the input and rhs types.
template <typename WeightType, typename RhsType, typename OutType>
typename std::enable_if<IsFixed16Type<WeightType>::value &&
                        IsFixed16Type<RhsType>::value &&
                        IsFixed16Type<OutType>::value>::type
SpMV_4x4(const WeightType* weights_ptr, const int16_t* col_deltas_bytes,
         const int32_t* nnz_per_row, const RhsType* rhs_ptr,
         const typename TypeOfProduct<WeightType, RhsType>::type* bias_ptr,
         OutType* out_ptr, int64_t assigned_rows,
         int64_t rows /* only used in SpMM variants */,
         int64_t cols /* only used in SpMM variants */, int relu) {
  constexpr int kShiftAmount = 15 - WeightType::kExponentBits -
                               RhsType::kExponentBits + OutType::kExponentBits;
  if (relu) {
    asm(
        // Load the first two column deltas.
        "ldrsh x7, [%[col_deltas_bytes]], #2\n"
        "ldrsh x8, [%[col_deltas_bytes]], #2\n"
        // ld1 doesn't support pre-index, so we do the first addition here.
        "add %[rhs_ptr], %[rhs_ptr], x7\n"

        "movi v25.4s, #0\n"

        LABEL_ROW_LOOP
        ":\n"

        // Load the bias.
        "ld1 {v27.4s}, [%[bias_ptr]], #16\n"

        // Zero out local accumulators.
        "dup v28.4s, v27.s[0]\n"  // accum_0 = 0
        "dup v29.4s, v27.s[1]\n"  // accum_1 = 0
        "dup v30.4s, v27.s[2]\n"  // accum_2 = 0
        "dup v31.4s, v27.s[3]\n"  // accum_3 = 0

        // Update the stopping condition for this set of rows.
        "ldr w6, [%[nnz_per_row]], #4\n"
        "cmp w6, #0\n"
        // Skip the body if there isn't anything in this row.
        "beq " LABEL_SKIP_COL_LOOP "f\n"

        LABEL_COL_LOOP
        ":\n"
        // Load 1 Rhs vectors of size 1x4 each.
        "ld1 {v0.4h}, [%[rhs_ptr]], x8\n"
        // Duplicate the lower half into the upper half.
        "mov v0.d[1], v0.d[0]\n"

        // Start this load now, which we won't need until the end of the loop.
        "ldrsh x8, [%[col_deltas_bytes]], #2\n"

        // Load 16 Lhs cells corresponding to a 4x4 block.
        "ld1 {v2.8h, v3.8h}, [%[weights_ptr]], #32\n"

        // Multiply-accumulate.
        "smlal v28.4s, v2.4h, v0.4h\n"
        "smlal2 v29.4s, v2.8h, v0.8h\n"
        "smlal v30.4s, v3.4h, v0.4h\n"
        "smlal2 v31.4s, v3.8h, v0.8h\n"

        // Loop. Decrement loop index.
        "subs w6, w6, #1\n"  // decrement (reduced) columns left
        "bne " LABEL_COL_LOOP "b\n"

        LABEL_SKIP_COL_LOOP
        ":\n"

        // Horizontally add accumulators and store result.
        "addp v28.4s, v28.4s, v29.4s\n"
        "addp v30.4s, v30.4s, v31.4s\n"
        "addp v28.4s, v28.4s, v30.4s\n"

        // Do relu if requested.
        "smax v28.4s, v28.4s, v25.4s\n"
        "sqrshrn v26.4h, v28.4s, %[shift_amount]\n"

        // Store accumulators.
        "st1 {v26.4h}, [%[out_ptr]], #8\n"

        // Decrement rows remaining.
        "subs %[assigned_rows], %[assigned_rows], #1\n"
        "bne " LABEL_ROW_LOOP "b\n"

        // clang-format off
        :  // outputs
        [out_ptr] "+r"(out_ptr), [weights_ptr] "+r"(weights_ptr),
        [col_deltas_bytes] "+r"(col_deltas_bytes), [bias_ptr] "+r"(bias_ptr),
        [nnz_per_row] "+r"(nnz_per_row), [assigned_rows] "+r"(assigned_rows),
        [rhs_ptr] "+r"(rhs_ptr)
        :  // inputs
        [shift_amount] "I"(kShiftAmount)
        :  // clobbers
        "cc", "memory", "x6", "x7", "x8", "v0", "v1", "v2", "v3", "v4", "v5",
        "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
        "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
        "v26", "v27", "v28", "v29", "v30", "v31");
    // clang-format on
  } else {
    asm(
        // Load the first two column deltas.
        "ldrsh x7, [%[col_deltas_bytes]], #2\n"
        "ldrsh x8, [%[col_deltas_bytes]], #2\n"
        // ld1 doesn't support pre-index, so we do the first addition here.
        "add %[rhs_ptr], %[rhs_ptr], x7\n"

        "movi v25.4s, #0\n"

        LABEL_ROW_LOOP
        ":\n"

        // Load the bias.
        "ld1 {v27.4s}, [%[bias_ptr]], #16\n"

        // Zero out local accumulators.
        "dup v28.4s, v27.s[0]\n"  // accum_0 = 0
        "dup v29.4s, v27.s[1]\n"  // accum_1 = 0
        "dup v30.4s, v27.s[2]\n"  // accum_2 = 0
        "dup v31.4s, v27.s[3]\n"  // accum_3 = 0

        // Update the stopping condition for this set of rows.
        "ldr w6, [%[nnz_per_row]], #4\n"
        "cmp w6, #0\n"
        // Skip the body if there isn't anything in this row.
        "beq " LABEL_SKIP_COL_LOOP "f\n"

        LABEL_COL_LOOP
        ":\n"
        // Load 1 Rhs vectors of size 1x4 each.
        "ld1 {v0.4h}, [%[rhs_ptr]], x8\n"
        // Duplicate the lower half into the upper half.
        "mov v0.d[1], v0.d[0]\n"

        // Start this load now, which we won't need until the end of the loop.
        "ldrsh x8, [%[col_deltas_bytes]], #2\n"

        // Load 16 Lhs cells corresponding to a 4x4 block.
        "ld1 {v2.8h, v3.8h}, [%[weights_ptr]], #32\n"

        // Multiply-accumulate.
        "smlal v28.4s, v2.4h, v0.4h\n"
        "smlal2 v29.4s, v2.8h, v0.8h\n"
        "smlal v30.4s, v3.4h, v0.4h\n"
        "smlal2 v31.4s, v3.8h, v0.8h\n"

        // Loop. Decrement loop index.
        "subs w6, w6, #1\n"  // decrement (reduced) columns left
        "bne " LABEL_COL_LOOP "b\n"

        LABEL_SKIP_COL_LOOP
        ":\n"

        // Horizontally add accumulators and store result.
        "addp v28.4s, v28.4s, v29.4s\n"
        "addp v30.4s, v30.4s, v31.4s\n"
        "addp v28.4s, v28.4s, v30.4s\n"
        "sqrshrn v26.4h, v28.4s, %[shift_amount]\n"

        // Store accumulators.
        "st1 {v26.4h}, [%[out_ptr]], #8\n"

        // Decrement rows remaining.
        "subs %[assigned_rows], %[assigned_rows], #1\n"
        "bne " LABEL_ROW_LOOP "b\n"

        // clang-format off
        :  // outputs
        [out_ptr] "+r"(out_ptr), [weights_ptr] "+r"(weights_ptr),
        [col_deltas_bytes] "+r"(col_deltas_bytes), [bias_ptr] "+r"(bias_ptr),
        [nnz_per_row] "+r"(nnz_per_row), [assigned_rows] "+r"(assigned_rows),
        [rhs_ptr] "+r"(rhs_ptr)
        :  // inputs
        [shift_amount] "I"(kShiftAmount)
        :  // clobbers
        "cc", "memory", "x6", "x7", "x8", "v0", "v1", "v2", "v3", "v4", "v5",
        "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
        "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
        "v26", "v27", "v28", "v29", "v30", "v31");
    // clang-format on
  }
}

// Note that the number of exponent bits in the output must exactly match
// the sum of the input and rhs types.
template <typename WeightType, typename RhsType, typename OutType>
typename std::enable_if<IsFixed16Type<WeightType>::value &&
                        IsFixed16Type<RhsType>::value &&
                        IsFixed16Type<OutType>::value>::type
SpMM5_4x4(const WeightType* weights_ptr, const int16_t* col_deltas_bytes,
          const int32_t* nnz_per_row, const RhsType* rhs_ptr,
          const typename TypeOfProduct<WeightType, RhsType>::type* bias_ptr,
          OutType* out_ptr, int64_t assigned_rows, int64_t rows, int64_t cols,
          int relu) {
  constexpr int kShiftAmount = 15 - WeightType::kExponentBits -
                               RhsType::kExponentBits + OutType::kExponentBits;
  // Pointers to the columns.
  const RhsType* rhs2_ptr = rhs_ptr + cols;
  OutType* out2_ptr = out_ptr + rows;
  const RhsType* rhs3_ptr = rhs_ptr + 2 * cols;
  OutType* out3_ptr = out_ptr + 2 * rows;
  const RhsType* rhs4_ptr = rhs_ptr + 3 * cols;
  OutType* out4_ptr = out_ptr + 3 * rows;
  const RhsType* rhs5_ptr = rhs_ptr + 4 * cols;
  OutType* out5_ptr = out_ptr + 4 * rows;
  if (relu) {
    asm(
        // Load the first two column deltas.
        "ldrsh x7, [%[col_deltas_bytes]], #2\n"
        "ldrsh x8, [%[col_deltas_bytes]], #2\n"
        // ld1 doesn't support pre-index, so we do the first addition here.
        "add %[rhs_ptr], %[rhs_ptr], x7\n"
        "add %[rhs2_ptr], %[rhs2_ptr], x7\n"
        "add %[rhs3_ptr], %[rhs3_ptr], x7\n"
        "add %[rhs4_ptr], %[rhs4_ptr], x7\n"
        "add %[rhs5_ptr], %[rhs5_ptr], x7\n"

        LABEL_ROW_LOOP
        ":\n"

        // Load the bias.
        "ld1 {v27.4s}, [%[bias_ptr]], #16\n"

        // Zero out local accumulators.
        "dup v28.4s, v27.s[0]\n"  // for 1st column
        "dup v29.4s, v27.s[1]\n"  // for 1st column
        "dup v30.4s, v27.s[2]\n"  // for 1st column
        "dup v31.4s, v27.s[3]\n"  // for 1st column
        "dup v23.4s, v27.s[0]\n"  // for 2nd column
        "dup v24.4s, v27.s[1]\n"  // for 2nd column
        "dup v25.4s, v27.s[2]\n"  // for 2nd column
        "dup v26.4s, v27.s[3]\n"  // for 2nd column
        "dup v19.4s, v27.s[0]\n"  // for 3rd column
        "dup v20.4s, v27.s[1]\n"  // for 3rd column
        "dup v21.4s, v27.s[2]\n"  // for 3rd column
        "dup v22.4s, v27.s[3]\n"  // for 3rd column
        "dup v15.4s, v27.s[0]\n"  // for 4th column
        "dup v16.4s, v27.s[1]\n"  // for 4th column
        "dup v17.4s, v27.s[2]\n"  // for 4th column
        "dup v18.4s, v27.s[3]\n"  // for 4th column
        "dup v11.4s, v27.s[0]\n"  // for 5th column
        "dup v12.4s, v27.s[1]\n"  // for 5th column
        "dup v13.4s, v27.s[2]\n"  // for 5th column
        "dup v14.4s, v27.s[3]\n"  // for 5th column

        // Update the stopping condition for this set of rows.
        "ldr w6, [%[nnz_per_row]], #4\n"
        "cmp w6, #0\n"
        // Skip the body if there isn't anything in this row.
        "beq " LABEL_SKIP_COL_LOOP "f\n"

        LABEL_COL_LOOP
        ":\n"
        // Load 1 Rhs vectors of size 1x4 each and duplicate into upper half.
        "ld1 {v0.4h}, [%[rhs_ptr]], x8\n"
        "mov v0.d[1], v0.d[0]\n"
        "ld1 {v1.4h}, [%[rhs2_ptr]], x8\n"
        "mov v1.d[1], v1.d[0]\n"
        "ld1 {v8.4h}, [%[rhs3_ptr]], x8\n"
        "mov v8.d[1], v8.d[0]\n"
        "ld1 {v9.4h}, [%[rhs4_ptr]], x8\n"
        "mov v9.d[1], v9.d[0]\n"
        "ld1 {v10.4h}, [%[rhs5_ptr]], x8\n"
        "mov v10.d[1], v10.d[0]\n"

        // Start this load now, which we won't need until the end of the loop.
        "ldrsh x8, [%[col_deltas_bytes]], #2\n"

        // Load 16 Lhs cells corresponding to a 4x4 block.
        "ld1 {v2.8h, v3.8h}, [%[weights_ptr]], #32\n"

        // Multiply-accumulate.
        "smlal v28.4s, v2.4h, v0.4h\n"    // for 1st column
        "smlal2 v29.4s, v2.8h, v0.8h\n"   // for 1st column
        "smlal v30.4s, v3.4h, v0.4h\n"    // for 1st column
        "smlal2 v31.4s, v3.8h, v0.8h\n"   // for 1st columh
        "smlal v23.4s, v2.4h, v1.4h\n"    // for 2nd column
        "smlal2 v24.4s, v2.8h, v1.8h\n"   // for 2nd column
        "smlal v25.4s, v3.4h, v1.4h\n"    // for 2nd column
        "smlal2 v26.4s, v3.8h, v1.8h\n"   // for 2nd column
        "smlal v19.4s, v2.4h, v8.4h\n"    // for 3rd column
        "smlal2 v20.4s, v2.8h, v8.8h\n"   // for 3rd column
        "smlal v21.4s, v3.4h, v8.4h\n"    // for 3rd column
        "smlal2 v22.4s, v3.8h, v8.8h\n"   // for 3rd column
        "smlal v15.4s, v2.4h, v9.4h\n"    // for 4th column
        "smlal2 v16.4s, v2.8h, v9.8h\n"   // for 4th column
        "smlal v17.4s, v3.4h, v9.4h\n"    // for 4th column
        "smlal2 v18.4s, v3.8h, v9.8h\n"   // for 4th column
        "smlal v11.4s, v2.4h, v10.4h\n"   // for 5th column
        "smlal2 v12.4s, v2.8h, v10.8h\n"  // for 5th column
        "smlal v13.4s, v3.4h, v10.4h\n"   // for 5th column
        "smlal2 v14.4s, v3.8h, v10.8h\n"  // for 5th column

        // Loop. Decrement loop index.
        "subs w6, w6, #1\n"  // decrement (reduced) columns left
        "bne " LABEL_COL_LOOP "b\n"

        LABEL_SKIP_COL_LOOP
        ":\n"

        "movi v0.4s, #0\n"
        "addp v28.4s, v28.4s, v29.4s\n"  // 1st column
        "addp v23.4s, v23.4s, v24.4s\n"  // 2nd column
        "addp v19.4s, v19.4s, v20.4s\n"  // 3rd column
        "addp v15.4s, v15.4s, v16.4s\n"  // 4th column
        "addp v11.4s, v11.4s, v12.4s\n"  // 5th column

        "addp v30.4s, v30.4s, v31.4s\n"  // 1st column
        "addp v25.4s, v25.4s, v26.4s\n"  // 2nd column
        "addp v21.4s, v21.4s, v22.4s\n"  // 3rd column
        "addp v17.4s, v17.4s, v18.4s\n"  // 4th column
        "addp v13.4s, v13.4s, v14.4s\n"  // 5th column

        "addp v28.4s, v28.4s, v30.4s\n"  // 1st column
        "addp v23.4s, v23.4s, v25.4s\n"  // 2nd column
        "addp v19.4s, v19.4s, v21.4s\n"  // 3rd column
        "addp v15.4s, v15.4s, v17.4s\n"  // 4th column
        "addp v11.4s, v11.4s, v13.4s\n"  // 5th column

        // Do relu as requested.
        "smax v28.4s, v28.4s, v0.4s\n"
        "smax v23.4s, v23.4s, v0.4s\n"
        "smax v19.4s, v19.4s, v0.4s\n"
        "smax v15.4s, v15.4s, v0.4s\n"
        "smax v11.4s, v11.4s, v0.4s\n"
        "sqrshrn v26.4h, v28.4s, %[shift_amount]\n"
        "sqrshrn v22.4h, v23.4s, %[shift_amount]\n"
        "sqrshrn v18.4h, v19.4s, %[shift_amount]\n"
        "sqrshrn v14.4h, v15.4s, %[shift_amount]\n"
        "sqrshrn v10.4h, v11.4s, %[shift_amount]\n"

        // Store accumulators.
        "st1 {v26.4h}, [%[out_ptr]], #8\n"
        "st1 {v22.4h}, [%[out2_ptr]], #8\n"
        "st1 {v18.4h}, [%[out3_ptr]], #8\n"
        "st1 {v14.4h}, [%[out4_ptr]], #8\n"
        "st1 {v10.4h}, [%[out5_ptr]], #8\n"

        // Decrement rows remaining.
        "subs %[assigned_rows], %[assigned_rows], #1\n"
        "bne " LABEL_ROW_LOOP "b\n"

        // clang-format off
        :  // outputs
        [out_ptr] "+r"(out_ptr), [out2_ptr] "+r"(out2_ptr),
        [out3_ptr] "+r"(out3_ptr), [out4_ptr] "+r"(out4_ptr),
        [out5_ptr] "+r"(out5_ptr), [weights_ptr] "+r"(weights_ptr),
        [col_deltas_bytes] "+r"(col_deltas_bytes), [bias_ptr] "+r"(bias_ptr),
        [nnz_per_row] "+r"(nnz_per_row), [assigned_rows] "+r"(assigned_rows),
        [rhs_ptr] "+r"(rhs_ptr), [rhs2_ptr] "+r"(rhs2_ptr),
        [rhs3_ptr] "+r"(rhs3_ptr), [rhs4_ptr] "+r"(rhs4_ptr),
        [rhs5_ptr] "+r"(rhs5_ptr)
        :  // inputs
        [shift_amount] "I"(kShiftAmount)
        :  // clobbers
        "cc", "memory", "x6", "x7", "x8", "v0", "v1", "v2", "v3", "v4", "v5",
        "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
        "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
        "v26", "v27", "v28", "v29", "v30", "v31");
    // clang-format on
  } else {
    asm(
        // Load the first two column deltas.
        "ldrsh x7, [%[col_deltas_bytes]], #2\n"
        "ldrsh x8, [%[col_deltas_bytes]], #2\n"
        // ld1 doesn't support pre-index, so we do the first addition here.
        "add %[rhs_ptr], %[rhs_ptr], x7\n"
        "add %[rhs2_ptr], %[rhs2_ptr], x7\n"
        "add %[rhs3_ptr], %[rhs3_ptr], x7\n"
        "add %[rhs4_ptr], %[rhs4_ptr], x7\n"
        "add %[rhs5_ptr], %[rhs5_ptr], x7\n"

        LABEL_ROW_LOOP
        ":\n"

        // Load the bias.
        "ld1 {v27.4s}, [%[bias_ptr]], #16\n"

        // Zero out local accumulators.
        "dup v28.4s, v27.s[0]\n"  // for 1st column
        "dup v29.4s, v27.s[1]\n"  // for 1st column
        "dup v30.4s, v27.s[2]\n"  // for 1st column
        "dup v31.4s, v27.s[3]\n"  // for 1st column
        "dup v23.4s, v27.s[0]\n"  // for 2nd column
        "dup v24.4s, v27.s[1]\n"  // for 2nd column
        "dup v25.4s, v27.s[2]\n"  // for 2nd column
        "dup v26.4s, v27.s[3]\n"  // for 2nd column
        "dup v19.4s, v27.s[0]\n"  // for 3rd column
        "dup v20.4s, v27.s[1]\n"  // for 3rd column
        "dup v21.4s, v27.s[2]\n"  // for 3rd column
        "dup v22.4s, v27.s[3]\n"  // for 3rd column
        "dup v15.4s, v27.s[0]\n"  // for 4th column
        "dup v16.4s, v27.s[1]\n"  // for 4th column
        "dup v17.4s, v27.s[2]\n"  // for 4th column
        "dup v18.4s, v27.s[3]\n"  // for 4th column
        "dup v11.4s, v27.s[0]\n"  // for 5th column
        "dup v12.4s, v27.s[1]\n"  // for 5th column
        "dup v13.4s, v27.s[2]\n"  // for 5th column
        "dup v14.4s, v27.s[3]\n"  // for 5th column

        // Update the stopping condition for this set of rows.
        "ldr w6, [%[nnz_per_row]], #4\n"
        "cmp w6, #0\n"
        // Skip the body if there isn't anything in this row.
        "beq " LABEL_SKIP_COL_LOOP "f\n"

        LABEL_COL_LOOP
        ":\n"
        // Load 1 Rhs vectors of size 1x4 each and duplicate into upper half.
        "ld1 {v0.4h}, [%[rhs_ptr]], x8\n"
        "mov v0.d[1], v0.d[0]\n"
        "ld1 {v1.4h}, [%[rhs2_ptr]], x8\n"
        "mov v1.d[1], v1.d[0]\n"
        "ld1 {v8.4h}, [%[rhs3_ptr]], x8\n"
        "mov v8.d[1], v8.d[0]\n"
        "ld1 {v9.4h}, [%[rhs4_ptr]], x8\n"
        "mov v9.d[1], v9.d[0]\n"
        "ld1 {v10.4h}, [%[rhs5_ptr]], x8\n"
        "mov v10.d[1], v10.d[0]\n"

        // Start this load now, which we won't need until the end of the loop.
        "ldrsh x8, [%[col_deltas_bytes]], #2\n"

        // Load 16 Lhs cells corresponding to a 4x4 block.
        "ld1 {v2.8h, v3.8h}, [%[weights_ptr]], #32\n"

        // Multiply-accumulate.
        "smlal v28.4s, v2.4h, v0.4h\n"    // for 1st column
        "smlal2 v29.4s, v2.8h, v0.8h\n"   // for 1st column
        "smlal v30.4s, v3.4h, v0.4h\n"    // for 1st column
        "smlal2 v31.4s, v3.8h, v0.8h\n"   // for 1st columh
        "smlal v23.4s, v2.4h, v1.4h\n"    // for 2nd column
        "smlal2 v24.4s, v2.8h, v1.8h\n"   // for 2nd column
        "smlal v25.4s, v3.4h, v1.4h\n"    // for 2nd column
        "smlal2 v26.4s, v3.8h, v1.8h\n"   // for 2nd column
        "smlal v19.4s, v2.4h, v8.4h\n"    // for 3rd column
        "smlal2 v20.4s, v2.8h, v8.8h\n"   // for 3rd column
        "smlal v21.4s, v3.4h, v8.4h\n"    // for 3rd column
        "smlal2 v22.4s, v3.8h, v8.8h\n"   // for 3rd column
        "smlal v15.4s, v2.4h, v9.4h\n"    // for 4th column
        "smlal2 v16.4s, v2.8h, v9.8h\n"   // for 4th column
        "smlal v17.4s, v3.4h, v9.4h\n"    // for 4th column
        "smlal2 v18.4s, v3.8h, v9.8h\n"   // for 4th column
        "smlal v11.4s, v2.4h, v10.4h\n"   // for 5th column
        "smlal2 v12.4s, v2.8h, v10.8h\n"  // for 5th column
        "smlal v13.4s, v3.4h, v10.4h\n"   // for 5th column
        "smlal2 v14.4s, v3.8h, v10.8h\n"  // for 5th column

        // Loop. Decrement loop index.
        "subs w6, w6, #1\n"  // decrement (reduced) columns left
        "bne " LABEL_COL_LOOP "b\n"

        LABEL_SKIP_COL_LOOP
        ":\n"

        "addp v28.4s, v28.4s, v29.4s\n"  // 1st column
        "addp v23.4s, v23.4s, v24.4s\n"  // 2nd column
        "addp v19.4s, v19.4s, v20.4s\n"  // 3rd column
        "addp v15.4s, v15.4s, v16.4s\n"  // 4th column
        "addp v11.4s, v11.4s, v12.4s\n"  // 5th column

        "addp v30.4s, v30.4s, v31.4s\n"  // 1st column
        "addp v25.4s, v25.4s, v26.4s\n"  // 2nd column
        "addp v21.4s, v21.4s, v22.4s\n"  // 3rd column
        "addp v17.4s, v17.4s, v18.4s\n"  // 4th column
        "addp v13.4s, v13.4s, v14.4s\n"  // 5th column

        "addp v28.4s, v28.4s, v30.4s\n"  // 1st column
        "addp v23.4s, v23.4s, v25.4s\n"  // 2nd column
        "addp v19.4s, v19.4s, v21.4s\n"  // 3rd column
        "addp v15.4s, v15.4s, v17.4s\n"  // 4th column
        "addp v11.4s, v11.4s, v13.4s\n"  // 5th column

        "sqrshrn v26.4h, v28.4s, %[shift_amount]\n"
        "sqrshrn v22.4h, v23.4s, %[shift_amount]\n"
        "sqrshrn v18.4h, v19.4s, %[shift_amount]\n"
        "sqrshrn v14.4h, v15.4s, %[shift_amount]\n"
        "sqrshrn v10.4h, v11.4s, %[shift_amount]\n"

        // Store accumulators.
        "st1 {v26.4h}, [%[out_ptr]], #8\n"
        "st1 {v22.4h}, [%[out2_ptr]], #8\n"
        "st1 {v18.4h}, [%[out3_ptr]], #8\n"
        "st1 {v14.4h}, [%[out4_ptr]], #8\n"
        "st1 {v10.4h}, [%[out5_ptr]], #8\n"

        // Decrement rows remaining.
        "subs %[assigned_rows], %[assigned_rows], #1\n"
        "bne " LABEL_ROW_LOOP "b\n"

        // clang-format off
        :  // outputs
        [out_ptr] "+r"(out_ptr), [out2_ptr] "+r"(out2_ptr),
        [out3_ptr] "+r"(out3_ptr), [out4_ptr] "+r"(out4_ptr),
        [out5_ptr] "+r"(out5_ptr), [weights_ptr] "+r"(weights_ptr),
        [col_deltas_bytes] "+r"(col_deltas_bytes), [bias_ptr] "+r"(bias_ptr),
        [nnz_per_row] "+r"(nnz_per_row), [assigned_rows] "+r"(assigned_rows),
        [rhs_ptr] "+r"(rhs_ptr), [rhs2_ptr] "+r"(rhs2_ptr),
        [rhs3_ptr] "+r"(rhs3_ptr), [rhs4_ptr] "+r"(rhs4_ptr),
        [rhs5_ptr] "+r"(rhs5_ptr)
        :  // inputs
        [shift_amount] "I"(kShiftAmount)
        :  // clobbers
        "cc", "memory", "x6", "x7", "x8", "v0", "v1", "v2", "v3", "v4", "v5",
        "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
        "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
        "v26", "v27", "v28", "v29", "v30", "v31");
    // clang-format on
  }
}

// Note that the number of exponent bits in the output must exactly match
// the sum of the input and rhs types.
template <typename WeightType, typename RhsType, typename OutType>
typename std::enable_if<
    IsFixed16Type<WeightType>::value && IsFixed16Type<RhsType>::value &&
    IsFixed32Type<OutType>::value &&
    !std::is_same<OutType, typename TypeOfProduct<WeightType,
                                                  RhsType>::type>::value>::type
SpMV_4x4(const WeightType* weights_ptr, const int16_t* col_deltas_bytes,
         const int32_t* nnz_per_row, const RhsType* rhs_ptr,
         const typename TypeOfProduct<WeightType, RhsType>::type* bias_ptr,
         OutType* out_ptr, int64_t assigned_rows,
         int64_t rows /* only used in SpMM variants */,
         int64_t cols /* only used in SpMM variants */, int relu) {
  constexpr int kShiftAmount =
      TypeOfProduct<WeightType, RhsType>::type::kMantissaBits -
      OutType::kMantissaBits;
  static_assert(kShiftAmount > 0,
                "Result must have fewer mantissa bits than product");
  if (relu) {
    asm(
        // Load the first two column deltas.
        "ldrsh x7, [%[col_deltas_bytes]], #2\n"
        "ldrsh x8, [%[col_deltas_bytes]], #2\n"
        // ld1 doesn't support pre-index, so we do the first addition here.
        "add %[rhs_ptr], %[rhs_ptr], x7\n"

        "movi v25.4s, #0\n"

        LABEL_ROW_LOOP
        ":\n"

        // Load the bias.
        "ld1 {v27.4s}, [%[bias_ptr]], #16\n"

        // Zero out local accumulators.
        "dup v28.4s, v27.s[0]\n"  // accum_0 = 0
        "dup v29.4s, v27.s[1]\n"  // accum_1 = 0
        "dup v30.4s, v27.s[2]\n"  // accum_2 = 0
        "dup v31.4s, v27.s[3]\n"  // accum_3 = 0

        // Update the stopping condition for this set of rows.
        "ldr w6, [%[nnz_per_row]], #4\n"
        "cmp w6, #0\n"
        // Skip the body if there isn't anything in this row.
        "beq " LABEL_SKIP_COL_LOOP "f\n"

        LABEL_COL_LOOP
        ":\n"
        // Load 1 Rhs vectors of size 1x4 each.
        "ld1 {v0.4h}, [%[rhs_ptr]], x8\n"
        // Duplicate the lower half into the upper half.
        "mov v0.d[1], v0.d[0]\n"

        // Start this load now, which we won't need until the end of the loop.
        "ldrsh x8, [%[col_deltas_bytes]], #2\n"

        // Load 16 Lhs cells corresponding to a 4x4 block.
        "ld1 {v2.8h, v3.8h}, [%[weights_ptr]], #32\n"

        // Multiply-accumulate.
        "smlal v28.4s, v2.4h, v0.4h\n"
        "smlal2 v29.4s, v2.8h, v0.8h\n"
        "smlal v30.4s, v3.4h, v0.4h\n"
        "smlal2 v31.4s, v3.8h, v0.8h\n"

        // Loop. Decrement loop index.
        "subs w6, w6, #1\n"  // decrement (reduced) columns left
        "bne " LABEL_COL_LOOP "b\n"

        LABEL_SKIP_COL_LOOP
        ":\n"

        // Horizontally add accumulators and store result.
        "addp v28.4s, v28.4s, v29.4s\n"
        "addp v30.4s, v30.4s, v31.4s\n"
        "addp v28.4s, v28.4s, v30.4s\n"

        // Do relu if requested.
        "smax v28.4s, v28.4s, v25.4s\n"
        "srshr v28.4s, v28.4s, %[shift_amount]\n"

        // Store accumulators.
        "st1 {v28.4s}, [%[out_ptr]], #16\n"

        // Decrement rows remaining.
        "subs %[assigned_rows], %[assigned_rows], #1\n"
        "bne " LABEL_ROW_LOOP "b\n"

        // clang-format off
        :  // outputs
        [out_ptr] "+r"(out_ptr), [weights_ptr] "+r"(weights_ptr),
        [col_deltas_bytes] "+r"(col_deltas_bytes), [bias_ptr] "+r"(bias_ptr),
        [nnz_per_row] "+r"(nnz_per_row), [assigned_rows] "+r"(assigned_rows),
        [rhs_ptr] "+r"(rhs_ptr)
        :  // inputs
        [shift_amount] "I"(kShiftAmount)
        :  // clobbers
        "cc", "memory", "x6", "x7", "x8", "v0", "v1", "v2", "v3", "v4", "v5",
        "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
        "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
        "v26", "v27", "v28", "v29", "v30", "v31");
    // clang-format on
  } else {
    asm(
        // Load the first two column deltas.
        "ldrsh x7, [%[col_deltas_bytes]], #2\n"
        "ldrsh x8, [%[col_deltas_bytes]], #2\n"
        // ld1 doesn't support pre-index, so we do the first addition here.
        "add %[rhs_ptr], %[rhs_ptr], x7\n"

        "movi v25.4s, #0\n"

        LABEL_ROW_LOOP
        ":\n"

        // Load the bias.
        "ld1 {v27.4s}, [%[bias_ptr]], #16\n"

        // Zero out local accumulators.
        "dup v28.4s, v27.s[0]\n"  // accum_0 = 0
        "dup v29.4s, v27.s[1]\n"  // accum_1 = 0
        "dup v30.4s, v27.s[2]\n"  // accum_2 = 0
        "dup v31.4s, v27.s[3]\n"  // accum_3 = 0

        // Update the stopping condition for this set of rows.
        "ldr w6, [%[nnz_per_row]], #4\n"
        "cmp w6, #0\n"
        // Skip the body if there isn't anything in this row.
        "beq " LABEL_SKIP_COL_LOOP "f\n"

        LABEL_COL_LOOP
        ":\n"
        // Load 1 Rhs vectors of size 1x4 each.
        "ld1 {v0.4h}, [%[rhs_ptr]], x8\n"
        // Duplicate the lower half into the upper half.
        "mov v0.d[1], v0.d[0]\n"

        // Start this load now, which we won't need until the end of the loop.
        "ldrsh x8, [%[col_deltas_bytes]], #2\n"

        // Load 16 Lhs cells corresponding to a 4x4 block.
        "ld1 {v2.8h, v3.8h}, [%[weights_ptr]], #32\n"

        // Multiply-accumulate.
        "smlal v28.4s, v2.4h, v0.4h\n"
        "smlal2 v29.4s, v2.8h, v0.8h\n"
        "smlal v30.4s, v3.4h, v0.4h\n"
        "smlal2 v31.4s, v3.8h, v0.8h\n"

        // Loop. Decrement loop index.
        "subs w6, w6, #1\n"  // decrement (reduced) columns left
        "bne " LABEL_COL_LOOP "b\n"

        LABEL_SKIP_COL_LOOP
        ":\n"

        // Horizontally add accumulators and store result.
        "addp v28.4s, v28.4s, v29.4s\n"
        "addp v30.4s, v30.4s, v31.4s\n"
        "addp v28.4s, v28.4s, v30.4s\n"

        "srshr v28.4s, v28.4s, %[shift_amount]\n"
        // Store accumulators.
        "st1 {v28.4s}, [%[out_ptr]], #16\n"

        // Decrement rows remaining.
        "subs %[assigned_rows], %[assigned_rows], #1\n"
        "bne " LABEL_ROW_LOOP "b\n"

        // clang-format off
        :  // outputs
        [out_ptr] "+r"(out_ptr), [weights_ptr] "+r"(weights_ptr),
        [col_deltas_bytes] "+r"(col_deltas_bytes), [bias_ptr] "+r"(bias_ptr),
        [nnz_per_row] "+r"(nnz_per_row), [assigned_rows] "+r"(assigned_rows),
        [rhs_ptr] "+r"(rhs_ptr)
        :  // inputs
        [shift_amount] "I"(kShiftAmount)
        :  // clobbers
        "cc", "memory", "x6", "x7", "x8", "v0", "v1", "v2", "v3", "v4", "v5",
        "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
        "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
        "v26", "v27", "v28", "v29", "v30", "v31");
    // clang-format on
  }
}

// Note that the number of exponent bits in the output must exactly match
// the sum of the input and rhs types.
template <typename WeightType, typename RhsType, typename OutType>
typename std::enable_if<
    IsFixed16Type<WeightType>::value && IsFixed16Type<RhsType>::value &&
    IsFixed32Type<OutType>::value &&
    !std::is_same<OutType, typename TypeOfProduct<WeightType,
                                                  RhsType>::type>::value>::type
SpMM5_4x4(const WeightType* weights_ptr, const int16_t* col_deltas_bytes,
          const int32_t* nnz_per_row, const RhsType* rhs_ptr,
          const typename TypeOfProduct<WeightType, RhsType>::type* bias_ptr,
          OutType* out_ptr, int64_t assigned_rows, int64_t rows, int64_t cols,
          int relu) {
  constexpr int kShiftAmount =
      TypeOfProduct<WeightType, RhsType>::type::kMantissaBits -
      OutType::kMantissaBits;
  static_assert(kShiftAmount > 0,
                "Result must have fewer mantissa bits than product");
  // Pointers to the columns.
  const RhsType* rhs2_ptr = rhs_ptr + cols;
  OutType* out2_ptr = out_ptr + rows;
  const RhsType* rhs3_ptr = rhs_ptr + 2 * cols;
  OutType* out3_ptr = out_ptr + 2 * rows;
  const RhsType* rhs4_ptr = rhs_ptr + 3 * cols;
  OutType* out4_ptr = out_ptr + 3 * rows;
  const RhsType* rhs5_ptr = rhs_ptr + 4 * cols;
  OutType* out5_ptr = out_ptr + 4 * rows;
  if (relu) {
    asm(
        // Load the first two column deltas.
        "ldrsh x7, [%[col_deltas_bytes]], #2\n"
        "ldrsh x8, [%[col_deltas_bytes]], #2\n"
        // ld1 doesn't support pre-index, so we do the first addition here.
        "add %[rhs_ptr], %[rhs_ptr], x7\n"
        "add %[rhs2_ptr], %[rhs2_ptr], x7\n"
        "add %[rhs3_ptr], %[rhs3_ptr], x7\n"
        "add %[rhs4_ptr], %[rhs4_ptr], x7\n"
        "add %[rhs5_ptr], %[rhs5_ptr], x7\n"

        LABEL_ROW_LOOP
        ":\n"

        // Load the bias.
        "ld1 {v27.4s}, [%[bias_ptr]], #16\n"

        // Zero out local accumulators.
        "dup v28.4s, v27.s[0]\n"  // for 1st column
        "dup v29.4s, v27.s[1]\n"  // for 1st column
        "dup v30.4s, v27.s[2]\n"  // for 1st column
        "dup v31.4s, v27.s[3]\n"  // for 1st column
        "dup v23.4s, v27.s[0]\n"  // for 2nd column
        "dup v24.4s, v27.s[1]\n"  // for 2nd column
        "dup v25.4s, v27.s[2]\n"  // for 2nd column
        "dup v26.4s, v27.s[3]\n"  // for 2nd column
        "dup v19.4s, v27.s[0]\n"  // for 3rd column
        "dup v20.4s, v27.s[1]\n"  // for 3rd column
        "dup v21.4s, v27.s[2]\n"  // for 3rd column
        "dup v22.4s, v27.s[3]\n"  // for 3rd column
        "dup v15.4s, v27.s[0]\n"  // for 4th column
        "dup v16.4s, v27.s[1]\n"  // for 4th column
        "dup v17.4s, v27.s[2]\n"  // for 4th column
        "dup v18.4s, v27.s[3]\n"  // for 4th column
        "dup v11.4s, v27.s[0]\n"  // for 5th column
        "dup v12.4s, v27.s[1]\n"  // for 5th column
        "dup v13.4s, v27.s[2]\n"  // for 5th column
        "dup v14.4s, v27.s[3]\n"  // for 5th column

        // Update the stopping condition for this set of rows.
        "ldr w6, [%[nnz_per_row]], #4\n"
        "cmp w6, #0\n"
        // Skip the body if there isn't anything in this row.
        "beq " LABEL_SKIP_COL_LOOP "f\n"

        LABEL_COL_LOOP
        ":\n"
        // Load 1 Rhs vectors of size 1x4 each and duplicate into upper half.
        "ld1 {v0.4h}, [%[rhs_ptr]], x8\n"
        "mov v0.d[1], v0.d[0]\n"
        "ld1 {v1.4h}, [%[rhs2_ptr]], x8\n"
        "mov v1.d[1], v1.d[0]\n"
        "ld1 {v8.4h}, [%[rhs3_ptr]], x8\n"
        "mov v8.d[1], v8.d[0]\n"
        "ld1 {v9.4h}, [%[rhs4_ptr]], x8\n"
        "mov v9.d[1], v9.d[0]\n"
        "ld1 {v10.4h}, [%[rhs5_ptr]], x8\n"
        "mov v10.d[1], v10.d[0]\n"

        // Start this load now, which we won't need until the end of the loop.
        "ldrsh x8, [%[col_deltas_bytes]], #2\n"

        // Load 16 Lhs cells corresponding to a 4x4 block.
        "ld1 {v2.8h, v3.8h}, [%[weights_ptr]], #32\n"

        // Multiply-accumulate.
        "smlal v28.4s, v2.4h, v0.4h\n"    // for 1st column
        "smlal2 v29.4s, v2.8h, v0.8h\n"   // for 1st column
        "smlal v30.4s, v3.4h, v0.4h\n"    // for 1st column
        "smlal2 v31.4s, v3.8h, v0.8h\n"   // for 1st columh
        "smlal v23.4s, v2.4h, v1.4h\n"    // for 2nd column
        "smlal2 v24.4s, v2.8h, v1.8h\n"   // for 2nd column
        "smlal v25.4s, v3.4h, v1.4h\n"    // for 2nd column
        "smlal2 v26.4s, v3.8h, v1.8h\n"   // for 2nd column
        "smlal v19.4s, v2.4h, v8.4h\n"    // for 3rd column
        "smlal2 v20.4s, v2.8h, v8.8h\n"   // for 3rd column
        "smlal v21.4s, v3.4h, v8.4h\n"    // for 3rd column
        "smlal2 v22.4s, v3.8h, v8.8h\n"   // for 3rd column
        "smlal v15.4s, v2.4h, v9.4h\n"    // for 4th column
        "smlal2 v16.4s, v2.8h, v9.8h\n"   // for 4th column
        "smlal v17.4s, v3.4h, v9.4h\n"    // for 4th column
        "smlal2 v18.4s, v3.8h, v9.8h\n"   // for 4th column
        "smlal v11.4s, v2.4h, v10.4h\n"   // for 5th column
        "smlal2 v12.4s, v2.8h, v10.8h\n"  // for 5th column
        "smlal v13.4s, v3.4h, v10.4h\n"   // for 5th column
        "smlal2 v14.4s, v3.8h, v10.8h\n"  // for 5th column

        // Loop. Decrement loop index.
        "subs w6, w6, #1\n"  // decrement (reduced) columns left
        "bne " LABEL_COL_LOOP "b\n"

        LABEL_SKIP_COL_LOOP
        ":\n"

        "movi v0.4s, #0\n"
        "addp v28.4s, v28.4s, v29.4s\n"  // 1st column
        "addp v23.4s, v23.4s, v24.4s\n"  // 2nd column
        "addp v19.4s, v19.4s, v20.4s\n"  // 3rd column
        "addp v15.4s, v15.4s, v16.4s\n"  // 4th column
        "addp v11.4s, v11.4s, v12.4s\n"  // 5th column

        "addp v30.4s, v30.4s, v31.4s\n"  // 1st column
        "addp v25.4s, v25.4s, v26.4s\n"  // 2nd column
        "addp v21.4s, v21.4s, v22.4s\n"  // 3rd column
        "addp v17.4s, v17.4s, v18.4s\n"  // 4th column
        "addp v13.4s, v13.4s, v14.4s\n"  // 5th column

        "addp v28.4s, v28.4s, v30.4s\n"  // 1st column
        "addp v23.4s, v23.4s, v25.4s\n"  // 2nd column
        "addp v19.4s, v19.4s, v21.4s\n"  // 3rd column
        "addp v15.4s, v15.4s, v17.4s\n"  // 4th column
        "addp v11.4s, v11.4s, v13.4s\n"  // 5th column

        // Do relu as requested.
        "smax v28.4s, v28.4s, v0.4s\n"
        "smax v23.4s, v23.4s, v0.4s\n"
        "smax v19.4s, v19.4s, v0.4s\n"
        "smax v15.4s, v15.4s, v0.4s\n"
        "smax v11.4s, v11.4s, v0.4s\n"

        "srshr v28.4s, v28.4s, %[shift_amount]\n"
        "srshr v23.4s, v23.4s, %[shift_amount]\n"
        "srshr v19.4s, v19.4s, %[shift_amount]\n"
        "srshr v15.4s, v15.4s, %[shift_amount]\n"
        "srshr v11.4s, v11.4s, %[shift_amount]\n"

        // Store accumulators.
        "st1 {v28.4s}, [%[out_ptr]], #16\n"
        "st1 {v23.4s}, [%[out2_ptr]], #16\n"
        "st1 {v19.4s}, [%[out3_ptr]], #16\n"
        "st1 {v15.4s}, [%[out4_ptr]], #16\n"
        "st1 {v11.4s}, [%[out5_ptr]], #16\n"

        // Decrement rows remaining.
        "subs %[assigned_rows], %[assigned_rows], #1\n"
        "bne " LABEL_ROW_LOOP "b\n"

        // clang-format off
        :  // outputs
        [out_ptr] "+r"(out_ptr), [out2_ptr] "+r"(out2_ptr),
        [out3_ptr] "+r"(out3_ptr), [out4_ptr] "+r"(out4_ptr),
        [out5_ptr] "+r"(out5_ptr), [weights_ptr] "+r"(weights_ptr),
        [col_deltas_bytes] "+r"(col_deltas_bytes), [bias_ptr] "+r"(bias_ptr),
        [nnz_per_row] "+r"(nnz_per_row), [assigned_rows] "+r"(assigned_rows),
        [rhs_ptr] "+r"(rhs_ptr), [rhs2_ptr] "+r"(rhs2_ptr),
        [rhs3_ptr] "+r"(rhs3_ptr), [rhs4_ptr] "+r"(rhs4_ptr),
        [rhs5_ptr] "+r"(rhs5_ptr)
        :  // inputs
        [shift_amount] "I"(kShiftAmount)
        :  // clobbers
        "cc", "memory", "x6", "x7", "x8", "v0", "v1", "v2", "v3", "v4", "v5",
        "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
        "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
        "v26", "v27", "v28", "v29", "v30", "v31");
    // clang-format on
  } else {
    asm(
        // Load the first two column deltas.
        "ldrsh x7, [%[col_deltas_bytes]], #2\n"
        "ldrsh x8, [%[col_deltas_bytes]], #2\n"
        // ld1 doesn't support pre-index, so we do the first addition here.
        "add %[rhs_ptr], %[rhs_ptr], x7\n"
        "add %[rhs2_ptr], %[rhs2_ptr], x7\n"
        "add %[rhs3_ptr], %[rhs3_ptr], x7\n"
        "add %[rhs4_ptr], %[rhs4_ptr], x7\n"
        "add %[rhs5_ptr], %[rhs5_ptr], x7\n"

        LABEL_ROW_LOOP
        ":\n"

        // Load the bias.
        "ld1 {v27.4s}, [%[bias_ptr]], #16\n"

        // Zero out local accumulators.
        "dup v28.4s, v27.s[0]\n"  // for 1st column
        "dup v29.4s, v27.s[1]\n"  // for 1st column
        "dup v30.4s, v27.s[2]\n"  // for 1st column
        "dup v31.4s, v27.s[3]\n"  // for 1st column
        "dup v23.4s, v27.s[0]\n"  // for 2nd column
        "dup v24.4s, v27.s[1]\n"  // for 2nd column
        "dup v25.4s, v27.s[2]\n"  // for 2nd column
        "dup v26.4s, v27.s[3]\n"  // for 2nd column
        "dup v19.4s, v27.s[0]\n"  // for 3rd column
        "dup v20.4s, v27.s[1]\n"  // for 3rd column
        "dup v21.4s, v27.s[2]\n"  // for 3rd column
        "dup v22.4s, v27.s[3]\n"  // for 3rd column
        "dup v15.4s, v27.s[0]\n"  // for 4th column
        "dup v16.4s, v27.s[1]\n"  // for 4th column
        "dup v17.4s, v27.s[2]\n"  // for 4th column
        "dup v18.4s, v27.s[3]\n"  // for 4th column
        "dup v11.4s, v27.s[0]\n"  // for 5th column
        "dup v12.4s, v27.s[1]\n"  // for 5th column
        "dup v13.4s, v27.s[2]\n"  // for 5th column
        "dup v14.4s, v27.s[3]\n"  // for 5th column

        // Update the stopping condition for this set of rows.
        "ldr w6, [%[nnz_per_row]], #4\n"
        "cmp w6, #0\n"
        // Skip the body if there isn't anything in this row.
        "beq " LABEL_SKIP_COL_LOOP "f\n"

        LABEL_COL_LOOP
        ":\n"
        // Load 1 Rhs vectors of size 1x4 each and duplicate into upper half.
        "ld1 {v0.4h}, [%[rhs_ptr]], x8\n"
        "mov v0.d[1], v0.d[0]\n"
        "ld1 {v1.4h}, [%[rhs2_ptr]], x8\n"
        "mov v1.d[1], v1.d[0]\n"
        "ld1 {v8.4h}, [%[rhs3_ptr]], x8\n"
        "mov v8.d[1], v8.d[0]\n"
        "ld1 {v9.4h}, [%[rhs4_ptr]], x8\n"
        "mov v9.d[1], v9.d[0]\n"
        "ld1 {v10.4h}, [%[rhs5_ptr]], x8\n"
        "mov v10.d[1], v10.d[0]\n"

        // Start this load now, which we won't need until the end of the loop.
        "ldrsh x8, [%[col_deltas_bytes]], #2\n"

        // Load 16 Lhs cells corresponding to a 4x4 block.
        "ld1 {v2.8h, v3.8h}, [%[weights_ptr]], #32\n"

        // Multiply-accumulate.
        "smlal v28.4s, v2.4h, v0.4h\n"    // for 1st column
        "smlal2 v29.4s, v2.8h, v0.8h\n"   // for 1st column
        "smlal v30.4s, v3.4h, v0.4h\n"    // for 1st column
        "smlal2 v31.4s, v3.8h, v0.8h\n"   // for 1st columh
        "smlal v23.4s, v2.4h, v1.4h\n"    // for 2nd column
        "smlal2 v24.4s, v2.8h, v1.8h\n"   // for 2nd column
        "smlal v25.4s, v3.4h, v1.4h\n"    // for 2nd column
        "smlal2 v26.4s, v3.8h, v1.8h\n"   // for 2nd column
        "smlal v19.4s, v2.4h, v8.4h\n"    // for 3rd column
        "smlal2 v20.4s, v2.8h, v8.8h\n"   // for 3rd column
        "smlal v21.4s, v3.4h, v8.4h\n"    // for 3rd column
        "smlal2 v22.4s, v3.8h, v8.8h\n"   // for 3rd column
        "smlal v15.4s, v2.4h, v9.4h\n"    // for 4th column
        "smlal2 v16.4s, v2.8h, v9.8h\n"   // for 4th column
        "smlal v17.4s, v3.4h, v9.4h\n"    // for 4th column
        "smlal2 v18.4s, v3.8h, v9.8h\n"   // for 4th column
        "smlal v11.4s, v2.4h, v10.4h\n"   // for 5th column
        "smlal2 v12.4s, v2.8h, v10.8h\n"  // for 5th column
        "smlal v13.4s, v3.4h, v10.4h\n"   // for 5th column
        "smlal2 v14.4s, v3.8h, v10.8h\n"  // for 5th column

        // Loop. Decrement loop index.
        "subs w6, w6, #1\n"  // decrement (reduced) columns left
        "bne " LABEL_COL_LOOP "b\n"

        LABEL_SKIP_COL_LOOP
        ":\n"

        "addp v28.4s, v28.4s, v29.4s\n"  // 1st column
        "addp v23.4s, v23.4s, v24.4s\n"  // 2nd column
        "addp v19.4s, v19.4s, v20.4s\n"  // 3rd column
        "addp v15.4s, v15.4s, v16.4s\n"  // 4th column
        "addp v11.4s, v11.4s, v12.4s\n"  // 5th column

        "addp v30.4s, v30.4s, v31.4s\n"  // 1st column
        "addp v25.4s, v25.4s, v26.4s\n"  // 2nd column
        "addp v21.4s, v21.4s, v22.4s\n"  // 3rd column
        "addp v17.4s, v17.4s, v18.4s\n"  // 4th column
        "addp v13.4s, v13.4s, v14.4s\n"  // 5th column

        "addp v28.4s, v28.4s, v30.4s\n"  // 1st column
        "addp v23.4s, v23.4s, v25.4s\n"  // 2nd column
        "addp v19.4s, v19.4s, v21.4s\n"  // 3rd column
        "addp v15.4s, v15.4s, v17.4s\n"  // 4th column
        "addp v11.4s, v11.4s, v13.4s\n"  // 5th column

        "srshr v28.4s, v28.4s, %[shift_amount]\n"
        "srshr v23.4s, v23.4s, %[shift_amount]\n"
        "srshr v19.4s, v19.4s, %[shift_amount]\n"
        "srshr v15.4s, v15.4s, %[shift_amount]\n"
        "srshr v11.4s, v11.4s, %[shift_amount]\n"

        // Store accumulators.
        "st1 {v28.4s}, [%[out_ptr]], #16\n"
        "st1 {v23.4s}, [%[out2_ptr]], #16\n"
        "st1 {v19.4s}, [%[out3_ptr]], #16\n"
        "st1 {v15.4s}, [%[out4_ptr]], #16\n"
        "st1 {v11.4s}, [%[out5_ptr]], #16\n"

        // Decrement rows remaining.
        "subs %[assigned_rows], %[assigned_rows], #1\n"
        "bne " LABEL_ROW_LOOP "b\n"

        // clang-format off
        :  // outputs
        [out_ptr] "+r"(out_ptr), [out2_ptr] "+r"(out2_ptr),
        [out3_ptr] "+r"(out3_ptr), [out4_ptr] "+r"(out4_ptr),
        [out5_ptr] "+r"(out5_ptr), [weights_ptr] "+r"(weights_ptr),
        [col_deltas_bytes] "+r"(col_deltas_bytes), [bias_ptr] "+r"(bias_ptr),
        [nnz_per_row] "+r"(nnz_per_row), [assigned_rows] "+r"(assigned_rows),
        [rhs_ptr] "+r"(rhs_ptr), [rhs2_ptr] "+r"(rhs2_ptr),
        [rhs3_ptr] "+r"(rhs3_ptr), [rhs4_ptr] "+r"(rhs4_ptr),
        [rhs5_ptr] "+r"(rhs5_ptr)
        :  // inputs
        [shift_amount] "I"(kShiftAmount)
        :  // clobbers
        "cc", "memory", "x6", "x7", "x8", "v0", "v1", "v2", "v3", "v4", "v5",
        "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
        "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
        "v26", "v27", "v28", "v29", "v30", "v31");
    // clang-format on
  }
}

template <typename Type>
typename std::enable_if<IsFixed32Type<Type>::value>::type SumVectors(
    int start, int end, const Type* add1, const Type* add2, Type* result) {
  constexpr int kSIMDWidth = 4;
  for (int i = start; i < end; i += kSIMDWidth) {
    int32x4_t add1_int = vld1q_s32(reinterpret_cast<const int32_t*>(add1 + i));
    int32x4_t add2_int = vld1q_s32(reinterpret_cast<const int32_t*>(add2 + i));
    int32x4_t result_int = vqaddq_s32(add1_int, add2_int);
    vst1q_s32(reinterpret_cast<int32_t*>(result + i), result_int);
  }
}

template <typename Type>
typename std::enable_if<IsFixed16Type<Type>::value>::type SumVectors(
    int start, int end, const Type* add1, const Type* add2, Type* result) {
  constexpr int kSIMDWidth = 8;
  for (int i = start; i < end; i += kSIMDWidth) {
    int16x8_t add1_int = vld1q_s16(reinterpret_cast<const int16_t*>(add1 + i));
    int16x8_t add2_int = vld1q_s16(reinterpret_cast<const int16_t*>(add2 + i));
    int16x8_t result_int = vqaddq_s16(add1_int, add2_int);
    vst1q_s16(reinterpret_cast<int16_t*>(result + i), result_int);
  }
}

}  // namespace detail
}  // namespace csrblocksparse

#undef LABEL_COL_LOOP
#undef LABEL_ROW_LOOP
#undef LABEL_SKIP_COL_LOOP
#undef LABEL_TOP_LOOP

#endif  // defined __aarch64__
#endif  // LYRA_CODEC_SPARSE_MATMUL_COMPUTE_KERNELS_ARM_H_
