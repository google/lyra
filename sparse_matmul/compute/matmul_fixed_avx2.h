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

#ifndef LYRA_CODEC_SPARSE_MATMUL_COMPUTE_MATMUL_FIXED_AVX2_H_
#define LYRA_CODEC_SPARSE_MATMUL_COMPUTE_MATMUL_FIXED_AVX2_H_

#include <cstdint>

namespace csrblocksparse {
namespace detail {

// Version that covers all possible combinations of the variable conditions:
// |relu|, |shift_out|, |replicas|, with int16 output.
void MatVec4x4FixedAVX2(const int16_t* weights_ptr, const int16_t* rhs,
                        const int32_t* bias, const int32_t* nnz_per_row,
                        const int16_t* rhs_indices, int start_row, int end_row,
                        bool relu, int shift_out, int replicas, int stride,
                        int16_t* output);
// Version that covers all possible combinations of the variable conditions:
// |relu|, |shift_out|, |replicas|, with int32 output.
void MatVec4x4FixedAVX2(const int16_t* weights_ptr, const int16_t* rhs,
                        const int32_t* bias, const int32_t* nnz_per_row,
                        const int16_t* rhs_indices, int start_row, int end_row,
                        bool relu, int shift_out, int replicas, int stride,
                        int32_t* output);
// Version that covers the main conditions used with 8x4:
// |relu|, |shift_out|, with int32 output.
void MatVec8x4FixedAVX2(const int16_t* weights_ptr, const int16_t* rhs,
                        const int32_t* bias, const int32_t* nnz_per_row,
                        const int16_t* rhs_indices, int start_row, int end_row,
                        bool relu, int shift_out, int32_t* output);

}  // namespace detail
}  // namespace csrblocksparse

#endif  // LYRA_CODEC_SPARSE_MATMUL_COMPUTE_MATMUL_FIXED_AVX2_H_
