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

#ifndef LYRA_CODEC_SPARSE_MATMUL_COMPUTE_MATMUL_GENERIC_H_
#define LYRA_CODEC_SPARSE_MATMUL_COMPUTE_MATMUL_GENERIC_H_

#include <cstdint>

namespace csrblocksparse {
namespace detail {

// Generic version uses plain C++ code.
void MatVecFloatGeneric(const float* weights, const float* rhs,
                        const float* bias, const int32_t* nnz_per_row,
                        const int16_t* rhs_indices, int start_row, int end_row,
                        int block_height, int block_width, bool relu,
                        int replicas, int stride, float* output);
void MatVecFixedGeneric(const int16_t* weights, const int16_t* rhs,
                        const int32_t* bias, const int32_t* nnz_per_row,
                        const int16_t* rhs_indices, int start_row, int end_row,
                        int block_height, int block_width, bool relu,
                        int bytes_out, int shift_out, int replicas, int stride,
                        void* output);

}  // namespace detail
}  // namespace csrblocksparse

#endif  // LYRA_CODEC_SPARSE_MATMUL_COMPUTE_MATMUL_GENERIC_H_
