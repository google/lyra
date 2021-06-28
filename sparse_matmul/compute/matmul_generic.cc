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

#include "sparse_matmul/compute/matmul_generic.h"

#include <cstdint>
#include <vector>

#include "sparse_matmul/compute/matmul.h"

namespace csrblocksparse {
namespace detail {

void MatVecFloatGeneric(const float* weights, const float* rhs,
                        const float* bias, const int32_t* nnz_per_row,
                        const int16_t* rhs_indices, int start_row, int end_row,
                        int block_height, int block_width, bool relu,
                        int replicas, int stride, float* output) {
  int weight_index = 0;
  int bias_index = 0;
  std::vector<float> accumulators(block_height);
  for (int row_block = start_row; row_block < end_row;
       ++row_block, output += block_height) {
    int nnz = nnz_per_row[row_block];
    // Biases are now stored and used directly without pre-division.
    for (int i = 0; i < block_height; ++i) accumulators[i] = bias[bias_index++];

    for (int c = 0; c < nnz; ++c) {
      int rhs_index = rhs_indices[c];
      const float* block_rhs = rhs + rhs_index * block_width;
      // Multiply this |block_height| x |block_width| block.
      for (int i = 0; i < block_height; ++i) {
        for (int j = 0; j < block_width; ++j) {
          accumulators[i] += weights[weight_index++] * block_rhs[j];
        }
      }
    }
    rhs_indices += nnz;
    // Apply relu if desired.
    if (relu) {
      for (int i = 0; i < block_height; ++i) {
        if (accumulators[i] < 0) accumulators[i] = 0;
      }
    }
    for (int r = 0; r < replicas; ++r) {
      for (int i = 0; i < block_height; ++i) {
        output[i + r * stride] = accumulators[i];
      }
    }
  }
}

void MatVecFixedGeneric(const int16_t* weights, const int16_t* rhs,
                        const int32_t* bias, const int32_t* nnz_per_row,
                        const int16_t* rhs_indices, int start_row, int end_row,
                        int block_height, int block_width, bool relu,
                        int bytes_out, int shift_out, int replicas, int stride,
                        void* output) {
  int weight_index = 0;
  int bias_index = 0;
  std::vector<int32_t> accumulators(block_height);
  for (int row_block = start_row; row_block < end_row; ++row_block) {
    int nnz = nnz_per_row[row_block];
    // Biases are now stored and used directly without pre-division.
    for (int i = 0; i < block_height; ++i) accumulators[i] = bias[bias_index++];

    for (int c = 0; c < nnz; ++c) {
      int rhs_index = rhs_indices[c];
      const int16_t* block_rhs = rhs + rhs_index * block_width;
      // Multiply this |block_height| x |block_width| block.
      for (int i = 0; i < block_height; ++i) {
        for (int j = 0; j < block_width; ++j) {
          accumulators[i] += weights[weight_index++] * block_rhs[j];
        }
      }
    }
    rhs_indices += nnz;
    // Apply relu if desired.
    if (relu) {
      for (int i = 0; i < block_height; ++i) {
        if (accumulators[i] < 0) accumulators[i] = 0;
      }
    }
    // Output shift.
    if (shift_out > 0) {
      for (int i = 0; i < block_height; ++i) {
        accumulators[i] >>= shift_out;
      }
    }
    if (bytes_out == 2) {
      int16_t* out16 = reinterpret_cast<int16_t*>(output);
      output = out16 + block_height;
      for (int r = 0; r < replicas; ++r, out16 += stride) {
        for (int i = 0; i < block_height; ++i) {
          out16[i] = accumulators[i];
        }
      }
    } else {
      int32_t* out32 = reinterpret_cast<int32_t*>(output);
      output = out32 + block_height;
      for (int r = 0; r < replicas; ++r, out32 += stride) {
        for (int i = 0; i < block_height; ++i) {
          out32[i] = accumulators[i];
        }
      }
    }
  }
}

}  // namespace detail
}  // namespace csrblocksparse
