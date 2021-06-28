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

#ifndef LYRA_CODEC_SPARSE_MATMUL_COMPUTE_THREAD_BOUNDS_H_
#define LYRA_CODEC_SPARSE_MATMUL_COMPUTE_THREAD_BOUNDS_H_

#include <vector>

namespace csrblocksparse {

// Class to compute and store the bounds of each thread used in a computation,
// and to provide corresponding spans of vectors.
class ThreadBounds {
 public:
  ThreadBounds() : block_width_(0), block_height_(0) {}

  void PrepareForThreads(int block_width, int block_height, int num_threads,
                         int reduced_rows_per_cache_row, int reduced_rows,
                         const int* nnz_per_row);

  // Functions that offset the appropriate type to the start of the data
  // needed by the given thread id (|tid|).
  template <typename WeightType>
  const WeightType* OffsetWeights(const WeightType* weights, int tid) const {
    return weights + weight_starts_[tid];
  }
  template <typename RhsIndType>
  const RhsIndType* OffsetRhsIndices(const RhsIndType* rhs_indices,
                                     int tid) const {
    return rhs_indices + rhs_indices_starts_[tid];
  }
  template <typename BiasType>
  const BiasType* OffsetBias(const BiasType* bias, int tid) const {
    return bias + bias_starts_[tid];
  }
  template <typename OutType>
  OutType* OffsetOutput(OutType* output, int tid) const {
    return output + block_height_ * row_starts_[tid];
  }
  int StartRow(int tid) const { return row_starts_[tid]; }
  const std::vector<int>& row_starts() const { return row_starts_; }

 private:
  // Computes the block row (reduced) index of the start of each thread.
  void ComputeThreadSplitPoints(int num_threads, int reduced_rows_per_cache_row,
                                int reduced_rows, const int* nnz_per_row);

  // Sizes of a sparse block.
  int block_width_;
  int block_height_;
  // Start indices of each data type by thread-id with an extra value at the
  // end.
  std::vector<int> row_starts_;
  std::vector<int> weight_starts_;
  std::vector<int> rhs_indices_starts_;
  std::vector<int> bias_starts_;
};

}  // namespace csrblocksparse

#endif  // LYRA_CODEC_SPARSE_MATMUL_COMPUTE_THREAD_BOUNDS_H_
