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

#include "sparse_matmul/compute/thread_bounds.h"

#include <vector>

#include "glog/logging.h"

namespace csrblocksparse {

void ThreadBounds::PrepareForThreads(int block_width, int block_height,
                                     int num_threads,
                                     int reduced_rows_per_cache_row,
                                     int reduced_rows, const int* nnz_per_row) {
  CHECK_GT(num_threads, 0);
  block_width_ = block_width;
  block_height_ = block_height;
  ComputeThreadSplitPoints(num_threads, reduced_rows_per_cache_row,
                           reduced_rows, nnz_per_row);
  weight_starts_.clear();
  rhs_indices_starts_.clear();
  bias_starts_.clear();
  weight_starts_.reserve(row_starts_.size());
  rhs_indices_starts_.reserve(row_starts_.size());
  bias_starts_.reserve(row_starts_.size());

  // Compute the start indices of each of the types, given what we know about
  // padding, and number of |nnz_per_row|.
  int weight_index = 0;
  int rhs_indices_index = 0;
  int bias_index = 0;
  int row = 0;
  for (int start : row_starts_) {
    while (row < start) {
      weight_index += nnz_per_row[row] * block_width_ * block_height_;
      rhs_indices_index += nnz_per_row[row];
      bias_index += block_height_;
      ++row;
    }
    weight_starts_.push_back(weight_index);
    rhs_indices_starts_.push_back(rhs_indices_index);
    bias_starts_.push_back(bias_index);
  }
}

// Computes the block row (reduced) index of the start of each thread.
void ThreadBounds::ComputeThreadSplitPoints(int num_threads,
                                            int reduced_rows_per_cache_row,
                                            int reduced_rows,
                                            const int* nnz_per_row) {
  row_starts_.assign(/*n=*/1, /*val=*/0);
  // Break the rule if the matrix is too small to allow one per thread, which
  // occurs only during tests.
  if (reduced_rows_per_cache_row * num_threads > reduced_rows)
    reduced_rows_per_cache_row = std::max(reduced_rows / num_threads, 1);
  int cache_rows = (reduced_rows + reduced_rows_per_cache_row - 1) /
                   reduced_rows_per_cache_row;

  // Compute exclusive prefix sum of the amount of work per row.
  std::vector<int> work_upto_row(cache_rows + 1, 0);
  int extra_row_work = 2 * reduced_rows_per_cache_row;
  for (int i = 0; i < cache_rows; ++i) {
    int new_nnz = 0;
    for (int j = 0; j < reduced_rows_per_cache_row; ++j) {
      // if |reduced_rows_per_cache_row| isn't an exact multiple of the
      // matrix size, then we need to be careful here.
      int index = i * reduced_rows_per_cache_row + j;
      if (index < reduced_rows) new_nnz += nnz_per_row[index];
    }
    work_upto_row[i + 1] = new_nnz + extra_row_work + work_upto_row[i];
  }
  int total_work = work_upto_row.back();
  // Find the split point point based on assigned approximately equal amount
  // of work for each thread.
  int prev_split = 0;
  for (int i = 1; i <= num_threads; ++i) {
    int split = std::distance(
        work_upto_row.begin(),
        std::lower_bound(work_upto_row.begin(), work_upto_row.end(),
                         i * total_work / num_threads));
    int split_row = split * reduced_rows_per_cache_row;
    if (i == num_threads) {
      split_row = reduced_rows;
    }

    VLOG(2) << "tid=" << i - 1 << " num rows=" << split_row - row_starts_.back()
            << " work=" << work_upto_row[split] - work_upto_row[prev_split];
    row_starts_.push_back(split_row);
    prev_split = split;
  }
  VLOG(2) << "total rows=" << reduced_rows << " total work=" << total_work;
}

}  // namespace csrblocksparse
