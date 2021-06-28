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

#ifndef LYRA_CODEC_SPARSE_MATMUL_LAYERS_CSR_BLOCKSPARSE_MATRIX_H_
#define LYRA_CODEC_SPARSE_MATMUL_LAYERS_CSR_BLOCKSPARSE_MATRIX_H_

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <memory>
#include <tuple>
#include <vector>

#include "glog/logging.h"
// IWYU pragma: begin_exports
#include "sparse_matmul/compute/kernels_generic.h"
#include "sparse_matmul/compute/matmul.h"
#include "sparse_matmul/compute/thread_bounds.h"
#include "sparse_matmul/layers/masked_sparse_matrix.h"
#include "sparse_matmul/numerics/fixed_types.h"
#include "sparse_matmul/numerics/float16_types.h"
#include "sparse_matmul/os/coop_threads.h"
#include "sparse_matmul/vector/cache_aligned_vector.h"
// IWYU pragma: end_exports
#include "absl/memory/memory.h"

namespace csrblocksparse {
// CsrBlockSparseMatrix stores a modified block compressed sparse row
// representation of a sparse matrix.  The ordering of the weights is modified
// in the 16x1 and 1x1 cases so that a certain number (4 and 8 respectively)
// of columns of weights are stored contiguously before moving on to the next
// row.  The 4x4 case stores each block contiguously.
//
// Currently it is constructed from a MaskedSparseMatrix which usees a dense
// binary mask representation.  The construction generates the compressed
// representation.  Further iterations will support a direct serialization
// of the compressed representation.
//
// MaskedSparseMatrix masked_matrix(rows, cols, existing_mask, existing_values)
// CsrBlockSparseMatrix matrix(masked_matrix)
//
// matrix.SpMV_bias(rhs, bias, &out);
//
// This class is thread compatible.
template <typename WeightType, typename RhsType, typename DeltaType = int16_t>
class CsrBlockSparseMatrix {
 public:
  CsrBlockSparseMatrix() {}

  // Reference used to indicate that this is an input and not an output.
  CsrBlockSparseMatrix(const uint8_t* const& buffer, const std::size_t& len) {
    ReadFromFlatBuffer(buffer, len);
    ComputeRHSIndices();
  }

  template <typename InputType>
  CsrBlockSparseMatrix(const MaskedSparseMatrix<InputType>& masked_matrix) {
    sparsity_ = masked_matrix.sparsity();
    rows_ = masked_matrix.rows();
    cols_ = masked_matrix.cols();

    DetermineBlockSize(masked_matrix);

    if (block_width_ == 1 && block_height_ == 1)
      col_multiple_ = 8;
    else
      col_multiple_ = 1;

    std::vector<InputType> weights(masked_matrix.values().begin(),
                                   masked_matrix.values().end());

    reduced_rows_ = (rows_ + block_height_ - 1) / block_height_;
    rows_ = reduced_rows_ * block_height_;
    reduced_cols_ = cols_ / block_width_;

    // Calculate the reduced CSR representation of the matrix.
    std::vector<int> reduced_mask(reduced_rows_ * reduced_cols_);
    std::vector<int> row_offsets = {0};
    int nnz = 0;
    const auto& mask = masked_matrix.mask();
    for (int r = 0; r < reduced_rows_; ++r) {
      for (int c = 0; c < reduced_cols_; ++c) {
        int mask_val = mask[r * block_height_ * cols_ + c * block_width_];
        reduced_mask[r * reduced_cols_ + c] = mask_val;
        nnz += mask_val;
      }
      row_offsets.push_back(nnz);
    }

    // Make sure the reduced representation has the correct number of columns.
    MakeColumnsMultiple(row_offsets, &reduced_mask, &weights);

    std::vector<int> col_indices;
    std::vector<WeightType> weights_csr;
    std::vector<int> nnz_per_row;
    MaskAndWeightsToCsr(reduced_mask, weights, &nnz_per_row, &col_indices,
                        &weights_csr);

    // Generate column deltas from |col_indices|.
    std::vector<DeltaType> col_deltas;
    for (int i = 0; i < col_indices.size(); ++i) {
      // |col_indices| are used to index the RHS vector which is always float.
      int64_t diff = sizeof(RhsType);
      if (i == 0)
        diff *= block_width_ * (col_indices[i]);
      else
        diff *= block_width_ * (col_indices[i] - col_indices[i - 1]);

      CHECK(diff < std::numeric_limits<DeltaType>::max())
          << "delta between column indices in bytes " << diff
          << " exceeded the maximum size of the DeltaType "
          << std::numeric_limits<DeltaType>::max();
      col_deltas.push_back(static_cast<DeltaType>(diff));
    }

    // Because of pre-fetching we need some extra values at the end.
    col_deltas.insert(col_deltas.end(), std::max(2, col_multiple_ + 1), 0);
    nnz_per_row.insert(nnz_per_row.end(), 2, nnz_per_row.back());

    weights_ = CacheAlignedVector<WeightType>(weights_csr);
    col_deltas_ = CacheAlignedVector<DeltaType>(col_deltas);
    nnz_per_row_ = CacheAlignedVector<int>(nnz_per_row);
    ComputeRHSIndices();

    num_threads_ = 0;
    PrepareForThreads(1);
  }

  // Constructor makes a matrix from the given weights, deltas and nnz, taking
  // the other parameters from |src_matrix|. |cols| is the number of raw columns
  // (NOT blocks) of the new matrix.
  CsrBlockSparseMatrix(
      const CsrBlockSparseMatrix<WeightType, RhsType, DeltaType>& src_matrix,
      const std::vector<WeightType>& new_weights,
      const std::vector<DeltaType>& new_deltas, const std::vector<int>& new_nnz,
      int cols) {
    num_threads_ = 0;
    col_multiple_ = src_matrix.col_multiple_;
    block_width_ = src_matrix.block_width_;
    block_height_ = src_matrix.block_height_;
    reduced_rows_ = new_nnz.size();
    rows_ = reduced_rows_ * block_height_;
    cols_ = cols;
    reduced_cols_ = cols_ / block_width_;
    weights_ = CacheAlignedVector<WeightType>(new_weights);
    col_deltas_ = CacheAlignedVector<DeltaType>(new_deltas);
    nnz_per_row_ = CacheAlignedVector<int>(new_nnz);
    sparsity_ = 1.0f - static_cast<float>(new_weights.size()) / (rows_ * cols_);
    ComputeRHSIndices();
    name_ = src_matrix.name_;
    PrepareForThreads(1);
  }

  // Factory method takes a column slice out of *this and returns a sparse
  // matrix that takes as inputs [|start_col|, |end_col|) of *this, and
  // returns the same number of outputs, but only a partial result.
  // If |keep_rhs_size|, then the new matrix takes the same rhs as the current
  // matrix, but uses a subset of it, instead of expecting just the reduced rhs.
  // If |start_col| > |end_col|, then we slice out the complement of the defined
  // interval, ie [0, |end_col|) + [|start_col|, current end).
  // NOTE That |start_col| and |end_col| are in raw column coordinates, NOT
  // block units.
  CsrBlockSparseMatrix SplitByColumn(int start_col, int end_col,
                                     bool keep_rhs_size = false) const {
    int weight_index = 0;
    int delta_index = 0;
    std::vector<DeltaType> new_deltas;
    std::vector<WeightType> new_weights;
    std::vector<int> new_nnz(reduced_rows_);
    int col = 0;
    int prev_col = keep_rhs_size ? 0 : start_col;
    for (int r = 0; r < reduced_rows_; ++r) {
      int reduced_col_count = nnz_per_row_[r];
      for (int c = 0; c < reduced_col_count; ++c, ++delta_index) {
        col += col_deltas_[delta_index] / sizeof(RhsType);
        if ((start_col < end_col && start_col <= col && col < end_col) ||
            (start_col > end_col && (col < end_col || col >= start_col))) {
          ++new_nnz[r];
          new_deltas.push_back((col - prev_col) * sizeof(RhsType));
          prev_col = col;
          for (int i = 0; i < block_width_ * block_height_;
               ++i, ++weight_index) {
            new_weights.push_back(weights_[weight_index]);
          }
        } else {
          weight_index += block_width_ * block_height_;
        }
      }
    }
    int new_cols = keep_rhs_size ? cols_ : end_col - start_col;
    return CsrBlockSparseMatrix(*this, new_weights, new_deltas, new_nnz,
                                new_cols);
  }

  // Factory method takes a row slice out of *this and returns a sparse
  // matrix that takes the sampe inputs as *this, and returns the outputs for
  // the range [|start_row|, |end_row|).
  // NOTE That |start_row| and |end_row| are in raw column coordinates, NOT
  // block units.
  CsrBlockSparseMatrix SplitByRow(int start_row, int end_row) const {
    int start_reduced = start_row / block_height_;
    int end_reduced = end_row / block_height_;
    std::vector<int> new_nnz(nnz_per_row_.data() + start_reduced,
                             nnz_per_row_.data() + end_reduced);
    int weight_start = 0;
    for (int r = 0; r < start_reduced; ++r) {
      weight_start += nnz_per_row_[r];
    }
    int weight_end = weight_start;
    for (int r = start_reduced; r < end_reduced; ++r) {
      weight_end += nnz_per_row_[r];
    }
    int delta_start = 0;
    for (int i = 0; i < weight_start; ++i) {
      delta_start += col_deltas_[i];
    }
    std::vector<DeltaType> new_deltas(col_deltas_.data() + weight_start,
                                      col_deltas_.data() + weight_end);
    new_deltas[0] += delta_start;
    int block_size = block_height_ * block_width_;
    std::vector<WeightType> new_weights(
        weights_.data() + weight_start * block_size,
        weights_.data() + weight_end * block_size);
    return CsrBlockSparseMatrix(*this, new_weights, new_deltas, new_nnz, cols_);
  }

  // Combines adjacent row blocks, doubling the block height.
  // This necessarily involves adding zero weights where the blocks don't align
  // across adjacent pairs of rows, so use with caution, as the resulting matrix
  // is most likely to run slower if very sparse to begin with.
  // In the few cases where the blocks do mostly align, the resulting matmul
  // could be much faster, as the number of reads of the rhs will be halved.
  void DoubleBlockHeight() {
    int new_rows = reduced_rows_ / 2;
    std::vector<int> new_nnz(new_rows);
    std::vector<DeltaType> new_rhs_indices;
    std::vector<WeightType> new_weights;
    int rhs_index1 = 0;
    int rhs_index2 = 0;
    int block_size = block_height_ * block_width_;
    for (int r = 0; r < new_rows; ++r) {
      int start_nnz = new_rhs_indices.size();
      rhs_index2 += nnz_per_row_[r * 2];
      int end1 = rhs_index1 + nnz_per_row_[r * 2];
      int end2 = rhs_index2 + nnz_per_row_[r * 2 + 1];
      // Run over a pair of rows with 2 iterators, combining blocks as we go, or
      // padding with zeros where the block positions don't match.
      while (rhs_index1 < end1 || rhs_index2 < end2) {
        int col1 = rhs_index1 < end1 ? rhs_indices_[rhs_index1] : reduced_cols_;
        int col2 = rhs_index2 < end2 ? rhs_indices_[rhs_index2] : reduced_cols_;
        if (col1 < col2) {
          // Need zero weights for row2 to pad out weights block.
          new_rhs_indices.push_back(col1);
          new_weights.insert(new_weights.end(),
                             weights_.data() + rhs_index1 * block_size,
                             weights_.data() + (rhs_index1 + 1) * block_size);
          new_weights.insert(new_weights.end(), block_size,
                             static_cast<WeightType>(0.0f));
          ++rhs_index1;
        } else if (col1 > col2) {
          // Need zero weights for row1 to pad out weights block.
          new_rhs_indices.push_back(col2);
          new_weights.insert(new_weights.end(), block_size,
                             static_cast<WeightType>(0.0f));
          new_weights.insert(new_weights.end(),
                             weights_.data() + rhs_index2 * block_size,
                             weights_.data() + (rhs_index2 + 1) * block_size);
          ++rhs_index2;
        } else {
          // Combine weights for both row1 and row2.
          new_rhs_indices.push_back(col1);
          new_weights.insert(new_weights.end(),
                             weights_.data() + rhs_index1 * block_size,
                             weights_.data() + (rhs_index1 + 1) * block_size);
          new_weights.insert(new_weights.end(),
                             weights_.data() + rhs_index2 * block_size,
                             weights_.data() + (rhs_index2 + 1) * block_size);
          ++rhs_index1;
          ++rhs_index2;
        }
      }
      rhs_index1 = rhs_index2;
      new_nnz[r] = new_rhs_indices.size() - start_nnz;
    }
    block_height_ *= 2;
    reduced_rows_ /= 2;
    weights_ = CacheAlignedVector<WeightType>(new_weights);
    rhs_indices_ = CacheAlignedVector<DeltaType>(new_rhs_indices);
    nnz_per_row_ = CacheAlignedVector<int>(new_nnz);
    sparsity_ = 1.0f - static_cast<float>(new_weights.size()) / (rows_ * cols_);
    ComputeColDeltas();
    if (num_threads_ > 0) {
      int num_threads = num_threads_;
      num_threads_ = 0;
      PrepareForThreads(num_threads);
    }
  }

  // Allocates memory and fills buffer.
  // Caller is responsible for the memory de-allocation.
  // TODO(b/189958858): Both Read and Write need to eventually handle the
  // different possible HalfType and DeltaType values, but punting for now as
  // there is only one supported combination.
  std::size_t WriteToFlatBuffer(std::string* csr_flatbuffer) {
    std::size_t bytes = 0;
    bytes += FixedParameterSize();
    bytes += weights_.size() * sizeof(WeightType);
    bytes += col_deltas_.size() * sizeof(DeltaType);
    bytes += nnz_per_row_.size() * sizeof(int);

    uint8_t* bytes_ptr_ptr =
        reinterpret_cast<uint8_t*>(CHECK_NOTNULL(malloc(bytes)));

    int* int_bytes_ptr = reinterpret_cast<int*>(bytes_ptr_ptr);

    *int_bytes_ptr++ = rows_;
    *int_bytes_ptr++ = cols_;
    *int_bytes_ptr++ = reduced_rows_;
    *int_bytes_ptr++ = reduced_cols_;
    *int_bytes_ptr++ = block_width_;
    *int_bytes_ptr++ = block_height_;
    *int_bytes_ptr++ = col_multiple_;
    *int_bytes_ptr++ = num_threads_;
    *int_bytes_ptr++ = weights_.size();
    *int_bytes_ptr++ = col_deltas_.size();
    *int_bytes_ptr++ = nnz_per_row_.size();

    float* float_bytes_ptr = reinterpret_cast<float*>(int_bytes_ptr);
    *float_bytes_ptr++ = sparsity_;

    uint8_t* bytes_ptr = reinterpret_cast<uint8_t*>(float_bytes_ptr);

    memcpy(bytes_ptr, weights_.data(), weights_.size() * sizeof(WeightType));
    bytes_ptr += weights_.size() * sizeof(WeightType);

    memcpy(bytes_ptr, col_deltas_.data(),
           col_deltas_.size() * sizeof(DeltaType));
    bytes_ptr += col_deltas_.size() * sizeof(DeltaType);

    memcpy(bytes_ptr, nnz_per_row_.data(), nnz_per_row_.size() * sizeof(int));
    bytes_ptr += nnz_per_row_.size() * sizeof(int);

    csr_flatbuffer->resize(bytes);
    csr_flatbuffer->assign(reinterpret_cast<char*>(bytes_ptr_ptr), bytes);
    free(bytes_ptr_ptr);

    return bytes;
  }

  void ReadFromFlatBuffer(const uint8_t* const& bytes, const std::size_t& len) {
    CHECK_GE(len, FixedParameterSize());

    const int* int_bytes_ptr = reinterpret_cast<const int*>(bytes);
    rows_ = *int_bytes_ptr++;
    cols_ = *int_bytes_ptr++;
    reduced_rows_ = *int_bytes_ptr++;
    reduced_cols_ = *int_bytes_ptr++;
    block_width_ = *int_bytes_ptr++;
    block_height_ = *int_bytes_ptr++;
    col_multiple_ = *int_bytes_ptr++;
    int num_threads = *int_bytes_ptr++;
    int32_t weights_size = *int_bytes_ptr++;
    int32_t col_deltas_size = *int_bytes_ptr++;
    int32_t nnz_per_row_size = *int_bytes_ptr++;

    // Make sure negative sizes don't mess things up.
    weights_size = std::max(0, weights_size);
    col_deltas_size = std::max(0, col_deltas_size);
    nnz_per_row_size = std::max(0, nnz_per_row_size);

    const float* float_bytes_ptr =
        reinterpret_cast<const float*>(int_bytes_ptr);
    sparsity_ = *float_bytes_ptr++;

    std::size_t total_bytes =
        FixedParameterSize() + weights_size * sizeof(WeightType) +
        col_deltas_size * sizeof(DeltaType) + nnz_per_row_size * sizeof(int);

    CHECK_EQ(total_bytes, len)
        << "total bytes: " << total_bytes << ", actual len given: " << len;

    const uint8_t* bytes_ptr =
        reinterpret_cast<const uint8_t*>(float_bytes_ptr);
    std::vector<WeightType> weights_raw(weights_size);
    memcpy(weights_raw.data(), bytes_ptr, weights_size * sizeof(WeightType));
    weights_ = CacheAlignedVector<WeightType>(weights_raw);
    bytes_ptr += weights_size * sizeof(WeightType);

    std::vector<DeltaType> deltas_raw(col_deltas_size);
    memcpy(deltas_raw.data(), bytes_ptr, col_deltas_size * sizeof(DeltaType));
    col_deltas_ = CacheAlignedVector<DeltaType>(deltas_raw);
    bytes_ptr += col_deltas_size * sizeof(DeltaType);

    std::vector<int> nnz_raw(nnz_per_row_size);
    memcpy(nnz_raw.data(), bytes_ptr, nnz_per_row_size * sizeof(int));
    nnz_per_row_ = CacheAlignedVector<int>(nnz_raw);
    num_threads_ = 0;
    PrepareForThreads(num_threads);
  }

  // Multiply a Sparse matrix by a possibly dense matrix.  Often the matrix is
  // a vector with a small number of columns, hence the term "fat vector".
  // 1x1 and 4x4 have specializations for output columns (ie fatness) > 5,
  // and often achieve twice as many GFlops when multiplying a right hand side
  // that has 5 or more columns.  (Best is a multiple of 5).
  // 16x1 doesn't have enough registers and just loops over the width 1 kernel.
  //
  // |rhs| and |out| are COLUMN MAJOR.

  // Fast Tuples WeightType, BiasType, RhsType, OutType are:
  // (float, float, float, float)
  // (bfloat16, float, float, float)
  // and only on ARM64.  All other cases use a slow generic implementation.
  template <typename RhsClass, typename BiasClass, typename OutClass,
            typename BiasType = typename BiasClass::value_type,
            typename OutType = typename OutClass::value_type>
  void SpMM_bias(const RhsClass& rhs, const BiasClass& bias, OutClass* out,
                 bool relu = false, int tid = 0,
                 SpinBarrier* barrier = nullptr) const {
    static_assert(std::is_same<typename RhsClass::value_type, RhsType>::value,
                  "Rhs types must match");
    CHECK_LT(tid, num_threads_);
    CHECK_EQ(rhs.cols(), out->cols());
    CHECK_EQ(rhs.rows(), cols_);
    CHECK_GE(out->rows(), rows_);
    int cols_to_go = out->cols();
    int rhs_index = *thread_bounds_.OffsetRhsIndices(rhs_indices_.data(), tid);
    const RhsType* rhs_ptr = rhs.data() + rhs_index * block_height_;
    OutType* out_ptr = thread_bounds_.OffsetOutput(out->data(), tid);
    const WeightType* weights_ptr =
        thread_bounds_.OffsetWeights(weights_.data(), tid);
    const DeltaType* delta_ptr =
        thread_bounds_.OffsetRhsIndices(col_deltas_.data(), tid);
    int offset = *delta_ptr / sizeof(RhsType);
    rhs_ptr -= offset;
    const int* nnz_ptr = nnz_per_row_.data() + thread_bounds_.StartRow(tid);
    int assigned_rows =
        thread_bounds_.StartRow(tid + 1) - thread_bounds_.StartRow(tid);
    const BiasType* bias_ptr = thread_bounds_.OffsetBias(bias.data(), tid);

    while (cols_to_go > 0) {
      if (block_width_ == 4 && block_height_ == 4) {
        if (cols_to_go >= 5) {
          detail::SpMM5_4x4<WeightType, RhsType, OutType>(
              weights_ptr, delta_ptr, nnz_ptr, rhs_ptr, bias_ptr, out_ptr,
              assigned_rows, out->col_stride(), rhs.col_stride(), relu);
        } else {
          detail::SpMV_4x4<WeightType, RhsType, OutType>(
              weights_ptr, delta_ptr, nnz_ptr, rhs_ptr, bias_ptr, out_ptr,
              assigned_rows, out->col_stride(), rhs.col_stride(), relu);
        }
      } else {
        if (cols_to_go >= 5) {
          detail::SpMM5_1x1<WeightType, RhsType, OutType>(
              weights_ptr, delta_ptr, nnz_ptr, rhs_ptr, bias_ptr, out_ptr,
              assigned_rows, out->col_stride(), rhs.col_stride(), relu);
        } else {
          detail::SpMV_1x1<WeightType, RhsType, OutType>(
              weights_ptr, delta_ptr, nnz_ptr, rhs_ptr, bias_ptr, out_ptr,
              assigned_rows, out->col_stride(), rhs.col_stride(), relu);
        }
      }

      if (cols_to_go >= 5) {
        cols_to_go -= 5;
        rhs_ptr += rhs.col_stride() * 5;
        out_ptr += out->col_stride() * 5;
      } else {
        cols_to_go--;
        rhs_ptr += rhs.col_stride();
        out_ptr += out->col_stride();
      }
      if (barrier) barrier->barrier();
    }
  }
  template <typename MVRhsType, typename MVBiasType, typename OutType>
  void MatVec(const MVRhsType* rhs, const MVBiasType* bias, bool relu, int tid,
              int replicas, int output_stride, OutType* output) {
    CHECK_LT(tid, num_threads_);
    CHECK_EQ(block_width_, 4) << "Block width must be 4!";
    if (block_height_ == 8) {
      matmul_.MatVec8x4(
          thread_bounds_.OffsetWeights(weights_.cast_data(), tid), rhs,
          thread_bounds_.OffsetBias(bias, tid), nnz_per_row_.data(),
          thread_bounds_.OffsetRhsIndices(rhs_indices_.data(), tid),
          thread_bounds_.StartRow(tid), thread_bounds_.StartRow(tid + 1), relu,
          replicas, output_stride, thread_bounds_.OffsetOutput(output, tid));
    } else {
      CHECK_EQ(block_height_, 4) << "Block height must be 4 or 8!";
      matmul_.MatVec4x4(
          thread_bounds_.OffsetWeights(weights_.cast_data(), tid), rhs,
          thread_bounds_.OffsetBias(bias, tid), nnz_per_row_.data(),
          thread_bounds_.OffsetRhsIndices(rhs_indices_.data(), tid),
          thread_bounds_.StartRow(tid), thread_bounds_.StartRow(tid + 1), relu,
          replicas, output_stride, thread_bounds_.OffsetOutput(output, tid));
    }
  }

  int rows() const { return rows_; }
  int cols() const { return cols_; }
  int block_height() const { return block_height_; }
  int block_width() const { return block_width_; }
  float sparsity() const { return sparsity_; }
  int num_threads() const { return num_threads_; }
  const ThreadBounds& thread_bounds() const { return thread_bounds_; }
  const CacheAlignedVector<DeltaType>& rhs_indices() const {
    return rhs_indices_;
  }
  const std::string& name() const { return name_; }
  void set_name(const std::string& name) { name_ = name; }
  const std::vector<int>& split_points() const {
    return thread_bounds_.row_starts();
  }

  std::size_t bytes() const {
    return weights_.size() * sizeof(WeightType) +
           col_deltas_.size() * sizeof(DeltaType) +
           nnz_per_row_.size() * sizeof(int);
  }

  // Multiplies a sparse matrix by a possibly dense matrix, as SpMM_bias above,
  // and then samples from the output (softmax distribution) layer.
  template <typename RhsClass, typename BiasClass, typename OutClass,
            typename BiasType = typename BiasClass::value_type,
            typename OutType = typename OutClass::value_type>
  typename std::enable_if<!IsFixed32Type<OutType>::value, int>::type
  SpMM_bias_Sample(const RhsClass& rhs, const BiasClass& bias, OutClass* out,
                   float temperature, int tid, SpinBarrier* barrier,
                   std::minstd_rand* gen,
                   CacheAlignedVector<float>* scratch) const {
    SpMM_bias(rhs, bias, out, /*relu=*/false, tid, barrier);
    return out->Sample(temperature, gen, scratch);
  }
  // Fixed32 version.
  template <typename RhsClass, typename BiasClass, typename OutClass,
            typename BiasType = typename BiasClass::value_type,
            typename OutType = typename OutClass::value_type>
  typename std::enable_if<IsFixed32Type<OutType>::value, int>::type
  SpMM_bias_Sample(const RhsClass& rhs, const BiasClass& bias, OutClass* out,
                   float temperature, int tid, SpinBarrier* barrier,
                   std::minstd_rand* gen,
                   CacheAlignedVector<float>* scratch) const {
    // We don't pass the barrier on, as we have more work to do.
    SpMM_bias(rhs, bias, out, /*relu=*/false, tid);
    return out->ReducingSample(gen, scratch, tid, temperature, barrier);
  }

  void Print() const {
    std::cout << "Weights\n";
    weights_.Print();
    std::cout << std::endl;
    std::cout << "Deltas\n";
    col_deltas_.Print();
    std::cout << std::endl;
    std::cout << "nnz\n";
    nnz_per_row_.Print();
    std::cout << std::endl;
  }

  // Split the computation amongst threads by rows based on the number of
  // non zeros, with the addition of a constant to account for the work of the
  // bias and the horizontal add at the end, and also guarantees that each
  // thread writes only whole cache lines, based on the size of OutType.
  // The |cache_line_size| arg is used only for testing. Normally it is provided
  // through the architecture #defines.
  // Each thread gets a contiguous row range (|split_points|).
  // Thread t does rows [ split_points[t], split_points[t + 1] )
  // Each thread also needs to know how many non zeros were before it to skip
  // (|nnz_to_skip|).  And finally it also needs to know what the offset into
  // the rhs vector would have been at the split point (|rhs_to_skip|).
  //
  // Some tricky corner cases where the number of non-zeros doesn't split
  // nicely amongst the number of requested threads are not handled and default
  // to one thread; these cases are only going to happen in tests and not in
  // the matrices that correspond in real models.
  //
  // Returns the maximum number of threads that can be used; <= |num_threads|.
  template <typename OutType = int32_t>
  int PrepareForThreads(int num_threads, int cache_line_size = -1) {
    CHECK_GT(num_threads, 0);
    // we've already prepared for this number of threads, nothing to do
    if (num_threads == num_threads_) return num_threads_;

    num_threads_ = num_threads;
    thread_bounds_.PrepareForThreads(
        block_width_, block_height_, num_threads_,
        ReducedRowsPerCacheLine<OutType>(cache_line_size), reduced_rows_,
        nnz_per_row_.data());
    return num_threads_;
  }

  // Computes and stores the |rhs_indices_| from the |col_deltas_|.
  void ComputeRHSIndices() {
    std::vector<int> cumulative_deltas = CumulativeColDeltas();
    std::vector<DeltaType> rhs_indices(cumulative_deltas.size() +
                                       reduced_rows_);
    int total_indices = 0;
    int delta_index = 0;
    for (int r = 0; r < reduced_rows_; ++r) {
      for (int n = 0; n < nnz_per_row_[r]; ++n, ++delta_index) {
        rhs_indices[total_indices++] =
            cumulative_deltas[delta_index] / block_width_;
      }
    }
    rhs_indices_ = CacheAlignedVector<DeltaType>(rhs_indices);
  }

  // Computes and stores the |col_deltas_| from the |rhs_indices_|.
  void ComputeColDeltas() {
    std::vector<int> col_deltas(rhs_indices_.size());
    int prev_index = 0;
    for (int i = 0; i < rhs_indices_.size(); ++i) {
      int offset = rhs_indices_[i] - prev_index;
      prev_index = rhs_indices_[i];
      col_deltas[i] = offset * block_width_ * sizeof(RhsType);
    }
    col_deltas_ = CacheAlignedVector<DeltaType>(col_deltas);
  }

  // Computes and returns the inclusive prefix sum of the deltas, ie absolute
  // positions.
  std::vector<int> CumulativeColDeltas() const {
    std::vector<int> cum_col_deltas(col_deltas_.size());
    for (int i = 0; i < col_deltas_.size(); ++i) {
      cum_col_deltas[i] = col_deltas_[i] / sizeof(RhsType);
      if (i > 0) cum_col_deltas[i] += cum_col_deltas[i - 1];
    }
    return cum_col_deltas;
  }

 private:
  constexpr std::size_t FixedParameterSize() const {
    return sizeof(int)      // rows
           + sizeof(int)    // cols
           + sizeof(int)    // reduced_rows
           + sizeof(int)    // reduced_cols
           + sizeof(int)    // block_width
           + sizeof(int)    // block_height
           + sizeof(float)  // sparsity
           + sizeof(int)    // col_multiple
           + sizeof(int)    // num_threads_
           + sizeof(int)    // weights_.size()
           + sizeof(int)    // col_deltas_.size()
           + sizeof(int);   // nnz_per_row_.size()
  }
  // Possible block sizes are only those that are supported by the computation
  // default is 1x1, other options are 4x4 and 16x1.
  template <typename InputType>
  void DetermineBlockSize(const MaskedSparseMatrix<InputType>& masked_matrix) {
    const std::vector<std::pair<int, int>> kPreferredOrder = {{4, 4}};
    int rows = masked_matrix.rows();
    int cols = masked_matrix.cols();

    for (const auto& block_size : kPreferredOrder) {
      int block_height, block_width;
      std::tie(block_height, block_width) = block_size;
      if (cols % block_width != 0) continue;

      int reduced_rows = (rows + block_height - 1) / block_height;
      int reduced_cols = cols / block_width;

      // For each possible block, confirm that it is either all 0s or all 1s.
      bool all_same = true;
      const auto& mask = masked_matrix.mask();
      for (int r = 0; r < reduced_rows; ++r) {
        for (int c = 0; c < reduced_cols; ++c) {
          int val = mask[r * block_height * cols + c * block_width];
          for (int i = 0; i < block_height; ++i) {
            for (int j = 0; j < block_width; ++j) {
              int index = (r * block_height + i) * cols + c * block_width + j;
              if (index < masked_matrix.mask().size()) {
                all_same &= (masked_matrix.mask()[index] == val);
              }
            }
          }
        }
      }

      // If this block configuration is possible, accept it.
      if (all_same) {
        block_height_ = block_height;
        block_width_ = block_width;
        return;
      }
    }

    // No large blocks were found, default to 1x1.
    block_height_ = 1;
    block_width_ = 1;
  }

  // CSR descriptors are for the reduced matrix, weights is the full matrix.
  template <typename InputType>
  void MakeColumnsMultiple(const std::vector<int>& row_offsets,
                           std::vector<int>* reduced_mask,
                           std::vector<InputType>* weights) {
    if (col_multiple_ > 0) {
      // Make sure each row has a number of columns that is a multiple of
      // |col_multiple|.
      for (int r = 1; r < row_offsets.size(); ++r) {
        int num_row = row_offsets[r] - row_offsets[r - 1];
        int num_needed = col_multiple_ - num_row % col_multiple_;
        if (num_needed < col_multiple_) {
          // Find gaps in the columns where we can insert a column of 0 weights.
          int num_added = 0;
          for (int c = 0; c < reduced_cols_; ++c) {
            if ((*reduced_mask)[(r - 1) * reduced_cols_ + c] == 0) {
              (*reduced_mask)[(r - 1) * reduced_cols_ + c] = 1;

              // Zero out the weights that correspond to this block.
              for (int i = 0; i < block_height_; ++i) {
                for (int j = 0; j < block_width_; ++j) {
                  (*weights)[((r - 1) * block_height_ + i) * cols_ +
                             block_width_ * c + j] = InputType(0.f);
                }
              }
              num_added++;
            }

            if (num_added == num_needed) break;
          }
        }
      }
    }
  }

  // Given the final dense mask and weights, convert to the compressed
  // block CSR representation.
  template <typename InputType>
  void MaskAndWeightsToCsr(const std::vector<int>& mask,
                           const std::vector<InputType>& weights,
                           std::vector<int>* nnz_per_row,
                           std::vector<int>* col_indices,
                           std::vector<WeightType>* weights_csr) {
    std::vector<int> row_offsets = {0};
    int nnz = 0;
    // Standard CSR format.
    if (block_width_ == 1 && block_height_ == 1) {
      for (int r = 0; r < rows_; ++r) {
        for (int c = 0; c < cols_; ++c) {
          if (mask[r * cols_ + c] == 1) {
            nnz++;
            col_indices->push_back(c);
            weights_csr->push_back(WeightType(weights[r * cols_ + c]));
          }
        }
        row_offsets.push_back(nnz);
      }
    } else if (block_width_ == 4 && block_height_ == 4) {
      // Weights are stored contiguously for each block in this case.
      for (int r = 0; r < reduced_rows_; ++r) {
        for (int c = 0; c < reduced_cols_; ++c) {
          if (mask[r * reduced_cols_ + c] == 1) {
            col_indices->push_back(c);
            nnz++;
            for (int i = 0; i < block_height_; ++i) {
              for (int j = 0; j < block_width_; ++j) {
                int row_index = (block_height_ * r + i) * cols_;
                int w_index = row_index + block_width_ * c + j;
                WeightType weight = w_index < weights.size()
                                        ? WeightType(weights[w_index])
                                        : WeightType(0.0f);
                weights_csr->push_back(weight);
              }
            }
          }
        }
        row_offsets.push_back(nnz);
      }
    }
    for (int i = 1; i < row_offsets.size(); ++i)
      nnz_per_row->push_back(row_offsets[i] - row_offsets[i - 1]);
  }

  // Returns the number of block rows per cache line. This is the minimum unit
  // into which the calculation is broken for threads.
  template <typename OutType>
  int ReducedRowsPerCacheLine(int override_cache_line_size = -1) const {
    int line_size = kCacheLineSize;
    if (override_cache_line_size >= 1) line_size = override_cache_line_size;
    return std::max<int>(line_size / (block_height_ * sizeof(OutType)), 1);
  }

  int col_multiple_;
  int rows_;
  int cols_;
  int reduced_rows_;
  int reduced_cols_;
  float sparsity_;
  int block_width_;
  int block_height_;
  int num_threads_;
  std::string name_;

  CacheAlignedVector<WeightType> weights_;
  CacheAlignedVector<DeltaType> col_deltas_;
  CacheAlignedVector<int> nnz_per_row_;
  // |thread_bounds_| and |rhs_indices_| don't need to be serialized as they are
  // always recalculated from serialized data.
  CacheAlignedVector<DeltaType> rhs_indices_;
  Matmul<WeightType, RhsType> matmul_;
  ThreadBounds thread_bounds_;
  static constexpr int kCacheLineSize = 64;
};

// Converts a sparse matrix represented with (|mask|, |weights|, |size|) into
// the CSR format, and returns that as a serialized string.
template <typename MaskType>
std::string ConvertDenseToSparseRepresentation_Int16Deltas(
    const std::vector<MaskType>& mask, const std::vector<float>& weights,
    const int rows, const int cols) {
  MaskedSparseMatrix<float> masked_weights(rows, cols, mask.data(),
                                           weights.data());
  CsrBlockSparseMatrix<csrblocksparse::bfloat16, float, int16_t>
      sparse_masked_weights(masked_weights);
  std::string buffer;
  sparse_masked_weights.WriteToFlatBuffer(&buffer);
  return buffer;
}

}  // namespace csrblocksparse
#endif  // LYRA_CODEC_SPARSE_MATMUL_LAYERS_CSR_BLOCKSPARSE_MATRIX_H_
