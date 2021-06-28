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

#ifndef LYRA_CODEC_SPARSE_MATMUL_LAYERS_MASKED_SPARSE_MATRIX_H_
#define LYRA_CODEC_SPARSE_MATMUL_LAYERS_MASKED_SPARSE_MATRIX_H_

#include <algorithm>
#include <cstdio>
#include <numeric>
#include <vector>

#include "absl/strings/str_format.h"
#include "sparse_matmul/vector/cache_aligned_vector.h"

namespace csrblocksparse {

// MaskedSparseMatrix serves two purposes:
// 1) It is useful as a reference implementation of SpMV for correctness
//    checking the much more complicated implementations in CSRBlockSparseMatrix
// 2) This is the format that sparse matrices are represented after pruning
//    in TF.  This class provides a bridge to getting these parameters into
//    a compressed form suitable for computation and serialization.
//
//  MaskedSparseMatrix<float> matrix(rows, cols, mask_from_tf, values_from_tf);
//  CSRBlockSparseMatrix<float, bfloat16, int16_t> csr_matrix(matrix);
//  csr_matrix.Multiply(rhs, bias, &out);
template <typename T>
class MaskedSparseMatrix {
 public:
  MaskedSparseMatrix() {}

  // Construct a MaskedSparseMatrix of the given size, sparsity and block size.
  // This is mainly useful for testing.
  MaskedSparseMatrix(int rows, int cols, float sparsity, int block_height = 1,
                     int block_width = 1, float constant = 1.f,
                     bool random = true)
      : rows_(rows), cols_(cols), sparsity_(sparsity) {
    CHECK_EQ(rows % block_height, 0);
    CHECK_EQ(cols % block_width, 0);

    init(sparsity, block_height, block_width, constant, random);
  }

  // Construct from an existing mask and values (most likely from a TF model).
  template <typename MaskType>
  MaskedSparseMatrix(int rows, int cols, const MaskType* mask, const T* values)
      : rows_(rows), cols_(cols) {
    mask_.resize(rows * cols);
    values_.resize(rows * cols);
    std::copy_n(mask, rows * cols, mask_.begin());
    std::copy_n(values, rows * cols, values_.begin());
    sparsity_ =
        1.f - std::accumulate(mask_.begin(), mask_.end(), 0.f) / mask_.size();
  }

  const std::vector<int>& mask() const { return mask_; }
  const std::vector<T>& values() const { return values_; }
  T* data() { return values_.data(); }
  const T* data() const { return values_.data(); }

  int rows() const { return rows_; }
  int cols() const { return cols_; }
  float sparsity() const { return sparsity_; }

  void Print() const {
    absl::PrintF("-------Values---------\n");
    for (int r = 0; r < rows_; ++r) {
      for (int c = 0; c < cols_; ++c) {
        absl::PrintF("%+6.3f ", static_cast<float>(values_[r * cols_ + c]));
      }
      absl::PrintF("\n");
    }
    absl::PrintF("-------Mask---------\n");
    for (int r = 0; r < rows_; ++r) {
      for (int c = 0; c < cols_; ++c) {
        printf("%2d ", mask_[r * cols_ + c]);
      }
      absl::PrintF("\n");
    }
  }

  // This routine is useful for rounding the possibly higher precision values
  // stored in this class to a lower precision, so that correctness checks
  // between this class and CSRBlockSparseMatrix can have a tighter tolerance.
  template <typename U>
  void CastWeights() {
    for (int i = 0; i < values_.size(); ++i) {
      values_[i] = static_cast<T>(U(values_[i]));
    }
  }

  // Only meant for correctness checking.
  // RhsClassType is meant to be either CacheAlignedVector OR
  // FatCacheAlignedVector.
  // The weight matrix is ROW MAJOR and RhsClassType is COLUMN MAJOR.
  // |bias| is broadcast if |rhs| has more than one column.
  template <typename RhsClassType, typename BiasType, typename OutClassType,
            typename RhsType = typename RhsClassType::value_type,
            typename OutType = typename OutClassType::value_type>
  void SpMM_bias(const RhsClassType& rhs,
                 const CacheAlignedVector<BiasType>& bias, OutClassType* out,
                 bool relu = false) {
    for (int r = 0; r < rows_; ++r) {
      for (int n = 0; n < rhs.cols(); ++n) {
        float sum = 0.f;
        const RhsType* rhs_ptr = rhs.data() + n * rhs.rows();
        OutType* out_ptr = out->data() + n * out->rows();
        const int* mask_ptr = mask_.data() + r * cols_;
        const T* value_ptr = values_.data() + r * cols_;
        for (int c = 0; c < cols_; ++c) {
          sum += mask_ptr[c] * static_cast<float>(value_ptr[c]) *
                 static_cast<float>(rhs_ptr[c]);
        }
        out_ptr[r] = static_cast<OutType>(
            relu ? std::max(sum + static_cast<float>(bias[r]), 0.f)
                 : sum + static_cast<float>(bias[r]));
      }
    }
  }

 private:
  // Generate a random matrix with the specified sparsity.
  // Useful for testing.
  void init(float sparsity, int block_height, int block_width, float constant,
            bool random = true) {
    int reduced_rows = rows_ / block_height;
    int reduced_cols = cols_ / block_width;
    mask_.resize(rows_ * cols_, 0);

    // Fill with non-zero value to make sure masking works.
    values_.resize(rows_ * cols_, static_cast<T>(2.f));

    std::mt19937 generator(0);
    std::uniform_real_distribution<float> dist_sparsity;
    std::uniform_real_distribution<float> dist_value(-1.f, 1.f);
    int nnz = 0;
    while (nnz == 0) {
      for (int r = 0; r < reduced_rows; ++r) {
        for (int c = 0; c < reduced_cols; ++c) {
          if (dist_sparsity(generator) > sparsity) {
            nnz++;
            for (int i = 0; i < block_height; ++i) {
              for (int j = 0; j < block_width; ++j) {
                mask_[(r * block_height + i) * cols_ + block_width * c + j] = 1;
                values_[(r * block_height + i) * cols_ + block_width * c + j] =
                    static_cast<T>(random ? dist_value(generator) : constant);
              }
            }
          }
        }
      }
    }
  }

  std::vector<int> mask_;
  std::vector<T> values_;
  int rows_;
  int cols_;
  float sparsity_;
};

template <typename T>
class MaskedLinearLayer {
 public:
  MaskedLinearLayer(MaskedSparseMatrix<T>&& weights,
                    CacheAlignedVector<T>&& bias)
      : weights_(std::move(weights)), bias_(std::move(bias)) {}

  MaskedLinearLayer() {}

  template <typename U>
  void CastWeights() {
    weights_.template CastWeights<U>();
  }

  // Does Ax + b where A is a masked sparse ROW MAJOR matrix and
  // x is a COLUMN MAJOR dense vector or matrix.  Bias is a vector that is
  // broadcast is rhs has more than one column.
  template <typename FatVector>
  void SpMM_bias(const FatVector& rhs, FatVector* out, bool relu = false) {
    static_assert(std::is_same<typename FatVector::value_type, T>::value,
                  "FatVector value_type must match masked_linear_layer type");
    weights_.SpMM_bias(rhs, bias_, out, relu);
  }

 private:
  MaskedSparseMatrix<T> weights_;
  CacheAlignedVector<T> bias_;
};

}  // namespace csrblocksparse

#endif  // LYRA_CODEC_SPARSE_MATMUL_LAYERS_MASKED_SPARSE_MATRIX_H_
