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

#ifndef LYRA_CODEC_SPARSE_MATMUL_LAYERS_SPARSE_LINEAR_LAYER_H_
#define LYRA_CODEC_SPARSE_MATMUL_LAYERS_SPARSE_LINEAR_LAYER_H_

#include <cstdint>

#include "absl/memory/memory.h"
#include "glog/logging.h"
#include "sparse_matmul/layers/csr_blocksparse_matrix.h"
#include "sparse_matmul/layers/masked_sparse_matrix.h"
#include "sparse_matmul/numerics/type_utils.h"
#include "sparse_matmul/os/coop_threads.h"
#include "sparse_matmul/vector/cache_aligned_vector.h"

namespace csrblocksparse {

template <typename WeightType, typename RhsType,
          typename BiasType = typename TypeOfProduct<WeightType, RhsType>::type,
          typename DeltaType = int16_t>
class SparseLinearLayer {
 public:
  SparseLinearLayer() {}

  SparseLinearLayer(CsrBlockSparseMatrix<WeightType, RhsType>&& sparse_matrix,
                    CacheAlignedVector<BiasType>&& bias)
      : sparse_matrix_(std::move(sparse_matrix)), full_bias_(std::move(bias)) {
    CHECK_EQ(sparse_matrix_.rows(), full_bias_.size());
    // Some kernels expect that the bias is divided by 4, so we store a second
    // copy of a quarter of the bias.
    // TODO(b/189958858): Remove the quartered bias if it can be done without
    // loss of speed, and rename the |full_bias_| member back to |bias_|.
    bias_ = full_bias_;
    for (int i = 0; i < bias_.size(); ++i) {
      bias_[i] = static_cast<BiasType>(.25f * static_cast<float>(bias_[i]));
    }
  }
  SparseLinearLayer(
      const SparseLinearLayer<WeightType, RhsType, BiasType, DeltaType>& src) {
    *this = src;
  }
  SparseLinearLayer& operator=(
      const SparseLinearLayer<WeightType, RhsType, BiasType, DeltaType>& src) {
    sparse_matrix_ = src.sparse_matrix_;
    bias_ = src.bias_;
    full_bias_ = src.full_bias_;
    mid_output_ = src.mid_output_;
    thread_layers_ = src.thread_layers_;
    num_threads_ = src.num_threads_;
    if (src.split_pc_) {
      split_pc_ = absl::make_unique<ProducerConsumer>(
          src.split_pc_->num_producers(), src.split_pc_->num_consumers());
    }
    return *this;
  }

  // Does Ax + b where A is a block sparse compressed sparse row matrix and
  // x is a COLUMN MAJOR dense vector or matrix.  Bias is a vector that is
  // broadcast if rhs has more than one column.
  template <typename RhsClassType, typename OutType>
  void SpMM_bias(const RhsClassType& rhs, OutType* out, bool relu = false,
                 int tid = 0, SpinBarrier* barrier = nullptr) const {
    static_assert(
        std::is_same<typename RhsClassType::value_type, RhsType>::value, "");
    sparse_matrix_.SpMM_bias(rhs, bias_, out, relu, tid, barrier);
  }
  // Multiplies a sparse matrix by a possibly dense matrix, as SpMM_bias above,
  // and then samples from the output (softmax distribution) layer.
  template <typename RhsClassType, typename OutType>
  int SpMM_bias_Sample(const RhsClassType& rhs, OutType* out, float temperature,
                       int tid, SpinBarrier* barrier, std::minstd_rand* gen,
                       CacheAlignedVector<float>* scratch) const {
    static_assert(
        std::is_same<typename RhsClassType::value_type, RhsType>::value, "");
    return sparse_matrix_.SpMM_bias_Sample(rhs, bias_, out, temperature, tid,
                                           barrier, gen, scratch);
  }
  template <typename RhsClassType, typename OutType>
  void MatVec(const RhsClassType& rhs, bool relu, int tid, int replicas,
              int output_stride, OutType* output,
              SpinBarrier* barrier = nullptr) {
    static_assert(
        std::is_same<typename RhsClassType::value_type, RhsType>::value, "");
#ifdef __AVX2__
    if (block_width() == 4 && (block_height() == 4 || block_height() == 8) &&
        !IsCustomFloatType<WeightType>::value) {
      if (!IsSplit()) {
        sparse_matrix_.MatVec(rhs.cast_data(), full_bias_.cast_data(), relu,
                              tid, replicas, output_stride, output->data());
        if (barrier != nullptr) barrier->barrier();
        return;
      }
      // NOTE: Until the quartered bias is removed it is a bad idea to split
      // for ARM in the same way, as we would have to quarter the output of
      // the first part of the split before running the second part.
      // Signal completion of the previous MatVec.
      split_pc_->produce();
      PartLinearLayer& thread_part = thread_layers_[tid];
      auto offset_output =
          sparse_matrix_.thread_bounds().OffsetOutput(output->data(), tid);
      auto mid_output =
          sparse_matrix_.thread_bounds().OffsetOutput(mid_output_.data(), tid);
      auto offset_bias = sparse_matrix_.thread_bounds().OffsetOutput(
          mid_output_.cast_data(), tid);
      // We can continue to consume the data that this thread produced and
      // compute just the |self_matrix| part.
      // No |relu| or |replicas|, as this is only a partial matmul.
      // |tid| is always zero because the matrix has been split by tid.
      thread_part.self_matrix.MatVec(
          rhs.cast_data(), thread_part.full_bias.cast_data(), /*relu=*/false,
          /*tid=*/0, /*replicas=*/1, output_stride, mid_output);
      // We have to wait for the other threads to finish working on the previous
      // MatMul before consuming the rest of |rhs|.
      split_pc_->consume();
      thread_part.other_matrix.MatVec(rhs.cast_data(), offset_bias, relu,
                                      /*tid=*/0, replicas, output_stride,
                                      offset_output);
      return;
    }
#endif
    DCHECK_EQ(replicas, 1) << "Must have single replica for SpMM API";
    if (IsSplit()) {
      // Generics aren't setup to use a split matrix. This will be inefficient.
      split_pc_->produce();
      split_pc_->consume();
    }
    if (block_height() == 8) {
      // We are currently forced to use MatVec generics for this case.
      LOG(WARNING) << "Need to implement MatVec for 8x4 for non-AVX2 targets!!";
      sparse_matrix_.MatVec(rhs.cast_data(), full_bias_.cast_data(), relu, tid,
                            replicas, output_stride, output->data());
      if (barrier != nullptr) barrier->barrier();
    } else {
      sparse_matrix_.SpMM_bias(rhs, bias_, output, relu, tid, barrier);
    }
  }

  int rows() const { return sparse_matrix_.rows(); }
  int cols() const { return sparse_matrix_.cols(); }
  float sparsity() const { return sparse_matrix_.sparsity(); }
  int block_width() const { return sparse_matrix_.block_width(); }
  int block_height() const { return sparse_matrix_.block_height(); }
  int num_threads() const { return sparse_matrix_.num_threads(); }
  const CacheAlignedVector<BiasType>& bias() const { return bias_; }
  const std::vector<int>& split_points() const {
    return sparse_matrix_.split_points();
  }
  bool IsSplit() const {
    return !thread_layers_.empty() && split_pc_ != nullptr;
  }

  std::size_t bytes() const { return sparse_matrix_.bytes() + bias_.bytes(); }
  void Print() const {
    printf("Matrix\n");
    sparse_matrix_.Print();
    printf("Bias\n");
    bias_.Print();
  }

  // Combines adjacent row blocks, doubling the block height.
  // This necessarily involves adding zero weights where the blocks don't align
  // across adjacent pairs of rows, so use with caution, as the resulting matrix
  // is most likely to run slower if very sparse to begin with.
  // In the few cases where the blocks do mostly align, the resulting matmul
  // could be much faster, as the number of reads of the rhs will be halved.
  void DoubleBlockHeight() { sparse_matrix_.DoubleBlockHeight(); }

  // Cache_line_size is provided only for testing. Normally uses a value for
  // the current architecture.
  int PrepareForThreads(int num_threads, int cache_line_size = -1) {
    num_threads_ = num_threads;
    if (num_threads_ > 1) {
      split_pc_ =
          absl::make_unique<ProducerConsumer>(num_threads_, num_threads_);
    } else {
      split_pc_.reset(nullptr);
    }
    return sparse_matrix_.PrepareForThreads(num_threads, cache_line_size);
  }

  // Partitions the matrix into pieces by thread.
  // In this matrix, we can go ahead and calculate the part that only depends
  // on rhs inputs that were generated by this thread in the previous matvec,
  // without having to use any thread synchronization, and only after that do we
  // have to wait for the other threads to finish the previous matvec.
  // So we split the matrix using the |split_points| from the previous matrix
  // into 2 * |num_threads_| pieces: self and other for each thread, being the
  //  parts that can be calculated before and after the other threads have
  // completed their calculation of the previous matvec.
  // We then have to use a ProducerConsumer lock instead of a SpinBarrier to
  // synchronize the data produced by the other threads.
  void SliceForThreads(const std::vector<int>& split_points) {
    thread_layers_.clear();
    thread_layers_.reserve(num_threads_);
    LOG(INFO) << "Slicing " << rows() << "x" << cols() << " matrix for "
              << num_threads_ << " threads";
    for (int tid = 0; tid < num_threads_; ++tid) {
      thread_layers_.emplace_back(
          sparse_matrix_, full_bias_, bias_, tid,
          split_points[tid] * sparse_matrix_.block_height(),
          split_points[tid + 1] * sparse_matrix_.block_height());
    }
    mid_output_ =
        std::move(csrblocksparse::CacheAlignedVector<BiasType>(rows()));
    mid_output_.FillZero();
  }

  // Splits the layer by inputs into 2 equal pieces. Each of the resulting
  // layers should be computed independently on the first and second halves of
  // the inputs respectively and the results added to achieve the same effect
  // as the original layer.
  void SplitInputs(
      SparseLinearLayer<WeightType, RhsType, BiasType, DeltaType>* part1,
      SparseLinearLayer<WeightType, RhsType, BiasType, DeltaType>* part2) {
    CsrBlockSparseMatrix<WeightType, RhsType> matrix1(
        sparse_matrix_.SplitByColumn(0, sparse_matrix_.cols() / 2));
    CsrBlockSparseMatrix<WeightType, RhsType> matrix2(
        sparse_matrix_.SplitByColumn(sparse_matrix_.cols() / 2,
                                     sparse_matrix_.cols()));
    *part1 =
        std::move(SparseLinearLayer<WeightType, RhsType, BiasType, DeltaType>(
            std::move(matrix1),
            std::move(CacheAlignedVector<BiasType>(full_bias_))));
    CacheAlignedVector<BiasType> bias2(sparse_matrix_.rows());
    bias2.FillZero();
    *part2 =
        std::move(SparseLinearLayer<WeightType, RhsType, BiasType, DeltaType>(
            std::move(matrix2), std::move(bias2)));
  }

  // Splits the layer by outputs into 2 equal pieces. Each of the resulting
  // layers should be computed independently on the full inputs and the results
  // concatenated to achieve the same effect as the original layer.
  void SplitOutputs(
      SparseLinearLayer<WeightType, RhsType, BiasType, DeltaType>* part1,
      SparseLinearLayer<WeightType, RhsType, BiasType, DeltaType>* part2) {
    LOG(INFO) << "input rows=" << sparse_matrix_.rows()
              << ", cols=" << sparse_matrix_.cols();
    CsrBlockSparseMatrix<WeightType, RhsType> matrix1(
        sparse_matrix_.SplitByRow(0, sparse_matrix_.rows() / 2));
    CsrBlockSparseMatrix<WeightType, RhsType> matrix2(sparse_matrix_.SplitByRow(
        sparse_matrix_.rows() / 2, sparse_matrix_.rows()));
    CacheAlignedVector<BiasType> bias1(full_bias_, 0, full_bias_.size() / 2);
    *part1 =
        std::move(SparseLinearLayer<WeightType, RhsType, BiasType, DeltaType>(
            std::move(matrix1), std::move(bias1)));
    CacheAlignedVector<BiasType> bias2(full_bias_, full_bias_.size() / 2,
                                       full_bias_.size());
    *part2 =
        std::move(SparseLinearLayer<WeightType, RhsType, BiasType, DeltaType>(
            std::move(matrix2), std::move(bias2)));
  }

 private:
  // Simple struct to hold a partitioned layer.
  struct PartLinearLayer {
    // The original matrix is first split by row to generate only the outputs
    // for the given tid. The |row_sub_matrix| is then split by column into two
    // partitions:
    // self is the part for which the rhs elements in [|start_col|, |end_col|)
    // were generated by this thread in some previous matmul.
    // |other| is the rest of the columns that require rhs elements from other
    // threads.
    // NOTE that| start_col|, |end_col| are in raw columns, not blocks.
    PartLinearLayer(const CsrBlockSparseMatrix<WeightType, RhsType>& matrix,
                    const CacheAlignedVector<BiasType>& bias,
                    const CacheAlignedVector<BiasType>& bias_4, int tid,
                    int start_col, int end_col) {
      int block_height = matrix.block_height();
      // Split the input matrix by row, selecting only the rows relevant to
      // thread tid.
      int start_row = matrix.split_points()[tid] * block_height;
      int end_row = matrix.split_points()[tid + 1] * block_height;
      LOG(INFO) << "input cols [" << start_col << "," << end_col << ") rows ["
                << start_row << "," << end_row << ")";
      CsrBlockSparseMatrix<WeightType, RhsType> row_sub_matrix =
          matrix.SplitByRow(start_row, end_row);
      // Partition into the columns that use rhs elements that thread tid
      // produced in a previous matmul, and the other rhs elements.
      // NOTE that we |keep_rhs_size|=true so that each matrix can operate on
      // the same rhs input vector. The self matrix just guarantees not to
      // access any of the elements that are generated by another thread.
      self_matrix = std::move(row_sub_matrix.SplitByColumn(
          start_col, end_col, /*keep_rhs_size=*/true));
      self_matrix.PrepareForThreads(1);
      // The reversed start and end slice out the complement of [start, end).
      other_matrix = std::move(row_sub_matrix.SplitByColumn(
          end_col, start_col, /*keep_rhs_size=*/true));
      other_matrix.PrepareForThreads(1);
      full_bias =
          std::move(CacheAlignedVector<BiasType>(bias, start_row, end_row));
      // TODO(b/189958858): Eliminate the quarter bias from all the code.
      quarter_bias =
          std::move(CacheAlignedVector<BiasType>(bias_4, start_row, end_row));
    }
    // The part of the matrix that only depends on this thread for rhs inputs.
    CsrBlockSparseMatrix<WeightType, RhsType> self_matrix;
    CacheAlignedVector<BiasType> full_bias;
    CacheAlignedVector<BiasType> quarter_bias;
    // The part of the matrix that uses rhs inputs from other threads.
    CsrBlockSparseMatrix<WeightType, RhsType> other_matrix;
  };
  CsrBlockSparseMatrix<WeightType, RhsType, DeltaType> sparse_matrix_;
  CacheAlignedVector<BiasType> bias_;
  CacheAlignedVector<BiasType> full_bias_;
  // Output from the self_matrix that will be given to |other_matrix| as bias.
  CacheAlignedVector<BiasType> mid_output_;
  // One partitioned pair of matrices for each thread.
  std::vector<PartLinearLayer> thread_layers_;
  // Producer-consumer lock used to wait between computing |self_matrix| and
  // |other_matrix| for the other threads to finish the *previous* matvec.
  std::unique_ptr<ProducerConsumer> split_pc_;
  int num_threads_ = 0;
};

template <typename WeightType, typename RhsType>
SparseLinearLayer<WeightType, RhsType> CreateRandomLayer(int rows, int cols,
                                                         float sparsity,
                                                         int block_height = 1,
                                                         int block_width = 1) {
  typedef typename TypeOfProduct<WeightType, RhsType>::type BiasType;
  CacheAlignedVector<BiasType> bias(rows);
  bias.FillRandom();

  auto masked_matrix = MaskedSparseMatrix<float>(rows, cols, sparsity,
                                                 block_height, block_width);
  auto sparse_matrix = CsrBlockSparseMatrix<WeightType, RhsType>(masked_matrix);

  return SparseLinearLayer<WeightType, RhsType>(std::move(sparse_matrix),
                                                std::move(bias));
}

template <typename WeightType, typename RhsType>
SparseLinearLayer<WeightType, RhsType> CreateConstantLayer(
    int rows, int cols, float sparsity, float constant = 1.f) {
  typedef typename TypeOfProduct<WeightType, RhsType>::type BiasType;
  CacheAlignedVector<BiasType> bias(rows);
  bias.FillOnes();

  MaskedSparseMatrix<float> masked_matrix(rows, cols, sparsity,
                                          /*block_height=*/1, /*block_width=*/1,
                                          constant, /*random=*/false);
  CsrBlockSparseMatrix<WeightType, RhsType> sparse_matrix(masked_matrix);

  return SparseLinearLayer<WeightType, RhsType>(std::move(sparse_matrix),
                                                std::move(bias));
}

}  // namespace csrblocksparse

#endif  // LYRA_CODEC_SPARSE_MATMUL_LAYERS_SPARSE_LINEAR_LAYER_H_
