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

#include "sparse_matmul/layers/sparse_linear_layer.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "sparse_matmul/numerics/test_utils.h"

namespace csrblocksparse {
namespace {

constexpr int kBlockSize = 4;
constexpr int kSize = 256;
constexpr int kNumThreads = 4;
constexpr int kCols = 1;

void SlicedThreadBody(SpinBarrier* spin_barrier, int tid,
                      const FatCacheAlignedVector<float>& rhs,
                      SparseLinearLayer<float, float>* sparse_linear_layer,
                      FatCacheAlignedVector<float>* out, bool use_relu) {
  sparse_linear_layer->MatVec(rhs, use_relu, tid, /*replicas=*/1,
                              /*output_stride=*/0, out);
  spin_barrier->barrier();
}

// Tests that a Layer that has been SliceForThreads computes the same result as
// the original layer. This is a basic test that all the slicing didn't mess up
// any of the computations.
TEST(CsrBlockSparseMatrix, SliceForThreads) {
  MaskedSparseMatrix<float> matrix(kSize, kSize, 0.95, kBlockSize, kBlockSize);
  FatCacheAlignedVector<float> rhs(kSize, kCols);
  CacheAlignedVector<float> bias(kSize);
  FatCacheAlignedVector<float> out1(kSize, kCols);

  bias.FillRandom();
  rhs.FillRandom();
  out1.FillZero();
  FatCacheAlignedVector<float> out_reference = out1;
  CsrBlockSparseMatrix<float, float> sparse_matrix(matrix);
  SparseLinearLayer<float, float> sparse_linear_layer(std::move(sparse_matrix),
                                                      std::move(bias));
  sparse_linear_layer.PrepareForThreads(1);
  sparse_linear_layer.MatVec(rhs, /*relu=*/true, /*tid=*/0, /*replicas=*/1,
                             /*output_stride=*/0, &out_reference);
  std::vector<int> fake_split_points = {0, 48 / kBlockSize, 128 / kBlockSize,
                                        208 / kBlockSize, kSize / kBlockSize};
  sparse_linear_layer.PrepareForThreads(kNumThreads);
  sparse_linear_layer.SliceForThreads(fake_split_points);
  csrblocksparse::LaunchOnThreadsWithBarrier(kNumThreads, SlicedThreadBody, rhs,
                                             &sparse_linear_layer, &out1,
                                             /*relu=*/true);

  CheckResult(out_reference, out1, kCols);
}

void LayersThreadBody(SpinBarrier* spin_barrier, int tid,
                      const FatCacheAlignedVector<float>& rhs,
                      SparseLinearLayer<float, float>* sparse_linear_layer1,
                      SparseLinearLayer<float, float>* sparse_linear_layer2,
                      FatCacheAlignedVector<float>* out1,
                      FatCacheAlignedVector<float>* out2, bool use_relu) {
  sparse_linear_layer1->MatVec(rhs, use_relu, tid, /*replicas=*/1,
                               /*output_stride=*/0, out1);
  // NOTE no barrier here!
  sparse_linear_layer2->MatVec(*out1, use_relu, tid, /*replicas=*/1,
                               /*output_stride=*/0, out2);
  spin_barrier->barrier();
}

// Tests that a pair of layers computes the same result whether or not the
// second layer has been SliceForThreads. This is a more critical test that
// the replacement of barriers with producer-consumer locks works.
// Must be run with tsan to really test it properly.
TEST(CsrBlockSparseMatrix, SliceForThreadsLayers) {
  MaskedSparseMatrix<float> matrix1(kSize, kSize, 0.95, kBlockSize, kBlockSize);
  FatCacheAlignedVector<float> rhs(kSize, kCols);
  CacheAlignedVector<float> bias1(kSize);
  FatCacheAlignedVector<float> out1(kSize, kCols);
  MaskedSparseMatrix<float> matrix2(kSize, kSize, 0.95, kBlockSize, kBlockSize);
  CacheAlignedVector<float> bias2(kSize);
  FatCacheAlignedVector<float> out2(kSize, kCols);

  bias1.FillRandom();
  rhs.FillRandom();
  bias2.FillRandom();
  out1.FillZero();
  out2.FillZero();
  FatCacheAlignedVector<float> out_reference = out2;
  CsrBlockSparseMatrix<float, float> sparse_matrix1(matrix1);
  SparseLinearLayer<float, float> layer1(std::move(sparse_matrix1),
                                         std::move(bias1));
  CsrBlockSparseMatrix<float, float> sparse_matrix2(matrix2);
  SparseLinearLayer<float, float> layer2(std::move(sparse_matrix2),
                                         std::move(bias2));
  layer1.PrepareForThreads(1);
  layer2.PrepareForThreads(1);
  layer1.MatVec(rhs, /*relu=*/true, /*tid=*/0, /*replicas=*/1,
                /*output_stride=*/0, &out1);
  layer2.MatVec(out1, /*relu=*/true, /*tid=*/0, /*replicas=*/1,
                /*output_stride=*/0, &out_reference);
  layer1.PrepareForThreads(kNumThreads);
  layer2.PrepareForThreads(kNumThreads);
  layer2.SliceForThreads(layer1.split_points());
  csrblocksparse::LaunchOnThreadsWithBarrier(kNumThreads, LayersThreadBody, rhs,
                                             &layer1, &layer2, &out1, &out2,
                                             /*relu=*/true);

  CheckResult(out_reference, out2, kCols);
}

// Tests that a Layer that has been DoubleBlockHeight()-ed computes the same
// result as original layer. (Float compute type).
TEST(CsrBlockSparseMatrix, Float8x4) {
  using ComputeType = float;
  using RhsType = float;
  using BiasType = float;
  MaskedSparseMatrix<float> matrix(kSize, kSize, 0.95, kBlockSize, kBlockSize);
  matrix.CastWeights<ComputeType>();
  FatCacheAlignedVector<RhsType> rhs(kSize, kCols);
  CacheAlignedVector<BiasType> bias(kSize);
  FatCacheAlignedVector<BiasType> out1(kSize, kCols);

  bias.FillRandom();
  rhs.FillRandom();
  out1.FillZero();
  FatCacheAlignedVector<BiasType> out_reference = out1;
  CsrBlockSparseMatrix<ComputeType, RhsType> sparse_matrix(matrix);
  SparseLinearLayer<ComputeType, RhsType> sparse_linear_layer(
      std::move(sparse_matrix), std::move(bias));
  sparse_linear_layer.PrepareForThreads(1);
  sparse_linear_layer.MatVec(rhs, /*relu=*/true, /*tid=*/0, /*replicas=*/1,
                             /*output_stride=*/0, &out_reference);
  sparse_linear_layer.DoubleBlockHeight();
  sparse_linear_layer.PrepareForThreads(1);
  sparse_linear_layer.MatVec(rhs, /*relu=*/true, /*tid=*/0, /*replicas=*/1,
                             /*output_stride=*/0, &out1);
  CheckResult(out_reference, out1, kCols);
}

// Tests that a Layer that has been DoubleBlockHeight()-ed computes the same
// result as original layer. (Fixed16 compute type).
TEST(CsrBlockSparseMatrix, Fixed8x4) {
  using ComputeType = csrblocksparse::fixed16<4>;
  using RhsType = csrblocksparse::fixed16<4>;
  using BiasType = typename TypeOfProduct<ComputeType, RhsType>::type;
  MaskedSparseMatrix<float> matrix(kSize, kSize, 0.95, kBlockSize, kBlockSize);
  matrix.CastWeights<ComputeType>();
  FatCacheAlignedVector<RhsType> rhs(kSize, kCols);
  CacheAlignedVector<BiasType> bias(kSize);
  FatCacheAlignedVector<BiasType> out1(kSize, kCols);

  bias.FillRandom();
  rhs.FillRandom();
  out1.FillZero();
  FatCacheAlignedVector<BiasType> out_reference = out1;
  CsrBlockSparseMatrix<ComputeType, RhsType> sparse_matrix(matrix);
  SparseLinearLayer<ComputeType, RhsType> sparse_linear_layer(
      std::move(sparse_matrix), std::move(bias));
  sparse_linear_layer.PrepareForThreads(1);
  sparse_linear_layer.MatVec(rhs, /*relu=*/false, /*tid=*/0, /*replicas=*/1,
                             /*output_stride=*/0, &out_reference);
  sparse_linear_layer.DoubleBlockHeight();
  sparse_linear_layer.PrepareForThreads(1);
  sparse_linear_layer.MatVec(rhs, /*relu=*/false, /*tid=*/0, /*replicas=*/1,
                             /*output_stride=*/0, &out1);
  CheckResult(out_reference, out1, kCols);
}

TEST(SparseLinearLayerTest, PrintCompiles) {
  SparseLinearLayer<float, float> sparse_linear_layer;
  sparse_linear_layer.Print();
}

}  // namespace
}  // namespace csrblocksparse
