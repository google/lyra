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

#include <array>
#include <cstdint>
#include <tuple>
#include <vector>

// Placeholder for get runfiles header.
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "gtest/gtest.h"
#include "include/ghc/filesystem.hpp"
#include "sparse_matmul/compute/matmul.h"
#include "sparse_matmul/layers/utils.h"
#include "sparse_matmul/numerics/test_utils.h"
#include "sparse_matmul/os/coop_threads.h"

namespace csrblocksparse {
namespace {

inline constexpr absl::string_view kTestdataPath = "layers/testdata";

TEST(CSRBlockSparseMatrix, FlatBufferSerialization) {
  const int kRows = 8;
  const int kCols = 8;
  std::vector<int> mask = {1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0,
                           1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0,
                           0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1,
                           0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1};
  std::vector<float> values(kRows * kCols, 1.f);
  values[1] = 2.f;
  values[3] = 3.f;
  values[36] = -1.f;
  values[45] = -2.f;

  csrblocksparse::CacheAlignedVector<float> bias(kRows);
  csrblocksparse::CacheAlignedVector<float> rhs(kCols);
  csrblocksparse::CacheAlignedVector<float> out_ref(kRows);
  csrblocksparse::CacheAlignedVector<float> out_test(kRows);

  bias.FillZero();
  rhs.FillOnes();

  csrblocksparse::MaskedSparseMatrix<float> matrix(kRows, kCols, mask.data(),
                                                   values.data());

  matrix.SpMM_bias(rhs, bias, &out_ref);

  csrblocksparse::CsrBlockSparseMatrix<csrblocksparse::bfloat16, float, int16_t>
      block_sparse_matrix(matrix);

  std::string buffer;
  std::size_t num_bytes = block_sparse_matrix.WriteToFlatBuffer(&buffer);

  csrblocksparse::CsrBlockSparseMatrix<csrblocksparse::bfloat16, float, int16_t>
      new_block_sparse_matrix(reinterpret_cast<const uint8_t*>(buffer.c_str()),
                              num_bytes);

  new_block_sparse_matrix.SpMM_bias(rhs, bias, &out_test);

  CheckResult(out_ref, out_test, kCols);
}

template <typename ComputeType, typename RhsType, typename OutType>
void CorrectnessCheckBlockSpMM(int rows, int cols, int block_height,
                               int block_width, float sparsity,
                               bool use_relu = false, int num_threads = 1,
                               int fatness = 1, bool test_matmul = false) {
  using BiasType = typename TypeOfProduct<ComputeType, RhsType>::type;
  MaskedSparseMatrix<float> matrix(rows, cols, sparsity, block_height,
                                   block_width);
  matrix.CastWeights<ComputeType>();
  FatCacheAlignedVector<RhsType> rhs(cols, fatness);
  CacheAlignedVector<BiasType> bias(rows);
  FatCacheAlignedVector<OutType> out(rows, fatness);

  bias.FillRandom();
  rhs.FillRandom();
  out.FillZero();
  FatCacheAlignedVector<OutType> out_reference = out;

  matrix.SpMM_bias(rhs, bias, &out_reference, use_relu);

  CsrBlockSparseMatrix<ComputeType, RhsType> sparse_matrix(matrix);

  SparseLinearLayer<ComputeType, RhsType> sparse_linear_layer(
      std::move(sparse_matrix), std::move(bias));
  num_threads = sparse_linear_layer.PrepareForThreads(num_threads);

  // Checks that the result of applying each thread's portion serially is
  // correct.
  for (int thread_id = 0; thread_id < num_threads; ++thread_id) {
    sparse_linear_layer.SpMM_bias(rhs, &out, use_relu, thread_id);
  }

  CheckResult(out_reference, out, sparse_linear_layer.cols());

  if (test_matmul) {
    for (int thread_id = 0; thread_id < num_threads; ++thread_id) {
      sparse_linear_layer.MatVec(rhs, use_relu, thread_id,
                                 /*replicas=*/1, /*output_stride=*/0, &out);
    }

    CheckResult(out_reference, out, sparse_linear_layer.cols());
  }
}

// Does:
// y = Ax + b;
// x = Ay + b;
// y = Ax + b;
//
// to make sure that dependent multiplies are correct.
template <typename ComputeType, typename RhsType, typename OutType>
void ThreadBody(
    SpinBarrier* spin_barrier, int tid,
    const SparseLinearLayer<ComputeType, RhsType>& sparse_linear_layer,
    FatCacheAlignedVector<RhsType>* rhs, FatCacheAlignedVector<OutType>* out,
    bool use_relu) {
  sparse_linear_layer.SpMM_bias(*rhs, out, use_relu, tid);
  spin_barrier->barrier();
  sparse_linear_layer.SpMM_bias(*out, rhs, use_relu, tid);
  spin_barrier->barrier();
  sparse_linear_layer.SpMM_bias(*rhs, out, use_relu, tid);
}

template <typename ComputeType, typename RhsType, typename OutType>
void CorrectnessCheckBlockSpMM_MultiThread(int rows, int cols, int block_height,
                                           int block_width, float sparsity,
                                           bool use_relu = false,
                                           int num_threads = 1,
                                           int fatness = 1) {
  typedef typename TypeOfProduct<ComputeType, RhsType>::type BiasType;
  CHECK(rows == cols);
  MaskedSparseMatrix<float> matrix(rows, cols, sparsity, block_height,
                                   block_width);
  matrix.CastWeights<ComputeType>();
  FatCacheAlignedVector<RhsType> rhs(cols, fatness);
  FatCacheAlignedVector<RhsType> rhs_mt(cols, fatness);
  CacheAlignedVector<BiasType> bias(rows);
  FatCacheAlignedVector<OutType> out(rows, fatness);

  bias.FillOnes();
  rhs.FillOnes();
  rhs_mt.FillOnes();
  out.FillZero();
  FatCacheAlignedVector<OutType> out_reference = out;

  matrix.SpMM_bias(rhs, bias, &out_reference, use_relu);
  matrix.SpMM_bias(out_reference, bias, &rhs, use_relu);
  matrix.SpMM_bias(rhs, bias, &out_reference, use_relu);

  CsrBlockSparseMatrix<ComputeType, RhsType> sparse_matrix(matrix);

  num_threads = sparse_matrix.PrepareForThreads(num_threads,
                                                /*cache_line_size=*/1);

  SparseLinearLayer<ComputeType, RhsType> sparse_linear_layer(
      std::move(sparse_matrix), std::move(bias));

  csrblocksparse::LaunchOnThreadsWithBarrier(
      num_threads, ThreadBody<ComputeType, RhsType, OutType>,
      sparse_linear_layer, &rhs_mt, &out, use_relu);

  CheckResult(out_reference, out, cols);
}

}  // namespace

TEST(MaskedSparseCorrectness, HandCoded) {
  const int kRows = 8;
  const int kCols = 8;
  // clang-format off
  std::vector<int> mask = {1, 1, 0, 0, 0, 1, 1, 1,
                           0, 1, 0, 1, 0, 1, 0, 1,
                           1, 0, 0, 1, 1, 1, 1, 0,
                           0, 0, 0, 0, 0, 0, 0, 0,
                           1, 1, 1, 1, 1, 1, 1, 1,
                           0, 0, 0, 0, 1, 1, 0, 0,
                           1, 1, 0, 0, 1, 1, 0, 0,
                           1, 0, 0, 0, 0, 1, 0, 1};
  // clang-format on
  std::vector<float> values(kRows * kCols, 1.f);

  std::vector<float> answer = {6.f, 5.f, 6.f, 1.f, 9.f, 3.f, 5.f, 4.f};

  MaskedSparseMatrix<float> matrix(kRows, kCols, mask.data(), values.data());
  CacheAlignedVector<float> rhs(kCols);
  CacheAlignedVector<float> bias(kRows);
  CacheAlignedVector<float> out(kRows);

  bias.FillOnes();
  rhs.FillOnes();
  out.FillZero();

  MaskedLinearLayer<float> masked_linear_layer(std::move(matrix),
                                               std::move(bias));

  masked_linear_layer.SpMM_bias(rhs, &out);

  for (int i = 0; i < kRows; ++i) {
    EXPECT_EQ(answer[i], out[i]);
  }
}

TEST(MaskedSparseCorrectness, HandCodedFatVector) {
  const int kRows = 8;
  const int kCols = 8;
  // clang-format off
  std::vector<int> mask = {1, 1, 0, 0, 0, 1, 1, 1,
                           0, 1, 0, 1, 0, 1, 0, 1,
                           1, 0, 0, 1, 1, 1, 1, 0,
                           0, 0, 0, 0, 0, 0, 0, 0,
                           1, 1, 1, 1, 1, 1, 1, 1,
                           0, 0, 0, 0, 1, 1, 0, 0,
                           1, 1, 0, 0, 1, 1, 0, 0,
                           1, 0, 0, 0, 0, 1, 0, 1};
  // clang-format on

  std::vector<float> values(kRows * kCols, 1.f);
  std::vector<float> answer = {6.f, 5.f, 6.f, 1.f, 9.f, 3.f, 5.f, 4.f};

  MaskedSparseMatrix<float> matrix(kRows, kCols, mask.data(), values.data());
  const int kMaxWidth = 5;
  for (int width = 5; width <= kMaxWidth; ++width) {
    FatCacheAlignedVector<float> rhs(kCols, width);
    CacheAlignedVector<float> bias(kRows);
    FatCacheAlignedVector<float> out(kRows, width);

    bias.FillOnes();
    rhs.FillOnes();
    out.FillZero();

    MaskedLinearLayer<float> masked_linear_layer(std::move(matrix),
                                                 std::move(bias));

    masked_linear_layer.SpMM_bias(rhs, &out);

    for (int i = 0; i < kRows; ++i) {
      for (int width = 0; width < kMaxWidth; ++width) {
        EXPECT_EQ(answer[i], out[i + width * kRows]);
      }
    }
  }
}

TEST(CsrBlockSparseMatrix, HandCodedMultiThread) {
  const int kRows = 8;
  const int kCols = 8;
  // clang-format off
  std::vector<int> mask = {1, 1, 0, 0, 0, 1, 1, 1,
                           0, 1, 0, 1, 0, 1, 0, 1,
                           1, 0, 0, 1, 1, 1, 1, 0,
                           0, 0, 0, 0, 0, 0, 0, 0,
                           1, 1, 1, 1, 1, 1, 1, 1,
                           0, 0, 0, 0, 1, 1, 0, 0,
                           1, 1, 0, 0, 1, 1, 0, 0,
                           1, 0, 0, 0, 0, 1, 0, 1};
  // clang-format on
  std::vector<float> values(kRows * kCols, 1.f);

  std::vector<float> answer = {6.f, 5.f, 6.f, 1.f, 9.f, 3.f, 5.f, 4.f};

  MaskedSparseMatrix<float> matrix(kRows, kCols, mask.data(), values.data());
  CacheAlignedVector<float> rhs(kCols);
  CacheAlignedVector<float> bias(kRows);
  CacheAlignedVector<float> out(kRows);

  bias.FillOnes();
  rhs.FillOnes();
  out.FillZero();

  CacheAlignedVector<float> bias_csr = bias;

  CsrBlockSparseMatrix<bfloat16, float> sparse_matrix(matrix);

  MaskedLinearLayer<float> masked_linear_layer(std::move(matrix),
                                               std::move(bias));

  masked_linear_layer.SpMM_bias(rhs, &out);

  SparseLinearLayer<bfloat16, float> sparse_linear_layer(
      std::move(sparse_matrix), std::move(bias_csr));
  sparse_linear_layer.PrepareForThreads(2, /*cache_line_size=*/1);

  CacheAlignedVector<float> out_tmp(kRows);
  const bool kUseRelu = false;
  sparse_linear_layer.SpMM_bias(rhs, &out_tmp, kUseRelu, /*tid=*/0);
  sparse_linear_layer.SpMM_bias(rhs, &out_tmp, kUseRelu, /*tid=*/1);

  for (int i = 0; i < kRows; ++i) {
    EXPECT_EQ(answer[i], out_tmp[i]);
  }
}

TEST(TestCasts, TestBfloat16) {
  const int kRows = 1000;
  const int kCols = 100;
  const float kSparsity = 0.f;

  MaskedSparseMatrix<float> matrix(kRows, kCols, kSparsity);
  MaskedSparseMatrix<float> matrix_bfloat16(kRows, kCols, matrix.mask().data(),
                                            matrix.values().data());

  matrix_bfloat16.CastWeights<bfloat16>();

  CheckResult(matrix.values(), matrix_bfloat16.values(), kCols);
}

TEST(TestCasts, TestFP16) {
  const int kRows = 1000;
  const int kCols = 100;
  const float kSparsity = 0.f;

  MaskedSparseMatrix<float> matrix(kRows, kCols, kSparsity);
#if !defined __arm__ && !defined __aarch64__
  // Conversion doesn't handle denormals, so flush denormals to zero first.
  for (int i = 0; i < matrix.values().size(); ++i) {
    if (matrix.data()[i] < 1. / static_cast<float>(1 << 14))
      matrix.data()[i] = 0.f;
  }
#endif
  MaskedSparseMatrix<float> matrix_fp16(kRows, kCols, matrix.mask().data(),
                                        matrix.values().data());

  matrix_fp16.CastWeights<csrblocksparse::fp16>();

  CheckResult(matrix.values(), matrix_fp16.values(), kCols);
}

TEST(TestCasts, TestFixed16) {
  const int kRows = 100000;
  const int kCols = 1;
  const float kSparsity = 0.f;

  MaskedSparseMatrix<float> matrix(kRows, kCols, kSparsity);

  // Relative error for fixed point is high near 0.
  for (int i = 0; i < matrix.values().size(); ++i) {
    // 1.1e-3 is based on the max error of .013 and a grid spacing of 1 / 2**16
    // == 3e-5.  3e-5 / .013 / 2 = 1.1e-3.
    if (std::abs(matrix.data()[i]) < 1.1e-3) {
      matrix.data()[i] = 0.f;
    }
  }

  MaskedSparseMatrix<float> matrix_fixed16 = matrix;

  matrix_fixed16.CastWeights<csrblocksparse::fixed16</*ExponentBits=*/0>>();

  CheckResult(matrix.values(), matrix_fixed16.values(), kCols);
}

TEST(TestCasts, TestFixed32) {
  const int kRows = 100000;
  const int kCols = 1;
  const float kSparsity = 0.f;

  MaskedSparseMatrix<float> matrix(kRows, kCols, kSparsity);
  MaskedSparseMatrix<float> matrix_fixed32 = matrix;

  matrix_fixed32.CastWeights<csrblocksparse::fixed32</*ExponentBits=*/0>>();

  CheckResult(matrix.values(), matrix_fixed32.values(), kCols);
}

template <typename ComputeType, typename RhsType, typename OutType>
void TestSpMM(int block_width, int block_height, int fatness,
              bool test_matmul = false) {
  std::array<bool, 2> use_relu = {false, true};
  std::vector<float> sparsity_levels = {.5, .8, .9, .95, .98};
  std::vector<std::pair<int, int>> sizes = {{8, 8},     {128, 128}, {128, 64},
                                            {256, 192}, {512, 512}, {1024, 512},
                                            {384, 384}, {512, 384}};
  for (int num_threads = 1; num_threads < 2 + test_matmul; ++num_threads) {
    for (const auto& relu : use_relu) {
      for (const auto& sparsity : sparsity_levels) {
        for (const auto& size : sizes) {
          int rows, cols;
          std::tie(rows, cols) = size;
          CorrectnessCheckBlockSpMM<ComputeType, RhsType, OutType>(
              rows, cols, block_height, block_width, sparsity, relu,
              num_threads, fatness, test_matmul);
        }
      }
    }
  }
}

template <typename ComputeType, typename RhsType, typename OutType>
void TestSpMM_MultiThread(int block_width, int block_height, int fatness) {
  std::array<bool, 2> use_relu = {false, true};
  std::vector<float> sparsity_levels = {.5, .8, .9, .95, .98};
  std::vector<std::pair<int, int>> sizes = {
      {48, 48}, {128, 128}, {512, 512}, {384, 384}};
  for (int num_threads = 1; num_threads < 5; ++num_threads) {
    for (const auto& relu : use_relu) {
      for (const auto& sparsity : sparsity_levels) {
        for (const auto& size : sizes) {
          int rows, cols;
          std::tie(rows, cols) = size;
          CorrectnessCheckBlockSpMM_MultiThread<ComputeType, RhsType, OutType>(
              rows, cols, block_height, block_width, sparsity, relu,
              num_threads, fatness);
        }
      }
    }
  }
}

template <typename DataType>
void TestSumVectors(int start = 0, int end = -1, int size = 6) {
  std::vector<DataType> values;
  std::vector<DataType> answer;

  for (int i = 1; i < size + 1; ++i) {
    const float x = static_cast<float>(i);
    values.push_back(static_cast<DataType>(x));
    answer.push_back(static_cast<DataType>(x * 2));
  }

  if (end == -1) {
    end = values.size();
  }

  csrblocksparse::CacheAlignedVector<DataType> result(values.size());
  csrblocksparse::CacheAlignedVector<DataType> values_aligned(values);
  detail::SumVectors(start, end, values_aligned.data(), values_aligned.data(),
                     result.data());
  for (int i = start; i < end; ++i) {
    EXPECT_EQ(static_cast<float>(answer[i]), static_cast<float>(result[i]));
  }
}

TEST(CsrBlockSparseMatrix, SumVectors_Generic) {
  TestSumVectors<float>();
  TestSumVectors<float>(1);
  TestSumVectors<float>(1, 4);
}

TEST(CsrBlockSparseMatrix, SumVectors_Bfloat16) {
  TestSumVectors<csrblocksparse::bfloat16>();
  TestSumVectors<csrblocksparse::bfloat16>(1);
  TestSumVectors<csrblocksparse::bfloat16>(1, 4);
}

// For SIMD-optimized SumVectors, the memory of the vector should be at least
// |kSIMDWidth * sizeof(float)| long, and the start position has to be an
// aligned memory location. So setting |size| to be 100 to be safe and
// |start| to be 0 (|start| == 1 is not aligned).
TEST(CsrBlockSparseMatrix, SumVectors_Fixed16) {
  TestSumVectors<csrblocksparse::fixed16<8>>(0, -1, 100);
  TestSumVectors<csrblocksparse::fixed16<8>>(0, 4, 100);
}

TEST(CsrBlockSparseMatrix, SumVectors_Fixed32) {
  TestSumVectors<csrblocksparse::fixed32<11>>(0, -1, 100);
  TestSumVectors<csrblocksparse::fixed32<11>>(0, 4, 100);
}

TEST(CsrBlockSparseMatrix, SpMM_Block4x4_Bfloat16) {
  TestSpMM<csrblocksparse::bfloat16, float, float>(/*block_width=*/4,
                                                   /*block_height=*/4,
                                                   /*fatness=*/7);
}

// This actually uses multiple threads, and uses the output as the input for
// multiple steps to test that synchronization and memory visibility is
// working correctly.Requires square matrices.
TEST(CsrBlockSparseMatrix, SpMV_4x4MultiThreading_Bfloat16) {
  TestSpMM_MultiThread<csrblocksparse::bfloat16, float, float>(
      /*block_width=*/4,
      /*block_height=*/4,
      /*fatness=*/1);
}

TEST(CsrBlockSparseMatrix, SpMM_4x4MultiThreading_Bfloat16) {
  TestSpMM_MultiThread<csrblocksparse::bfloat16, float, float>(
      /*block_width=*/4,
      /*block_height=*/4,
      /*fatness=*/7);
}

TEST(CsrBlockSparseMatrix, SpMV_Block1x1_Bfloat16) {
  TestSpMM<csrblocksparse::bfloat16, float, float>(/*block_width=*/1,
                                                   /*block_height=*/1,
                                                   /*fatness=*/1);
}

TEST(CsrBlockSparseMatrix, SpMM_Block1x1_Bfloat16) {
  TestSpMM<csrblocksparse::bfloat16, float, float>(/*block_width=*/1,
                                                   /*block_height=*/1,
                                                   /*fatness=*/7);
}

// This actually uses multiple threads, and uses the output as the input for
// multiple steps to test that synchronization and memory visibility is
// working correctly.Requires square matrices.
TEST(CsrBlockSparseMatrix, SpMV_1x1MultiThreading_Bfloat16) {
  TestSpMM_MultiThread<csrblocksparse::bfloat16, float, float>(
      /*block_width=*/1,
      /*block_height=*/1,
      /*fatness=*/1);
}

TEST(CsrBlockSparseMatrix, SpMM_1x1MultiThreading_Bfloat16) {
  TestSpMM_MultiThread<csrblocksparse::bfloat16, float, float>(
      /*block_width=*/1,
      /*block_height=*/1,
      /*fatness=*/7);
}

TEST(CsrBlockSparseMatrix, SpMV_Block4x4_float) {
  TestSpMM<float, float, float>(/*block_width=*/4,
                                /*block_height=*/4,
                                /*fatness=*/1,
                                /*test_matmul=*/true);
}

TEST(CsrBlockSparseMatrix, SpMM_Block4x4_float) {
  TestSpMM<float, float, float>(/*block_width=*/4,
                                /*block_height=*/4,
                                /*fatness=*/7);
}

// This actually uses multiple threads, and uses the output as the input for
// multiple steps to test that synchronization and memory visibility is
// working correctly.Requires square matrices.
TEST(CsrBlockSparseMatrix, SpMV_4x4MultiThreading_float) {
  TestSpMM_MultiThread<float, float, float>(/*block_width=*/4,
                                            /*block_height=*/4,
                                            /*fatness=*/1);
}

TEST(CsrBlockSparseMatrix, SpMM_4x4MultiThreading_float) {
  TestSpMM_MultiThread<float, float, float>(/*block_width=*/4,
                                            /*block_height=*/4,
                                            /*fatness=*/7);
}

TEST(CsrBlockSparseMatrix, SpMV_Block1x1_float) {
  TestSpMM<float, float, float>(/*block_width=*/1,
                                /*block_height=*/1,
                                /*fatness=*/1);
}

TEST(CsrBlockSparseMatrix, SpMM_Block1x1_float) {
  TestSpMM<float, float, float>(/*block_width=*/1,
                                /*block_height=*/1,
                                /*fatness=*/7);
}

// This actually uses multiple threads, and uses the output as the input for
// multiple steps to test that synchronization and memory visibility is
// working correctly.Requires square matrices.
TEST(CsrBlockSparseMatrix, SpMV_1x1MultiThreading_float) {
  TestSpMM_MultiThread<float, float, float>(/*block_width=*/1,
                                            /*block_height=*/1,
                                            /*fatness=*/1);
}

TEST(CsrBlockSparseMatrix, SpMM_1x1MultiThreading_float) {
  TestSpMM_MultiThread<float, float, float>(/*block_width=*/1,
                                            /*block_height=*/1,
                                            /*fatness=*/7);
}

TEST(CsrBlockSparseMatrix, SpMV_Block4x4_fixed16x16_32) {
  TestSpMM<csrblocksparse::fixed16<4>, csrblocksparse::fixed16<4>,
           typename csrblocksparse::TypeOfProduct<
               csrblocksparse::fixed16<4>, csrblocksparse::fixed16<4>>::type>(
      /*block_width=*/4,
      /*block_height=*/4,
      /*fatness=*/1,
      /*test_matmul=*/true);
}

TEST(CsrBlockSparseMatrix, SpMM_Block4x4_fixed16x16_32) {
  TestSpMM<csrblocksparse::fixed16<4>, csrblocksparse::fixed16<4>,
           typename csrblocksparse::TypeOfProduct<
               csrblocksparse::fixed16<4>, csrblocksparse::fixed16<4>>::type>(
      /*block_width=*/4,
      /*block_height=*/4,
      /*fatness=*/7);
}

TEST(CsrBlockSparseMatrix, SpMV_Block1x1_fixed16x16_32) {
  TestSpMM<csrblocksparse::fixed16<4>, csrblocksparse::fixed16<4>,
           typename csrblocksparse::TypeOfProduct<
               csrblocksparse::fixed16<4>, csrblocksparse::fixed16<4>>::type>(
      /*block_width=*/1,
      /*block_height=*/1,
      /*fatness=*/1);
}

TEST(CsrBlockSparseMatrix, SpMM_Block1x1_fixed16x16_32) {
  TestSpMM<csrblocksparse::fixed16<4>, csrblocksparse::fixed16<4>,
           typename csrblocksparse::TypeOfProduct<
               csrblocksparse::fixed16<4>, csrblocksparse::fixed16<4>>::type>(
      /*block_width=*/1,
      /*block_height=*/1,
      /*fatness=*/7);
}

TEST(CsrBlockSparseMatrix, SpMV_Block4x4_fixed16x16_16) {
  TestSpMM<csrblocksparse::fixed16<5>, csrblocksparse::fixed16<5>,
           csrblocksparse::fixed16<8>>(
      /*block_width=*/4,
      /*block_height=*/4,
      /*fatness=*/1,
      /*test_matmul=*/true);
}

TEST(CsrBlockSparseMatrix, SpMM_Block4x4_fixed16x16_16) {
  TestSpMM<csrblocksparse::fixed16<5>, csrblocksparse::fixed16<5>,
           csrblocksparse::fixed16<8>>(
      /*block_width=*/4,
      /*block_height=*/4,
      /*fatness=*/7);
}

TEST(CsrBlockSparseMatrix, SpMV_Block1x1_fixed16x16_16) {
  TestSpMM<csrblocksparse::fixed16<5>, csrblocksparse::fixed16<5>,
           csrblocksparse::fixed16<8>>(
      /*block_width=*/1,
      /*block_height=*/1,
      /*fatness=*/1);
}

TEST(CsrBlockSparseMatrix, SpMM_Block1x1_fixed16x16_16) {
  TestSpMM<csrblocksparse::fixed16<5>, csrblocksparse::fixed16<5>,
           csrblocksparse::fixed16<8>>(
      /*block_width=*/1,
      /*block_height=*/1,
      /*fatness=*/7);
}

TEST(CsrBlockSparseMatrix, SpMV_Block4x4_fixed16x16_32_unmatched) {
  TestSpMM<csrblocksparse::fixed16<5>, csrblocksparse::fixed16<5>,
           csrblocksparse::fixed32<13>>(
      /*block_width=*/4,
      /*block_height=*/4,
      /*fatness=*/1,
      /*test_matmul=*/true);
}

TEST(CsrBlockSparseMatrix, SpMM_Block4x4_fixed16x16_32_unmatched) {
  TestSpMM<csrblocksparse::fixed16<5>, csrblocksparse::fixed16<5>,
           csrblocksparse::fixed32<13>>(
      /*block_width=*/4,
      /*block_height=*/4,
      /*fatness=*/7);
}

TEST(CsrBlockSparseMatrix, SpMV_Block1x1_fixed16x16_32_unmatched) {
  TestSpMM<csrblocksparse::fixed16<5>, csrblocksparse::fixed16<5>,
           csrblocksparse::fixed32<13>>(
      /*block_width=*/1,
      /*block_height=*/1,
      /*fatness=*/1);
}

TEST(CsrBlockSparseMatrix, SpMM_Block1x1_fixed16x16_32_unmatched) {
  TestSpMM<csrblocksparse::fixed16<5>, csrblocksparse::fixed16<5>,
           csrblocksparse::fixed32<13>>(
      /*block_width=*/1,
      /*block_height=*/1,
      /*fatness=*/7);
}

TEST(CsrBlockSparseMatrix, RhsIndicesDeltasRoundTrip) {
  MaskedSparseMatrix<float> matrix(/*rows=*/256, /*cols=*/256,
                                   /*sparsity=*/0.9, /*block_height=*/4,
                                   /*block_width=*/4);
  CsrBlockSparseMatrix<float, float> sparse_matrix(matrix);
  CacheAlignedVector<int16_t> copy_indices = sparse_matrix.rhs_indices();
  sparse_matrix.ComputeColDeltas();
  sparse_matrix.ComputeRHSIndices();
  // They get padded when created, so the newer one could be bigger.
  EXPECT_LE(copy_indices.size(), sparse_matrix.rhs_indices().size());
  for (int i = 0; i < copy_indices.size(); ++i) {
    EXPECT_EQ(copy_indices[i], sparse_matrix.rhs_indices()[i]) << "i=" << i;
  }
}

// Tests that a Layer that is split into 2 by columns (inputs) computes the same
// result as the original layer.
TEST(CsrBlockSparseMatrix, SplitByCol) {
  int kRows = 1024;
  int kCols = 1024;
  MaskedSparseMatrix<float> matrix(kRows, kCols, 0.95, /*block_height=*/4,
                                   /*block_width=*/4);
  FatCacheAlignedVector<float> rhs(kCols, /*cols=*/1);
  CacheAlignedVector<float> bias(kRows);
  FatCacheAlignedVector<float> out1(kRows, /*cols=*/1);
  FatCacheAlignedVector<float> out2(kRows, /*cols=*/1);

  bias.FillRandom();
  rhs.FillRandom();
  out1.FillZero();
  out2.FillZero();
  FatCacheAlignedVector<float> out_reference = out1;

  CsrBlockSparseMatrix<float, float> sparse_matrix(matrix);

  SparseLinearLayer<float, float> sparse_linear_layer(std::move(sparse_matrix),
                                                      std::move(bias));
  sparse_linear_layer.PrepareForThreads(1);
  sparse_linear_layer.SpMM_bias(rhs, &out_reference, /*relu=*/false,
                                /*tid=*/0);
  // Split the layer into 2 parts.
  SparseLinearLayer<float, float> part1, part2;
  sparse_linear_layer.SplitInputs(&part1, &part2);
  part1.PrepareForThreads(1);
  part2.PrepareForThreads(1);
  EXPECT_EQ(kRows, part1.rows());
  EXPECT_EQ(kCols / 2, part1.cols());
  EXPECT_EQ(kRows, part2.rows());
  EXPECT_EQ(kCols / 2, part2.cols());
  MutableVectorView<float> rhs1(&rhs, 0, kCols / 2);
  MutableVectorView<float> rhs2(&rhs, kCols / 2, kCols / 2);
  for (int i = 0; i < kCols / 2; ++i) {
    EXPECT_FLOAT_EQ(rhs[i], rhs1.data()[i]);
    EXPECT_FLOAT_EQ(rhs[i + kCols / 2], rhs2.data()[i]);
  }
  part1.SpMM_bias(rhs1, &out1, /*relu=*/false, /*tid=*/0);
  part2.SpMM_bias(rhs2, &out2, /*relu=*/false, /*tid=*/0);
  // Check that out1 + out2 = out_reference.
  for (int i = 0; i < kRows; ++i) {
    EXPECT_NEAR(out_reference[i], out1[i] + out2[i], 2e-5)
        << " i=" << i << " out1=" << out1[i] << " out2=" << out2[i];
  }
}
// Tests that a Layer that is split into 2 by rows (outputs) computes the same
// result as the original layer.
TEST(CsrBlockSparseMatrix, SplitByRow) {
  int kRows = 1024;
  int kCols = 1024;
  MaskedSparseMatrix<float> matrix(kRows, kCols, 0.95, /*block_height=*/4,
                                   /*block_width=*/4);
  FatCacheAlignedVector<float> rhs(kCols, /*cols=*/1);
  CacheAlignedVector<float> bias(kRows);
  FatCacheAlignedVector<float> out1(kRows, /*cols=*/1);
  FatCacheAlignedVector<float> out2(kRows, /*cols=*/1);

  bias.FillRandom();
  rhs.FillRandom();
  out1.FillZero();
  out2.FillZero();
  FatCacheAlignedVector<float> out_reference = out1;

  CsrBlockSparseMatrix<float, float> sparse_matrix(matrix);

  SparseLinearLayer<float, float> sparse_linear_layer(std::move(sparse_matrix),
                                                      std::move(bias));
  sparse_linear_layer.PrepareForThreads(1);
  sparse_linear_layer.SpMM_bias(rhs, &out_reference, /*relu=*/false,
                                /*tid=*/0);
  // Split the layer into 2 parts.
  SparseLinearLayer<float, float> part1, part2;
  sparse_linear_layer.SplitOutputs(&part1, &part2);
  part1.PrepareForThreads(1);
  part2.PrepareForThreads(1);
  EXPECT_EQ(kRows / 2, part1.rows());
  EXPECT_EQ(kCols, part1.cols());
  EXPECT_EQ(kRows / 2, part2.rows());
  EXPECT_EQ(kCols, part2.cols());
  MutableVectorView<float> out2a(&out2, 0, kRows / 2);
  MutableVectorView<float> out2b(&out2, kRows / 2, kRows / 2);
  part1.SpMM_bias(rhs, &out2a, /*relu=*/false, /*tid=*/0);
  part2.SpMM_bias(rhs, &out2b, /*relu=*/false, /*tid=*/0);
  // Check that out2 = out_reference.
  for (int i = 0; i < kRows; ++i) {
    EXPECT_NEAR(out_reference[i], out2[i], 2e-5)
        << " i=" << i << " out1=" << out_reference[i] << " out2=" << out2[i];
  }
}

TEST(CsrBlockSparseMatrix, MutableVectorView) {
  const int kRows = 1024;
  const int kCols = 1024;
  const int kFatness = 2;

  std::vector<float> values(kRows * kCols, 1.f);
  std::vector<int> mask(kRows * kCols);
  for (int i = 0; i < mask.size(); ++i) mask[i] = i % 2;

  auto masked_matrix =
      MaskedSparseMatrix<float>(kRows, kCols, mask.data(), values.data());
  auto sparse_matrix = CsrBlockSparseMatrix<bfloat16, float>(masked_matrix);
  FatCacheAlignedVector<float> x(kCols, kFatness);
  x.FillOnes();

  CacheAlignedVector<float> bias(kRows);
  bias.FillZero();

  // First check that we can use spans as output.  Split a multiplication
  // into upper and lower halves times the full vector:
  // ---------------  x   t
  // |             |  x   t
  // |             |  x   t
  // ---------------    =
  // |             |  x   b
  // |             |  x   b
  // ---------------  x   b

  FatCacheAlignedVector<float> out(kRows, kFatness);
  FatCacheAlignedVector<float> out_view(kRows, kFatness);

  MutableVectorView<float> out_view_top(&out_view, 0, kRows / 2);
  MutableVectorView<float> out_view_bottom(&out_view, kRows / 2, kRows / 2);

  sparse_matrix.SpMM_bias(x, bias, &out);

  auto masked_matrix_top =
      MaskedSparseMatrix<float>(kRows / 2, kCols, mask.data(), values.data());
  auto masked_matrix_bottom = MaskedSparseMatrix<float>(
      kRows / 2, kCols, mask.data() + kRows * kCols / 2,
      values.data() + kRows * kCols / 2);
  auto sparse_matrix_top =
      CsrBlockSparseMatrix<bfloat16, float>(masked_matrix_top);
  auto sparse_matrix_bottom =
      CsrBlockSparseMatrix<bfloat16, float>(masked_matrix_bottom);

  sparse_matrix_top.SpMM_bias(x, bias, &out_view_top);
  sparse_matrix_bottom.SpMM_bias(x, bias, &out_view_bottom);

  CheckResult(out, out_view, kCols);

  // Check that we can use a span as an input vector.  Multiply upper left
  // portion of the matrix by the top half of the vector.
  // ---------------
  // |oooooo       |   x   q
  // |oooooo       |   x   q
  // |             |     =
  // |             |
  // ---------------

  auto masked_matrix_quarter = MaskedSparseMatrix<float>(
      kRows / 2, kCols / 2, mask.data(), values.data());
  auto sparse_matrix_quarter =
      CsrBlockSparseMatrix<bfloat16, float>(masked_matrix_quarter);

  MutableVectorView<float> x_top(&x, 0, kCols / 2);
  FatCacheAlignedVector<float> out_correct(kRows / 2, /*cols=*/2);

  for (int i = 0; i < kFatness * (kRows / 2); ++i) out_correct[i] = 256.f;

  MutableVectorView<float> bias_top(&bias, 0, kRows / 2);
  FatCacheAlignedVector<float> out_quarter(kRows / 2, kFatness);

  sparse_matrix_quarter.SpMM_bias(x_top, bias_top, &out_quarter);

  CheckResult(out_correct, out_quarter, kCols / 2);
}

namespace {

bool skip_test(const absl::Status& status, absl::string_view msg) {
  if (!status.ok()) {
    LOG(INFO) << "Couldn't load " << msg << ", skipping test " << status;
    return true;
  }

  return false;
}

}  // namespace

TEST(CsrBlockSparseMatrix, ModelMatrices_Bfloat16) {
  std::vector<std::string> names = {
      "768_512_95_4x4_wavernn_gru_", "768_512_95_4x4_coarseproj_",
      "768_512_95_4x4_coarselogit_", "768_512_95_4x4_fineproj_",
      "768_512_95_4x4_finelogit_",   "lyra_conv1d_"};
  const std::string kPath =
#if defined __arm__ || defined __aarch64__
      "/data/local/tmp/";
#else
      (ghc::filesystem::current_path() / kTestdataPath).string();
#endif
  for (auto& layer_name : names) {
    SparseLinearLayer<bfloat16, float> sparse_linear_layer;
    auto status = LoadSparseLayer<bfloat16, float>(layer_name, /*zipped=*/true,
                                                   &sparse_linear_layer, kPath);
    // If the files don't exist on the device we're running on, just skip this
    // test and log that it was skipped.
    if (skip_test(status, layer_name)) return;

    int rows = sparse_linear_layer.rows();
    int cols = sparse_linear_layer.cols();

    MaskedLinearLayer<float> masked_linear_layer;
    status = LoadMaskedLayer<float>(layer_name, /*zipped=*/true,
                                    &masked_linear_layer, kPath);
    if (skip_test(status, layer_name)) return;
    masked_linear_layer.CastWeights<csrblocksparse::bfloat16>();

    CacheAlignedVector<float> rhs(cols);
    CacheAlignedVector<float> out_ref(rows);
    CacheAlignedVector<float> out_spmv(rows);

    rhs.FillRandom();
    out_ref.FillZero();
    out_spmv.FillZero();

    std::array<bool, 2> use_relus = {false, true};
    for (bool use_relu : use_relus) {
      masked_linear_layer.SpMM_bias(rhs, &out_ref, use_relu);
      sparse_linear_layer.SpMM_bias(rhs, &out_spmv, use_relu);

      CheckResult(out_ref, out_spmv, cols);
    }
  }
}

TEST(CsrBlockSparseMatrix, ModelMatrices_float) {
  std::vector<std::string> names = {
      "768_512_95_4x4_wavernn_gru_", "768_512_95_4x4_coarseproj_",
      "768_512_95_4x4_coarselogit_", "768_512_95_4x4_fineproj_",
      "768_512_95_4x4_finelogit_",   "lyra_conv1d_"};
  const std::string kPath =
#if defined __arm__ || defined __aarch64__
      "/data/local/tmp/";
#else
      (ghc::filesystem::current_path() / kTestdataPath).string();
#endif
  for (auto& layer_name : names) {
    SparseLinearLayer<float, float> sparse_linear_layer;
    auto status = LoadSparseLayer<float, float>(layer_name, /*zipped=*/true,
                                                &sparse_linear_layer, kPath);
    // If the files don't exist on the device we're running on, just skip this
    // test and log that it was skipped.
    if (skip_test(status, layer_name)) return;

    int rows = sparse_linear_layer.rows();
    int cols = sparse_linear_layer.cols();

    MaskedLinearLayer<float> masked_linear_layer;
    status = LoadMaskedLayer<float>(layer_name, /*zipped=*/true,
                                    &masked_linear_layer, kPath);
    if (skip_test(status, layer_name)) return;

    CacheAlignedVector<float> rhs(cols);
    CacheAlignedVector<float> out_ref(rows);
    CacheAlignedVector<float> out_spmv(rows);

    rhs.FillRandom();
    out_ref.FillZero();
    out_spmv.FillZero();

    std::array<bool, 2> use_relus = {false, true};
    for (bool use_relu : use_relus) {
      masked_linear_layer.SpMM_bias(rhs, &out_ref, use_relu);
      sparse_linear_layer.SpMM_bias(rhs, &out_spmv, use_relu);

      CheckResult(out_ref, out_spmv, cols);
    }
  }
}

#undef SKIP_TEST

}  // namespace csrblocksparse
