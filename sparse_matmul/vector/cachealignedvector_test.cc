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

#include "sparse_matmul/vector/cache_aligned_vector.h"

#if defined __aarch64__
#include <arm_neon.h>
#endif

#include <stdio.h>

#include <array>
#include <cmath>
#include <random>
#include <tuple>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "sparse_matmul/numerics/test_utils.h"
#include "sparse_matmul/os/coop_threads.h"

namespace csrblocksparse {

const float kExpRelTolerance = .03f;  // 3% relative
#ifdef SIGMOID_AS_TANH
const float kSigmoidRelTolerance = .09f;  // 9.0% relative
const float kSigmoidAbsTolerance = .003f;
#else
const float kSigmoidRelTolerance = .031f;  // 3.1% relative
const float kSigmoidAbsTolerance = .006f;
#endif
const float kTanhRelTolerance = .014f;  // 1.4% relative
const float kTanhAbsTolerance = .00525f;

TEST(Transcendentals, CacheAlignedVectorExp) {
  const int kTestSize = 1 << 16;
  CacheAlignedVector<float> values(kTestSize);
  values.FillRandom();
  CacheAlignedVector<float> values_ref = values;

  values.Exp();
  for (int i = 0; i < kTestSize; ++i) {
    float exact_val = std::exp(values_ref[i]);
    float rel_diff = RelDiff(exact_val, values[i]);

    EXPECT_LT(rel_diff, kExpRelTolerance)
        << exact_val << " " << values[i] << " " << values_ref[i];
  }
}

TEST(Transcendentals, CacheAlignedVectorSigmoid) {
  const int kTestSize = 1 << 16;
  CacheAlignedVector<float> values(kTestSize);
  values.FillRandom();
  CacheAlignedVector<float> values_ref = values;

  values.Sigmoid();
  for (int i = 0; i < kTestSize; ++i) {
    float exact_val = 1. / (1. + std::exp(-values_ref[i]));
    float rel_diff = RelDiff(exact_val, values[i]);

    EXPECT_LT(rel_diff, kSigmoidRelTolerance)
        << exact_val << " " << values[i] << " " << values_ref[i];
    EXPECT_NEAR(values[i], exact_val, kSigmoidAbsTolerance) << values_ref[i];
  }
}

TEST(Transcendentals, CacheAlignedVectorTanh) {
  const int kTestSize = 1 << 16;
  CacheAlignedVector<float> values(kTestSize);
  values.FillRandom();
  CacheAlignedVector<float> values_ref = values;

  values.Tanh();
  for (int i = 0; i < kTestSize; ++i) {
    float exact_val = std::tanh(values_ref[i]);
    float rel_diff = RelDiff(exact_val, values[i]);

    EXPECT_LT(rel_diff, kTanhRelTolerance)
        << exact_val << " " << values[i] << " " << values_ref[i];
    EXPECT_NEAR(values[i], exact_val, kTanhAbsTolerance) << values_ref[i];
  }
}

// Uniformly sample logits and check that the resulting sample choices are
// also (nearly) uniformly distributed.
TEST(Sampling, Random) {
  const int kSize = 256;

  CacheAlignedVector<float> logits(kSize);
  logits.FillZero();

  double histogram[kSize] = {};

  const int kIterations = 10000;
  for (int i = 0; i < kIterations; ++i) {
    histogram[logits.Sample()]++;
  }

  for (int i = 0; i < kSize; ++i) {
    // .002 is an empirical bound
    EXPECT_GT(histogram[i] / kIterations, 1. / kSize - .002f);
    EXPECT_LT(histogram[i] / kIterations, 1. / kSize + .002f);
  }
}

// Put (nearly) all the probability mass on one bin and make sure only that bin
// is chosen.
TEST(Sampling, FixedDistribution) {
  const int kSize = 256;

  CacheAlignedVector<float> logits(kSize);

  int histogram[kSize] = {};

  const int kIterations = 1000;
  const int kIndex = 3;
  const int kAllProbabilityMass = 10;
  const int kNoProbabilityMass = -10;
  for (int i = 0; i < kIterations; ++i) {
    for (int i = 1; i <= kSize; ++i) {
      logits.data()[i - 1] =
          i == (kIndex + 1) ? kAllProbabilityMass : kNoProbabilityMass;
    }

    histogram[logits.Sample()]++;
  }

  EXPECT_EQ(histogram[kIndex], 1000);
}

// Put (nearly) all the probability mass on one bin outside the target range,
// and make sure that bin is not chosen.
TEST(ScalarSample, ThreadedMasked) {
  const int kSize = 256;
  const int mindex = 2;
  const int maxdex = 3;
  const int kNumThreads = 4;
  const int kIterations = 1000;
  const int kIndex = 3;
  const int kMostProbabilityMass = 3;
  const int kLittleProbabilityMass = -3;

  CacheAlignedVector<float> logits(kSize);
  std::vector<CacheAlignedVector<float>> tmp_vectors;
  std::vector<std::minstd_rand> generators(kNumThreads);

  for (int i = 0; i < kNumThreads; ++i) {
    tmp_vectors.emplace_back(kSize);
  }

  for (int i = 0; i < kSize; ++i) {
    logits.data()[i] =
        (i + 1) == (kIndex + 1) ? kMostProbabilityMass : kLittleProbabilityMass;
  }

  std::vector<std::vector<int>> histograms;
  for (int i = 0; i < kNumThreads; ++i) {
    histograms.emplace_back(kSize);
  }

  auto f = [&](csrblocksparse::SpinBarrier* /*barrier*/, int tid) {
    for (int i = 0; i < kIterations; ++i) {
      histograms[tid][logits.ScalarSample(
          1.f, &generators[tid], &tmp_vectors[tid], 0, mindex, maxdex)]++;
    }
  };

  csrblocksparse::LaunchOnThreadsWithBarrier(kNumThreads, f);

  // Every thread should generate the exact same set of samples.
  for (int i = 0; i < kSize; ++i) {
    int val = histograms[0][i];
    for (int tid = 1; tid < kNumThreads; ++tid) {
      EXPECT_EQ(val, histograms[tid][i]);
    }
  }

  // The most probable sample should be the only one we're sampling.
  for (int tid = 0; tid < kNumThreads; ++tid) {
    EXPECT_EQ(std::distance(histograms[tid].begin(),
                            std::max_element(histograms[tid].begin(),
                                             histograms[tid].end())),
              mindex);
  }
}

TEST(Sampling, Threaded) {
  const int kSize = 256;
  const int kNumThreads = 4;
  const int kIterations = 1000;
  const int kIndex = 3;
  const int kMostProbabilityMass = 3;
  const int kLittleProbabilityMass = -3;

  CacheAlignedVector<float> logits(kSize);
  std::vector<CacheAlignedVector<float>> tmp_vectors;
  std::vector<std::minstd_rand> generators(kNumThreads);

  for (int i = 0; i < kNumThreads; ++i) {
    tmp_vectors.emplace_back(kSize);
  }

  for (int i = 0; i < kSize; ++i) {
    logits.data()[i] =
        (i + 1) == (kIndex + 1) ? kMostProbabilityMass : kLittleProbabilityMass;
  }

  std::vector<std::vector<int>> histograms;
  for (int i = 0; i < kNumThreads; ++i) {
    histograms.emplace_back(kSize);
  }

  auto f = [&](csrblocksparse::SpinBarrier* /*barrier*/, int tid) {
    for (int i = 0; i < kIterations; ++i) {
      histograms[tid]
                [logits.Sample(1.f, &generators[tid], &tmp_vectors[tid])]++;
    }
  };

  csrblocksparse::LaunchOnThreadsWithBarrier(kNumThreads, f);

  // Every thread should generate the exact same set of samples.
  for (int i = 0; i < kSize; ++i) {
    int val = histograms[0][i];
    for (int tid = 1; tid < kNumThreads; ++tid) {
      EXPECT_EQ(val, histograms[tid][i]);
    }
  }

  // The most probable sample should be the one with the most probability mass.
  for (int tid = 0; tid < kNumThreads; ++tid) {
    EXPECT_EQ(std::distance(histograms[tid].begin(),
                            std::max_element(histograms[tid].begin(),
                                             histograms[tid].end())),
              kIndex);
  }
}

void CreateVectorHelper(
    csrblocksparse::FatCacheAlignedVector<float>* fat_vector, int cols,
    int rows, std::unique_ptr<csrblocksparse::VectorView<float>>* view) {
  *view = absl::make_unique<csrblocksparse::VectorView<float>>(*fat_vector,
                                                               cols, rows);
}

void CreateVectorHelper(
    csrblocksparse::FatCacheAlignedVector<float>* fat_vector, int cols,
    int rows, std::unique_ptr<csrblocksparse::MutableVectorView<float>>* view) {
  *view = absl::make_unique<csrblocksparse::MutableVectorView<float>>(
      fat_vector, cols, rows);
}

csrblocksparse::FatCacheAlignedVector<float> CreateFatAlignedVector(int rows,
                                                                    int cols) {
  csrblocksparse::FatCacheAlignedVector<float> fat_vector(rows, cols);
  // Usage intent of FatCacheAlignedVector is that they are COLUMN MAJOR.
  float v = 0;
  for (int c = 0; c < cols; ++c) {
    for (int r = 0; r < rows; ++r) {
      fat_vector.data()[c * rows + r] = v++;
    }
  }

  return fat_vector;
}

template <typename VectorViewType>
void TestFatVectorView() {
  const int kRows = 6;
  const int kCols = 6;
  auto fat_vector = CreateFatAlignedVector(kRows, kCols);

  std::unique_ptr<VectorViewType> top;
  CreateVectorHelper(&fat_vector, 0, kRows / 2, &top);
  std::unique_ptr<VectorViewType> bottom;
  CreateVectorHelper(&fat_vector, kRows / 2, kRows / 2, &bottom);

  EXPECT_EQ(top->cols(), kCols);
  EXPECT_EQ(bottom->cols(), kCols);
  EXPECT_EQ(top->rows(), kRows / 2);
  EXPECT_EQ(bottom->rows(), kRows / 2);
  EXPECT_EQ(top->col_stride(), kRows);
  EXPECT_EQ(bottom->col_stride(), kRows);

  for (int c = 0; c < kCols; ++c) {
    for (int r = 0; r < kRows; ++r) {
      if (r < kRows / 2) {
        EXPECT_EQ(fat_vector[c * kRows + r],
                  top->data()[c * top->col_stride() + r]);
      } else {
        EXPECT_EQ(fat_vector[c * kRows + r],
                  bottom->data()[c * top->col_stride() + r - kRows / 2]);
      }
    }
  }
}

TEST(FatVector, View) {
  TestFatVectorView<csrblocksparse::VectorView<float>>();
}
TEST(FatVector, MutableView) {
  TestFatVectorView<csrblocksparse::MutableVectorView<float>>();
}

TEST(FatVector, SliceMutableView) {
  const int kRows = 6;
  const int kCols = 3;
  auto fat_vector = CreateFatAlignedVector(kRows, kCols);

  int c = 1;
  csrblocksparse::MutableVectorView<float> slice = fat_vector.slice(c);
  for (int r = 0; r < kRows; ++r) {
    EXPECT_EQ(slice[r], c * kRows + r);
  }
}

TEST(FatVector, SliceConstView) {
  const int kRows = 6;
  const int kCols = 3;
  auto fat_vector = CreateFatAlignedVector(kRows, kCols);

  int c = 1;
  csrblocksparse::VectorView<float> const_slice;
  {
    // Take a VectorView from a non-const slice.
    const_slice = fat_vector.slice(c);
    for (int r = 0; r < kRows; ++r) {
      EXPECT_EQ(const_slice[r], c * kRows + r);
    }
  }

  {
    // Take a VectorView from a const slice.
    const auto& const_fat_vector = fat_vector;
    const_slice = const_fat_vector.slice(c);
    for (int r = 0; r < kRows; ++r) {
      EXPECT_EQ(const_slice[r], c * kRows + r);
    }
  }
}

TEST(View, FromMutableToConst) {
  const int kRows = 6;
  const int kCols = 3;
  auto fat_vector = CreateFatAlignedVector(kRows, kCols);
  csrblocksparse::MutableVectorView<float> slice = fat_vector.slice(0);

  csrblocksparse::VectorView<float> const_slice(slice);
  for (int r = 0; r < kRows; ++r) {
    EXPECT_EQ(const_slice[r], r);
  }
}

TEST(View, CopyTest) {
  const int kRows = 6;
  const int kCols = 3;
  auto fat_vector = CreateFatAlignedVector(kRows, kCols);
  csrblocksparse::MutableVectorView<float> slice = fat_vector.slice(0);
  csrblocksparse::MutableVectorView<float> slice2(slice);

  for (int r = 0; r < kRows; ++r) {
    EXPECT_EQ(slice2[r], r);
  }
}

TEST(Vector, CopyNull) {
  // Check that we can copy a vector with a null generator without segfault.
  CacheAlignedVector<float> foo((CacheAlignedVector<float>()));
  // This is here to prevent foo from being optimized out.
  CHECK_EQ(foo.size(), 0);
  CacheAlignedVector<float> foo_bar = CacheAlignedVector<float>();
  CHECK_EQ(foo_bar.size(), 0);
}

TEST(Vector, FromRawPointer) {
  std::vector<float> input;
  for (int i = 0; i < 5; ++i) {
    input.push_back(i * 2);
  }

  // Calls first constructor.
  CacheAlignedVector<float> foo(input.data(), 5);
  CHECK_EQ(foo.size(), 5);
  EXPECT_THAT(input, testing::ElementsAreArray(foo.data(), 5));

  // Calls the second constructor.
  CacheAlignedVector<double> foo2(input.data(), 5);
  CHECK_EQ(foo2.size(), 5);
  EXPECT_THAT(input, testing::ElementsAreArray(foo2.data(), 5));
}

}  // namespace csrblocksparse
