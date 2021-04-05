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

// Tests for sparse_inference_matrixvector library.
// This can be useful for checking runtime ABI compatibility on a unit.
#include "sparse_inference_matrixvector.h"

#include "gtest/gtest.h"

namespace chromemedia {
namespace codec {
namespace {

TEST(SparseInferenceTest, ScalarSample) {
  // The vector size must be a multiple of 8.
  constexpr int kOutputBins = 96;
  // The scratch size must be at least as big as the vector size.
  csrblocksparse::CacheAlignedVector<float> scratch_space(kOutputBins);

  const std::minstd_rand::result_type kSeed = 42;
  std::minstd_rand gen{std::minstd_rand(kSeed)};


  csrblocksparse::CacheAlignedVector<float> mat =
      csrblocksparse::CacheAlignedVector<float>(kOutputBins);
  mat.ScalarSample(1.0, &gen, &scratch_space);
  // If we've reached this point, there hasn't been anything so mismatched
  // as to cause a segfault.
  // This is useful for testing our library ABI.
  SUCCEED();
}

}  // namespace
}  // namespace codec
}  // namespace chromemedia
