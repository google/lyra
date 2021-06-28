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

#ifndef LYRA_CODEC_SPARSE_MATMUL_NUMERICS_TEST_UTILS_H_
#define LYRA_CODEC_SPARSE_MATMUL_NUMERICS_TEST_UTILS_H_

#include <cmath>
#include <limits>
#include <type_traits>

#include "gtest/gtest.h"
#include "sparse_matmul/numerics/type_utils.h"

namespace csrblocksparse {

// Computes the relative difference between two floating point numbers
// std::abs(b - a) / a. If the a is < 10 * epsilon, then use the absolute
// difference instead of the relative one.
template <typename T>
T RelDiff(T a, T b) {
  static_assert(std::is_floating_point<T>::value,
                "RelDiff should only be used on floating point types.");
  if (std::abs(a) < 600 * std::numeric_limits<T>::epsilon()) {
    return std::abs(b - a);
  }
  return std::abs((b - a) / a);
}

// Compares two CacheAlignedVectors elementwise, checks if each pair passes a
// RelDiff check.  The result of RelDiff is scaled by the log of the size of the
// column to account for increasing summation errors as the number of summands
// increases.
template <typename VectorType>
void CheckResult(const VectorType& lhs, const VectorType& rhs, int columns) {
  ASSERT_EQ(lhs.size(), rhs.size());
  float epsilon =
      1.0f /
      (1 << (MantissaBitsOf<typename VectorType::value_type>::value - 1));

  // if we're summing a large number of values, then we can relax the tolerance
  float log_scale = std::max(1.f, logf(columns));

  // The tolerance is so large because it is a relative tolerance used to test
  // numbers that are close to zero at the limit of the resolution of the
  // representation. It would probably be better to focus on an absolute
  // tolerance, based on the epsilon above.
  const float tolerance = 0.026f;
  for (int i = 0; i < lhs.size(); ++i) {
    float lhs_value = static_cast<float>(lhs.data()[i]);
    float rhs_value = static_cast<float>(rhs.data()[i]);
    // If the absolute difference is no more than the epsilon for the
    // representation, then it is OK.
    if (std::abs(lhs_value - rhs_value) <= epsilon) continue;
    float rel_diff = RelDiff(lhs_value, rhs_value) / log_scale;
    EXPECT_LT(rel_diff, tolerance) << i % columns << " " << i / columns << " "
                                   << lhs_value << " " << rhs_value;
  }
}

}  // namespace csrblocksparse

#endif  // LYRA_CODEC_SPARSE_MATMUL_NUMERICS_TEST_UTILS_H_
