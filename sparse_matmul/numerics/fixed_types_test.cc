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

#include "sparse_matmul/numerics/fixed_types.h"

#include <cstdint>

#include "gtest/gtest.h"
#include "sparse_matmul/numerics/test_utils.h"
#include "sparse_matmul/numerics/type_utils.h"

namespace csrblocksparse {

// Basic test that makes sure basic multiplication and TypeOfProduct work
// correctly.
TEST(FixedPoint, Multiplication) {
  fixed16<4> a(.1f);
  fixed16<4> b(1.f);

  TypeOfProduct<fixed16<4>, fixed16<4>>::type c(a.raw_val() * b.raw_val());

  EXPECT_NEAR(static_cast<float>(c), .1f,
              1. / (1 << fixed16<2>::kMantissaBits));
}

TEST(FixedPoint, SafeCastingIntMax) {
  const float int_max_float = std::numeric_limits<int32_t>::max();
  const csrblocksparse::fixed32<31> int_max_fixed(int_max_float);
  EXPECT_FLOAT_EQ(int_max_float, static_cast<float>(int_max_fixed));
}

}  // namespace csrblocksparse
