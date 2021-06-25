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

#include "sparse_matmul/compute/gru_gates.h"

#include <cstdint>
#include <cstring>
#include <numeric>

#include "absl/memory/memory.h"
#include "absl/types/span.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace {

using csrblocksparse::ARInputsMode;

template <typename GRUStateType, typename InputType, typename SampleType = void,
          csrblocksparse::ARInputsMode kInputsMode, bool kSplitGates>
csrblocksparse::CacheAlignedVector<GRUStateType> TestGruGates() {
  using SampleWeightType = float;
  constexpr int kStateSize = 16;
  csrblocksparse::CacheAlignedVector<SampleWeightType> qr(6 * kStateSize);
  csrblocksparse::CacheAlignedVector<SampleWeightType> w(3 * kStateSize);
  csrblocksparse::CacheAlignedVector<InputType> gru_gates(3 * kStateSize);
  csrblocksparse::CacheAlignedVector<InputType> gru_other_gates(3 * kStateSize);
  csrblocksparse::CacheAlignedVector<InputType> conditioning(3 * kStateSize);
  csrblocksparse::CacheAlignedVector<GRUStateType> gru_h(kStateSize);
  csrblocksparse::GruGates<GRUStateType, InputType, SampleType> gru_gates_impl;
  const SampleType kCoarseAtSMinus1(0.03f);
  const SampleType kFineAtSMinus1(0.07f);
  const SampleType kCoarseAtS(-0.02f);

  qr.FillOnes();
  w.FillOnes();
  gru_gates.FillRandom();
  gru_other_gates.FillRandom();
  conditioning.FillRandom();
  gru_h.FillZero();

  gru_gates_impl.template GruWithARInput<kInputsMode, kSplitGates>(
      /*start=*/0, /*end=*/kStateSize, kStateSize, gru_gates.data(),
      conditioning.data(), gru_h.data(), &kCoarseAtSMinus1, &kFineAtSMinus1,
      qr.data(),
      /*num_replicas=*/1, /*replica_stride=*/0, &kCoarseAtS, w.data(),
      gru_other_gates.data());
  return gru_h;
}

TEST(GruGates, FloatWaveRNNCoarseMatchesGolden) {
  // If the RNG in csrblocksparse::CacheAlignedVector changes, these numbers
  // will also need to change.
  const std::vector<float> kGoldenValues = {
      0.0f, 0.0f, 0.0f,   0.0f, 1.0f, 0.746f, 0.0f, 0.0f,
      0.0f, 0.0f, 0.970f, 0.0f, 0.0f, 1.0f,   0.0f, -0.993f};
  csrblocksparse::CacheAlignedVector<float> gru_h =
      TestGruGates<float, float, float, ARInputsMode::k2ARInputs,
                   /*kSplitGates=*/true>();

  ASSERT_EQ(kGoldenValues.size(), gru_h.size());
  for (int i = 0; i < gru_h.size(); ++i) {
    EXPECT_NEAR(kGoldenValues[i], gru_h[i], 1e-3) << "i=" << i;
  }
}

TEST(GruGates, FloatWaveRNNFineMatchesGolden) {
  // If the RNG in csrblocksparse::CacheAlignedVector changes, these numbers
  // will also need to change.
  const std::vector<float> kGoldenValues = {
      0.0f, 0.0f, 0.0f,   0.0f, 1.0f, 0.737f, 0.0f, 0.0f,
      0.0f, 0.0f, 0.969f, 0.0f, 0.0f, 1.0f,   0.0f, -0.994f};
  csrblocksparse::CacheAlignedVector<float> gru_h =
      TestGruGates<float, float, float, ARInputsMode::k3ARInputs,
                   /*kSplitGates=*/true>();

  ASSERT_EQ(kGoldenValues.size(), gru_h.size());
  for (int i = 0; i < gru_h.size(); ++i) {
    EXPECT_NEAR(kGoldenValues[i], gru_h[i], 1e-3) << "i=" << i;
  }
}

TEST(GruGates, FloatTwoArInputsNonSplitGateMatchesGolden) {
  // If the RNG in csrblocksparse::CacheAlignedVector changes, these numbers
  // will also need to change.
  const std::vector<float> kGoldenValues = {
      0.0f, 0.0f, 0.0f,   0.0f, 1.0f, 0.714f, 0.0f, -0.002f,
      0.0f, 0.0f, 0.970f, 0.0f, 0.0f, 1.0f,   0.0f, -0.965f};
  csrblocksparse::CacheAlignedVector<float> gru_h =
      TestGruGates<float, float, float, ARInputsMode::k2ARInputs,
                   /*kSplitGates=*/false>();

  ASSERT_EQ(kGoldenValues.size(), gru_h.size());
  for (int i = 0; i < gru_h.size(); ++i) {
    EXPECT_NEAR(kGoldenValues[i], gru_h[i], 1e-3) << "i=" << i;
  }
}

TEST(GruGates, FixedWaveRNNCoarseMatchesFloat) {
  using GRUMatMulOutType = csrblocksparse::fixed32<11>;
  using GRUStateType = csrblocksparse::fixed16<2>;
  using SampleType = csrblocksparse::fixed16<0>;
  csrblocksparse::CacheAlignedVector<float> float_gru_h =
      TestGruGates<float, float, float, ARInputsMode::k2ARInputs,
                   /*kSplitGates=*/true>();
  csrblocksparse::CacheAlignedVector<GRUStateType> fixed_gru_h =
      TestGruGates<GRUStateType, GRUMatMulOutType, SampleType,
                   ARInputsMode::k2ARInputs, /*kSplitGates=*/true>();

  ASSERT_EQ(float_gru_h.size(), fixed_gru_h.size());
  for (int i = 0; i < fixed_gru_h.size(); ++i) {
    EXPECT_NEAR(float_gru_h[i], static_cast<float>(fixed_gru_h[i]), 1e-3)
        << "i=" << i;
  }
}

TEST(GruGates, FixedWaveRNNFineMatchesFloat) {
  using GRUMatMulOutType = csrblocksparse::fixed32<11>;
  using GRUStateType = csrblocksparse::fixed16<2>;
  using SampleType = csrblocksparse::fixed16<0>;
  csrblocksparse::CacheAlignedVector<float> float_gru_h =
      TestGruGates<float, float, float, ARInputsMode::k3ARInputs,
                   /*kSplitGates=*/true>();
  csrblocksparse::CacheAlignedVector<GRUStateType> fixed_gru_h =
      TestGruGates<GRUStateType, GRUMatMulOutType, SampleType,
                   ARInputsMode::k3ARInputs, /*kSplitGates=*/true>();

  ASSERT_EQ(float_gru_h.size(), fixed_gru_h.size());
  for (int i = 0; i < fixed_gru_h.size(); ++i) {
    EXPECT_NEAR(float_gru_h[i], static_cast<float>(fixed_gru_h[i]), 1e-3)
        << "i=" << i;
  }
}

TEST(GruGates, FixedTwoArInputsNonSplitGateMatchesFloat) {
  using GRUMatMulOutType = csrblocksparse::fixed32<11>;
  using GRUStateType = csrblocksparse::fixed16<2>;
  using SampleType = csrblocksparse::fixed16<0>;
  csrblocksparse::CacheAlignedVector<float> float_gru_h =
      TestGruGates<float, float, float, ARInputsMode::k2ARInputs,
                   /*kSplitGates=*/false>();
  csrblocksparse::CacheAlignedVector<GRUStateType> fixed_gru_h =
      TestGruGates<GRUStateType, GRUMatMulOutType, SampleType,
                   ARInputsMode::k2ARInputs, /*kSplitGates=*/false>();

  ASSERT_EQ(float_gru_h.size(), fixed_gru_h.size());
  for (int i = 0; i < fixed_gru_h.size(); ++i) {
    EXPECT_NEAR(float_gru_h[i], static_cast<float>(fixed_gru_h[i]), 1e-3)
        << "i=" << i;
  }
}

}  // namespace
