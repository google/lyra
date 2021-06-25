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

#include "dsp_util.h"

#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <vector>

#include "absl/types/span.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace chromemedia {
namespace codec {
namespace {

TEST(DspUtilTest, LogSpectralDistanceTest) {
  std::vector<float> first_log_spectrum(10);
  std::iota(first_log_spectrum.begin(), first_log_spectrum.end(), 0);
  std::vector<float> second_log_spectrum(10);
  std::iota(second_log_spectrum.begin(), second_log_spectrum.end(), 1);
  const auto log_spectral_distance_or =
      LogSpectralDistance(absl::MakeConstSpan(first_log_spectrum),
                          absl::MakeConstSpan(second_log_spectrum));
  ASSERT_TRUE(log_spectral_distance_or.has_value());
  EXPECT_NEAR(log_spectral_distance_or.value(), 10.0f, 0.0001);
}

TEST(ClipToInt16Test, ClipsExtremeValues) {
  const int16_t kMaxExceeded = ClipToInt16(10000000);
  EXPECT_EQ(kMaxExceeded, std::numeric_limits<int16_t>::max());
  const int16_t kMinExceeded = ClipToInt16(-10000000);
  EXPECT_EQ(kMinExceeded, std::numeric_limits<int16_t>::min());
}

TEST(ClipToInt16Test, TruncatesDecimal) {
  const int16_t kJustAboveZero = ClipToInt16(.0001);
  EXPECT_EQ(kJustAboveZero, 0);

  const int16_t kJustBelowOne = ClipToInt16(.999);
  EXPECT_EQ(kJustBelowOne, 0);

  const int16_t kShouldTruncateNegativeDecimalToZero = ClipToInt16(-.0001);
  EXPECT_EQ(kShouldTruncateNegativeDecimalToZero, 0);
}

TEST(ClipToInt16Test, BoundaryIdentity) {
  const int16_t kZero = ClipToInt16(0);
  EXPECT_EQ(kZero, 0);

  const int16_t kMaxBoundary = ClipToInt16(std::numeric_limits<int16_t>::max());
  EXPECT_EQ(kMaxBoundary, std::numeric_limits<int16_t>::max());

  const int16_t kMinBoundary = ClipToInt16(std::numeric_limits<int16_t>::min());
  EXPECT_EQ(kMinBoundary, std::numeric_limits<int16_t>::min());
}

TEST(UnitFloatToInt16ScalarTest, ExtremeValues) {
  const int16_t kMaxExceeded = UnitFloatToInt16Scalar(100000.0);
  EXPECT_EQ(kMaxExceeded, std::numeric_limits<int16_t>::max());

  const int16_t kMinExceeded = UnitFloatToInt16Scalar(-100000.0);
  EXPECT_EQ(kMinExceeded, std::numeric_limits<int16_t>::min());
}

TEST(UnitFloatToInt16ScalarTest, RoundsTowardsZero) {
  const int16_t kShouldRoundDownToZero = UnitFloatToInt16Scalar(1e-10);
  EXPECT_EQ(kShouldRoundDownToZero, 0);

  const int16_t kShouldRoundNegativeUpToZero = UnitFloatToInt16Scalar(-1e-10);
  EXPECT_EQ(kShouldRoundNegativeUpToZero, 0);
}

TEST(UnitFloatToInt16ScalarTest, BoundariesMapToLimits) {
  const int16_t kZero = UnitFloatToInt16Scalar(0.0);
  EXPECT_EQ(kZero, 0);

  const int16_t kMaxBoundary = UnitFloatToInt16Scalar(1.0);
  EXPECT_EQ(kMaxBoundary, std::numeric_limits<int16_t>::max());

  const int16_t kMinBoundary = UnitFloatToInt16Scalar(-1.0);
  EXPECT_EQ(kMinBoundary, std::numeric_limits<int16_t>::min());
}

// Pair of input and output types to be tested for casting and their
// relevant data.
template <typename I, typename O>
struct InputOutputTypes {
  using InputType = I;
  using OutputType = O;

  // The range of values that are safe to put in the input vector vector
  // without clipping.
  static float InputAbsMax() {
    if constexpr (csrblocksparse::IsFixedType<InputType>::value) {
      return (1 << InputType::kExponentBits) - 0.5;
    } else {
      return 10.0f;
    }
  }

  // For fixed point types, set the tolerance to be the resolution. For floats,
  // use a constant 1e-7f.
  static float Tolerance() {
    if constexpr (csrblocksparse::IsFixedType<InputType>::value) {
      return 1.0f / (1 << OutputType::kMantissaBits);
    } else {
      return 1e-7f;
    }
  }
};

template <typename InputOutputTypes>
class CastVectorTest : public ::testing::Test {
 public:
  CastVectorTest()
      : input_(kNumElements),
        output_(kNumElements),
        expected_output_(kNumElements),
        tolerance_(InputOutputTypes::Tolerance()) {
    input_.FillRandom(-InputOutputTypes::InputAbsMax(),
                      InputOutputTypes::InputAbsMax());
    for (int i = 0; i < kNumElements; ++i) {
      expected_output_[i] =
          static_cast<float>(static_cast<typename InputOutputTypes::OutputType>(
              static_cast<float>(input_[i])));
    }
  }

 protected:
  static constexpr int kNumElements = 11;

  csrblocksparse::CacheAlignedVector<typename InputOutputTypes::InputType>
      input_;
  csrblocksparse::CacheAlignedVector<typename InputOutputTypes::OutputType>
      output_;
  std::vector<float> expected_output_;
  const float tolerance_;
};

using CastVectorTypes = ::testing::Types<
    // fixed16 --> fixed16: casting to equal, more, and fewer exponent bits.
    InputOutputTypes<csrblocksparse::fixed16<4>, csrblocksparse::fixed16<4>>,
    InputOutputTypes<csrblocksparse::fixed16<5>, csrblocksparse::fixed16<4>>,
    InputOutputTypes<csrblocksparse::fixed16<3>, csrblocksparse::fixed16<4>>,
    // fixed32 --> fixed32: casting to equal, more, and fewer exponent bits.
    InputOutputTypes<csrblocksparse::fixed32<8>, csrblocksparse::fixed32<8>>,
    InputOutputTypes<csrblocksparse::fixed32<11>, csrblocksparse::fixed32<8>>,
    InputOutputTypes<csrblocksparse::fixed32<5>, csrblocksparse::fixed32<8>>,
    // fixed32 --> fixed16: {s < 0, s == 0, 0 < s <=16, s > 16}, where |s| is
    // the shift amount: 16 + |output exponent bits| - |input exponent bits|.
    InputOutputTypes<csrblocksparse::fixed32<22>, csrblocksparse::fixed16<4>>,
    InputOutputTypes<csrblocksparse::fixed32<20>, csrblocksparse::fixed16<4>>,
    InputOutputTypes<csrblocksparse::fixed32<5>, csrblocksparse::fixed16<4>>,
    InputOutputTypes<csrblocksparse::fixed32<3>, csrblocksparse::fixed16<4>>,
    // float --> float.
    InputOutputTypes<float, float>>;
TYPED_TEST_SUITE(CastVectorTest, CastVectorTypes);

TYPED_TEST(CastVectorTest, ResultsCloseToCastingThroughFloat) {
  CastVector(0, this->kNumElements, this->input_.data(), this->output_.data());
  EXPECT_THAT(std::vector<float>(this->output_.begin(), this->output_.end()),
              testing::Pointwise(testing::FloatNear(this->tolerance_),
                                 this->expected_output_));
}

}  // namespace
}  // namespace codec
}  // namespace chromemedia
