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

#include "lyra/dsp_utils.h"

#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

#include "absl/types/span.h"
#include "gtest/gtest.h"

namespace chromemedia {
namespace codec {
namespace {

TEST(DspUtilTest, LogSpectralDistanceTest) {
  std::vector<float> first_log_spectrum(10);
  std::iota(first_log_spectrum.begin(), first_log_spectrum.end(), 0);
  std::vector<float> second_log_spectrum(10);
  std::iota(second_log_spectrum.begin(), second_log_spectrum.end(), 1);
  const auto log_spectral_distance =
      LogSpectralDistance(absl::MakeConstSpan(first_log_spectrum),
                          absl::MakeConstSpan(second_log_spectrum));
  ASSERT_TRUE(log_spectral_distance.has_value());
  EXPECT_NEAR(log_spectral_distance.value(), 10.0f, 0.0001);
}

using FloatingPointTypes = testing::Types<float, double>;
template <typename T>
class ConversionTest : public ::testing::Test {};
TYPED_TEST_SUITE(ConversionTest, FloatingPointTypes);

TYPED_TEST(ConversionTest, ClipToInt16ScalarTestClipsExtremeValues) {
  const int16_t kMaxExceeded =
      ClipToInt16Scalar(static_cast<TypeParam>(10000000));
  EXPECT_EQ(kMaxExceeded, std::numeric_limits<int16_t>::max());
  const int16_t kMinExceeded =
      ClipToInt16Scalar(static_cast<TypeParam>(-10000000));
  EXPECT_EQ(kMinExceeded, std::numeric_limits<int16_t>::min());
}

TYPED_TEST(ConversionTest, ClipToInt16ScalarTestTruncatesDecimal) {
  const int16_t kJustAboveZero =
      ClipToInt16Scalar(static_cast<TypeParam>(.0001));
  EXPECT_EQ(kJustAboveZero, 0);

  const int16_t kJustBelowOne = ClipToInt16Scalar(static_cast<TypeParam>(.999));
  EXPECT_EQ(kJustBelowOne, 0);

  const int16_t kShouldTruncateNegativeDecimalToZero =
      ClipToInt16Scalar(static_cast<TypeParam>(-.0001));
  EXPECT_EQ(kShouldTruncateNegativeDecimalToZero, 0);
}

TYPED_TEST(ConversionTest, ClipToInt16ScalarTestBoundaryIdentity) {
  const int16_t kZero = ClipToInt16Scalar(static_cast<TypeParam>(0));
  EXPECT_EQ(kZero, 0);

  const int16_t kMaxBoundary = ClipToInt16Scalar(
      static_cast<TypeParam>(std::numeric_limits<int16_t>::max()));
  EXPECT_EQ(kMaxBoundary, std::numeric_limits<int16_t>::max());

  const int16_t kMinBoundary = ClipToInt16Scalar(
      static_cast<TypeParam>(std::numeric_limits<int16_t>::min()));
  EXPECT_EQ(kMinBoundary, std::numeric_limits<int16_t>::min());
}

TYPED_TEST(ConversionTest, UnitToInt16ScalarTestExtremeValues) {
  const int16_t kMaxExceeded =
      UnitToInt16Scalar(static_cast<TypeParam>(100000.0));
  EXPECT_EQ(kMaxExceeded, std::numeric_limits<int16_t>::max());

  const int16_t kMinExceeded =
      UnitToInt16Scalar(static_cast<TypeParam>(-100000.0));
  EXPECT_EQ(kMinExceeded, std::numeric_limits<int16_t>::min());
}

TYPED_TEST(ConversionTest, UnitToInt16ScalarTestRoundsTowardsZero) {
  const int16_t kShouldRoundDownToZero =
      UnitToInt16Scalar(static_cast<TypeParam>(1e-10));
  EXPECT_EQ(kShouldRoundDownToZero, 0);

  const int16_t kShouldRoundNegativeUpToZero =
      UnitToInt16Scalar(static_cast<TypeParam>(-1e-10));
  EXPECT_EQ(kShouldRoundNegativeUpToZero, 0);
}

TYPED_TEST(ConversionTest, UnitToInt16ScalarTestBoundariesMapToLimits) {
  const int16_t kZero = UnitToInt16Scalar(static_cast<TypeParam>(0.0));
  EXPECT_EQ(kZero, 0);

  const int16_t kMaxBoundary = UnitToInt16Scalar(static_cast<TypeParam>(1.0));
  EXPECT_EQ(kMaxBoundary, std::numeric_limits<int16_t>::max());

  const int16_t kMinBoundary = UnitToInt16Scalar(static_cast<TypeParam>(-1.0));
  EXPECT_EQ(kMinBoundary, std::numeric_limits<int16_t>::min());
}

TYPED_TEST(ConversionTest, Int16ToUnitScalarTestBoundariesMapToLimits) {
  const TypeParam kZero = Int16ToUnitScalar<TypeParam>(0);
  EXPECT_EQ(kZero, 0);

  const TypeParam kMaxBoundary =
      Int16ToUnitScalar<TypeParam>(std::numeric_limits<int16_t>::max());
  const TypeParam kStep = Int16ToUnitScalar<TypeParam>(1);
  EXPECT_EQ(kMaxBoundary + kStep, 1.0);

  const TypeParam kMinBoundary =
      Int16ToUnitScalar<TypeParam>(std::numeric_limits<int16_t>::min());
  EXPECT_EQ(kMinBoundary, -1.0);
}

}  // namespace
}  // namespace codec
}  // namespace chromemedia
