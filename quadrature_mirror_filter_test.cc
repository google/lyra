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

#include "quadrature_mirror_filter.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include "gtest/gtest.h"

namespace chromemedia::codec {
namespace {

constexpr int kNumBands = 2;
constexpr int kNumBandSamples = 320;
constexpr int kNumSignalSamples = kNumBands * kNumBandSamples;

template <typename T>
double MaxFactor();
template <>
double MaxFactor<int16_t>() {
  return std::pow(std::numeric_limits<int16_t>().max(), 2.f);
}
template <>
double MaxFactor<float>() {
  return 1;
}

struct MaxCorrelation {
  float correlation;
  int delay;
};

template <typename T>
MaxCorrelation ComputeMaxCorrelation(const std::vector<T>& a,
                                     const std::vector<T>& b) {
  float max_correlation = 0.f;
  int max_delay = 0;
  const auto update_max = [&](float correlation, int delay) {
    if (correlation > max_correlation) {
      max_correlation = correlation;
      max_delay = delay;
    }
  };
  for (int delay = 0; delay < a.size(); ++delay) {
    float a_correlation = 0.f;
    float b_correlation = 0.f;
    for (int i = 0; i < a.size() - delay; ++i) {
      a_correlation +=
          static_cast<float>(a[i]) * static_cast<float>(b[i + delay]);
      b_correlation +=
          static_cast<float>(a[i + delay]) * static_cast<float>(b[i]);
    }
    a_correlation /= a.size() * MaxFactor<T>();
    b_correlation /= a.size() * MaxFactor<T>();
    update_max(a_correlation, delay);
    update_max(b_correlation, delay);
  }
  return {.correlation = max_correlation, .delay = max_delay};
}

void PopulateWithNoise(std::vector<int16_t>* signal) {
  std::mt19937 generator;
  std::uniform_int_distribution<> distribution(
      std::numeric_limits<int16_t>().min(),
      std::numeric_limits<int16_t>().max());
  for (int i = 0; i < signal->size(); ++i) {
    (*signal)[i] = distribution(generator);
  }
}

void PopulateWithNoise(std::vector<float>* signal) {
  std::mt19937 generator;
  std::uniform_real_distribution<float> distribution(-1.f, 1.f);
  for (int i = 0; i < signal->size(); ++i) {
    (*signal)[i] = distribution(generator);
  }
}

template <typename T>
class QuadratureMirrorFiltersTest : public ::testing::Test {};
using QuadratureMirrorFiltersTypes = ::testing::Types<int16_t, float>;
TYPED_TEST_SUITE(QuadratureMirrorFiltersTest, QuadratureMirrorFiltersTypes);

TYPED_TEST(QuadratureMirrorFiltersTest, OddSignalSize) {
  const std::vector<TypeParam> signal(kNumSignalSamples + 1);
  SplitQuadratureMirrorFilter<TypeParam> split_filter;

  EXPECT_DEATH(split_filter.Split(signal), "");
}

TYPED_TEST(QuadratureMirrorFiltersTest, InvalidLowBandSize) {
  Bands<TypeParam> bands(kNumBandSamples);
  bands.low_band.resize(kNumBands + 1);
  MergeQuadratureMirrorFilter<TypeParam> merge_filter;

  EXPECT_DEATH(merge_filter.Merge(bands), "");
}

TYPED_TEST(QuadratureMirrorFiltersTest, InvalidHighBandSize) {
  Bands<TypeParam> bands(kNumBandSamples);
  bands.high_band.resize(kNumBands + 1);
  MergeQuadratureMirrorFilter<TypeParam> merge_filter;

  EXPECT_DEATH(merge_filter.Merge(bands), "");
}

TYPED_TEST(QuadratureMirrorFiltersTest, NoSignal) {
  const std::vector<TypeParam> signal(kNumSignalSamples, 0);
  SplitQuadratureMirrorFilter<TypeParam> split_filter;
  MergeQuadratureMirrorFilter<TypeParam> merge_filter;

  const Bands<TypeParam> bands = split_filter.Split(signal);
  EXPECT_EQ(kNumBandSamples, bands.num_samples_per_band);
  EXPECT_FLOAT_EQ(
      0.f, ComputeMaxCorrelation(bands.low_band, bands.low_band).correlation);
  EXPECT_FLOAT_EQ(
      0.f, ComputeMaxCorrelation(bands.high_band, bands.high_band).correlation);

  const std::vector<TypeParam> merged_signal = merge_filter.Merge(bands);
  EXPECT_EQ(kNumSignalSamples, merged_signal.size());
  EXPECT_FLOAT_EQ(
      0.f, ComputeMaxCorrelation(merged_signal, merged_signal).correlation);
}

TYPED_TEST(QuadratureMirrorFiltersTest, Noise) {
  constexpr float kMinCorrelation = 0.2f;
  std::vector<TypeParam> signal(kNumSignalSamples);
  PopulateWithNoise(&signal);
  SplitQuadratureMirrorFilter<TypeParam> split_filter;
  MergeQuadratureMirrorFilter<TypeParam> merge_filter;

  const Bands<TypeParam> bands = split_filter.Split(signal);
  EXPECT_EQ(kNumBandSamples, bands.num_samples_per_band);
  EXPECT_LT(kMinCorrelation / kNumBands,
            ComputeMaxCorrelation(bands.low_band, bands.low_band).correlation);
  EXPECT_LT(
      kMinCorrelation / kNumBands,
      ComputeMaxCorrelation(bands.high_band, bands.high_band).correlation);

  const std::vector<TypeParam> merged_signal = merge_filter.Merge(bands);
  EXPECT_EQ(kNumSignalSamples, merged_signal.size());
  EXPECT_LT(kMinCorrelation,
            ComputeMaxCorrelation(merged_signal, merged_signal).correlation);
}

class QuadratureMirrorFiltersSineTest : public testing::TestWithParam<int> {};

TEST_P(QuadratureMirrorFiltersSineTest, Sine) {
  constexpr float kTolerance = 0.03;
  const int kSineBand = GetParam();

  std::vector<int16_t> signal(kNumSignalSamples);
  for (int i = 0; i < signal.size(); ++i) {
    signal[i] =
        std::numeric_limits<int16_t>().max() *
        std::sin(M_PI * (2.f * kSineBand + 1.f) * i / (2.f * kNumBands));
  }
  SplitQuadratureMirrorFilter<int16_t> split_filter;
  MergeQuadratureMirrorFilter<int16_t> merge_filter;

  const Bands<int16_t> bands = split_filter.Split(signal);
  EXPECT_EQ(kNumBandSamples, bands.num_samples_per_band);
  const auto low_max_correlation =
      ComputeMaxCorrelation(bands.low_band, bands.low_band);
  const auto high_max_correlation =
      ComputeMaxCorrelation(bands.high_band, bands.high_band);
  EXPECT_NEAR(kSineBand == 0 ? 0.5f : 0.f, low_max_correlation.correlation,
              kTolerance);
  EXPECT_NEAR(kSineBand == 1 ? 0.5f : 0.f, high_max_correlation.correlation,
              kTolerance);

  const std::vector<int16_t> merged_signal = merge_filter.Merge(bands);
  EXPECT_EQ(kNumSignalSamples, merged_signal.size());
  const auto merged_max_correlation =
      ComputeMaxCorrelation(signal, merged_signal);
  EXPECT_NEAR(0.5f, merged_max_correlation.correlation, kTolerance);
  EXPECT_LE(merged_max_correlation.delay, 4);
}

INSTANTIATE_TEST_SUITE_P(DifferentFrequencies, QuadratureMirrorFiltersSineTest,
                         testing::Range(0, kNumBands));

}  // namespace
}  // namespace chromemedia::codec
