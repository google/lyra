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

#include "filter_banks.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include "gtest/gtest.h"

namespace chromemedia {
namespace codec {
namespace {

constexpr int kNumBands = 8;
constexpr int kNumBandSamples = 320;
constexpr int kNumSignalSamples = kNumBands * kNumBandSamples;

float MaxCorrelation(const std::vector<int16_t>& a,
                     const std::vector<int16_t>& b) {
  float max_correlation = 0.f;
  for (int delay = 0; delay < a.size(); ++delay) {
    float a_correlation = 0.f;
    float b_correlation = 0.f;
    for (int i = 0; i < a.size() - delay; ++i) {
      a_correlation +=
          static_cast<float>(a[i]) * static_cast<float>(b[i + delay]);
      b_correlation +=
          static_cast<float>(a[i + delay]) * static_cast<float>(b[i]);
    }
    a_correlation /=
        a.size() * std::pow(std::numeric_limits<int16_t>().max(), 2.f);
    b_correlation /=
        a.size() * std::pow(std::numeric_limits<int16_t>().max(), 2.f);
    max_correlation = std::max(max_correlation, a_correlation);
    max_correlation = std::max(max_correlation, b_correlation);
  }
  return max_correlation;
}

TEST(FilterBanksTest, InvalidNumberOfSplitBands) {
  std::unique_ptr<SplitFilter> split_filter =
      SplitFilter::Create(kNumBands + 1);
  EXPECT_EQ(nullptr, split_filter);
}

TEST(FilterBanksTest, InvalidNumberOfMergeBands) {
  std::unique_ptr<MergeFilter> merge_filter = MergeFilter::Create(-kNumBands);
  EXPECT_EQ(nullptr, merge_filter);
}

TEST(FilterBanksTest, InvalidSignalSize) {
  const std::vector<int16_t> signal(kNumSignalSamples + 1);
  std::unique_ptr<SplitFilter> split_filter = SplitFilter::Create(kNumBands);
  ASSERT_NE(nullptr, split_filter);
  EXPECT_EQ(kNumBands, split_filter->num_bands());

  EXPECT_DEATH(split_filter->Split(signal), "");
}

TEST(FilterBanksTest, DifferentNumberOfBands) {
  const std::vector<std::vector<int16_t>> bands(2 * kNumBands);
  std::unique_ptr<MergeFilter> merge_filter = MergeFilter::Create(kNumBands);
  ASSERT_NE(nullptr, merge_filter);
  EXPECT_EQ(kNumBands, merge_filter->num_bands());

  EXPECT_DEATH(merge_filter->Merge(bands), "");
}

TEST(FilterBanksTest, DifferentBandSize) {
  std::vector<std::vector<int16_t>> bands(
      kNumBands, std::vector<int16_t>(kNumBandSamples));
  bands[kNumBands - 1].resize(2 * kNumBandSamples);
  std::unique_ptr<MergeFilter> merge_filter = MergeFilter::Create(kNumBands);
  ASSERT_NE(nullptr, merge_filter);
  EXPECT_EQ(kNumBands, merge_filter->num_bands());

  EXPECT_DEATH(merge_filter->Merge(bands), "");
}

TEST(FilterBanksTest, NoSignal) {
  const std::vector<int16_t> signal(kNumSignalSamples, 0);
  std::unique_ptr<SplitFilter> split_filter = SplitFilter::Create(kNumBands);
  ASSERT_NE(nullptr, split_filter);
  EXPECT_EQ(kNumBands, split_filter->num_bands());
  std::unique_ptr<MergeFilter> merge_filter = MergeFilter::Create(kNumBands);
  ASSERT_NE(nullptr, merge_filter);
  EXPECT_EQ(kNumBands, merge_filter->num_bands());

  const std::vector<std::vector<int16_t>> bands = split_filter->Split(signal);
  EXPECT_EQ(kNumBands, bands.size());
  for (const std::vector<int16_t>& band : bands) {
    EXPECT_EQ(kNumBandSamples, band.size());
    EXPECT_FLOAT_EQ(0.f, MaxCorrelation(band, band));
  }

  const std::vector<int16_t> merged_signal = merge_filter->Merge(bands);
  EXPECT_EQ(kNumSignalSamples, merged_signal.size());
  EXPECT_FLOAT_EQ(0.f, MaxCorrelation(merged_signal, merged_signal));
}

TEST(FilterBanksTest, Noise) {
  constexpr float kMinCorrelation = 0.2f;
  std::vector<int16_t> signal(kNumSignalSamples);
  std::mt19937 generator;
  std::uniform_int_distribution<> distribution(
      std::numeric_limits<int16_t>().min(),
      std::numeric_limits<int16_t>().max());
  for (int i = 0; i < signal.size(); ++i) {
    signal[i] = distribution(generator);
  }
  std::unique_ptr<SplitFilter> split_filter = SplitFilter::Create(kNumBands);
  ASSERT_NE(nullptr, split_filter);
  EXPECT_EQ(kNumBands, split_filter->num_bands());
  std::unique_ptr<MergeFilter> merge_filter = MergeFilter::Create(kNumBands);
  ASSERT_NE(nullptr, merge_filter);
  EXPECT_EQ(kNumBands, merge_filter->num_bands());

  const std::vector<std::vector<int16_t>> bands = split_filter->Split(signal);
  EXPECT_EQ(kNumBands, bands.size());
  for (const std::vector<int16_t>& band : bands) {
    EXPECT_EQ(kNumBandSamples, band.size());
    EXPECT_LT(kMinCorrelation / kNumBands, MaxCorrelation(band, band));
  }

  const std::vector<int16_t> merged_signal = merge_filter->Merge(bands);
  EXPECT_EQ(kNumSignalSamples, merged_signal.size());
  EXPECT_LT(kMinCorrelation, MaxCorrelation(merged_signal, merged_signal));
}

class FilterBanksSineTest : public testing::TestWithParam<int> {};

TEST_P(FilterBanksSineTest, Sine) {
  constexpr float kTolerance = 0.03;
  const int kSineBand = GetParam();

  std::vector<int16_t> signal(kNumSignalSamples);
  for (int i = 0; i < signal.size(); ++i) {
    signal[i] =
        std::numeric_limits<int16_t>().max() *
        std::sin(M_PI * (2.f * kSineBand + 1.f) * i / (2.f * kNumBands));
  }
  std::unique_ptr<SplitFilter> split_filter = SplitFilter::Create(kNumBands);
  ASSERT_NE(nullptr, split_filter);
  EXPECT_EQ(kNumBands, split_filter->num_bands());
  std::unique_ptr<MergeFilter> merge_filter = MergeFilter::Create(kNumBands);
  ASSERT_NE(nullptr, merge_filter);
  EXPECT_EQ(kNumBands, merge_filter->num_bands());

  const std::vector<std::vector<int16_t>> bands = split_filter->Split(signal);
  EXPECT_EQ(kNumBands, bands.size());
  for (int band = 0; band < kNumBands; ++band) {
    EXPECT_EQ(kNumBandSamples, bands[band].size());
    const float expected_correlation = band == kSineBand ? 0.5f : 0.f;
    EXPECT_NEAR(expected_correlation, MaxCorrelation(bands[band], bands[band]),
                kTolerance);
  }

  const std::vector<int16_t> merged_signal = merge_filter->Merge(bands);
  EXPECT_EQ(kNumSignalSamples, merged_signal.size());
  EXPECT_NEAR(0.5f, MaxCorrelation(signal, merged_signal), kTolerance);
}

INSTANTIATE_TEST_SUITE_P(DifferentFrequencies, FilterBanksSineTest,
                         testing::Range(0, kNumBands));

}  // namespace
}  // namespace codec
}  // namespace chromemedia
