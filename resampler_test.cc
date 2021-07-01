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

#include "resampler.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <tuple>
#include <vector>

#include "absl/types/span.h"
#include "audio/dsp/signal_vector_util.h"
#include "gtest/gtest.h"
#include "lyra_config.h"

namespace chromemedia {
namespace codec {

class ResamplerSampleRateTest
    : public testing::TestWithParam<std::tuple<int, int>> {};

TEST_P(ResamplerSampleRateTest, ResamplingWorksAllZeros) {
  const int input_sample_rate_hz = std::get<0>(GetParam());
  const int output_sample_rate_hz = std::get<1>(GetParam());
  std::vector<int16_t> samples(GetNumSamplesPerHop(input_sample_rate_hz), 0);
  auto resampler =
      Resampler::Create(input_sample_rate_hz, output_sample_rate_hz);
  const auto resampled = resampler->Resample(absl::MakeConstSpan(samples));
  std::vector<int16_t> expected(GetNumSamplesPerHop(output_sample_rate_hz), 0);
  EXPECT_EQ(resampled, expected);
}

INSTANTIATE_TEST_SUITE_P(
    UpsampleAndDownsample, ResamplerSampleRateTest,
    testing::Combine(::testing::Values(kInternalSampleRateHz),
                     ::testing::ValuesIn(kSupportedSampleRates)));

TEST(ResamplerTest, UpsampleThenDownsampleSimilar) {
  const double base_sample_rate_hz = 16000;
  const double upsample_sample_rate_hz = 32000;

  std::vector<double> doubles_samples;
  audio_dsp::ComputeSineWaveVector(1000, base_sample_rate_hz, 0.0, 100,
                                   &doubles_samples);
  std::vector<int16_t> samples;
  for (auto val : doubles_samples) {
    samples.push_back(val * 100);
  }

  // Upsample.
  auto resampler =
      Resampler::Create(base_sample_rate_hz, upsample_sample_rate_hz);
  const auto upsampled = resampler->Resample(absl::MakeConstSpan(samples));

  // Downsample.
  resampler = Resampler::Create(upsample_sample_rate_hz, base_sample_rate_hz);
  const auto downsampled = resampler->Resample(absl::MakeConstSpan(upsampled));

  // Formula for delay:
  // (delay from upsample filter) + (delay from downsample filter)
  // = 17 + floor(17/2) = 25
  constexpr int delay = 25;
  for (int i = delay; i < samples.size(); ++i) {
    EXPECT_NEAR(samples[i - delay], downsampled[i], 25);
  }
}

// This test will fail without clipping, as ubsan will catch the overflow when
// converting from float to int16_t after resampling.
TEST(ResamplerExtremeValuesTest, AlternatingExtremeValuesTest) {
  constexpr double kInputSampleRate = 16000;
  constexpr double kOutputSampleRate = 32000;
  std::vector<int16_t> samples(GetNumSamplesPerHop(kInputSampleRate));
  for (int i = 0; i < samples.size(); ++i) {
    samples[i] = ((i / 2) % 2 == 0) ? std::numeric_limits<int16_t>::min()
                                    : std::numeric_limits<int16_t>::max();
  }

  auto resampler = Resampler::Create(kInputSampleRate, kOutputSampleRate);
  const auto resampled = resampler->Resample(absl::MakeConstSpan(samples));

  EXPECT_EQ(resampled.size(), GetNumSamplesPerHop(kOutputSampleRate));
}

}  // namespace codec
}  // namespace chromemedia
