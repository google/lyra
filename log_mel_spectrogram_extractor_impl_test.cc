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

#include "log_mel_spectrogram_extractor_impl.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/types/span.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace chromemedia {
namespace codec {
namespace {

static constexpr int kTestSampleRateHz = 16000;
static constexpr int kNumMelBins = 10;
static constexpr int kHopLengthSamples = 5;
static constexpr int kWindowLengthSamples = 10;
static constexpr int kNumOutputMelFeatures = 3;

static constexpr int16_t kWavData[] = {7954,   10085, 8733,   10844,  29949,
                                       -549,   20833, 30345,  18086,  11375,
                                       -27309, 12323, -22891, -23360, 11958};

// These results were obtained by running
// audio/dsp/mfcc/mfcc_mel.LogMelSpectrogram on kWavData pre-pended with 5
// zeros and then dividing the results by 10. The parameters used were:
// audio_sample_rate=16000
// log_additive_offset=0.0
// log_floor=500
// window_length_secs=0.000625 (window_length_samples=10)
// hop_length_secs=0.0003125 (hop_length_samples=5)
// window_type="hann"
// fft_length=None
// upper_edge_hz=0.99*(16000/2)
// lower_edge_hz=0
static constexpr float kMelBins[][10] = {
    {0.62146081, 0.62146081, 0.79771997, 1.00416802, 0.73013308, 0.96676503,
     0.87643814, 0.89284485, 0.90586112, 0.8633126},
    {0.62146081, 0.62146081, 0.89000145, 1.09644949, 0.76740002, 1.00403196,
     0.8919037, 0.99746922, 1.06052462, 1.08220812},
    {0.62146081, 0.62146081, 0.83526758, 1.04171563, 0.82093681, 1.05756876,
     0.96348656, 1.01345318, 1.07686605, 1.12100911}};

class LogMelSpectrogramExtractorImplTest : public testing::Test {
 protected:
  void SetUp() override {
    feature_extractor_ = LogMelSpectrogramExtractorImpl::Create(
        kTestSampleRateHz, kHopLengthSamples, kWindowLengthSamples,
        kNumMelBins);
    ASSERT_NE(feature_extractor_, nullptr);
  }

  std::unique_ptr<LogMelSpectrogramExtractorImpl> feature_extractor_;
};

TEST_F(LogMelSpectrogramExtractorImplTest, ThreeFeaturesEqualExpected) {
  for (int i = 0; i < kNumOutputMelFeatures; ++i) {
    const absl::Span<const int16_t> audio = absl::MakeConstSpan(
        &kWavData[i * kHopLengthSamples], kHopLengthSamples);

    auto features = feature_extractor_->Extract(audio);

    EXPECT_TRUE(features.has_value());
    EXPECT_THAT(features.value(),
                testing::Pointwise(testing::FloatEq(), kMelBins[i]));
  }
}

TEST_F(LogMelSpectrogramExtractorImplTest, SamplesLongerThanExpected) {
  std::vector<int16_t> audio(kHopLengthSamples + 1);

  auto features = feature_extractor_->Extract(absl::MakeConstSpan(audio));

  EXPECT_FALSE(features.has_value());
}

TEST_F(LogMelSpectrogramExtractorImplTest, SamplesShorterThanExpected) {
  std::vector<int16_t> audio(kWavData, kWavData + kHopLengthSamples - 1);

  auto features = feature_extractor_->Extract(absl::MakeConstSpan(audio));

  EXPECT_FALSE(features.has_value());
}

}  // namespace
}  // namespace codec
}  // namespace chromemedia
