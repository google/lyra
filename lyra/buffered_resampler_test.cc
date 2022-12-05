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

#include "lyra/buffered_resampler.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "lyra/lyra_config.h"
#include "lyra/resampler_interface.h"
#include "lyra/testing/mock_resampler.h"

namespace chromemedia {
namespace codec {

class BufferedResamplerPeer {
 public:
  explicit BufferedResamplerPeer(std::unique_ptr<MockResampler> mock_resampler)
      : buffered_resampler_(new BufferedResampler(std::move(mock_resampler))) {}

  std::optional<std::vector<int16_t>> FilterAndBuffer(
      std::optional<std::vector<int16_t>> new_split_samples,
      int num_external_samples_requested) {
    std::function<std::optional<std::vector<int16_t>>(int)> sample_generator =
        [&new_split_samples](int num_samples_to_generate)
        -> std::optional<std::vector<int16_t>> { return new_split_samples; };

    return buffered_resampler_->FilterAndBuffer(sample_generator,
                                                num_external_samples_requested);
  }

  int GetInternalNumSamplesToGenerate(int num_external_samples_requested) {
    return buffered_resampler_->GetInternalNumSamplesToGenerate(
        num_external_samples_requested);
  }

  void SetLeftoverSamples(const std::vector<int16_t> samples) {
    buffered_resampler_->leftover_samples_ = samples;
  }

  std::unique_ptr<BufferedResampler> buffered_resampler_;
};

namespace {

using testing::_;
using testing::Exactly;
using testing::Return;

const int16_t k16kHzSignalValue = 2;

TEST(BufferedResamplerTest, ResamplerNotCalledMatchingRates) {
  // Resample ratio is 1.
  auto mock_resampler = std::make_unique<MockResampler>(kInternalSampleRateHz,
                                                        kInternalSampleRateHz);
  const std::vector<int16_t> internal_samples(75);

  // Expect the resampler to be never called.
  EXPECT_CALL(*mock_resampler, Resample(_)).Times(Exactly(0));

  BufferedResamplerPeer buffered_resampler_peer(std::move(mock_resampler));
  auto samples = buffered_resampler_peer.FilterAndBuffer(internal_samples, 75);
  ASSERT_TRUE(samples.has_value());
  EXPECT_EQ(75, samples->size());
}

TEST(BufferedResamplerTest, FilterAndBufferResultSizesCorrect) {
  auto mock_resampler =
      std::make_unique<MockResampler>(kInternalSampleRateHz, 48000);
  // Set up mock calls for 37 and 66 requested samples.
  // 37 of 39 samples will be used, 2 will be leftover.
  const std::vector<int16_t> expected_internal_samples_0(13, k16kHzSignalValue);
  // The next request for 66 will only need 64 newly generated samples at the
  // external sample rate, meaning 22 at the internal sample rate.
  const std::vector<int16_t> expected_internal_samples_1(22, k16kHzSignalValue);

  EXPECT_CALL(*mock_resampler,
              Resample(absl::MakeConstSpan(expected_internal_samples_0)))
      .Times(Exactly(1));
  EXPECT_CALL(*mock_resampler,
              Resample(absl::MakeConstSpan(expected_internal_samples_1)))
      .Times(Exactly(1));

  BufferedResamplerPeer buffered_resampler_peer(std::move(mock_resampler));

  auto samples =
      buffered_resampler_peer.FilterAndBuffer(expected_internal_samples_0, 37);
  ASSERT_TRUE(samples.has_value());
  EXPECT_EQ(37, samples->size());

  samples =
      buffered_resampler_peer.FilterAndBuffer(expected_internal_samples_1, 66);
  ASSERT_TRUE(samples.has_value());
  EXPECT_EQ(66, samples->size());

  // Now that there are 2 leftover samples, if the next request is 2 samples,
  // no new samples are needed.
  EXPECT_EQ(0, buffered_resampler_peer.GetInternalNumSamplesToGenerate(2));
  // But if we request one more than 2 we need kNumBands samples.
  EXPECT_EQ(1, buffered_resampler_peer.GetInternalNumSamplesToGenerate(3));
}

TEST(BufferedResamplerTest, RequestingFewerThanLeftoversNotResampling) {
  auto mock_resampler =
      std::make_unique<MockResampler>(kInternalSampleRateHz, 48000);
  // Set up a mock call for 73 samples.
  const std::vector<int16_t> expected_internal_samples_0(25, k16kHzSignalValue);
  const std::vector<int16_t> empty_vector(0);

  {  // Enforce mocks are called in a specific order.
    ::testing::InSequence in;
    EXPECT_CALL(*mock_resampler,
                Resample(absl::MakeConstSpan(expected_internal_samples_0)))
        .Times(Exactly(1));
    // We are not expecting the resampler to be called if there are enough
    // samples to fulfill the request.
    EXPECT_CALL(*mock_resampler, Resample(absl::MakeConstSpan(empty_vector)))
        .Times(Exactly(1));
  }
  BufferedResamplerPeer buffered_resampler_peer(std::move(mock_resampler));

  // If 73 samples are requested, then there will be 3 * 75 - 73 =  2 leftover
  // samples.
  auto samples =
      buffered_resampler_peer.FilterAndBuffer(expected_internal_samples_0, 73);
  ASSERT_TRUE(samples.has_value());
  EXPECT_EQ(73, samples->size());

  // If subsequently 2 samples are requested, because the 2 leftovers
  // are enough to fulfill the requests, no new samples are generated, and
  // |mock_resampler->Resample()| is not called.
  for (int num_external_samples_requested : {2, 1}) {
    const int internal_num_samples_to_generate =
        buffered_resampler_peer.GetInternalNumSamplesToGenerate(
            num_external_samples_requested);
    EXPECT_EQ(0, internal_num_samples_to_generate);
  }
  samples = buffered_resampler_peer.FilterAndBuffer(empty_vector, 2);
  ASSERT_TRUE(samples.has_value());
  EXPECT_EQ(2, samples->size());
}

TEST(BufferedResamplerTest, SubsequentFilterAndBufferUseLeftovers) {
  auto mock_resampler =
      std::make_unique<MockResampler>(kInternalSampleRateHz, 48000);
  // Test that leftovers are kept and used to return in the next call of
  // FilterAndBuffer().
  // For example:
  //
  //   16khz samples: {0, 1, 2, 3, 4}
  //   48khz samples with value 0 come from 16khz samples with value 0, etc.
  //   48khz samples: {0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4}
  //                  |<--   1st call -->| |<--      2nd      -->|
  const std::vector<int16_t> expected_internal_samples_0({0, 1, 2});
  const std::vector<int16_t> expected_resampled_samples_0(
      {0, 0, 0, 1, 1, 1, 2, 2, 2});
  const std::vector<int16_t> expected_results_0({0, 0, 0, 1, 1, 1, 2});
  const std::vector<int16_t> expected_internal_samples_1({3, 4});
  const std::vector<int16_t> expected_resampled_samples_1({3, 3, 3, 4, 4, 4});
  const std::vector<int16_t> expected_results_1({2, 2, 3, 3, 3, 4, 4, 4});

  {  // Enforce mocks are called in a specific order.
    ::testing::InSequence in;
    EXPECT_CALL(*mock_resampler,
                Resample(absl::MakeSpan(expected_internal_samples_0)))
        .Times(Exactly(1))
        .WillOnce(Return(expected_resampled_samples_0));
    EXPECT_CALL(*mock_resampler,
                Resample(absl::MakeSpan(expected_internal_samples_1)))
        .Times(Exactly(1))
        .WillOnce(Return(expected_resampled_samples_1));
  }
  BufferedResamplerPeer buffered_resampler_peer(std::move(mock_resampler));

  // First call to FilterAndBuffer().
  auto result_0 =
      buffered_resampler_peer.FilterAndBuffer(expected_internal_samples_0, 7);
  EXPECT_EQ(result_0, expected_results_0);

  // Second call to FilterAndBuffer(). Note the leading 2s of the expected
  // result, which comes from the left over of the first call.
  auto result_1 =
      buffered_resampler_peer.FilterAndBuffer(expected_internal_samples_1, 8);
  EXPECT_EQ(result_1, expected_results_1);
}

class BufferedResamplerSampleRatesTest : public testing::TestWithParam<int> {
 protected:
  BufferedResamplerSampleRatesTest() : external_sample_rate_hz_(GetParam()) {}
  const int external_sample_rate_hz_;
};

TEST_P(BufferedResamplerSampleRatesTest, CreationSucceeds) {
  auto buffered_generator = BufferedResampler::Create(
      kInternalSampleRateHz, static_cast<double>(external_sample_rate_hz_));
  EXPECT_NE(nullptr, buffered_generator);
}

TEST_P(BufferedResamplerSampleRatesTest, NumSamplesToGenerateCorrect) {
  auto mock_resampler =
      std::make_unique<MockResampler>(kInternalSampleRateHz, 48000);
  BufferedResamplerPeer buffered_resampler_peer(std::move(mock_resampler));
  EXPECT_EQ(0, buffered_resampler_peer.GetInternalNumSamplesToGenerate(0));
  // 118 samples at 48 khz means we need ceil(118 / 3) samples at 16khz.
  EXPECT_EQ(40, buffered_resampler_peer.GetInternalNumSamplesToGenerate(118));
}

TEST_P(BufferedResamplerSampleRatesTest, RequestingTooManySamplesFails) {
  auto mock_resampler =
      std::make_unique<MockResampler>(kInternalSampleRateHz, 48000);
  BufferedResamplerPeer buffered_resampler_peer(std::move(mock_resampler));

  // The upper limit of number of samples is one internal hop's worth of
  // samples.
  const int num_too_many_samples = GetNumSamplesPerHop(16000) + 1;
  const auto too_many_samples = std::vector<int16_t>(num_too_many_samples);
  EXPECT_DEATH(buffered_resampler_peer.FilterAndBuffer(too_many_samples,
                                                       num_too_many_samples),
               "");
}

INSTANTIATE_TEST_SUITE_P(AllSampleRates, BufferedResamplerSampleRatesTest,
                         testing::ValuesIn(kSupportedSampleRates));

}  // namespace
}  // namespace codec
}  // namespace chromemedia
