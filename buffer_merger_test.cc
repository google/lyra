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

#include "buffer_merger.h"

#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "filter_banks_interface.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "lyra_config.h"
#include "testing/mock_filter_banks.h"

namespace chromemedia {
namespace codec {

class BufferMergerPeer {
 public:
  explicit BufferMergerPeer(std::unique_ptr<MockMergeFilter> mock_merge_filter)
      : buffer_merger_(new BufferMerger(std::move(mock_merge_filter))) {}

  std::vector<int16_t> BufferAndMerge(
      const std::vector<std::vector<int16_t>>& new_split_samples,
      int num_samples) {
    std::function<const std::vector<std::vector<int16_t>>&(int)>
        sample_generator = [&new_split_samples](int num_samples_to_generate)
        -> const std::vector<std::vector<int16_t>>& {
      return new_split_samples;
    };
    return buffer_merger_->BufferAndMerge(sample_generator, num_samples);
  }

  int GetNumSamplesToGenerate(int num_samples) {
    return buffer_merger_->GetNumSamplesToGenerate(num_samples);
  }
  void Reset() { return buffer_merger_->Reset(); }

  std::unique_ptr<BufferMerger> buffer_merger_;
};

namespace {

using testing::_;
using testing::ElementsAre;
using testing::Exactly;
using testing::Invoke;
using testing::IsSupersetOf;
using testing::SizeIs;

// A simple merge function that just interleaves a vector of vectors into a
// vector. For example:
//   bands[0]: {0, 2, 4, ..., 2N}
//   bands[1]: {1, 3, 5, ..., 2N + 1}
//   merged:   {0, 1, 2, 3, 4, 5, ..., 2N, 2N + 1}
std::vector<int16_t> InterleaveMerge(
    const std::vector<std::vector<int16_t>>& bands) {
  const int num_bands = bands.size();
  const int band_length = bands[0].size();
  std::vector<int16_t> merged(num_bands * band_length);
  for (int j = 0; j < band_length; ++j) {
    for (int i = 0; i < num_bands; ++i) {
      merged[j * num_bands + i] = bands[i][j];
    }
  }
  return merged;
}

class BufferMergerTest : public testing::Test {
 protected:
  BufferMergerTest() {}

  // Set up a vector of vectors like this:
  //   split_samples[0] = {0, 1, 2, ..., M}
  //   split_samples[1] = {0, 1, 2, ..., M}
  //   ...
  //   split_samples[N] = {0, 1, 2, ..., M}
  std::vector<std::vector<int16_t>> SetUpSplitSamples(int num_bands,
                                                      int num_samples) {
    const int num_samples_per_band = static_cast<int>(std::ceil(
        static_cast<float>(num_samples) / static_cast<float>(num_bands)));
    std::vector<std::vector<int16_t>> split_samples(num_bands);
    for (int16_t i = 0; i < num_bands; ++i) {
      split_samples[i] = std::vector<int16_t>(num_samples_per_band);
      std::iota(split_samples[i].begin(), split_samples[i].end(), 0);
    }
    return split_samples;
  }
};

TEST_F(BufferMergerTest, NumSamplesToGenerateCorrect) {
  auto mock_merge_filter = absl::make_unique<MockMergeFilter>(4);
  BufferMergerPeer buffer_merger_peer(std::move(mock_merge_filter));
  EXPECT_EQ(0, buffer_merger_peer.GetNumSamplesToGenerate(0));
  // Expect the next multiple of 4 after 117.
  EXPECT_EQ(120, buffer_merger_peer.GetNumSamplesToGenerate(117));
}

TEST_F(BufferMergerTest, RequestingTooManySamplesFails) {
  const int kNumBands = 4;
  auto mock_merge_filter = absl::make_unique<MockMergeFilter>(kNumBands);
  BufferMergerPeer buffer_merger_peer(std::move(mock_merge_filter));

  // The upper limit of number of samples is one frame's worth of samples.
  const int num_too_many_samples =
      GetNumSamplesPerHop(kInternalSampleRateHz) + 1;
  const auto split_samples = SetUpSplitSamples(kNumBands, num_too_many_samples);
  EXPECT_DEATH(
      buffer_merger_peer.BufferAndMerge(split_samples, num_too_many_samples),
      "");
}

TEST_F(BufferMergerTest, MergeFilterNotCalledForOneBand) {
  // 1 band.
  auto mock_merge_filter = absl::make_unique<MockMergeFilter>(1);

  // Expect the merge filter to be never called.
  const auto split_samples = SetUpSplitSamples(1, 75);
  EXPECT_CALL(*mock_merge_filter, Merge(_)).Times(Exactly(0));

  BufferMergerPeer buffer_merger_peer(std::move(mock_merge_filter));
  std::vector<int16_t> samples =
      buffer_merger_peer.BufferAndMerge(split_samples, 75);
  EXPECT_EQ(75, samples.size());
}

TEST_F(BufferMergerTest, MergeFilterCalledWithBufferContainingSamples) {
  auto mock_merge_filter = absl::make_unique<MockMergeFilter>(4);
  const auto split_samples = SetUpSplitSamples(4, 37);

  // Match a buffer that is a vector of 4 vectors, each is contains the samples
  // from all the split bands.
  auto buffer_matcher = ElementsAre(
      IsSupersetOf(split_samples[0]), IsSupersetOf(split_samples[1]),
      IsSupersetOf(split_samples[2]), IsSupersetOf(split_samples[3]));
  EXPECT_CALL(*mock_merge_filter, Merge(buffer_matcher))
      .WillOnce(Invoke(InterleaveMerge));
  BufferMergerPeer buffer_merger_peer(std::move(mock_merge_filter));
  std::vector<int16_t> samples =
      buffer_merger_peer.BufferAndMerge(split_samples, 37);
  EXPECT_EQ(37, samples.size());
}

TEST_F(BufferMergerTest, BufferAndMergeResultSizesCorrect) {
  const int kNumBands = 4;
  auto mock_merge_filter = absl::make_unique<MockMergeFilter>(kNumBands);
  // Set up mock calls for 37 and 65 requested samples.
  // Match a buffer that is a vector of 4 vectors, each of size 10.
  auto buffer_matcher_37_samples =
      ElementsAre(SizeIs(10), SizeIs(10), SizeIs(10), SizeIs(10));
  EXPECT_CALL(*mock_merge_filter, Merge(buffer_matcher_37_samples))
      .Times(Exactly(1))
      .WillOnce(Invoke(InterleaveMerge));
  // Match a buffer that is a vector of 4 vectors, each of size 16.
  auto buffer_matcher_65_samples =
      ElementsAre(SizeIs(16), SizeIs(16), SizeIs(16), SizeIs(16));
  EXPECT_CALL(*mock_merge_filter, Merge(buffer_matcher_65_samples))
      .Times(Exactly(1))
      .WillOnce(Invoke(InterleaveMerge));

  BufferMergerPeer buffer_merger_peer(std::move(mock_merge_filter));

  // If 37 samples are requested, then there will be 3 leftover samples.
  auto split_samples = SetUpSplitSamples(kNumBands, 37);
  std::vector<int16_t> samples =
      buffer_merger_peer.BufferAndMerge(split_samples, 37);
  EXPECT_EQ(37, samples.size());

  // If another 65 samples are to be requested, then because of the 3
  // leftover samples, we only need 62 new samples. However we need this number
  // to be the next multiple of kNumBands. This means 16 samples per band are
  // needed, and 64 are generated. The 65 requested take 3 from the leftovers
  // and 62 of the 64 newly generated, leaving 2 leftover samples.
  EXPECT_EQ(64, buffer_merger_peer.GetNumSamplesToGenerate(65));
  split_samples = SetUpSplitSamples(kNumBands, 64);
  samples = buffer_merger_peer.BufferAndMerge(split_samples, 65);
  EXPECT_EQ(65, samples.size());

  // Now that there are 2 leftover samples, if the next request is 2 samples,
  // no new samples are needed.
  EXPECT_EQ(0, buffer_merger_peer.GetNumSamplesToGenerate(2));
  // But if we request one more than 2 we need kNumBands samples.
  EXPECT_EQ(kNumBands, buffer_merger_peer.GetNumSamplesToGenerate(3));
}

TEST_F(BufferMergerTest, RequestingFewerThanLeftoversNotMerging) {
  const int kNumBands = 8;
  auto mock_merge_filter = absl::make_unique<MockMergeFilter>(kNumBands);

  // Match a buffer that is a vector of 8 vectors, each of size 10. Expect
  // |mock_merge_filter->Merge()| to be called only once.
  auto buffer_matcher_73_samples =
      ElementsAre(SizeIs(10), SizeIs(10), SizeIs(10), SizeIs(10), SizeIs(10),
                  SizeIs(10), SizeIs(10), SizeIs(10));
  // After the first call expect this to be called once per |BufferAndMerge|
  // with 0 samples because after the first time
  // there will be sufficient leftovers to use.
  auto buffer_matcher_0_samples =
      ElementsAre(SizeIs(0), SizeIs(0), SizeIs(0), SizeIs(0), SizeIs(0),
                  SizeIs(0), SizeIs(0), SizeIs(0));
  EXPECT_CALL(*mock_merge_filter, Merge(buffer_matcher_73_samples))
      .Times(Exactly(1))
      .WillOnce(Invoke(InterleaveMerge));
  // Called once per element in {4, 2, 1}.
  EXPECT_CALL(*mock_merge_filter, Merge(buffer_matcher_0_samples))
      .Times(Exactly(3))
      .WillOnce(Invoke(InterleaveMerge));
  BufferMergerPeer buffer_merger_peer(std::move(mock_merge_filter));

  // If 73 samples are requested, then there will be 7 leftover samples.
  auto split_samples = SetUpSplitSamples(kNumBands, 73);
  auto samples = buffer_merger_peer.BufferAndMerge(split_samples, 73);
  EXPECT_EQ(73, samples.size());

  // If subsequently 4, 2, 1 samples are requested, because the 7 leftovers
  // are enough to fulfill the requests, no new samples are generated, and
  // |mock_merge_filter->Merge()| is not called.
  for (int num_samples : {4, 2, 1}) {
    const int num_samples_to_generate =
        buffer_merger_peer.GetNumSamplesToGenerate(num_samples);
    EXPECT_EQ(0, num_samples_to_generate);
    split_samples = SetUpSplitSamples(kNumBands, num_samples_to_generate);
    samples = buffer_merger_peer.BufferAndMerge(split_samples, num_samples);
    EXPECT_EQ(num_samples, samples.size());
  }
}

TEST_F(BufferMergerTest, ResetClearsLeftover) {
  const int kNumBands = 4;
  auto mock_merge_filter = absl::make_unique<MockMergeFilter>(kNumBands);
  auto buffer_matcher_78_samples =
      ElementsAre(SizeIs(20), SizeIs(20), SizeIs(20), SizeIs(20));
  EXPECT_CALL(*mock_merge_filter, Merge(buffer_matcher_78_samples))
      .Times(Exactly(1))
      .WillOnce(Invoke(InterleaveMerge));
  BufferMergerPeer buffer_merger_peer(std::move(mock_merge_filter));

  // If 78 samples are requested, then each band will contain 20 samples,
  // and there will be 2 leftover samples.
  const auto split_samples = SetUpSplitSamples(kNumBands, 78);
  std::vector<int16_t> samples =
      buffer_merger_peer.BufferAndMerge(split_samples, 78);
  EXPECT_EQ(78, samples.size());

  // Because of the 2 leftover samples, if the next request is 2 samples, no
  // new sample is needed.
  EXPECT_EQ(0, buffer_merger_peer.GetNumSamplesToGenerate(2));

  // However, calling Reset() clears the leftover samples, so requesting 2
  // samples would mean 4 samples need to be generated.
  buffer_merger_peer.Reset();
  EXPECT_EQ(kNumBands, buffer_merger_peer.GetNumSamplesToGenerate(2));
}

TEST_F(BufferMergerTest, SubsequentBufferAndMergeUseLeftovers) {
  // Test that leftovers are kept and used to return in the next call of
  // BufferAndMerge().
  // For example:
  //
  //   bands[0]: {0, 1, 2, 3}, {0, 1, 2, 3, 4}
  //   bands[1]: {0, 1, 2, 3}, {0, 1, 2, 3, 4}
  //             |<-- 1st -->| |<--  2nd  -->|
  //
  //   merged:   {0, 0, 1, 1, 2, 2, 3}, {3, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4}
  //             |<---  1st call  -->|  |<--         2nd call        -->|
  //
  // One of the ending 3's of the first call (requesting 7 samples) is the
  // "leftover" and will be used in the second call (requesting 11 samples).
  auto mock_merge_filter = absl::make_unique<MockMergeFilter>(2);
  EXPECT_CALL(*mock_merge_filter, Merge(_))
      .WillRepeatedly(Invoke(InterleaveMerge));
  BufferMergerPeer buffer_merger_peer(std::move(mock_merge_filter));

  // First call to BufferAndMerge().
  const int num_samples_1 = buffer_merger_peer.GetNumSamplesToGenerate(7);
  const auto split_samples_1 = SetUpSplitSamples(2, num_samples_1);
  auto result_1 = buffer_merger_peer.BufferAndMerge(split_samples_1, 7);
  EXPECT_THAT(result_1, ElementsAre(0, 0, 1, 1, 2, 2, 3));

  // Second call to BufferAndMerge(). Note the leading 3 of the expected
  // result, which comes from the left over of the first call.
  const int num_samples_2 = buffer_merger_peer.GetNumSamplesToGenerate(11);
  const auto split_samples_2 = SetUpSplitSamples(2, num_samples_2);
  auto result_2 = buffer_merger_peer.BufferAndMerge(split_samples_2, 11);
  EXPECT_THAT(result_2, ElementsAre(3, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4));
}

class BufferMergerNumBandTest : public testing::TestWithParam<int> {
 protected:
  BufferMergerNumBandTest() : num_bands_(GetParam()) {}
  const int num_bands_;
};

TEST_P(BufferMergerNumBandTest, CreationSucceeds) {
  auto buffer_merger = BufferMerger::Create(num_bands_);
  EXPECT_NE(nullptr, buffer_merger);
}

TEST_P(BufferMergerNumBandTest, CreationFailsNotPowerOfTwo) {
  // 2^N + 5 is guaranteed to not be a power of two.
  const int bad_num_bands = num_bands_ + 5;
  auto buffer_merger = BufferMerger::Create(bad_num_bands);
  EXPECT_EQ(nullptr, buffer_merger);
}

INSTANTIATE_TEST_SUITE_P(NumBands, BufferMergerNumBandTest,
                         testing::Values(1, 2, 4, 8, 16));

}  // namespace
}  // namespace codec
}  // namespace chromemedia
