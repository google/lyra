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

#include "lyra_encoder.h"

#include <algorithm>
#include <bitset>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

// Placeholder for get runfiles header.
#include "absl/memory/memory.h"   // IWYU pragma: keep
#include "absl/types/optional.h"  // IWYU pragma: keep
#include "absl/types/span.h"
#include "denoiser_interface.h"
#include "feature_extractor_interface.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "include/ghc/filesystem.hpp"
#include "lyra_config.h"
#include "noise_estimator_interface.h"
#include "packet.h"
#include "packet_interface.h"
#include "resampler_interface.h"
#include "testing/mock_denoiser.h"
#include "testing/mock_feature_extractor.h"
#include "testing/mock_noise_estimator.h"
#include "testing/mock_resampler.h"
#include "testing/mock_vector_quantizer.h"
#include "vector_quantizer_interface.h"

namespace chromemedia {
namespace codec {
namespace {

constexpr int kNumQuantizedBits = 120;
constexpr int kNumHeaderBits = 0;

}  // namespace

// Use a test peer to access the private constructor of LyraEncoder in order
// to inject mock dependencies.
class LyraEncoderPeer {
 public:
  explicit LyraEncoderPeer(
      std::unique_ptr<MockResampler> mock_resampler,
      std::unique_ptr<MockFeatureExtractor> mock_feature_extractor,
      std::unique_ptr<MockNoiseEstimator> mock_noise_estimator,
      std::unique_ptr<MockVectorQuantizer> mock_vector_quantizer,
      std::unique_ptr<MockDenoiser> mock_denoiser, int sample_rate_hz,
      int num_frames_per_packet, bool enable_dtx)
      : encoder_(std::move(mock_resampler), std::move(mock_feature_extractor),
                 std::move(mock_noise_estimator),
                 std::move(mock_vector_quantizer), std::move(mock_denoiser),
                 absl::make_unique<Packet<kNumQuantizedBits, kNumHeaderBits>>(),
                 sample_rate_hz, kNumChannels, kBitrate, num_frames_per_packet,
                 enable_dtx) {}

  absl::optional<std::vector<uint8_t>> Encode(
      const absl::Span<const int16_t> audio) {
    return encoder_.EncodeInternal(audio, false);
  }

  absl::optional<std::vector<uint8_t>> EncodeWithFiltering(
      const absl::Span<const int16_t> audio) {
    return encoder_.EncodeInternal(audio, true);
  }

 private:
  LyraEncoder encoder_;
};

namespace {

using testing::_;
using testing::Combine;
using testing::IsSupersetOf;
using testing::Return;
using testing::Values;
using testing::ValuesIn;

class LyraEncoderTest
    : public testing::TestWithParam<testing::tuple<int, int>> {
 protected:
  LyraEncoderTest()
      : sample_rate_hz_(std::get<0>(GetParam())),
        internal_sample_rate_hz_(GetInternalSampleRate(sample_rate_hz_)),
        internal_num_samples_per_hop_(
            GetNumSamplesPerHop(internal_sample_rate_hz_)),
        num_frames_per_packet_(std::get<1>(GetParam())),
        samples_(num_frames_per_packet_ * GetNumSamplesPerHop(sample_rate_hz_)),
        samples_span_(absl::MakeConstSpan(samples_)),
        internal_samples_(num_frames_per_packet_ *
                          internal_num_samples_per_hop_),
        internal_samples_span_(absl::MakeConstSpan(internal_samples_)),
        mock_feature_extractor_(absl::make_unique<MockFeatureExtractor>()),
        mock_noise_estimator_(absl::make_unique<MockNoiseEstimator>()),
        mock_vector_quantizer_(absl::make_unique<MockVectorQuantizer>()),
        mock_resampler_(absl::make_unique<MockResampler>()),
        mock_denoiser_(absl::make_unique<MockDenoiser>()),
        mock_features_(kNumFeatures),
        mock_concatenated_features_(num_frames_per_packet_ * kNumFeatures) {
    // Populate the mock features, samples, and internal samples with
    // increasing values starting at 0.
    std::iota(mock_features_.value().begin(), mock_features_.value().end(),
              0.f);
    std::iota(samples_.begin(), samples_.end(), 0);
    std::iota(internal_samples_.begin(), internal_samples_.end(), 0);

    // Mock concatenated features contains copies of mock features.
    for (int i = 0; i < num_frames_per_packet_; ++i) {
      std::copy(mock_features_.value().begin(), mock_features_.value().end(),
                mock_concatenated_features_.begin() + i * kNumFeatures);
    }

    // Mock quantized contains all 1s at every bit position.
    std::bitset<kNumQuantizedBits> mock_quantized_bits(0);
    mock_quantized_bits.flip();
    mock_quantized_ = mock_quantized_bits.to_string();
  }

  bool DoesPacketContainQuantized(const std::vector<uint8_t>& packet,
                                  const std::string& quantized_string) {
    if (packet.size() < kPacketSize) {
      return false;
    }
    std::string packet_data;
    for (const auto& byte : packet) {
      packet_data.append(std::bitset<CHAR_BIT>(byte).to_string());
    }
    // Remove header bits.
    packet_data = packet_data.substr(kNumHeaderBits, packet_data.size());
    // Remove extra bits at the end of packet_data if the number of bits stored
    // in the packet is not evenly divisible by CHAR_BITS.
    packet_data = packet_data.substr(0, kNumQuantizedBits);
    return packet_data == quantized_string;
  }

  void SetResamplerExpectation(int times_if_resampling) {
    if (internal_sample_rate_hz_ == sample_rate_hz_) {
      EXPECT_CALL(*mock_resampler_, Resample(samples_span_)).Times(0);
    } else {
      EXPECT_CALL(*mock_resampler_, Resample(samples_span_))
          .Times(times_if_resampling)
          .WillRepeatedly((Return(internal_samples_)));
    }
  }

  const int sample_rate_hz_;
  const int internal_sample_rate_hz_;
  const int internal_num_samples_per_hop_;
  const int num_frames_per_packet_;
  std::vector<int16_t> samples_;
  absl::Span<const int16_t> samples_span_;
  std::vector<int16_t> internal_samples_;
  absl::Span<const int16_t> internal_samples_span_;
  std::unique_ptr<MockFeatureExtractor> mock_feature_extractor_;
  std::unique_ptr<MockNoiseEstimator> mock_noise_estimator_;
  std::unique_ptr<MockVectorQuantizer> mock_vector_quantizer_;
  std::unique_ptr<MockResampler> mock_resampler_;
  std::unique_ptr<MockDenoiser> mock_denoiser_;
  absl::optional<std::vector<float>> mock_features_;
  std::vector<float> mock_concatenated_features_;
  std::string mock_quantized_;
};

TEST_P(LyraEncoderTest, InvalidSizedAudioFails) {
  // Adjust the size of |samples_| and |internal_samples_| to be
  // one fewer than the valid values.
  samples_.pop_back();
  internal_samples_.pop_back();
  samples_span_ = absl::MakeConstSpan(samples_);
  internal_samples_span_ = absl::MakeConstSpan(internal_samples_);

  SetResamplerExpectation(1);
  EXPECT_CALL(*mock_feature_extractor_, Extract(_)).Times(0);
  EXPECT_CALL(*mock_noise_estimator_, IsSimilarNoise(_)).Times(0);
  EXPECT_CALL(*mock_noise_estimator_, Update(_)).Times(0);
  EXPECT_CALL(*mock_vector_quantizer_, Quantize(_)).Times(0);

  LyraEncoderPeer encoder_peer(
      std::move(mock_resampler_), std::move(mock_feature_extractor_),
      std::move(mock_noise_estimator_), std::move(mock_vector_quantizer_),
      nullptr, sample_rate_hz_, num_frames_per_packet_,
      /*enable_dtx=*/false);
  auto encoded_or = encoder_peer.Encode(samples_span_);

  EXPECT_FALSE(encoded_or.has_value());
}

TEST_P(LyraEncoderTest, FeatureExtractionFails) {
  SetResamplerExpectation(1);
  EXPECT_CALL(*mock_feature_extractor_, Extract(internal_samples_span_.subspan(
                                            0, internal_num_samples_per_hop_)))
      .WillOnce(Return(absl::nullopt));
  EXPECT_CALL(*mock_noise_estimator_, IsSimilarNoise(_)).Times(0);
  EXPECT_CALL(*mock_noise_estimator_, Update(_)).Times(0);
  EXPECT_CALL(*mock_vector_quantizer_, Quantize(_)).Times(0);

  LyraEncoderPeer encoder_peer(
      std::move(mock_resampler_), std::move(mock_feature_extractor_),
      std::move(mock_noise_estimator_), std::move(mock_vector_quantizer_),
      nullptr, sample_rate_hz_, num_frames_per_packet_,
      /*enable_dtx=*/false);
  auto encoded_or = encoder_peer.Encode(samples_span_);

  EXPECT_FALSE(encoded_or.has_value());
}

TEST_P(LyraEncoderTest, SimilarNoiseEvaluationFails) {
  SetResamplerExpectation(1);
  EXPECT_CALL(*mock_feature_extractor_, Extract(internal_samples_span_.subspan(
                                            0, internal_num_samples_per_hop_)))
      .WillOnce(Return(mock_features_));
  EXPECT_CALL(*mock_noise_estimator_, IsSimilarNoise(_))
      .WillOnce(Return(absl::nullopt));
  EXPECT_CALL(*mock_noise_estimator_, Update(_)).Times(0);
  EXPECT_CALL(*mock_vector_quantizer_, Quantize(_)).Times(0);

  LyraEncoderPeer encoder_peer(
      std::move(mock_resampler_), std::move(mock_feature_extractor_),
      std::move(mock_noise_estimator_), std::move(mock_vector_quantizer_),
      nullptr, sample_rate_hz_, num_frames_per_packet_,
      /*enable_dtx=*/true);
  auto encoded_or = encoder_peer.Encode(samples_span_);

  EXPECT_FALSE(encoded_or.has_value());
}

TEST_P(LyraEncoderTest, NoiseEstimationUpdateFails) {
  SetResamplerExpectation(1);
  EXPECT_CALL(*mock_feature_extractor_, Extract(internal_samples_span_.subspan(
                                            0, internal_num_samples_per_hop_)))
      .WillOnce(Return(mock_features_));
  EXPECT_CALL(*mock_noise_estimator_, IsSimilarNoise(_))
      .WillOnce(Return(false));
  EXPECT_CALL(*mock_noise_estimator_, Update(mock_features_.value()))
      .WillOnce(Return(false));
  EXPECT_CALL(*mock_vector_quantizer_, Quantize(_)).Times(0);

  LyraEncoderPeer encoder_peer(
      std::move(mock_resampler_), std::move(mock_feature_extractor_),
      std::move(mock_noise_estimator_), std::move(mock_vector_quantizer_),
      nullptr, sample_rate_hz_, num_frames_per_packet_,
      /*enable_dtx=*/true);
  auto encoded_or = encoder_peer.Encode(samples_span_);

  EXPECT_FALSE(encoded_or.has_value());
}

TEST_P(LyraEncoderTest, NoiseEstimationUpdateSucceeds) {
  SetResamplerExpectation(1);
  for (int i = 0; i < num_frames_per_packet_; ++i) {
    EXPECT_CALL(
        *mock_feature_extractor_,
        Extract(internal_samples_span_.subspan(
            i * internal_num_samples_per_hop_, internal_num_samples_per_hop_)))
        .WillOnce(Return(mock_features_));
  }
  EXPECT_CALL(*mock_noise_estimator_, IsSimilarNoise(_))
      .Times(num_frames_per_packet_)
      .WillRepeatedly(Return(false));
  EXPECT_CALL(*mock_noise_estimator_, Update(_))
      .Times(num_frames_per_packet_)
      .WillRepeatedly(Return(true));
  EXPECT_CALL(*mock_vector_quantizer_, Quantize(_))
      .WillOnce(Return(mock_quantized_));

  LyraEncoderPeer encoder_peer(
      std::move(mock_resampler_), std::move(mock_feature_extractor_),
      std::move(mock_noise_estimator_), std::move(mock_vector_quantizer_),
      nullptr, sample_rate_hz_, num_frames_per_packet_,
      /*enable_dtx=*/true);
  auto encoded_or = encoder_peer.Encode(samples_span_);

  EXPECT_TRUE(encoded_or.has_value());

  Packet<0, 0> empty_packet;
  const auto packed = empty_packet.PackQuantized(std::bitset<0>{}.to_string());
  EXPECT_NE(packed, encoded_or.value());
}

TEST_P(LyraEncoderTest, NoiseDetectionReturnsEmptyPacket) {
  SetResamplerExpectation(1);
  for (int i = 0; i < num_frames_per_packet_; ++i) {
    EXPECT_CALL(
        *mock_feature_extractor_,
        Extract(internal_samples_span_.subspan(
            i * internal_num_samples_per_hop_, internal_num_samples_per_hop_)))
        .WillOnce(Return(mock_features_));
  }
  EXPECT_CALL(*mock_noise_estimator_, IsSimilarNoise(_))
      .Times(num_frames_per_packet_)
      .WillRepeatedly(Return(true));
  EXPECT_CALL(*mock_noise_estimator_, Update(_)).Times(0);
  EXPECT_CALL(*mock_vector_quantizer_, Quantize(_)).Times(0);

  LyraEncoderPeer encoder_peer(
      std::move(mock_resampler_), std::move(mock_feature_extractor_),
      std::move(mock_noise_estimator_), std::move(mock_vector_quantizer_),
      nullptr, sample_rate_hz_, num_frames_per_packet_,
      /*enable_dtx=*/true);
  auto encoded_or = encoder_peer.Encode(samples_span_);

  EXPECT_TRUE(encoded_or.has_value());

  Packet<0, 0> empty_packet;
  const auto packed = empty_packet.PackQuantized(std::bitset<0>{}.to_string());
  EXPECT_EQ(packed, encoded_or.value());
}

TEST_P(LyraEncoderTest, QuantizationFails) {
  SetResamplerExpectation(1);
  EXPECT_CALL(*mock_feature_extractor_, Extract(_))
      .Times(num_frames_per_packet_)
      .WillRepeatedly(Return(mock_features_));
  EXPECT_CALL(*mock_noise_estimator_, IsSimilarNoise(_)).Times(0);
  EXPECT_CALL(*mock_noise_estimator_, Update(_)).Times(0);
  EXPECT_CALL(*mock_vector_quantizer_, Quantize(mock_concatenated_features_))
      .WillOnce(Return(absl::nullopt));

  LyraEncoderPeer encoder_peer(
      std::move(mock_resampler_), std::move(mock_feature_extractor_),
      std::move(mock_noise_estimator_), std::move(mock_vector_quantizer_),
      nullptr, sample_rate_hz_, num_frames_per_packet_,
      /*enable_dtx=*/false);
  auto encoded_or = encoder_peer.Encode(samples_span_);

  EXPECT_FALSE(encoded_or.has_value());
}

TEST_P(LyraEncoderTest, EncodeWithFilteringModifiesSignal) {
  // Before filtering, the signal contains only a DC of 5.
  std::fill(samples_.begin(), samples_.end(), 5);
  std::fill(internal_samples_.begin(), internal_samples_.end(), 5);
  SetResamplerExpectation(1);

  // After filtering, the extractor gets an input that is filtered, which is
  // supposed to have the DC component removed. We test that at least 80% of
  // the samples_ are turned into zero.
  std::vector<int16_t> zeros_subset(
      static_cast<size_t>(internal_num_samples_per_hop_ * 0.8), 0);
  EXPECT_CALL(*mock_feature_extractor_, Extract(IsSupersetOf(zeros_subset)))
      .Times(num_frames_per_packet_)
      .WillRepeatedly(Return(mock_features_));
  EXPECT_CALL(*mock_noise_estimator_, IsSimilarNoise(_)).Times(0);
  EXPECT_CALL(*mock_noise_estimator_, Update(_)).Times(0);
  EXPECT_CALL(*mock_vector_quantizer_, Quantize(mock_concatenated_features_))
      .WillOnce(Return(mock_quantized_));

  LyraEncoderPeer encoder_peer(
      std::move(mock_resampler_), std::move(mock_feature_extractor_),
      std::move(mock_noise_estimator_), std::move(mock_vector_quantizer_),
      nullptr, sample_rate_hz_, num_frames_per_packet_,
      /*enable_dtx=*/false);
  auto encoded_or = encoder_peer.EncodeWithFiltering(samples_span_);

  EXPECT_TRUE(encoded_or.has_value());
  EXPECT_EQ(encoded_or.value().size(), kPacketSize);
  EXPECT_TRUE(DoesPacketContainQuantized(encoded_or.value(), mock_quantized_));
}

TEST_P(LyraEncoderTest, EncodeWithDenoiser) {
  std::fill(samples_.begin(), samples_.end(), 5);
  std::fill(internal_samples_.begin(), internal_samples_.end(), 5);
  SetResamplerExpectation(1);
  std::vector<int16_t> zeros_subset(
      static_cast<size_t>(internal_num_samples_per_hop_ * 0.8), 0);
  EXPECT_CALL(*mock_feature_extractor_, Extract(IsSupersetOf(zeros_subset)))
      .Times(num_frames_per_packet_)
      .WillRepeatedly(Return(mock_features_));
  EXPECT_CALL(*mock_noise_estimator_, IsSimilarNoise(_)).Times(0);
  EXPECT_CALL(*mock_noise_estimator_, Update(_)).Times(0);
  EXPECT_CALL(*mock_vector_quantizer_, Quantize(mock_concatenated_features_))
      .WillOnce(Return(mock_quantized_));

  const int denoiser_num_samples_per_hop = internal_num_samples_per_hop_ / 4;
  std::vector<int16_t> denoised_frame(denoiser_num_samples_per_hop, 5);

  EXPECT_CALL(*mock_denoiser_, SamplesPerHop())
      .WillRepeatedly(Return(denoiser_num_samples_per_hop));
  EXPECT_CALL(*mock_denoiser_, Denoise(_))
      .Times(4 * num_frames_per_packet_)
      .WillRepeatedly(Return(denoised_frame));

  LyraEncoderPeer encoder_peer(
      std::move(mock_resampler_), std::move(mock_feature_extractor_),
      std::move(mock_noise_estimator_), std::move(mock_vector_quantizer_),
      std::move(mock_denoiser_), sample_rate_hz_, num_frames_per_packet_,
      /*enable_dtx=*/false);
  auto encoded_or = encoder_peer.EncodeWithFiltering(samples_span_);

  EXPECT_TRUE(encoded_or.has_value());
  EXPECT_EQ(encoded_or.value().size(), kPacketSize);
  EXPECT_TRUE(DoesPacketContainQuantized(encoded_or.value(), mock_quantized_));
}

TEST_P(LyraEncoderTest, MultipleEncodeCalls) {
  const int kNumEncodeCalls = 5;
  SetResamplerExpectation(kNumEncodeCalls);
  for (int i = 0; i < num_frames_per_packet_; ++i) {
    EXPECT_CALL(
        *mock_feature_extractor_,
        Extract(internal_samples_span_.subspan(
            i * internal_num_samples_per_hop_, internal_num_samples_per_hop_)))
        .Times(kNumEncodeCalls)
        .WillRepeatedly(Return(mock_features_));
  }
  EXPECT_CALL(*mock_noise_estimator_, IsSimilarNoise(_)).Times(0);
  EXPECT_CALL(*mock_noise_estimator_, Update(_)).Times(0);
  EXPECT_CALL(*mock_vector_quantizer_, Quantize(mock_concatenated_features_))
      .Times(kNumEncodeCalls)
      .WillRepeatedly(Return(mock_quantized_));

  LyraEncoderPeer encoder_peer(
      std::move(mock_resampler_), std::move(mock_feature_extractor_),
      std::move(mock_noise_estimator_), std::move(mock_vector_quantizer_),
      nullptr, sample_rate_hz_, num_frames_per_packet_,
      /*enable_dtx=*/false);
  for (int i = 0; i < kNumEncodeCalls; ++i) {
    auto encoded_or = encoder_peer.Encode(samples_span_);

    EXPECT_TRUE(encoded_or.has_value());
    EXPECT_EQ(encoded_or.value().size(), kPacketSize);
    EXPECT_TRUE(
        DoesPacketContainQuantized(encoded_or.value(), mock_quantized_));
  }
}

TEST_P(LyraEncoderTest, GoodCreationParametersReturnNotNullptr) {
  const auto valid_model_path = ghc::filesystem::current_path() / "wavegru";

  EXPECT_NE(nullptr,
            LyraEncoder::Create(sample_rate_hz_, kNumChannels, kBitrate,
                                /*enable_dtx=*/false, valid_model_path));
  EXPECT_NE(nullptr,
            LyraEncoder::Create(sample_rate_hz_, kNumChannels, kBitrate,
                                /*enable_dtx=*/true, valid_model_path));
}

TEST_P(LyraEncoderTest, BadCreationParametersReturnNullptr) {
  const auto valid_model_path = ghc::filesystem::current_path() / "wavegru";

  EXPECT_EQ(nullptr,
            LyraEncoder::Create(0, kNumChannels, kBitrate,
                                /*enable_dtx=*/false, valid_model_path));
  EXPECT_EQ(nullptr,
            LyraEncoder::Create(0, kNumChannels, kBitrate,
                                /*enable_dtx=*/true, valid_model_path));
  EXPECT_EQ(nullptr,
            LyraEncoder::Create(sample_rate_hz_, -3, kBitrate,
                                /*enable_dtx=*/false, valid_model_path));
  EXPECT_EQ(nullptr,
            LyraEncoder::Create(sample_rate_hz_, -3, kBitrate,
                                /*enable_dtx=*/true, valid_model_path));
  EXPECT_EQ(nullptr,
            LyraEncoder::Create(sample_rate_hz_, kNumChannels, -2,
                                /*enable_dtx=*/false, valid_model_path));
  EXPECT_EQ(nullptr,
            LyraEncoder::Create(sample_rate_hz_, kNumChannels, -2,
                                /*enable_dtx=*/true, valid_model_path));
  EXPECT_EQ(nullptr,
            LyraEncoder::Create(sample_rate_hz_, kNumChannels, kBitrate,
                                /*enable_dtx=*/false, "bad_model_path"));
  EXPECT_EQ(nullptr,
            LyraEncoder::Create(sample_rate_hz_, kNumChannels, kBitrate,
                                /*enable_dtx=*/true, "bad_model_path"));
}

INSTANTIATE_TEST_SUITE_P(SampleRates, LyraEncoderTest,
                         Combine(ValuesIn(kSupportedSampleRates),
                                 Values(1, 2, 3)));

}  // namespace
}  // namespace codec
}  // namespace chromemedia
