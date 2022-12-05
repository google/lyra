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

#include "lyra/lyra_encoder.h"

#include <bitset>
#include <climits>
#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

// Placeholder for get runfiles header.
#include "absl/types/span.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "include/ghc/filesystem.hpp"
#include "lyra/feature_extractor_interface.h"
#include "lyra/lyra_config.h"
#include "lyra/noise_estimator_interface.h"
#include "lyra/packet.h"
#include "lyra/resampler_interface.h"
#include "lyra/testing/mock_feature_extractor.h"
#include "lyra/testing/mock_noise_estimator.h"
#include "lyra/testing/mock_resampler.h"
#include "lyra/testing/mock_vector_quantizer.h"
#include "lyra/vector_quantizer_interface.h"

namespace chromemedia {
namespace codec {

// Use a test peer to access the private constructor of LyraEncoder in order
// to inject mock dependencies.
class LyraEncoderPeer {
 public:
  explicit LyraEncoderPeer(
      std::unique_ptr<MockResampler> mock_resampler,
      std::unique_ptr<MockFeatureExtractor> mock_feature_extractor,
      std::unique_ptr<MockNoiseEstimator> mock_noise_estimator,
      std::unique_ptr<MockVectorQuantizer> mock_vector_quantizer,
      int sample_rate_hz, int num_quantized_bits, bool enable_dtx)
      : encoder_(std::move(mock_resampler), std::move(mock_feature_extractor),
                 std::move(mock_noise_estimator),
                 std::move(mock_vector_quantizer), sample_rate_hz, kNumChannels,
                 num_quantized_bits, enable_dtx) {}

  std::optional<std::vector<uint8_t>> Encode(
      const absl::Span<const int16_t> audio) {
    return encoder_.Encode(audio);
  }

  bool set_bitrate(int bitrate) { return encoder_.set_bitrate(bitrate); }

 private:
  LyraEncoder encoder_;
};

namespace {

using testing::_;
using testing::Combine;
using testing::Return;
using testing::ValuesIn;

class LyraEncoderTest
    : public testing::TestWithParam<testing::tuple<int, int>> {
 protected:
  LyraEncoderTest()
      : external_sample_rate_hz_(std::get<0>(GetParam())),
        num_quantized_bits_(std::get<1>(GetParam())),
        internal_num_samples_per_hop_(
            GetNumSamplesPerHop(kInternalSampleRateHz)),
        samples_(GetNumSamplesPerHop(external_sample_rate_hz_)),
        samples_span_(absl::MakeConstSpan(samples_)),
        internal_samples_(internal_num_samples_per_hop_),
        internal_samples_span_(absl::MakeConstSpan(internal_samples_)),
        mock_feature_extractor_(std::make_unique<MockFeatureExtractor>()),
        mock_noise_estimator_(std::make_unique<MockNoiseEstimator>()),
        mock_vector_quantizer_(std::make_unique<MockVectorQuantizer>()),
        mock_resampler_(std::make_unique<MockResampler>(
            kInternalSampleRateHz, external_sample_rate_hz_)),
        mock_features_(kNumFeatures),
        mock_noise_features_(kNumMelBins),
        mock_quantized_(num_quantized_bits_, '1') {
    // Populate the mock features, samples, and internal samples with
    // increasing values starting at 0.
    std::iota(mock_features_.begin(), mock_features_.end(), 0.f);
    std::iota(mock_noise_features_.begin(), mock_noise_features_.end(), 0.f);
    std::iota(samples_.begin(), samples_.end(), 0);
    std::iota(internal_samples_.begin(), internal_samples_.end(), 0);
  }

  bool DoesPacketContainQuantized(const std::vector<uint8_t>& packet,
                                  const std::string& quantized_string) {
    if (packet.size() < GetPacketSize(num_quantized_bits_)) {
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
    packet_data = packet_data.substr(0, num_quantized_bits_);
    return packet_data == quantized_string;
  }

  void SetResamplerExpectation(int times_if_resampling) {
    if (kInternalSampleRateHz == external_sample_rate_hz_) {
      EXPECT_CALL(*mock_resampler_, Resample(samples_span_)).Times(0);
    } else {
      EXPECT_CALL(*mock_resampler_, Resample(samples_span_))
          .Times(times_if_resampling)
          .WillRepeatedly((Return(internal_samples_)));
    }
  }

  const int external_sample_rate_hz_;
  const int num_quantized_bits_;
  const int internal_num_samples_per_hop_;
  std::vector<int16_t> samples_;
  absl::Span<const int16_t> samples_span_;
  std::vector<int16_t> internal_samples_;
  absl::Span<const int16_t> internal_samples_span_;
  std::unique_ptr<MockFeatureExtractor> mock_feature_extractor_;
  std::unique_ptr<MockNoiseEstimator> mock_noise_estimator_;
  std::unique_ptr<MockVectorQuantizer> mock_vector_quantizer_;
  std::unique_ptr<MockResampler> mock_resampler_;
  std::vector<float> mock_features_;
  std::vector<float> mock_noise_features_;
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
  EXPECT_CALL(*mock_vector_quantizer_, Quantize(_, _)).Times(0);

  LyraEncoderPeer encoder_peer(std::move(mock_resampler_),
                               std::move(mock_feature_extractor_), nullptr,
                               std::move(mock_vector_quantizer_),
                               external_sample_rate_hz_, num_quantized_bits_,
                               /*enable_dtx=*/false);
  auto encoded = encoder_peer.Encode(samples_span_);

  EXPECT_FALSE(encoded.has_value());
}

TEST_P(LyraEncoderTest, FeatureExtractionFails) {
  SetResamplerExpectation(1);
  EXPECT_CALL(*mock_feature_extractor_, Extract(internal_samples_span_))
      .WillOnce(Return(std::nullopt));
  EXPECT_CALL(*mock_vector_quantizer_, Quantize(_, _)).Times(0);

  LyraEncoderPeer encoder_peer(std::move(mock_resampler_),
                               std::move(mock_feature_extractor_), nullptr,
                               std::move(mock_vector_quantizer_),
                               external_sample_rate_hz_, num_quantized_bits_,
                               /*enable_dtx=*/false);
  auto encoded = encoder_peer.Encode(samples_span_);

  EXPECT_FALSE(encoded.has_value());
}

TEST_P(LyraEncoderTest, ReceiveSamplesFails) {
  SetResamplerExpectation(1);
  EXPECT_CALL(*mock_noise_estimator_, ReceiveSamples(internal_samples_span_))
      .Times(1)
      .WillOnce(Return(false));
  EXPECT_CALL(*mock_noise_estimator_, is_noise()).Times(0);
  EXPECT_CALL(*mock_feature_extractor_, Extract(_)).Times(0);
  EXPECT_CALL(*mock_vector_quantizer_, Quantize(_, _)).Times(0);

  LyraEncoderPeer encoder_peer(
      std::move(mock_resampler_), std::move(mock_feature_extractor_),
      std::move(mock_noise_estimator_), std::move(mock_vector_quantizer_),
      external_sample_rate_hz_, num_quantized_bits_,
      /*enable_dtx=*/true);
  auto encoded = encoder_peer.Encode(samples_span_);

  EXPECT_FALSE(encoded.has_value());
}

TEST_P(LyraEncoderTest, ReceiveSamplesSucceeds) {
  SetResamplerExpectation(1);
  EXPECT_CALL(*mock_feature_extractor_, Extract(internal_samples_span_))
      .WillOnce(Return(mock_features_));
  EXPECT_CALL(*mock_noise_estimator_, is_noise())
      .Times(1)
      .WillRepeatedly(Return(false));
  EXPECT_CALL(*mock_noise_estimator_, ReceiveSamples(_))
      .Times(1)
      .WillRepeatedly(Return(true));
  EXPECT_CALL(*mock_vector_quantizer_, Quantize(_, _))
      .WillOnce(Return(mock_quantized_));

  LyraEncoderPeer encoder_peer(
      std::move(mock_resampler_), std::move(mock_feature_extractor_),
      std::move(mock_noise_estimator_), std::move(mock_vector_quantizer_),
      external_sample_rate_hz_, num_quantized_bits_,
      /*enable_dtx=*/true);
  auto encoded = encoder_peer.Encode(samples_span_);

  EXPECT_TRUE(encoded.has_value());

  auto empty_packet = Packet<0>::Create(0, 0);
  const auto packed = empty_packet->PackQuantized(std::bitset<0>{}.to_string());
  EXPECT_NE(packed, encoded.value());
}

TEST_P(LyraEncoderTest, NoiseDetectionReturnsEmptyPacket) {
  SetResamplerExpectation(1);
  EXPECT_CALL(*mock_noise_estimator_, is_noise())
      .Times(1)
      .WillOnce(Return(true));
  EXPECT_CALL(*mock_noise_estimator_, ReceiveSamples(_))
      .Times(1)
      .WillOnce(Return(true));
  EXPECT_CALL(*mock_feature_extractor_, Extract(_)).Times(0);
  EXPECT_CALL(*mock_vector_quantizer_, Quantize(_, _)).Times(0);

  LyraEncoderPeer encoder_peer(
      std::move(mock_resampler_), std::move(mock_feature_extractor_),
      std::move(mock_noise_estimator_), std::move(mock_vector_quantizer_),
      external_sample_rate_hz_, num_quantized_bits_,
      /*enable_dtx=*/true);
  auto encoded = encoder_peer.Encode(samples_span_);

  EXPECT_TRUE(encoded.has_value());

  auto empty_packet = Packet<0>::Create(0, 0);
  const auto packed = empty_packet->PackQuantized(std::bitset<0>{}.to_string());
  EXPECT_EQ(packed, encoded.value());
}

TEST_P(LyraEncoderTest, QuantizationFails) {
  SetResamplerExpectation(1);
  EXPECT_CALL(*mock_feature_extractor_, Extract(_))
      .Times(1)
      .WillRepeatedly(Return(mock_features_));
  EXPECT_CALL(*mock_vector_quantizer_,
              Quantize(mock_features_, num_quantized_bits_))
      .WillOnce(Return(std::nullopt));

  LyraEncoderPeer encoder_peer(std::move(mock_resampler_),
                               std::move(mock_feature_extractor_), nullptr,
                               std::move(mock_vector_quantizer_),
                               external_sample_rate_hz_, num_quantized_bits_,
                               /*enable_dtx=*/false);
  auto encoded = encoder_peer.Encode(samples_span_);

  EXPECT_FALSE(encoded.has_value());
}

TEST_P(LyraEncoderTest, MultipleEncodeCalls) {
  const int kNumEncodeCalls = 5;
  SetResamplerExpectation(kNumEncodeCalls);
  EXPECT_CALL(*mock_feature_extractor_, Extract(_))
      .Times(kNumEncodeCalls)
      .WillRepeatedly(Return(mock_features_));
  EXPECT_CALL(*mock_vector_quantizer_,
              Quantize(mock_features_, num_quantized_bits_))
      .Times(kNumEncodeCalls)
      .WillRepeatedly(Return(mock_quantized_));

  LyraEncoderPeer encoder_peer(std::move(mock_resampler_),
                               std::move(mock_feature_extractor_), nullptr,
                               std::move(mock_vector_quantizer_),
                               external_sample_rate_hz_, num_quantized_bits_,
                               /*enable_dtx=*/false);
  for (int i = 0; i < kNumEncodeCalls; ++i) {
    auto encoded = encoder_peer.Encode(samples_span_);

    EXPECT_TRUE(encoded.has_value());
    EXPECT_EQ(encoded.value().size(), GetPacketSize(num_quantized_bits_));
    EXPECT_TRUE(DoesPacketContainQuantized(encoded.value(), mock_quantized_));
  }
}

TEST_P(LyraEncoderTest, GoodCreationParametersReturnNotNullptr) {
  const auto valid_model_path =
      ghc::filesystem::current_path() / "lyra/model_coeffs";

  EXPECT_NE(nullptr,
            LyraEncoder::Create(external_sample_rate_hz_, kNumChannels,
                                GetBitrate(num_quantized_bits_),
                                /*enable_dtx=*/false, valid_model_path));
  EXPECT_NE(nullptr,
            LyraEncoder::Create(external_sample_rate_hz_, kNumChannels,
                                GetBitrate(num_quantized_bits_),
                                /*enable_dtx=*/true, valid_model_path));
}

TEST_P(LyraEncoderTest, BadCreationParametersReturnNullptr) {
  const auto valid_model_path =
      ghc::filesystem::current_path() / "lyra/model_coeffs";

  EXPECT_EQ(nullptr, LyraEncoder::Create(
                         0, kNumChannels, GetBitrate(num_quantized_bits_),
                         /*enable_dtx=*/false, valid_model_path));
  EXPECT_EQ(nullptr, LyraEncoder::Create(
                         0, kNumChannels, GetBitrate(num_quantized_bits_),
                         /*enable_dtx=*/true, valid_model_path));
  EXPECT_EQ(nullptr,
            LyraEncoder::Create(external_sample_rate_hz_, -3,
                                GetBitrate(num_quantized_bits_),
                                /*enable_dtx=*/false, valid_model_path));
  EXPECT_EQ(nullptr,
            LyraEncoder::Create(external_sample_rate_hz_, -3,
                                GetBitrate(num_quantized_bits_),
                                /*enable_dtx=*/true, valid_model_path));
  EXPECT_EQ(nullptr,
            LyraEncoder::Create(external_sample_rate_hz_, kNumChannels, -2,
                                /*enable_dtx=*/false, valid_model_path));
  EXPECT_EQ(nullptr,
            LyraEncoder::Create(external_sample_rate_hz_, kNumChannels, -2,
                                /*enable_dtx=*/true, valid_model_path));
  EXPECT_EQ(nullptr,
            LyraEncoder::Create(external_sample_rate_hz_, kNumChannels,
                                GetBitrate(num_quantized_bits_),
                                /*enable_dtx=*/false, "bad_model_path"));
  EXPECT_EQ(nullptr,
            LyraEncoder::Create(external_sample_rate_hz_, kNumChannels,
                                GetBitrate(num_quantized_bits_),
                                /*enable_dtx=*/true, "bad_model_path"));
}

TEST_P(LyraEncoderTest, SetBitrateSucceeds) {
  LyraEncoderPeer encoder_peer(std::move(mock_resampler_),
                               std::move(mock_feature_extractor_), nullptr,
                               std::move(mock_vector_quantizer_),
                               external_sample_rate_hz_, num_quantized_bits_,
                               /*enable_dtx=*/false);
  EXPECT_TRUE(encoder_peer.set_bitrate(GetBitrate(num_quantized_bits_)));
}

TEST_P(LyraEncoderTest, SetBitrateFails) {
  LyraEncoderPeer encoder_peer(std::move(mock_resampler_),
                               std::move(mock_feature_extractor_), nullptr,
                               std::move(mock_vector_quantizer_),
                               external_sample_rate_hz_, num_quantized_bits_,
                               /*enable_dtx=*/false);
  EXPECT_FALSE(encoder_peer.set_bitrate(0));
}

INSTANTIATE_TEST_SUITE_P(SampleRatesQuantizedBitsAndHopsPerPacket,
                         LyraEncoderTest,
                         Combine(ValuesIn(kSupportedSampleRates),
                                 ValuesIn(GetSupportedQuantizedBits())));

}  // namespace
}  // namespace codec
}  // namespace chromemedia
