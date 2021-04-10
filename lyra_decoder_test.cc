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

#include "lyra_decoder.h"

#include <algorithm>
#include <bitset>
#include <climits>
#include <cstdint>
#include <iterator>
#include <memory>
#include <numeric>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

// placeholder for get runfiles header.
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/memory/memory.h"
#include "absl/random/random.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"  // IWYU pragma: keep
#include "absl/types/span.h"
#include "include/ghc/filesystem.hpp"
#include "generative_model_interface.h"
#include "log_mel_spectrogram_extractor_impl.h"
#include "lyra_config.h"
#include "packet.h"
#include "packet_interface.h"
#include "packet_loss_handler_interface.h"
#include "resampler.h"
#include "resampler_interface.h"
#include "testing/mock_generative_model.h"
#include "testing/mock_packet_loss_handler.h"
#include "testing/mock_resampler.h"
#include "testing/mock_vector_quantizer.h"
#include "vector_quantizer_interface.h"

namespace chromemedia {
namespace codec {
namespace {

constexpr int kNumQuantizedBits = 120;
constexpr int kNumHeaderBits = 0;

}  // namespace

// Use a test peer to access the private constructor of LyraDecoder in order
// to inject a MockGenerativeModel.
class LyraDecoderPeer {
 public:
  explicit LyraDecoderPeer(
      std::unique_ptr<MockGenerativeModel> mock_generative_model,
      std::unique_ptr<MockGenerativeModel> mock_comfort_noise_generator,
      std::unique_ptr<MockVectorQuantizer> mock_vector_quantizer,
      std::unique_ptr<MockPacketLossHandler> mock_packet_loss_handler,
      std::unique_ptr<ResamplerInterface> resampler, int sample_rate_hz,
      int num_frames_per_packet)
      : decoder_(std::move(mock_generative_model),
                 std::move(mock_comfort_noise_generator),
                 std::move(mock_vector_quantizer),
                 absl::make_unique<Packet<kNumQuantizedBits, kNumHeaderBits>>(),
                 std::move(mock_packet_loss_handler), std::move(resampler),
                 sample_rate_hz, kNumChannels, kBitrate,
                 num_frames_per_packet) {}

  bool SetEncodedPacket(const absl::Span<const uint8_t> encoded) {
    return decoder_.SetEncodedPacket(encoded);
  }

  absl::optional<std::vector<int16_t>> DecodeSamples(int num_samples) {
    return decoder_.DecodeSamples(num_samples);
  }

  absl::optional<std::vector<int16_t>> DecodePacketLoss(int num_samples) {
    return decoder_.DecodePacketLoss(num_samples);
  }

  absl::optional<std::vector<int16_t>> OverlapFrames(
      const std::vector<int16_t>& preceding_frame,
      const std::vector<int16_t>& following_frame) {
    return decoder_.OverlapFrames(preceding_frame, following_frame);
  }

 private:
  LyraDecoder decoder_;
};

namespace {

using testing::Return;

static constexpr absl::string_view kExportedModelPath =
    "wavegru";

class LyraDecoderTest
    : public testing::TestWithParam<testing::tuple<int, int>> {
 protected:
  typedef Packet<kNumQuantizedBits, kNumHeaderBits> PacketType;

  LyraDecoderTest()
      : sample_rate_hz_(std::get<0>(GetParam())),
        num_frames_per_packet_(std::get<1>(GetParam())),
        model_path_(ghc::filesystem::current_path() /
                    kExportedModelPath),
        mock_concatenated_features_(num_frames_per_packet_ * kNumFeatures),
        mock_feature_frames_(num_frames_per_packet_),
        mock_samples_(
            GetNumSamplesPerHop(GetInternalSampleRate(sample_rate_hz_))),
        output_mock_samples_(GetNumSamplesPerHop(sample_rate_hz_)) {
    // Fill |mock_concatenated_features_| with monotonically increasing values
    // at each index.
    std::iota(mock_concatenated_features_.begin(),
              mock_concatenated_features_.end(), 0);

    // Organize |mock_concatenated_features_| into frames.
    for (int i = 0; i < num_frames_per_packet_; ++i) {
      mock_feature_frames_[i] = std::vector<float>(
          mock_concatenated_features_.begin() + kNumFeatures * i,
          mock_concatenated_features_.begin() + kNumFeatures * (i + 1));
    }

    // Fill |mock_samples_| with monotonically increasing values at each index.
    std::iota(mock_samples_->begin(), mock_samples_->end(), 0);
    // Fill |output_mock_samples_| with monotonically increasing values at each
    // index.
    std::iota(output_mock_samples_.begin(), output_mock_samples_.end(), 0);
  }

  std::unique_ptr<MockResampler> GetResampler(int num_calls) {
    auto resampler = absl::make_unique<MockResampler>();
    if (GetInternalSampleRate(sample_rate_hz_) == sample_rate_hz_) {
      EXPECT_CALL(*resampler,
                  Resample(absl::MakeConstSpan(mock_samples_.value())))
          .Times(0);
    } else {
      EXPECT_CALL(*resampler,
                  Resample(absl::MakeConstSpan(mock_samples_.value())))
          .Times(testing::Exactly(num_calls))
          .WillRepeatedly(Return(output_mock_samples_));
    }
    return resampler;
  }

  const int sample_rate_hz_;
  const int num_frames_per_packet_;
  const ghc::filesystem::path model_path_;
  std::vector<float> mock_concatenated_features_;
  std::vector<std::vector<float>> mock_feature_frames_;
  absl::optional<std::vector<int16_t>> mock_samples_;
  std::vector<int16_t> output_mock_samples_;
};

TEST_P(LyraDecoderTest, PacketAllZerosSucceeds) {
  // Fill a packet with the bit pattern 00000000 at each byte.
  std::bitset<kNumQuantizedBits> quantized(0);
  PacketType packet;
  std::vector<uint8_t> encoded = packet.PackQuantized(quantized.to_string());
  auto mock_vector_quantizer = absl::make_unique<MockVectorQuantizer>();
  EXPECT_CALL(*mock_vector_quantizer,
              DecodeToLossyFeatures(quantized.to_string()))
      .WillOnce(Return(mock_concatenated_features_));
  auto mock_packet_loss_handler = absl::make_unique<MockPacketLossHandler>();
  auto mock_generative_model = absl::make_unique<MockGenerativeModel>();
  for (const auto& mock_features : mock_feature_frames_) {
    EXPECT_CALL(*mock_packet_loss_handler, SetReceivedFeatures(mock_features))
        .WillOnce(Return(true));
    EXPECT_CALL(*mock_generative_model, AddFeatures(mock_features));
  }
  const int num_requested_samples = output_mock_samples_.size();
  const int num_samples_to_generate = mock_samples_->size();
  EXPECT_CALL(*mock_generative_model, GenerateSamples(num_samples_to_generate))
      .WillOnce(Return(mock_samples_));
  auto mock_comfort_noise_generator = absl::make_unique<MockGenerativeModel>();
  EXPECT_CALL(*mock_comfort_noise_generator, AddFeatures(testing::_)).Times(0);
  EXPECT_CALL(*mock_comfort_noise_generator, GenerateSamples(testing::_))
      .Times(0);
  auto lyra_decoder_peer = absl::make_unique<LyraDecoderPeer>(
      std::move(mock_generative_model), std::move(mock_comfort_noise_generator),
      std::move(mock_vector_quantizer), std::move(mock_packet_loss_handler),
      GetResampler(1), sample_rate_hz_, num_frames_per_packet_);

  ASSERT_TRUE(lyra_decoder_peer->SetEncodedPacket(encoded));
  auto decoded_or = lyra_decoder_peer->DecodeSamples(num_requested_samples);

  ASSERT_TRUE(decoded_or.has_value());
  EXPECT_EQ(decoded_or.value(), output_mock_samples_);
}

TEST_P(LyraDecoderTest, DecodePacketLossWithoutPriorPacketSucceeds) {
  const std::vector<float> estimated_features(kNumFeatures, 23.0f);
  auto mock_vector_quantizer = absl::make_unique<MockVectorQuantizer>();
  auto mock_generative_model = absl::make_unique<MockGenerativeModel>();

  const int internal_num_samples = mock_samples_->size();
  EXPECT_CALL(*mock_generative_model, GenerateSamples(internal_num_samples))
      .WillOnce(Return(mock_samples_));
  EXPECT_CALL(*mock_generative_model, AddFeatures(estimated_features));
  auto mock_comfort_noise_generator = absl::make_unique<MockGenerativeModel>();
  EXPECT_CALL(*mock_comfort_noise_generator, AddFeatures(testing::_)).Times(0);
  EXPECT_CALL(*mock_comfort_noise_generator, GenerateSamples(testing::_))
      .Times(0);
  auto mock_packet_loss_handler = absl::make_unique<MockPacketLossHandler>();
  EXPECT_CALL(*mock_packet_loss_handler,
              EstimateLostFeatures(internal_num_samples))
      .WillOnce(Return(estimated_features));
  auto lyra_decoder_peer = absl::make_unique<LyraDecoderPeer>(
      std::move(mock_generative_model), std::move(mock_comfort_noise_generator),
      std::move(mock_vector_quantizer), std::move(mock_packet_loss_handler),
      GetResampler(1), sample_rate_hz_, num_frames_per_packet_);

  // Without first calling SetEncodedPacket() with any prior packet,
  // DecodePacketLoss() can still return audio samples.
  const int num_samples = output_mock_samples_.size();
  auto decoded_or = lyra_decoder_peer->DecodePacketLoss(num_samples);
  ASSERT_TRUE(decoded_or.has_value());
  EXPECT_EQ(decoded_or.value(), output_mock_samples_);
}

TEST_P(LyraDecoderTest, DecodeSamplesWithoutPriorPacketFails) {
  auto mock_vector_quantizer = absl::make_unique<MockVectorQuantizer>();
  auto mock_generative_model = absl::make_unique<MockGenerativeModel>();

  EXPECT_CALL(*mock_generative_model, GenerateSamples(testing::_)).Times(0);
  EXPECT_CALL(*mock_generative_model, AddFeatures(testing::_)).Times(0);
  auto mock_comfort_noise_generator = absl::make_unique<MockGenerativeModel>();
  EXPECT_CALL(*mock_comfort_noise_generator, AddFeatures(testing::_)).Times(0);
  EXPECT_CALL(*mock_comfort_noise_generator, GenerateSamples(testing::_))
      .Times(0);
  auto mock_packet_loss_handler = absl::make_unique<MockPacketLossHandler>();
  auto lyra_decoder_peer = absl::make_unique<LyraDecoderPeer>(
      std::move(mock_generative_model), std::move(mock_comfort_noise_generator),
      std::move(mock_vector_quantizer), std::move(mock_packet_loss_handler),
      GetResampler(0), sample_rate_hz_, num_frames_per_packet_);

  // Without first calling SetEncodedPacket() with any prior packet,
  // DecodeSamples() does not return audio samples.
  auto decoded_or =
      lyra_decoder_peer->DecodeSamples(GetNumSamplesPerHop(sample_rate_hz_));
  EXPECT_FALSE(decoded_or.has_value());
}

TEST_P(LyraDecoderTest, PlcSamplesStraddlePacketBoundary) {
  // Step 1: Add one encoded packet containing N frames.
  // Step 2: Completely decode N - 1 frames worth of (M) samples.
  // Step 3: Partially decode the last frame (M - 20 out of M samples).
  // Step 4: Try and fail to decode more samples than are left (62 vs 20).
  // Step 5: Make one PLC call requesting 62 samples, which use up all the
  //         remaining 20 samples from a normal packet and then add new PLC
  //         features to generate 42 samples.
  // Step 6: Make one PLC call requesting M - 40 samples, which use up the
  //         remaining M - 42 samples and then add new PLC features to generate
  //         2 samples.
  const int kNumTotalPackets = 3;
  std::bitset<kNumQuantizedBits> quantized_zeros(0);
  PacketType packet;
  std::vector<uint8_t> encoded_zeros =
      packet.PackQuantized(quantized_zeros.to_string());

  // The vector quantizer will be called only once.
  auto mock_vector_quantizer = absl::make_unique<MockVectorQuantizer>();
  EXPECT_CALL(*mock_vector_quantizer,
              DecodeToLossyFeatures(quantized_zeros.to_string()))
      .WillOnce(Return(mock_concatenated_features_));

  // Add N frames of features from encoded packet in Step 1.
  auto mock_packet_loss_handler = absl::make_unique<MockPacketLossHandler>();
  auto mock_generative_model = absl::make_unique<MockGenerativeModel>();
  for (const auto& mock_features : mock_feature_frames_) {
    EXPECT_CALL(*mock_packet_loss_handler, SetReceivedFeatures(mock_features))
        .WillOnce(Return(true));
    EXPECT_CALL(*mock_generative_model, AddFeatures(mock_features)).Times(1);
  }

  // All mocks have their sample vector sizes expressed at
  // |kInternalSampleRateHz|.
  // Generate N - 1 frames worth of (M) samples in Step 2.
  const int internal_frame_size = mock_samples_->size();
  absl::optional<std::vector<int16_t>> complete_decode_samples_call(
      {mock_samples_->begin(), mock_samples_->begin() + internal_frame_size});
  if (num_frames_per_packet_ > 1) {
    EXPECT_CALL(*mock_generative_model, GenerateSamples(internal_frame_size))
        .Times(num_frames_per_packet_ - 1)
        .WillRepeatedly(Return(complete_decode_samples_call));
  }

  // Generate M - 20 samples from the last frame in Step 3.
  absl::optional<std::vector<int16_t>> decode_samples_call_0(
      {mock_samples_->begin(),
       mock_samples_->begin() + internal_frame_size - 20});
  EXPECT_CALL(*mock_generative_model, GenerateSamples(internal_frame_size - 20))
      .WillOnce(Return(decode_samples_call_0));

  // Add features estimated by PLC in Step 5 & 6.
  const std::vector<float> estimated_features(kNumFeatures, 11.0f);
  EXPECT_CALL(*mock_generative_model, AddFeatures(estimated_features))
      .Times(kNumTotalPackets - 1);

  // Request 62 samples in PLC mode in Step 5, internally break this up into
  // two calls: the remaining 20 samples from the normal packet and 42
  // samples from a PLC packet.
  absl::optional<std::vector<int16_t>> plc_samples_call_0(
      {mock_samples_->begin() + internal_frame_size - 20,
       mock_samples_->begin() + internal_frame_size});
  EXPECT_CALL(*mock_generative_model, GenerateSamples(20))
      .WillOnce(Return(plc_samples_call_0));
  absl::optional<std::vector<int16_t>> plc_samples_call_1(
      {mock_samples_->begin(), mock_samples_->begin() + 42});
  EXPECT_CALL(*mock_generative_model, GenerateSamples(42))
      .WillOnce(Return(plc_samples_call_1));

  // Request M - 40 samples in PLC mode in Step 6, internally break this up into
  // two calls: the remaining M - 42 samples from the first PLC packet and 2
  // samples from a second PLC packet.
  absl::optional<std::vector<int16_t>> plc_samples_call_2(
      {mock_samples_->begin() + 42,
       mock_samples_->begin() + internal_frame_size});
  EXPECT_CALL(*mock_generative_model, GenerateSamples(internal_frame_size - 42))
      .WillOnce(Return(plc_samples_call_2));
  absl::optional<std::vector<int16_t>> plc_samples_call_3(
      {mock_samples_->begin(), mock_samples_->begin() + 2});
  EXPECT_CALL(*mock_generative_model, GenerateSamples(2))
      .WillOnce(Return(plc_samples_call_3));

  // Comfort noise generator not involved.
  auto mock_comfort_noise_generator = absl::make_unique<MockGenerativeModel>();
  EXPECT_CALL(*mock_comfort_noise_generator, AddFeatures(testing::_)).Times(0);
  EXPECT_CALL(*mock_comfort_noise_generator, GenerateSamples(testing::_))
      .Times(0);

  // Two estimated features will be used for decoding in Step 5 & 6.
  EXPECT_CALL(*mock_packet_loss_handler, EstimateLostFeatures(testing::_))
      .Times(kNumTotalPackets - 1)
      .WillRepeatedly(Return(estimated_features));

  // Use a real resampler as this test is only concerned with behaviour before
  // any resampling. We need the real resampler to avoid failing resampling
  // CHECKs in lyra_decoder.
  auto resampler = Resampler::Create(GetInternalSampleRate(sample_rate_hz_),
                                     sample_rate_hz_);
  auto lyra_decoder_peer = absl::make_unique<LyraDecoderPeer>(
      std::move(mock_generative_model), std::move(mock_comfort_noise_generator),
      std::move(mock_vector_quantizer), std::move(mock_packet_loss_handler),
      std::move(resampler), sample_rate_hz_, num_frames_per_packet_);

  // Step 1: Add a packet with N frames.
  ASSERT_TRUE(lyra_decoder_peer->SetEncodedPacket(encoded_zeros));

  // Step 2: Decode N - 1 complete frames.
  // The literals in |num_samples_request_*| expressions are at
  // |kInternalSampleRateHz| and then converted to |sample_rate_hz_| to match
  // the interface expectation.
  const int complete_num_samples_request = ConvertNumSamplesBetweenSampleRate(
      internal_frame_size, GetInternalSampleRate(sample_rate_hz_),
      sample_rate_hz_);
  for (int i = 0; i < num_frames_per_packet_ - 1; ++i) {
    EXPECT_TRUE(lyra_decoder_peer->DecodeSamples(complete_num_samples_request)
                    .has_value());
  }

  // Step 3: Decode M - 20 samples from the last frame.
  const int num_samples_request_0 = ConvertNumSamplesBetweenSampleRate(
      internal_frame_size - 20, GetInternalSampleRate(sample_rate_hz_),
      sample_rate_hz_);
  EXPECT_TRUE(
      lyra_decoder_peer->DecodeSamples(num_samples_request_0).has_value());

  // Step 4: A |DecodeSamples| request for more than 20 samples should return
  // nullopt.
  const int num_samples_request_1 = ConvertNumSamplesBetweenSampleRate(
      62, GetInternalSampleRate(sample_rate_hz_), sample_rate_hz_);
  EXPECT_FALSE(
      lyra_decoder_peer->DecodeSamples(num_samples_request_1).has_value());

  // Step 5: Requesting 62 samples in PLC mode is valid.
  EXPECT_TRUE(
      lyra_decoder_peer->DecodePacketLoss(num_samples_request_1).has_value());

  // Step 6: Requesting M - 40 samples in PLC mode is valid too.
  const int num_samples_request_2 = ConvertNumSamplesBetweenSampleRate(
      internal_frame_size - 40, GetInternalSampleRate(sample_rate_hz_),
      sample_rate_hz_);
  EXPECT_TRUE(lyra_decoder_peer->DecodePacketLoss(num_samples_request_2));
}

TEST_P(LyraDecoderTest, MultipleLostPackets) {
  // Requests 3 PLC packets without adding real features.
  static constexpr int kNumLostPackets = 3;
  const std::vector<float> estimated_features(kNumFeatures, 17.0f);
  auto mock_vector_quantizer = absl::make_unique<MockVectorQuantizer>();
  auto mock_generative_model = absl::make_unique<MockGenerativeModel>();
  const int internal_num_samples = mock_samples_->size();
  // PLC calls are made before any real packets are added.
  EXPECT_CALL(*mock_generative_model, GenerateSamples(internal_num_samples))
      .Times(kNumLostPackets)
      .WillRepeatedly(Return(mock_samples_));
  EXPECT_CALL(*mock_generative_model, AddFeatures(estimated_features))
      .Times(kNumLostPackets);
  auto mock_comfort_noise_generator = absl::make_unique<MockGenerativeModel>();
  EXPECT_CALL(*mock_comfort_noise_generator, AddFeatures(testing::_)).Times(0);
  EXPECT_CALL(*mock_comfort_noise_generator, GenerateSamples(testing::_))
      .Times(0);
  auto mock_packet_loss_handler = absl::make_unique<MockPacketLossHandler>();
  EXPECT_CALL(*mock_packet_loss_handler,
              EstimateLostFeatures(internal_num_samples))
      .Times(kNumLostPackets)
      .WillRepeatedly(Return(estimated_features));
  auto lyra_decoder_peer = absl::make_unique<LyraDecoderPeer>(
      std::move(mock_generative_model), std::move(mock_comfort_noise_generator),
      std::move(mock_vector_quantizer), std::move(mock_packet_loss_handler),
      GetResampler(kNumLostPackets), sample_rate_hz_, num_frames_per_packet_);

  const int num_samples = output_mock_samples_.size();
  for (int i = 0; i < kNumLostPackets; ++i) {
    auto decoded_or = lyra_decoder_peer->DecodePacketLoss(num_samples);
    ASSERT_TRUE(decoded_or.has_value());
    EXPECT_EQ(decoded_or.value(), output_mock_samples_);
  }
}

TEST_P(LyraDecoderTest, OneLostPacketMultipleRequests) {
  // A packet is lost, but there are several calls to DecodePacketLoss(), each
  // requesting a different number of samples. The total number of samples
  // requested does not exceed the frame size.
  std::vector<float> estimated_features(kNumFeatures, 13.0f);
  auto mock_vector_quantizer = absl::make_unique<MockVectorQuantizer>();
  auto mock_generative_model = absl::make_unique<MockGenerativeModel>();
  auto mock_comfort_noise_generator = absl::make_unique<MockGenerativeModel>();
  auto mock_packet_loss_handler = absl::make_unique<MockPacketLossHandler>();
  auto resampler = absl::make_unique<MockResampler>();

  // Prepare 4 requested numbers of samples whose sum < frame size M:
  // 7 + (M / 4 - 15) + (M / 4 + 2) + (M / 2 - 5) <= M - 11.
  const int frame_size = output_mock_samples_.size();
  const std::vector<int> num_samples_requested = {
      7, frame_size / 4 - 15, frame_size / 4 + 2, frame_size / 2 - 5};

  std::vector<std::vector<int16_t>> partial_mock_samples;
  std::vector<std::vector<int16_t>> partial_output_mock_samples;
  for (int i = 0; i < num_samples_requested.size(); ++i) {
    const int internal_num_samples = ConvertNumSamplesBetweenSampleRate(
        num_samples_requested[i], sample_rate_hz_,
        GetInternalSampleRate(sample_rate_hz_));
    partial_mock_samples.push_back(
        {mock_samples_->begin(),
         mock_samples_->begin() + internal_num_samples});
    if (i == 0) {
      const std::vector<int16_t> kEmptySamples;
      EXPECT_CALL(*mock_generative_model, GenerateSamples(internal_num_samples))
          .WillOnce(Return(kEmptySamples))
          .WillOnce(Return(partial_mock_samples.back()));
      EXPECT_CALL(*mock_generative_model, AddFeatures(estimated_features));
    } else {
      EXPECT_CALL(*mock_generative_model, GenerateSamples(internal_num_samples))
          .WillOnce(Return(partial_mock_samples.back()));
    }

    EXPECT_CALL(*mock_packet_loss_handler,
                EstimateLostFeatures(internal_num_samples))
        .WillOnce(Return(estimated_features));

    EXPECT_CALL(*mock_comfort_noise_generator, AddFeatures(testing::_))
        .Times(0);
    EXPECT_CALL(*mock_comfort_noise_generator, GenerateSamples(testing::_))
        .Times(0);

    // Set expectation for |resampler|.
    partial_output_mock_samples.push_back(
        {output_mock_samples_.begin(),
         output_mock_samples_.begin() + num_samples_requested[i]});
    if (GetInternalSampleRate(sample_rate_hz_) == sample_rate_hz_) {
      EXPECT_CALL(*resampler,
                  Resample(absl::MakeConstSpan(partial_mock_samples.back())))
          .Times(0);
    } else {
      EXPECT_CALL(*resampler,
                  Resample(absl::MakeConstSpan(partial_mock_samples.back())))
          .WillOnce(Return(partial_output_mock_samples.back()));
    }
  }

  auto lyra_decoder_peer = absl::make_unique<LyraDecoderPeer>(
      std::move(mock_generative_model), std::move(mock_comfort_noise_generator),
      std::move(mock_vector_quantizer), std::move(mock_packet_loss_handler),
      std::move(resampler), sample_rate_hz_, num_frames_per_packet_);
  for (int i = 0; i < num_samples_requested.size(); ++i) {
    auto decoded_or =
        lyra_decoder_peer->DecodePacketLoss(num_samples_requested[i]);
    ASSERT_TRUE(decoded_or.has_value());
    EXPECT_EQ(decoded_or.value(), partial_output_mock_samples[i]);
  }
}

TEST_P(LyraDecoderTest, ModelsTransitionsAndTriggerOverlap) {
  // Test that models transition and are overlapped when necessary. This test is
  // solely concerned with verifying that overlap happens and does not check for
  // correctness. The generative model is called once on a packet, then two
  // packets are lost, then the generative model is called one more time on a
  // packet.
  static constexpr int kNumPacketDecodes = 2;
  static constexpr int kNumLostPackets = 2;
  const int internal_num_samples = mock_samples_->size();
  std::bitset<kNumQuantizedBits> quantized(0);
  PacketType packet;
  std::vector<uint8_t> encoded = packet.PackQuantized(quantized.to_string());
  auto mock_vector_quantizer = absl::make_unique<MockVectorQuantizer>();
  EXPECT_CALL(*mock_vector_quantizer,
              DecodeToLossyFeatures(quantized.to_string()))
      .Times(kNumPacketDecodes)
      .WillRepeatedly(Return(mock_concatenated_features_));

  auto mock_packet_loss_handler = absl::make_unique<MockPacketLossHandler>();
  auto mock_generative_model = absl::make_unique<MockGenerativeModel>();

  // Called in SetEncodedPacket() in Step 1 and 3 below.
  for (const auto& mock_features : mock_feature_frames_) {
    EXPECT_CALL(*mock_packet_loss_handler, SetReceivedFeatures(mock_features))
        .Times(kNumPacketDecodes)
        .WillRepeatedly(Return(true));
    EXPECT_CALL(*mock_generative_model, AddFeatures(mock_features))
        .Times(kNumPacketDecodes);
  }

  // Called in DecodeSamples() in Step 1 and 3 below, as well as one extra time
  // in Step 2 below to account for the overlap when transitioning from CNG
  // to the generative model.
  EXPECT_CALL(*mock_generative_model, GenerateSamples(internal_num_samples))
      .Times(kNumPacketDecodes + 1)
      .WillRepeatedly(Return(mock_samples_));

  // In Step 2, in order to account for the overlap, the generative model's
  // AddFeatures() may be called one extra time using the estimated features.
  // This happens only if |num_frames_per_packet_| is 1. Otherwise there would
  // be enough leftover features from the previous packet to generate samples.
  const std::vector<float> mock_estimated_features(kNumFeatures, 10.0f);
  if (num_frames_per_packet_ == 1) {
    EXPECT_CALL(*mock_generative_model, AddFeatures(mock_estimated_features))
        .Times(1);
  }

  // CNG's AddFeatures() and GenerateSamples() are called |kNumLostPackets|
  // |kNumLostPackets| times in Step 2, as well as one extra time to account
  // for the overlap when transitioning from CNG to the generative model.
  auto mock_comfort_noise_generator = absl::make_unique<MockGenerativeModel>();
  EXPECT_CALL(*mock_comfort_noise_generator,
              AddFeatures(mock_estimated_features))
      .Times(kNumLostPackets + 1);
  EXPECT_CALL(*mock_comfort_noise_generator,
              GenerateSamples(internal_num_samples))
      .Times(kNumLostPackets + 1)
      .WillRepeatedly(Return(mock_samples_));

  // This test is not concerned with the behavior of the resampler, so use real
  // one.
  auto resampler = Resampler::Create(GetInternalSampleRate(sample_rate_hz_),
                                     sample_rate_hz_);

  // EstimateLostFeatures() is |kNumLostPackets| times in Step 2 and one extra
  // time in Step 3 because of the overlap. This extra call does not mess with
  // the class's internal comfort noise identification logic, as it happens
  // when the transition is from CNG to the generative model.
  EXPECT_CALL(*mock_packet_loss_handler,
              EstimateLostFeatures(internal_num_samples))
      .Times(kNumLostPackets + 1)
      .WillRepeatedly(Return(mock_estimated_features));
  EXPECT_CALL(*mock_packet_loss_handler, is_comfort_noise())
      .Times(kNumLostPackets)
      .WillRepeatedly(Return(true));

  auto lyra_decoder_peer = absl::make_unique<LyraDecoderPeer>(
      std::move(mock_generative_model), std::move(mock_comfort_noise_generator),
      std::move(mock_vector_quantizer), std::move(mock_packet_loss_handler),
      std::move(resampler), sample_rate_hz_, num_frames_per_packet_);

  // Step 1: Decode a packet with the generative model.
  const int num_samples = output_mock_samples_.size();
  ASSERT_TRUE(lyra_decoder_peer->SetEncodedPacket(encoded));
  EXPECT_TRUE(lyra_decoder_peer->DecodeSamples(num_samples).has_value());

  // Step 2: Call DecodePacketLoss() |kNumLostPackets| = 2 times to ensure
  // |prev_frame_was_comfort_noise_| = true. Overlap is expected here.
  EXPECT_TRUE(lyra_decoder_peer->DecodePacketLoss(num_samples).has_value());
  EXPECT_TRUE(lyra_decoder_peer->DecodePacketLoss(num_samples).has_value());

  // Step 3: Decode one more packet with the generative model. Overlap is
  // expected here.
  ASSERT_TRUE(lyra_decoder_peer->SetEncodedPacket(encoded));
  EXPECT_TRUE(lyra_decoder_peer->DecodeSamples(num_samples).has_value());
}

TEST_P(LyraDecoderTest, FrameSizesDiffer) {
  // Test that OverlapFrames() does not try to overlap two frames of different
  // sizes.
  auto mock_vector_quantizer = absl::make_unique<MockVectorQuantizer>();
  auto mock_generative_model = absl::make_unique<MockGenerativeModel>();
  auto mock_comfort_noise_generator = absl::make_unique<MockGenerativeModel>();
  auto mock_packet_loss_handler = absl::make_unique<MockPacketLossHandler>();
  auto resampler = absl::make_unique<MockResampler>();

  auto lyra_decoder_peer = absl::make_unique<LyraDecoderPeer>(
      std::move(mock_generative_model), std::move(mock_comfort_noise_generator),
      std::move(mock_vector_quantizer), std::move(mock_packet_loss_handler),
      std::move(resampler), sample_rate_hz_, num_frames_per_packet_);

  const int frame_size = output_mock_samples_.size();
  const std::vector<int16_t> preceding_frame(frame_size, 0.0);
  const std::vector<int16_t> following_frame(frame_size + 1, 0.0);

  EXPECT_FALSE(
      lyra_decoder_peer->OverlapFrames(preceding_frame, following_frame)
          .has_value());
}

TEST_P(LyraDecoderTest, FramesAreOverlappedCorrectly) {
  // Test overlap for correctness. Given two frames, where the values of one are
  // always above the values are the other, the values of the overlapped frame
  // should land somewhere in between.
  auto mock_vector_quantizer = absl::make_unique<MockVectorQuantizer>();
  auto mock_generative_model = absl::make_unique<MockGenerativeModel>();
  auto mock_comfort_noise_generator = absl::make_unique<MockGenerativeModel>();
  auto mock_packet_loss_handler = absl::make_unique<MockPacketLossHandler>();
  auto resampler = absl::make_unique<MockResampler>();

  auto lyra_decoder_peer = absl::make_unique<LyraDecoderPeer>(
      std::move(mock_generative_model), std::move(mock_comfort_noise_generator),
      std::move(mock_vector_quantizer), std::move(mock_packet_loss_handler),
      std::move(resampler), sample_rate_hz_, num_frames_per_packet_);

  const int frame_size = output_mock_samples_.size();
  // Generate random frames, making sure to keep preceding frame values above
  // following frame values always.
  absl::BitGen gen;
  std::vector<int16_t> preceding_frame(frame_size);
  std::vector<int16_t> following_frame(frame_size);
  for (int i = 0; i < frame_size; ++i) {
    preceding_frame[i] = absl::Uniform<int16_t>(gen, 0, 100);
    following_frame[i] = absl::Uniform<int16_t>(gen, -100, 0);
  }

  const auto overlapped_frame =
      lyra_decoder_peer->OverlapFrames(preceding_frame, following_frame);
  ASSERT_TRUE(overlapped_frame.has_value());
  for (int i = 0; i < overlapped_frame.value().size(); ++i) {
    EXPECT_LE(overlapped_frame.value()[i], preceding_frame[i]);
    EXPECT_GE(overlapped_frame.value()[i], following_frame[i]);
  }
}

TEST_P(LyraDecoderTest, OverlapSucceedsWithConsecutiveFramesOfDifferentSize) {
  // Test that overlaps does not fail given two consecutive frames of different
  // sizes.
  auto mock_vector_quantizer = absl::make_unique<MockVectorQuantizer>();
  auto mock_generative_model = absl::make_unique<MockGenerativeModel>();
  auto mock_comfort_noise_generator = absl::make_unique<MockGenerativeModel>();
  auto mock_packet_loss_handler = absl::make_unique<MockPacketLossHandler>();
  auto resampler = absl::make_unique<MockResampler>();

  auto lyra_decoder_peer = absl::make_unique<LyraDecoderPeer>(
      std::move(mock_generative_model), std::move(mock_comfort_noise_generator),
      std::move(mock_vector_quantizer), std::move(mock_packet_loss_handler),
      std::move(resampler), sample_rate_hz_, num_frames_per_packet_);

  const int frame_size = output_mock_samples_.size();
  std::vector<int16_t> preceding_frame(frame_size, 0);
  std::vector<int16_t> following_frame(frame_size, 0);
  EXPECT_TRUE(lyra_decoder_peer->OverlapFrames(preceding_frame, following_frame)
                  .has_value());

  preceding_frame.resize(frame_size + 1);
  following_frame.resize(frame_size + 1);
  EXPECT_TRUE(lyra_decoder_peer->OverlapFrames(preceding_frame, following_frame)
                  .has_value());
}

TEST_P(LyraDecoderTest, DecodeLatestPacketOnly) {
  // Fill a packet with the bit pattern 00000000 at each byte.
  std::bitset<kNumQuantizedBits> quantized_zeros(0);
  PacketType packet_zeros;
  std::vector<uint8_t> encoded_zeros =
      packet_zeros.PackQuantized(quantized_zeros.to_string());

  // Fill a packet with the bit pattern 11111111 at each byte.
  std::bitset<kNumQuantizedBits> quantized_ones(0);
  quantized_ones.flip();
  PacketType packet_ones;
  std::vector<uint8_t> encoded_ones =
      packet_ones.PackQuantized(quantized_ones.to_string());

  // First the all 0s packet is passed to the decoder, then the all 1s packet is
  // passed to the decoder.
  const std::vector<float> zeros_concatenated_features(
      num_frames_per_packet_ * kNumFeatures, 0.0f);
  auto mock_vector_quantizer = absl::make_unique<MockVectorQuantizer>();
  EXPECT_CALL(*mock_vector_quantizer,
              DecodeToLossyFeatures(quantized_zeros.to_string()))
      .WillOnce(Return(zeros_concatenated_features));
  EXPECT_CALL(*mock_vector_quantizer,
              DecodeToLossyFeatures(quantized_ones.to_string()))
      .WillOnce(Return(mock_concatenated_features_));

  auto mock_packet_loss_handler = absl::make_unique<MockPacketLossHandler>();
  auto mock_generative_model = absl::make_unique<MockGenerativeModel>();
  const std::vector<float> zeros_features(kNumFeatures, 0.0f);
  EXPECT_CALL(*mock_packet_loss_handler, SetReceivedFeatures(zeros_features))
      .Times(num_frames_per_packet_)
      .WillRepeatedly(Return(true));
  EXPECT_CALL(*mock_generative_model, AddFeatures(zeros_features))
      .Times(num_frames_per_packet_);
  for (const auto& mock_features : mock_feature_frames_) {
    EXPECT_CALL(*mock_packet_loss_handler, SetReceivedFeatures(mock_features))
        .WillOnce(Return(true));
    EXPECT_CALL(*mock_generative_model, AddFeatures(mock_features));
  }

  // The mock generative model should only be run once.
  const int num_requested_samples = output_mock_samples_.size();
  const int num_samples_to_generate = mock_samples_->size();
  EXPECT_CALL(*mock_generative_model, GenerateSamples(num_samples_to_generate))
      .WillOnce(Return(mock_samples_));
  auto mock_comfort_noise_generator = absl::make_unique<MockGenerativeModel>();
  EXPECT_CALL(*mock_comfort_noise_generator, AddFeatures(testing::_)).Times(0);
  EXPECT_CALL(*mock_comfort_noise_generator, GenerateSamples(testing::_))
      .Times(0);
  auto lyra_decoder_peer = absl::make_unique<LyraDecoderPeer>(
      std::move(mock_generative_model), std::move(mock_comfort_noise_generator),
      std::move(mock_vector_quantizer), std::move(mock_packet_loss_handler),
      GetResampler(1), sample_rate_hz_, num_frames_per_packet_);

  ASSERT_TRUE(lyra_decoder_peer->SetEncodedPacket(encoded_zeros));
  ASSERT_TRUE(lyra_decoder_peer->SetEncodedPacket(encoded_ones));
  auto decoded_or = lyra_decoder_peer->DecodeSamples(num_requested_samples);
  ASSERT_TRUE(decoded_or.has_value());
  EXPECT_EQ(decoded_or.value(), output_mock_samples_);
}

TEST_P(LyraDecoderTest, MoreSamplesRequestedThanInAPacket) {
  // Fill a packet with the bit pattern 00000000 at each byte.
  std::bitset<kNumQuantizedBits> quantized_zeros(0);
  PacketType packet;
  std::vector<uint8_t> encoded_zeros =
      packet.PackQuantized(quantized_zeros.to_string());

  auto mock_vector_quantizer = absl::make_unique<MockVectorQuantizer>();
  auto mock_packet_loss_handler = absl::make_unique<MockPacketLossHandler>();
  auto mock_generative_model = absl::make_unique<MockGenerativeModel>();
  EXPECT_CALL(*mock_packet_loss_handler, SetReceivedFeatures(testing::_))
      .Times(num_frames_per_packet_)
      .WillRepeatedly(Return(true));
  EXPECT_CALL(*mock_generative_model, AddFeatures(testing::_))
      .Times(num_frames_per_packet_);

  // Functions expected to not be called due to early exits.
  EXPECT_CALL(*mock_generative_model, GenerateSamples(testing::_)).Times(0);
  auto mock_comfort_noise_generator = absl::make_unique<MockGenerativeModel>();
  EXPECT_CALL(*mock_comfort_noise_generator, AddFeatures(testing::_)).Times(0);
  EXPECT_CALL(*mock_comfort_noise_generator, GenerateSamples(testing::_))
      .Times(0);

  auto lyra_decoder_peer = absl::make_unique<LyraDecoderPeer>(
      std::move(mock_generative_model), std::move(mock_comfort_noise_generator),
      std::move(mock_vector_quantizer), std::move(mock_packet_loss_handler),
      GetResampler(0), sample_rate_hz_, num_frames_per_packet_);
  ASSERT_TRUE(lyra_decoder_peer->SetEncodedPacket(encoded_zeros));

  // One more sample than there are in a packet.
  const int num_samples_more_than_packet =
      num_frames_per_packet_ * GetNumSamplesPerHop(sample_rate_hz_) + 1;
  auto decoded_or =
      lyra_decoder_peer->DecodeSamples(num_samples_more_than_packet);
  EXPECT_FALSE(decoded_or.has_value());
}

TEST_P(LyraDecoderTest, InvalidConfig) {
  for (const auto& invalid_num_channels : {-1, 0, 2}) {
    EXPECT_EQ(LyraDecoder::Create(sample_rate_hz_, invalid_num_channels,
                                  kBitrate, model_path_),
              nullptr);
  }
  for (const auto& invalid_bitrates : {-1, 0}) {
    EXPECT_EQ(LyraDecoder::Create(sample_rate_hz_, kNumChannels,
                                  invalid_bitrates, model_path_),
              nullptr);
  }
}

INSTANTIATE_TEST_SUITE_P(
    SampleRateAndNumFramePerPacket, LyraDecoderTest,
    testing::Combine(testing::ValuesIn(kSupportedSampleRates),
                     testing::Values(1, 2, 3)));

TEST(LyraDecoderCreate, InvalidCreateReturnsNullptr) {
  for (const auto& invalid_sample_rate : {0, -1, 16001}) {
    EXPECT_EQ(LyraDecoder::Create(invalid_sample_rate, kNumChannels, kBitrate,
                                  kExportedModelPath),
              nullptr);
  }
  for (const auto& valid_sample_rate : kSupportedSampleRates) {
    EXPECT_EQ(LyraDecoder::Create(valid_sample_rate, kNumChannels, kBitrate,
                                  "/does/not/exist"),
              nullptr);
  }
}

}  // namespace
}  // namespace codec
}  // namespace chromemedia
