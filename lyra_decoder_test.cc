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
#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

// Placeholder for get runfiles header.
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "buffered_filter_interface.h"
#include "buffered_resampler.h"
#include "dsp_utils.h"
#include "feature_estimator_interface.h"
#include "generative_model_interface.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "include/ghc/filesystem.hpp"
#include "lyra_components.h"
#include "lyra_config.h"
#include "packet_interface.h"
#include "resampler.h"
#include "testing/mock_generative_model.h"
#include "testing/mock_noise_estimator.h"
#include "testing/mock_vector_quantizer.h"
#include "vector_quantizer_interface.h"

namespace chromemedia {
namespace codec {

// Use a test peer to access the private constructor of LyraDecoder in order
// to inject a MockGenerativeModel.
class LyraDecoderPeer {
 public:
  explicit LyraDecoderPeer(
      std::unique_ptr<MockGenerativeModel> mock_generative_model,
      std::unique_ptr<MockGenerativeModel> mock_comfort_noise_generator,
      std::unique_ptr<MockVectorQuantizer> mock_vector_quantizer,
      std::unique_ptr<MockNoiseEstimator> mock_noise_estimator,
      std::unique_ptr<FeatureEstimatorInterface> feature_estimator,
      std::unique_ptr<BufferedFilterInterface> buffered_resampler,
      int external_sample_rate_hz)
      : decoder_(std::move(mock_generative_model),
                 std::move(mock_comfort_noise_generator),
                 std::move(mock_vector_quantizer),
                 std::move(mock_noise_estimator), std::move(feature_estimator),
                 std::move(buffered_resampler), external_sample_rate_hz,
                 kNumChannels) {}

  bool SetEncodedPacket(const absl::Span<const uint8_t> encoded) {
    return decoder_.SetEncodedPacket(encoded);
  }

  std::optional<std::vector<int16_t>> DecodeSamples(int num_samples) {
    return decoder_.DecodeSamples(num_samples);
  }

  void SetConcealmentProgress(int samples) {
    decoder_.concealment_progress_ = samples;
  }

  void SetFadeProgress(int samples) { decoder_.fade_progress_ = samples; }

  void SetFadeToCNG() { decoder_.fade_direction_ = LyraDecoder::kFadeToCNG; }

  void SetFadeFromCNG() {
    decoder_.fade_direction_ = LyraDecoder::kFadeFromCNG;
  }

 private:
  LyraDecoder decoder_;
};

namespace {

using testing::Exactly;
using testing::Return;

static constexpr absl::string_view kExportedModelPath = "model_coeffs";

// Duration of pure packet loss concealment.
inline int GetConcealmentDurationSamples() {
  static constexpr float kConcealmentDurationSeconds = 0.08;
  static constexpr int kConcealmentDurationSamples =
      kConcealmentDurationSeconds * kInternalSampleRateHz;
  CHECK_EQ(
      kConcealmentDurationSamples % GetNumSamplesPerHop(kInternalSampleRateHz),
      0);
  return kConcealmentDurationSamples;
}

// Duration it takes to fade from concealment to comfort noise, and from
// comfort noise to received packets.
inline int GetFadeDurationSamples() {
  static constexpr float kFadeDurationSeconds = 0.04;
  static constexpr int kFadeDurationSamples =
      kFadeDurationSeconds * kInternalSampleRateHz;
  CHECK_EQ(kFadeDurationSamples % GetNumSamplesPerHop(kInternalSampleRateHz),
           0);
  return kFadeDurationSamples;
}

class LyraDecoderTest
    : public testing::TestWithParam<testing::tuple<int, int>> {
 protected:
  enum ModelTypeSamples {
    kGenerative = -10000,
    kComfort = 10000,
    kFade = 20000,
  };

  LyraDecoderTest()
      : external_sample_rate_hz_(std::get<0>(GetParam())),
        num_quantized_bits_(std::get<1>(GetParam())),
        external_num_samples_per_hop_(
            GetNumSamplesPerHop(external_sample_rate_hz_)),
        internal_num_samples_per_hop_(
            GetNumSamplesPerHop(kInternalSampleRateHz)),
        concealment_duration_packets_(GetConcealmentDurationSamples() /
                                      internal_num_samples_per_hop_),
        fade_duration_packets_(GetFadeDurationSamples() /
                               internal_num_samples_per_hop_),
        quantized_zeros_(num_quantized_bits_, '0'),
        packet_(CreatePacket(kNumHeaderBits, num_quantized_bits_)),
        encoded_zeros_(packet_->PackQuantized(quantized_zeros_)),
        model_path_(ghc::filesystem::current_path() / kExportedModelPath),
        mock_features_(kNumFeatures),
        mock_noise_features_(kNumMelBins),
        mock_samples_(internal_num_samples_per_hop_),
        generative_model_mock_samples_(internal_num_samples_per_hop_,
                                       ModelTypeSamples::kGenerative),
        comfort_noise_generator_mock_samples_(internal_num_samples_per_hop_,
                                              ModelTypeSamples::kComfort) {
    // Fill |mock_features_| and |mock_noise_features_| with monotonically
    // increasing values at each index, with a constant offset from one another.
    std::iota(mock_features_.begin(), mock_features_.end(), 0);
    std::iota(mock_noise_features_.begin(), mock_noise_features_.end(), 10.f);
    // Fill |mock_samples_| with monotonically increasing values at each index.
    std::iota(mock_samples_->begin(), mock_samples_->end(), 0);
  }

  void SetUp() override {
    mock_generative_model_ = std::make_unique<MockGenerativeModel>(
        ModelTypeSamples::kGenerative, internal_num_samples_per_hop_,
        kNumFeatures);
    mock_comfort_noise_generator_ = std::make_unique<MockGenerativeModel>(
        ModelTypeSamples::kComfort, internal_num_samples_per_hop_, kNumMelBins);
    mock_vector_quantizer_ = std::make_unique<MockVectorQuantizer>();
    mock_noise_estimator_ = std::make_unique<MockNoiseEstimator>();
    feature_estimator_ = CreateFeatureEstimator(kNumFeatures);
    buffered_resampler_ = BufferedResampler::Create(kInternalSampleRateHz,
                                                    external_sample_rate_hz_);
    real_resampler_ =
        Resampler::Create(kInternalSampleRateHz, external_sample_rate_hz_);
  }

  void CreateDecoder() {
    lyra_decoder_peer_ = std::make_unique<LyraDecoderPeer>(
        std::move(mock_generative_model_),
        std::move(mock_comfort_noise_generator_),
        std::move(mock_vector_quantizer_), std::move(mock_noise_estimator_),
        std::move(feature_estimator_), std::move(buffered_resampler_),
        external_sample_rate_hz_);
  }

  // The packet loss mechanism can be thought of as a 6-state machine.
  // State 1: Normal decoding.
  //   Possible transitions functions:
  //    a-> Stay in State 1, while samples are available from received packets.
  //    b-> Transition to State 2, if more samples requested than available from
  //        received packets.
  // State 2: Pure concealment.
  //   Possible transitions functions:
  //    a-> Stay in State 2, while we have not received a packet and have not
  //        exceeded the pure concealment duration.
  //    b-> Transition to State 3, while we have not received a packet and have
  //        exceeded the pure concealment duration.
  //    c-> Transition to State 1, if we have received a packet.
  // State 3: Fade from concealment to comfort noise.
  //   Possible transitions functions:
  //    a-> Stay in State 3, while we have not received a packet and have not
  //        exceeded the fade duration.
  //    b-> Transition to State 4, while we have not received a packet and have
  //        exceeded the fade duration.
  //    c-> Transition to State 6, if we have received a packet.
  // State 4: Pure comfort noise generation.
  //   Possible transitions functions:
  //    a-> Stay in State 4, while we have not received a packet.
  //    b-> Transition to State 5, if we have received a packet.
  // State 5: Fade from comfort noise to normal decoding.
  //   Possible transitions functions:
  //    a-> Stay in State 5, while we receive packets and have not exceeded the
  //        fade duration.
  //    b-> Transition to State 1, if we have not lost a packet and have
  //    exceeded
  //        the fade duration.
  //    c-> Transition to State 6, if we have lost a packet.
  // State 6: Fade from comfort noise to concealment.
  //   Possible transitions functions:
  //    a-> Stay in State 6, while we do not receive packets and have not
  //    exceeded the fade duration.
  //    b-> Transition to State 5, if we have received a packet and have not
  //        exceeded the fade duration.
  //    c-> Transition to State 2, if do not receive a packet and have exceeded
  //        the fade duration.
  // State transitions only occur at |GetNumSamplesPerHop(kInternalSampleRate)|
  // intervals, meaning if we have decoded at least one sample from a
  // concealment 'packet' from State 2 and then receive a real packet, we must
  // completley play out the remaining samples from concealment 'packet' before
  // transitioning back to State 1.
  // TODO(b/227492039): Add tests for State 5->6 and 3->5 when hop size
  // decreases.

  void ExpectSetEncodedPacket(int num_calls) {
    EXPECT_CALL(*mock_vector_quantizer_,
                DecodeToLossyFeatures(quantized_zeros_))
        .Times(Exactly(num_calls))
        .WillRepeatedly(Return(mock_features_));
    EXPECT_CALL(*mock_generative_model_, AddFeatures(mock_features_))
        .Times(num_calls);
  }

  void ExpectNormalDecoding(
      const std::vector<int16_t>& expected_merged_samples) {
    EXPECT_CALL(*mock_generative_model_,
                GenerateSamples(expected_merged_samples.size()))
        .Times(Exactly(1));
    // Zero comfort noise samples requested in normal decoding.
    EXPECT_CALL(*mock_noise_estimator_, noise_estimate()).Times(Exactly(0));
    EXPECT_CALL(*mock_comfort_noise_generator_, AddFeatures(::testing::_))
        .Times(Exactly(0));
    EXPECT_CALL(*mock_comfort_noise_generator_, GenerateSamples(0))
        .Times(Exactly(1));
    EXPECT_CALL(
        *mock_noise_estimator_,
        ReceiveSamples(absl::MakeConstSpan(generative_model_mock_samples_)))
        .Times(Exactly(1))
        .WillOnce(Return(true));
  }

  void ExpectConcealment(const std::vector<int16_t>& expected_merged_samples,
                         bool expect_add_features) {
    if (expect_add_features) {
      // Estimated features are added to the model from a concrete class - we
      // are not concerned with their exact value in this test.
      EXPECT_CALL(*mock_generative_model_, AddFeatures(::testing::_))
          .Times(Exactly(1));
    }
    EXPECT_CALL(*mock_generative_model_,
                GenerateSamples(expected_merged_samples.size()))
        .Times(Exactly(1));
    EXPECT_CALL(*mock_comfort_noise_generator_, GenerateSamples(0))
        .Times(Exactly(1));
  }

  void ExpectFadeToComfortNoise(
      const std::vector<int16_t>& expected_merged_samples,
      bool expect_add_features) {
    if (expect_add_features) {
      // Still using estimated features since we have not received a packet.
      EXPECT_CALL(*mock_generative_model_, AddFeatures(::testing::_))
          .Times(Exactly(1));
    }
    EXPECT_CALL(*mock_generative_model_,
                GenerateSamples(expected_merged_samples.size()))
        .Times(Exactly(1));
    // Use the default noise estimate since we have not received a packet.
    if (expect_add_features) {
      EXPECT_CALL(*mock_noise_estimator_, noise_estimate())
          .Times(Exactly(1))
          .WillOnce(Return(mock_noise_features_));
      EXPECT_CALL(*mock_comfort_noise_generator_,
                  AddFeatures(mock_noise_features_))
          .Times(Exactly(1));
    }
    EXPECT_CALL(*mock_comfort_noise_generator_,
                GenerateSamples(expected_merged_samples.size()))
        .Times(Exactly(1));
  }

  void ExpectComfortNoise(const std::vector<int16_t>& expected_merged_samples,
                          bool expect_add_features) {
    // Still using estimated features since we have not received a packet.
    EXPECT_CALL(*mock_generative_model_, AddFeatures(::testing::_))
        .Times(Exactly(0));
    EXPECT_CALL(*mock_generative_model_, GenerateSamples(0)).Times(Exactly(1));
    if (expect_add_features) {
      EXPECT_CALL(*mock_noise_estimator_, noise_estimate())
          .Times(Exactly(1))
          .WillOnce(Return(mock_noise_features_));
      EXPECT_CALL(*mock_comfort_noise_generator_,
                  AddFeatures(mock_noise_features_))
          .Times(Exactly(1));
    }
    EXPECT_CALL(*mock_comfort_noise_generator_,
                GenerateSamples(expected_merged_samples.size()))
        .Times(Exactly(1));
  }

  void ExpectFadeToNormalDecoding(
      const std::vector<int16_t>& expected_merged_samples,
      bool expect_add_features) {
    EXPECT_CALL(*mock_generative_model_,
                GenerateSamples(expected_merged_samples.size()))
        .Times(Exactly(1));
    if (expect_add_features) {
      EXPECT_CALL(*mock_noise_estimator_, noise_estimate())
          .Times(Exactly(1))
          .WillOnce(Return(mock_noise_features_));
      EXPECT_CALL(*mock_comfort_noise_generator_,
                  AddFeatures(mock_noise_features_))
          .Times(Exactly(1));
    }
    EXPECT_CALL(*mock_comfort_noise_generator_,
                GenerateSamples(expected_merged_samples.size()))
        .Times(Exactly(1));
    // Noise estimator is called on the model output from the received packet.
    EXPECT_CALL(*mock_noise_estimator_, ReceiveSamples(::testing::_))
        .Times(Exactly(1))
        .WillOnce(Return(true));
  }

  std::unique_ptr<LyraDecoderPeer> lyra_decoder_peer_;

  std::unique_ptr<MockGenerativeModel> mock_generative_model_;
  std::unique_ptr<MockGenerativeModel> mock_comfort_noise_generator_;
  std::unique_ptr<MockVectorQuantizer> mock_vector_quantizer_;
  std::unique_ptr<MockNoiseEstimator> mock_noise_estimator_;
  std::unique_ptr<FeatureEstimatorInterface> feature_estimator_;
  std::unique_ptr<BufferedFilterInterface> buffered_resampler_;

  std::unique_ptr<Resampler> real_resampler_;

  const int external_sample_rate_hz_;
  const int num_quantized_bits_;
  const int external_num_samples_per_hop_;
  const int internal_num_samples_per_hop_;
  const int concealment_duration_packets_;
  const int fade_duration_packets_;
  const std::string quantized_zeros_;
  const std::unique_ptr<PacketInterface> packet_;
  const std::vector<uint8_t> encoded_zeros_;
  const ghc::filesystem::path model_path_;
  std::vector<float> mock_features_;
  std::vector<float> mock_noise_features_;
  std::optional<std::vector<int16_t>> mock_samples_;
  // Samples returned from the generative_model.
  std::vector<int16_t> generative_model_mock_samples_;
  // Samples returned from the comfort noise generator.
  std::vector<int16_t> comfort_noise_generator_mock_samples_;
};

// Test State 1: Normal decoding. -> State 2 -> State 1: Normal decoding.
TEST_P(LyraDecoderTest, EntirePacketRequests_NormalToConcealmentToNormal) {
  const int kNumRequests = 3;
  std::vector<std::vector<int16_t>> expected_merged_samples(kNumRequests);
  expected_merged_samples.at(0) = std::vector<int16_t>(
      internal_num_samples_per_hop_, ModelTypeSamples::kGenerative);
  expected_merged_samples.at(1) = expected_merged_samples.at(0);
  expected_merged_samples.at(2) = expected_merged_samples.at(0);

  std::vector<int> external_sample_requests(kNumRequests,
                                            external_num_samples_per_hop_);

  {  // Enforce mocks are called in a specific order.
    ::testing::InSequence in;
    ExpectSetEncodedPacket(1);
    ExpectNormalDecoding(expected_merged_samples.at(0));

    ExpectConcealment(expected_merged_samples.at(1),
                      /*expect_add_features=*/true);

    ExpectSetEncodedPacket(1);
    ExpectNormalDecoding(expected_merged_samples.at(2));
  }

  CreateDecoder();

  // State 1: Normal decoding.
  ASSERT_TRUE(lyra_decoder_peer_->SetEncodedPacket(encoded_zeros_));
  ASSERT_TRUE(
      lyra_decoder_peer_->DecodeSamples(external_sample_requests.at(0)));
  // State 2: Concealment.
  ASSERT_TRUE(
      lyra_decoder_peer_->DecodeSamples(external_sample_requests.at(1)));
  // State 1: Normal decoding.
  ASSERT_TRUE(lyra_decoder_peer_->SetEncodedPacket(encoded_zeros_));
  ASSERT_TRUE(
      lyra_decoder_peer_->DecodeSamples(external_sample_requests.at(2)));
}

// State 2: Concealment -> State 3: Fade to comfort noise -> State 4: Comfort
// noise.
TEST_P(LyraDecoderTest, TestFinishDecoding_ConcealmentToComfortNoise) {
  std::vector<std::vector<int16_t>> expected_merged_samples;
  // State 2: Concealment.
  for (int i = 0; i < concealment_duration_packets_; ++i) {
    expected_merged_samples.push_back(std::vector<int16_t>(
        internal_num_samples_per_hop_, ModelTypeSamples::kGenerative));
  }
  // State 3: Fade to comfort noise.
  for (int i = 0; i < fade_duration_packets_; ++i) {
    expected_merged_samples.push_back(std::vector<int16_t>(
        internal_num_samples_per_hop_, ModelTypeSamples::kFade));
  }
  // State 4: Comfort noise.
  const int kNumComfortNoisePackets = 3;
  for (int i = 0; i < kNumComfortNoisePackets; ++i) {
    expected_merged_samples.push_back(std::vector<int16_t>(
        internal_num_samples_per_hop_, ModelTypeSamples::kComfort));
  }

  std::vector<int16_t> external_sample_requests(expected_merged_samples.size(),
                                                external_num_samples_per_hop_);

  int request_index = 0;
  {
    ::testing::InSequence in;
    // State 2: Concealment.
    for (int i = 0; i < concealment_duration_packets_; ++i) {
      ExpectConcealment(expected_merged_samples.at(request_index++),
                        /*expect_add_features=*/true);
    }
    // State 3: Fade to comfort noise.
    for (int i = 0; i < fade_duration_packets_; ++i) {
      ExpectFadeToComfortNoise(expected_merged_samples.at(request_index++),
                               /*expect_add_features=*/true);
    }
    // State 4: Comfort noise.
    for (int i = 0; i < kNumComfortNoisePackets; ++i) {
      ExpectComfortNoise(expected_merged_samples.at(request_index++),
                         /*expect_add_features=*/true);
    }
  }

  CreateDecoder();

  request_index = 0;
  // State 2: Concealment.
  for (int i = 0; i < concealment_duration_packets_; ++i) {
    ASSERT_TRUE(
        lyra_decoder_peer_
            ->DecodeSamples(external_sample_requests.at(request_index++))
            .has_value());
  }
  // State 3: Fade to comfort noise.
  for (int i = 0; i < fade_duration_packets_; ++i) {
    ASSERT_TRUE(
        lyra_decoder_peer_
            ->DecodeSamples(external_sample_requests.at(request_index++))
            .has_value());
  }
  // State 4: Comfort noise.
  for (int i = 0; i < kNumComfortNoisePackets; ++i) {
    ASSERT_TRUE(
        lyra_decoder_peer_
            ->DecodeSamples(external_sample_requests.at(request_index++))
            .has_value());
  }
}

// State 4: Comfort noise -> SetEncodedPacket -> State 4: Comfort
// noise -> State 5: Fade to normal.
TEST_P(LyraDecoderTest, TestFinishDecoding_ComfortNoiseFadetoNormal) {
  std::vector<std::vector<int16_t>> expected_merged_samples;
  // State 4: Comfort noise.
  expected_merged_samples.push_back(
      std::vector<int16_t>(100, ModelTypeSamples::kComfort));
  // Packet added here, decode the remainder of the State 4: Comfort noise
  // packet.
  expected_merged_samples.push_back(std::vector<int16_t>(
      internal_num_samples_per_hop_ - 100, ModelTypeSamples::kComfort));
  // State 5: Fade to normal decoding.
  for (int i = 0; i < fade_duration_packets_; ++i) {
    expected_merged_samples.push_back(std::vector<int16_t>(
        internal_num_samples_per_hop_, ModelTypeSamples::kFade));
  }

  std::vector<int16_t> external_sample_requests(expected_merged_samples.size());
  std::transform(expected_merged_samples.begin(), expected_merged_samples.end(),
                 external_sample_requests.begin(),
                 [this](std::vector<int16_t> internal_samples) {
                   return ConvertNumSamplesBetweenSampleRate(
                       internal_samples.size(), kInternalSampleRateHz,
                       external_sample_rate_hz_);
                 });

  int request_index = 0;
  {
    ::testing::InSequence in;
    // State 4: Comfort noise, call for 100 samples.
    ExpectComfortNoise(expected_merged_samples.at(request_index++),
                       /*expect_add_features=*/true);
    ExpectSetEncodedPacket(1);

    // State 4: Comfort noise, call for the rest of the comfort noise packet.
    ExpectComfortNoise(expected_merged_samples.at(request_index++),
                       /*expect_add_features=*/false);
    // State 5: Fade to normal decoding, call for the entire packet.
    for (int i = 0; i < fade_duration_packets_; ++i) {
      if (i > 0) {
        ExpectSetEncodedPacket(1);
      }
      ExpectFadeToNormalDecoding(expected_merged_samples.at(request_index++),
                                 /*expect_add_features=*/true);
    }
  }

  CreateDecoder();
  lyra_decoder_peer_->SetConcealmentProgress(GetConcealmentDurationSamples());
  lyra_decoder_peer_->SetFadeProgress(GetFadeDurationSamples());
  lyra_decoder_peer_->SetFadeToCNG();

  request_index = 0;
  // Partially decode fake packet, State 4: Comfort noise.
  ASSERT_TRUE(lyra_decoder_peer_
                  ->DecodeSamples(external_sample_requests.at(request_index++))
                  .has_value());
  // Add Packet.
  ASSERT_TRUE(lyra_decoder_peer_->SetEncodedPacket(encoded_zeros_));
  // Finish decoding packet partially started, State 4: Comfort noise.
  ASSERT_TRUE(lyra_decoder_peer_
                  ->DecodeSamples(external_sample_requests.at(request_index++))
                  .has_value());
  // State 5: Fade to normal decoding.
  for (int i = 0; i < fade_duration_packets_; ++i) {
    if (i > 0) {
      ASSERT_TRUE(lyra_decoder_peer_->SetEncodedPacket(encoded_zeros_));
    }
    ASSERT_TRUE(
        lyra_decoder_peer_
            ->DecodeSamples(external_sample_requests.at(request_index++))
            .has_value());
  }
}

TEST_P(LyraDecoderTest, MultipleHopsOneRequestNormalDecode) {
  const int kNumHopsToDecode = 4;
  std::vector<int16_t> expected_merged_samples(
      kNumHopsToDecode * internal_num_samples_per_hop_,
      ModelTypeSamples::kGenerative);

  const int sample_request = ConvertNumSamplesBetweenSampleRate(
      expected_merged_samples.size(), kInternalSampleRateHz,
      external_sample_rate_hz_);

  ExpectSetEncodedPacket(/*num_calls=*/kNumHopsToDecode);

  EXPECT_CALL(*mock_noise_estimator_, ReceiveSamples(::testing::_))
      .Times(Exactly(kNumHopsToDecode))
      .WillRepeatedly(Return(true));

  CreateDecoder();

  // Add Packets.
  for (int i = 0; i < kNumHopsToDecode; ++i) {
    ASSERT_TRUE(lyra_decoder_peer_->SetEncodedPacket(
        absl::MakeConstSpan(encoded_zeros_)));
  }
  // Decode all |kNumHopsToDecode| at once.
  ASSERT_TRUE(lyra_decoder_peer_->DecodeSamples(sample_request).has_value());
}

TEST_P(LyraDecoderTest, HopsAreOverlappedCorrectly) {
  // Test overlap for correctness. Given two hops, where the values of one
  // are always above the values are the other, the values of the overlapped
  // hop should land somewhere in between.

  // Set up expectations for the generative model.
  EXPECT_CALL(*mock_generative_model_,
              GenerateSamples(internal_num_samples_per_hop_))
      .WillRepeatedly(Return(generative_model_mock_samples_));
  EXPECT_CALL(*mock_generative_model_, GenerateSamples(0))
      .WillRepeatedly(Return(std::vector<int16_t>()));
  EXPECT_CALL(*mock_generative_model_, num_samples_available())
      .WillRepeatedly(Return(internal_num_samples_per_hop_));
  // Set up expectations for the comfort noise generator.
  EXPECT_CALL(*mock_comfort_noise_generator_,
              GenerateSamples(internal_num_samples_per_hop_))
      .WillRepeatedly(Return(comfort_noise_generator_mock_samples_));
  EXPECT_CALL(*mock_comfort_noise_generator_, GenerateSamples(0))
      .WillRepeatedly(Return(std::vector<int16_t>()));
  EXPECT_CALL(*mock_comfort_noise_generator_, num_samples_available())
      .WillRepeatedly(Return(internal_num_samples_per_hop_));
  // Set up expectations for the noise estimator, it receives the generative
  // model samples when fading from comfort noise to normal decoding.
  EXPECT_CALL(*mock_noise_estimator_, ReceiveSamples(::testing::_))
      .WillRepeatedly(Return(true));

  // The resampled value may oscillate around the input value.
  const int kMonotonicityTolerance = 2;
  // The resampler FIR filter will not reach steady state immediately.
  const int kSamplesUntilSteadyState =
      real_resampler_->samples_until_steady_state();

  const std::vector<int16_t> expected_concealment_samples =
      kInternalSampleRateHz == external_sample_rate_hz_
          ? generative_model_mock_samples_
          : real_resampler_->Resample(generative_model_mock_samples_);
  const std::vector<int16_t> expected_comfort_noise_samples =
      kInternalSampleRateHz == external_sample_rate_hz_
          ? comfort_noise_generator_mock_samples_
          : real_resampler_->Resample(comfort_noise_generator_mock_samples_);

  CreateDecoder();

  // Request samples without adding a packet.
  auto samples =
      lyra_decoder_peer_->DecodeSamples(external_num_samples_per_hop_);
  ASSERT_TRUE(samples.has_value());
  // State 2: Concealment should return all generative model output from the
  // fade.
  EXPECT_EQ(samples.value(), expected_concealment_samples);
  // State 3: Fade to comfort noise should fade from generative model samples
  // to comfort noise samples.
  lyra_decoder_peer_->SetFadeProgress(0);
  lyra_decoder_peer_->SetConcealmentProgress(GetConcealmentDurationSamples());
  lyra_decoder_peer_->SetFadeToCNG();
  samples = lyra_decoder_peer_->DecodeSamples(external_num_samples_per_hop_);
  ASSERT_TRUE(samples.has_value());
  for (int i = kSamplesUntilSteadyState; i < samples->size(); ++i) {
    EXPECT_GE(samples->at(i), expected_concealment_samples.at(i))
        << "Expecting faded samples to be greater than or equal to generative"
           "model samples at index "
        << i;
    EXPECT_LE(samples->at(i), expected_comfort_noise_samples.at(i))
        << "Expecting faded samples to be less than or equal to comfort noise"
           "samples at index "
        << i;
    if (i > 0) {
      EXPECT_GE(samples->at(i), samples->at(i - 1) - kMonotonicityTolerance)
          << "Samples should be monotonically increasing at index " << i;
    }
  }
  // State 4: Comfort noise should be all comfort noise samples.
  lyra_decoder_peer_->SetFadeProgress(GetFadeDurationSamples());
  lyra_decoder_peer_->SetConcealmentProgress(GetConcealmentDurationSamples());
  lyra_decoder_peer_->SetFadeToCNG();
  samples = lyra_decoder_peer_->DecodeSamples(external_num_samples_per_hop_);
  ASSERT_TRUE(samples.has_value());
  EXPECT_EQ(std::vector<int16_t>(samples->begin() + kSamplesUntilSteadyState,
                                 samples->end()),
            std::vector<int16_t>(expected_comfort_noise_samples.begin() +
                                     kSamplesUntilSteadyState,
                                 expected_comfort_noise_samples.end()));
  // State 5: Fade to normal decoding noise should fade from comfort noise to
  // generative model samples.
  lyra_decoder_peer_->SetFadeProgress(GetFadeDurationSamples());
  lyra_decoder_peer_->SetConcealmentProgress(0);
  lyra_decoder_peer_->SetFadeFromCNG();
  samples = lyra_decoder_peer_->DecodeSamples(external_num_samples_per_hop_);
  ASSERT_TRUE(samples.has_value());
  for (int i = kSamplesUntilSteadyState; i < samples->size(); ++i) {
    EXPECT_GE(samples->at(i), expected_concealment_samples.at(i))
        << "Expecting faded samples to be greater than or equal to generative"
           "model samples at index "
        << i;
    EXPECT_LE(samples->at(i), expected_comfort_noise_samples.at(i))
        << "Expecting faded samples to be less than or equal to comfort noise"
           "samples at index "
        << i;
    if (i > 0) {
      EXPECT_LE(samples->at(i), samples->at(i - 1) + kMonotonicityTolerance)
          << "Samples should be monotonically decreasing at index " << i;
    }
  }
}

TEST_P(LyraDecoderTest, ArbitraryNumSamplesNormalDecode) {
  ExpectSetEncodedPacket(external_num_samples_per_hop_);
  EXPECT_CALL(*mock_noise_estimator_, ReceiveSamples(::testing::_))
      .WillRepeatedly(Return(true));
  CreateDecoder();

  for (int num_samples = 0; num_samples < external_num_samples_per_hop_;
       ++num_samples) {
    lyra_decoder_peer_->SetFadeProgress(0);
    lyra_decoder_peer_->SetConcealmentProgress(0);
    lyra_decoder_peer_->SetFadeFromCNG();
    ASSERT_TRUE(lyra_decoder_peer_->SetEncodedPacket(encoded_zeros_));
    auto samples = lyra_decoder_peer_->DecodeSamples(num_samples);
    ASSERT_TRUE(samples.has_value());
    EXPECT_EQ(samples->size(), num_samples)
        << "Could not generate " << num_samples
        << " samples in normal decoding mode.";
  }
}

TEST_P(LyraDecoderTest, ArbitraryNumSamplesConcealment) {
  CreateDecoder();

  for (int num_samples = 0; num_samples < external_num_samples_per_hop_;
       ++num_samples) {
    lyra_decoder_peer_->SetFadeProgress(0);
    lyra_decoder_peer_->SetConcealmentProgress(1);
    lyra_decoder_peer_->SetFadeFromCNG();
    auto samples = lyra_decoder_peer_->DecodeSamples(num_samples);
    ASSERT_TRUE(samples.has_value());
    EXPECT_EQ(samples->size(), num_samples)
        << "Could not generate " << num_samples
        << " samples in concealment mode.";
  }
}

TEST_P(LyraDecoderTest, ArbitraryNumSamplesComfortNoise) {
  EXPECT_CALL(*mock_noise_estimator_, noise_estimate())
      .WillRepeatedly(Return(mock_noise_features_));
  CreateDecoder();

  for (int num_samples = 0; num_samples < external_num_samples_per_hop_;
       ++num_samples) {
    lyra_decoder_peer_->SetConcealmentProgress(GetConcealmentDurationSamples());
    lyra_decoder_peer_->SetFadeProgress(GetFadeDurationSamples());
    lyra_decoder_peer_->SetFadeToCNG();
    auto samples = lyra_decoder_peer_->DecodeSamples(num_samples);
    ASSERT_TRUE(samples.has_value());
    EXPECT_EQ(samples->size(), num_samples)
        << "Could not generate " << num_samples
        << " samples in comfort noise mode .";
  }
}

TEST_P(LyraDecoderTest, ArbitraryNumSamplesFadeToComfortNoise) {
  EXPECT_CALL(*mock_noise_estimator_, noise_estimate())
      .WillRepeatedly(Return(mock_noise_features_));
  CreateDecoder();

  for (int num_samples = 0; num_samples < external_num_samples_per_hop_;
       ++num_samples) {
    lyra_decoder_peer_->SetConcealmentProgress(GetConcealmentDurationSamples());
    lyra_decoder_peer_->SetFadeProgress(0);
    lyra_decoder_peer_->SetFadeToCNG();
    auto samples = lyra_decoder_peer_->DecodeSamples(num_samples);
    ASSERT_TRUE(samples.has_value());
    EXPECT_EQ(samples->size(), num_samples)
        << "Could not generate " << num_samples
        << " samples while fading to comfort noise.";
  }
}

TEST_P(LyraDecoderTest, ArbitraryNumSamplesFadeFromComfortNoise) {
  ExpectSetEncodedPacket(external_num_samples_per_hop_);
  EXPECT_CALL(*mock_noise_estimator_, ReceiveSamples(::testing::_))
      .WillRepeatedly(Return(true));
  EXPECT_CALL(*mock_noise_estimator_, noise_estimate())
      .WillRepeatedly(Return(mock_noise_features_));
  CreateDecoder();

  for (int num_samples = 0; num_samples < external_num_samples_per_hop_;
       ++num_samples) {
    lyra_decoder_peer_->SetConcealmentProgress(GetConcealmentDurationSamples());
    lyra_decoder_peer_->SetFadeProgress(GetFadeDurationSamples());
    lyra_decoder_peer_->SetFadeFromCNG();
    ASSERT_TRUE(lyra_decoder_peer_->SetEncodedPacket(encoded_zeros_));

    auto samples = lyra_decoder_peer_->DecodeSamples(num_samples);
    ASSERT_TRUE(samples.has_value());
    EXPECT_EQ(samples->size(), num_samples)
        << "Could not generate " << num_samples
        << " samples while fading from comfort noise.";
  }
}

TEST_P(LyraDecoderTest, ValidConfig) {
  EXPECT_NE(
      LyraDecoder::Create(external_sample_rate_hz_, kNumChannels, model_path_),
      nullptr);
}

TEST_P(LyraDecoderTest, InvalidConfig) {
  for (const auto& invalid_num_channels : {-1, 0, 2}) {
    EXPECT_EQ(LyraDecoder::Create(external_sample_rate_hz_,
                                  invalid_num_channels, model_path_),
              nullptr);
  }
}

INSTANTIATE_TEST_SUITE_P(
    SampleRateQuantizedBitsAndNumHopsPerPacket, LyraDecoderTest,
    testing::Combine(testing::ValuesIn(kSupportedSampleRates),
                     testing::ValuesIn(GetSupportedQuantizedBits())));

TEST(LyraDecoderCreate, InvalidCreateReturnsNullptr) {
  for (const auto& invalid_sample_rate : {0, -1, 16001}) {
    EXPECT_EQ(LyraDecoder::Create(invalid_sample_rate, kNumChannels,
                                  kExportedModelPath),
              nullptr);
  }
  for (const auto& valid_sample_rate : kSupportedSampleRates) {
    EXPECT_EQ(
        LyraDecoder::Create(valid_sample_rate, kNumChannels, "/does/not/exist"),
        nullptr);
  }
}

}  // namespace
}  // namespace codec
}  // namespace chromemedia
