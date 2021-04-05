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

#include "packet_loss_handler.h"

#include <memory>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "lyra_config.h"
#include "noise_estimator_interface.h"
#include "spectrogram_predictor_interface.h"
#include "testing/mock_noise_estimator.h"
#include "testing/mock_spectrogram_predictor.h"

namespace chromemedia {
namespace codec {

namespace {

using testing::Return;

static constexpr int kSampleRateHz = 16000;
static const int kNumSamplesPerFrame = GetNumSamplesPerFrame(kSampleRateHz);
static constexpr float kMaxLostSeconds = 0.1;
static const int kMaxConsecutiveLostSamples = kMaxLostSeconds * kSampleRateHz;

}  // namespace

// Use a test peer to access the private constructor of PacketLossHandler and
// inject MockNoiseEstimator and MockSpectrogramPredictor.
class PacketLossHandlerPeer {
 public:
  explicit PacketLossHandlerPeer(
      std::unique_ptr<MockNoiseEstimator> mock_noise_estimator,
      std::unique_ptr<MockSpectrogramPredictor> mock_spectrogram_predictor)
      : packet_loss_handler_(kSampleRateHz, std::move(mock_noise_estimator),
                             std::move(mock_spectrogram_predictor)) {}

  bool SetReceivedFeatures(const std::vector<float>& features) {
    return packet_loss_handler_.SetReceivedFeatures(features);
  }

  absl::optional<std::vector<float>> EstimateLostFeatures(int num_samples) {
    return packet_loss_handler_.EstimateLostFeatures(num_samples);
  }

  bool is_comfort_noise() { return packet_loss_handler_.is_comfort_noise(); }

  int FetchConsecutiveLostSamples() {
    return packet_loss_handler_.consecutive_lost_samples_;
  }

 private:
  PacketLossHandler packet_loss_handler_;
};

// Creates a PacketLossHandler with valid parameters and ensures that the
// returned handler is not nullptr.
TEST(PacketLossHandlerTest, ValidCreateReturnsHandler) {
  EXPECT_NE(nullptr,
            PacketLossHandler::Create(
                kSampleRateHz, kNumFeatures,
                static_cast<float>(kNumSamplesPerFrame) / kSampleRateHz));
}

// Creates a PacketLossHandler with parameters such that the contained
// NoiseEstimator will be invalid and ensures that the returned
// PacketLossHandler is nullptr as a result.
TEST(PacketLossHandlerTest, InvalidNoiseEstimatorCreatesNullHandler) {
  // A seconds_per_frame value of 0.0 causes NoiseEstimator::Create to fail.
  float seconds_per_frame = 0.0;
  EXPECT_EQ(nullptr, PacketLossHandler::Create(kSampleRateHz, kNumFeatures,
                                               seconds_per_frame));
}

// Calls SetReceivedFeatures on a valid PacketLossHandler with a valid feature
// vector and ensures |consecutive_lost_samples_| remains 0 and that the
// SpectrogramPredictor's FeedPacket method is invoked.
TEST(PacketLossHandlerTest, SetReceivedFeaturesWithValidFeatures) {
  std::vector<float> mock_features(kNumFeatures, 1.0);
  auto mock_noise_estimator = absl::make_unique<MockNoiseEstimator>();
  auto mock_spectrogram_predictor =
      absl::make_unique<MockSpectrogramPredictor>();
  EXPECT_CALL(*mock_noise_estimator, Update(mock_features))
      .WillOnce(Return(true));
  EXPECT_CALL(*mock_spectrogram_predictor, FeedFrame(mock_features)).Times(1);
  auto packet_loss_handler_peer = absl::make_unique<PacketLossHandlerPeer>(
      std::move(mock_noise_estimator), std::move(mock_spectrogram_predictor));
  EXPECT_EQ(0, packet_loss_handler_peer->FetchConsecutiveLostSamples());
  EXPECT_FALSE(packet_loss_handler_peer->is_comfort_noise());
  EXPECT_TRUE(packet_loss_handler_peer->SetReceivedFeatures(mock_features));
  EXPECT_EQ(0, packet_loss_handler_peer->FetchConsecutiveLostSamples());
  EXPECT_FALSE(packet_loss_handler_peer->is_comfort_noise());
}

// Calls SetReceivedFeatures on a valid PacketLossHandler with an invalid
// feature vector and ensures |consecutive_lost_samples_| remains 0 and that the
// SpectrogramPredictor's FeedPacket method is invoked.
TEST(PacketLossHandlerTest, SetReceivedFeaturesWithInvalidFeatures) {
  float invalid_num_features = kNumFeatures - 1;
  std::vector<float> mock_features(invalid_num_features, 1.0);
  auto mock_noise_estimator = absl::make_unique<MockNoiseEstimator>();
  auto mock_spectrogram_predictor =
      absl::make_unique<MockSpectrogramPredictor>();
  EXPECT_CALL(*mock_noise_estimator, Update(mock_features))
      .WillOnce(Return(false));
  EXPECT_CALL(*mock_spectrogram_predictor, FeedFrame(mock_features)).Times(0);
  auto packet_loss_handler_peer = absl::make_unique<PacketLossHandlerPeer>(
      std::move(mock_noise_estimator), std::move(mock_spectrogram_predictor));
  EXPECT_FALSE(packet_loss_handler_peer->SetReceivedFeatures(mock_features));
  EXPECT_EQ(0, packet_loss_handler_peer->FetchConsecutiveLostSamples());
}

// Calls EstimateLostFeatures on a PacketLossHandler with
// |consecutive_lost_samples_| less than kMaxConsecutiveLostSamples and makes
// sure that |consecutive_lost_samples_| is incremented and
// |spectrogram_predictor_|'s PredictFrame method is invoked.
TEST(PacketLossHandlerTest,
     EstimateLostFeaturesWithConsecutiveLostSamplesBelowMax) {
  static constexpr int kNumSamplesToRequest = 100;
  std::vector<float> mock_prediction(kNumFeatures, 2.0);
  auto mock_noise_estimator = absl::make_unique<MockNoiseEstimator>();
  auto mock_spectrogram_predictor =
      absl::make_unique<MockSpectrogramPredictor>();
  EXPECT_CALL(*mock_spectrogram_predictor, PredictFrame())
      .WillOnce(Return(mock_prediction));

  auto packet_loss_handler_peer = absl::make_unique<PacketLossHandlerPeer>(
      std::move(mock_noise_estimator), std::move(mock_spectrogram_predictor));
  int consecutive_lost_samples_prior =
      packet_loss_handler_peer->FetchConsecutiveLostSamples();
  auto prediction =
      packet_loss_handler_peer->EstimateLostFeatures(kNumSamplesToRequest);
  EXPECT_EQ(consecutive_lost_samples_prior + kNumSamplesToRequest,
            packet_loss_handler_peer->FetchConsecutiveLostSamples());
  EXPECT_EQ(mock_prediction, prediction);
  EXPECT_FALSE(packet_loss_handler_peer->is_comfort_noise());
}

// Calls EstimateLostFeatures on a PacketLossHandler and ensures that as a
// result |consecutive_lost_samples_| is nonzero, then calls SetReceivedFeatures
// with a valid feature vector and ensures |consecutive_lost_samples_| is reset
// to 0.
TEST(PacketLossHandlerTest, SetFeaturesResetsConsecutiveLostSamples) {
  static constexpr int kNumSamplesToRequest = 100;
  std::vector<float> mock_features(kNumFeatures, 1.0);
  auto mock_noise_estimator = absl::make_unique<MockNoiseEstimator>();
  auto mock_spectrogram_predictor =
      absl::make_unique<MockSpectrogramPredictor>();

  EXPECT_CALL(*mock_noise_estimator, Update(mock_features))
      .WillOnce(Return(true));

  auto packet_loss_handler_peer = absl::make_unique<PacketLossHandlerPeer>(
      std::move(mock_noise_estimator), std::move(mock_spectrogram_predictor));
  packet_loss_handler_peer->EstimateLostFeatures(kNumSamplesToRequest);
  EXPECT_NE(0, packet_loss_handler_peer->FetchConsecutiveLostSamples());
  EXPECT_TRUE(packet_loss_handler_peer->SetReceivedFeatures(mock_features));
  EXPECT_EQ(0, packet_loss_handler_peer->FetchConsecutiveLostSamples());
  EXPECT_FALSE(packet_loss_handler_peer->is_comfort_noise());
}

// Calls EstimateLostFeatures with out of bound values for |num_samples| and
// ensures that a nullopt is returned.
TEST(PacketLossHandlerTest, EstimateLostFeaturesWithInvalidNumSamples) {
  auto mock_noise_estimator = absl::make_unique<MockNoiseEstimator>();
  auto mock_spectrogram_predictor =
      absl::make_unique<MockSpectrogramPredictor>();
  EXPECT_CALL(*mock_spectrogram_predictor, PredictFrame()).Times(0);
  EXPECT_CALL(*mock_spectrogram_predictor, FeedFrame(testing::_)).Times(0);
  EXPECT_CALL(*mock_noise_estimator, NoiseEstimate()).Times(0);

  auto packet_loss_handler_peer = absl::make_unique<PacketLossHandlerPeer>(
      std::move(mock_noise_estimator), std::move(mock_spectrogram_predictor));

  int num_samples_to_request = -1;
  ASSERT_FALSE(
      packet_loss_handler_peer->EstimateLostFeatures(num_samples_to_request)
          .has_value());

  num_samples_to_request = 0;
  ASSERT_FALSE(
      packet_loss_handler_peer->EstimateLostFeatures(num_samples_to_request)
          .has_value());
}

// Calls EstimateLostFeatures on a PacketLossHandler to move the handler into a
// state in which it should return noise on the next invocation of
// EstimateLostFeatures. Then calls EstimateLostFeatures an additional time to
// verify noise is returned and fed back into |spectrogram_predictor_|.
TEST(PacketLossHandlerTest,
     EstimateLostFeaturesReturnsNoiseAfterTooManyLostSamples) {
  static const int kNumSamplesToRequest = 100;
  std::vector<float> mock_prediction(kNumFeatures, 2.0);
  std::vector<float> mock_noise(kNumFeatures, 3.0);
  auto mock_noise_estimator = absl::make_unique<MockNoiseEstimator>();
  auto mock_spectrogram_predictor =
      absl::make_unique<MockSpectrogramPredictor>();
  EXPECT_CALL(*mock_spectrogram_predictor, PredictFrame())
      // The floor of the result is wanted, so integer division is fine.
      .Times(kMaxConsecutiveLostSamples / kNumSamplesToRequest)
      .WillRepeatedly(Return(mock_prediction));
  EXPECT_CALL(*mock_spectrogram_predictor, FeedFrame(mock_noise)).Times(1);
  EXPECT_CALL(*mock_noise_estimator, NoiseEstimate())
      .WillOnce(Return(mock_noise));

  auto packet_loss_handler_peer = absl::make_unique<PacketLossHandlerPeer>(
      std::move(mock_noise_estimator), std::move(mock_spectrogram_predictor));
  int num_samples_requested_aggregated = kNumSamplesToRequest;
  while (num_samples_requested_aggregated <= kMaxConsecutiveLostSamples) {
    auto prediction =
        packet_loss_handler_peer->EstimateLostFeatures(kNumSamplesToRequest);
    ASSERT_TRUE(prediction.has_value());
    num_samples_requested_aggregated += kNumSamplesToRequest;
    EXPECT_EQ(mock_prediction, prediction.value());
    EXPECT_FALSE(packet_loss_handler_peer->is_comfort_noise());
  }
  auto prediction =
      packet_loss_handler_peer->EstimateLostFeatures(kNumSamplesToRequest);
  ASSERT_TRUE(prediction.has_value());
  EXPECT_EQ(mock_noise, prediction.value());
  EXPECT_TRUE(packet_loss_handler_peer->is_comfort_noise());
}

// Puts a PacketLossHandler into a state in which it should return noise upon
// invocation of EstimateLostFeatures and then calls SetReceivedFeatures on it
// with a valid feature vector and ensures that this puts the handler back into
// a state in which EstimateLostFeatures returns the result of calling
// PredictFrame on |spectrogram_predictor_|.
TEST(PacketLossHandlerTest,
     SetReceivedFeaturesReturnsHandlerFromNoisePredictionState) {
  static const int kNumSamplesToRequest = 100;
  std::vector<float> mock_features(kNumFeatures, 1.0);
  std::vector<float> mock_prediction(kNumFeatures, 2.0);
  std::vector<float> mock_noise(kNumFeatures, 3.0);
  auto mock_noise_estimator = absl::make_unique<MockNoiseEstimator>();
  auto mock_spectrogram_predictor =
      absl::make_unique<MockSpectrogramPredictor>();
  EXPECT_CALL(*mock_spectrogram_predictor, PredictFrame())
      // The floor of the result is wanted, so integer division is fine.
      .Times(kMaxConsecutiveLostSamples / kNumSamplesToRequest + 1)
      .WillRepeatedly(Return(mock_prediction));
  EXPECT_CALL(*mock_spectrogram_predictor, FeedFrame(mock_noise)).Times(1);
  EXPECT_CALL(*mock_spectrogram_predictor, FeedFrame(mock_features)).Times(1);
  EXPECT_CALL(*mock_noise_estimator, Update(mock_features))
      .WillOnce(Return(true));
  EXPECT_CALL(*mock_noise_estimator, NoiseEstimate())
      .WillOnce(Return(mock_noise));

  auto packet_loss_handler_peer = absl::make_unique<PacketLossHandlerPeer>(
      std::move(mock_noise_estimator), std::move(mock_spectrogram_predictor));
  int num_samples_requested_aggregated = kNumSamplesToRequest;
  while (num_samples_requested_aggregated <= kMaxConsecutiveLostSamples) {
    auto prediction =
        packet_loss_handler_peer->EstimateLostFeatures(kNumSamplesToRequest);
    ASSERT_TRUE(prediction.has_value());
    num_samples_requested_aggregated += kNumSamplesToRequest;
    EXPECT_EQ(mock_prediction, prediction.value());
    EXPECT_FALSE(packet_loss_handler_peer->is_comfort_noise());
  }
  auto prediction =
      packet_loss_handler_peer->EstimateLostFeatures(kNumSamplesToRequest);
  ASSERT_TRUE(prediction.has_value());
  EXPECT_EQ(mock_noise, prediction.value());
  EXPECT_TRUE(packet_loss_handler_peer->is_comfort_noise());
  EXPECT_TRUE(packet_loss_handler_peer->SetReceivedFeatures(mock_features));
  prediction =
      packet_loss_handler_peer->EstimateLostFeatures(kNumSamplesToRequest);
  ASSERT_TRUE(prediction.has_value());
  EXPECT_EQ(mock_prediction, prediction.value());
  EXPECT_FALSE(packet_loss_handler_peer->is_comfort_noise());
}

}  // namespace codec
}  // namespace chromemedia
