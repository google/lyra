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

#include "naive_spectrogram_predictor.h"

#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "log_mel_spectrogram_extractor_impl.h"
#include "lyra_config.h"

namespace chromemedia {
namespace codec {

class NaiveSpectrogramPredictorPeer {
 public:
  explicit NaiveSpectrogramPredictorPeer(int num_features)
      : naive_spectrogram_predictor_(num_features) {}

  void FeedFrame(const std::vector<float>& features) {
    return naive_spectrogram_predictor_.FeedFrame(features);
  }

  std::vector<float> PredictFrame() {
    return naive_spectrogram_predictor_.PredictFrame();
  }

  std::vector<float> FetchLastPacket() {
    return naive_spectrogram_predictor_.last_packet_;
  }

 private:
  NaiveSpectrogramPredictor naive_spectrogram_predictor_;
};

// Initializes a NaiveSpectrogramPredictor and ensures that its |last_packet_|
// value is all silence.
TEST(NaiveSpectrogramPredictorTest, LastPacketInitializesToSilence) {
  std::vector<float> silence(kNumFeatures,
                             LogMelSpectrogramExtractorImpl::GetSilenceValue());
  auto naive_spectrogram_predictor_peer =
      absl::make_unique<NaiveSpectrogramPredictorPeer>(kNumFeatures);
  EXPECT_EQ(silence, naive_spectrogram_predictor_peer->FetchLastPacket());
}

// Calls FeedFrame on a NaiveSpectrogramPredictor with a valid feature vector
// and ensures that |last_packet_| is set to the input feature vector.
TEST(NaiveSpectrogramPredictorTest, FeedFrameSetsLastPacket) {
  std::vector<float> features(kNumFeatures, 1.0);
  auto naive_spectrogram_predictor_peer =
      absl::make_unique<NaiveSpectrogramPredictorPeer>(kNumFeatures);
  naive_spectrogram_predictor_peer->FeedFrame(features);
  EXPECT_EQ(features, naive_spectrogram_predictor_peer->FetchLastPacket());
}

// Sets the value of |last_packet_| for a NaiveSpectrogramPredictor to a value
// other than the default silence value and calls PredictFrame and ensures this
// nondefault value is returned.
TEST(NaiveSpectrogramPredictorTest, PredictFrameReturnsLastPacket) {
  std::vector<float> features(kNumFeatures, 1.0);
  auto naive_spectrogram_predictor_peer =
      absl::make_unique<NaiveSpectrogramPredictorPeer>(kNumFeatures);
  naive_spectrogram_predictor_peer->FeedFrame(features);
  auto prediction = naive_spectrogram_predictor_peer->PredictFrame();
  EXPECT_EQ(features, prediction);
}

}  // namespace codec
}  // namespace chromemedia
