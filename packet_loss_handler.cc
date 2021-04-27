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

#include "absl/memory/memory.h"
#include "glog/logging.h"
#include "naive_spectrogram_predictor.h"
#include "noise_estimator.h"
#include "noise_estimator_interface.h"
#include "spectrogram_predictor_interface.h"

namespace chromemedia {
namespace codec {

std::unique_ptr<PacketLossHandler> PacketLossHandler::Create(
    int sample_rate_hz, int num_features, float seconds_per_frame) {
  auto noise_estimator =
      NoiseEstimator::Create(num_features, seconds_per_frame);
  if (noise_estimator == nullptr) {
    return nullptr;
  }
  auto spectrogram_predictor =
      absl::make_unique<NaiveSpectrogramPredictor>(num_features);
  if (spectrogram_predictor == nullptr) {
    return nullptr;
  }
  return absl::WrapUnique(
      new PacketLossHandler(sample_rate_hz, std::move(noise_estimator),
                            std::move(spectrogram_predictor)));
}

PacketLossHandler::PacketLossHandler(
    int sample_rate_hz,
    std::unique_ptr<NoiseEstimatorInterface> noise_estimator,
    std::unique_ptr<SpectrogramPredictorInterface> spectrogram_predictor)
    : consecutive_lost_samples_(0),
      noise_estimator_(std::move(noise_estimator)),
      spectrogram_predictor_(std::move(spectrogram_predictor)) {
  constexpr float kMaxLostSeconds = 0.1;
  max_lost_samples_ = kMaxLostSeconds * sample_rate_hz;
}

bool PacketLossHandler::SetReceivedFeatures(
    const std::vector<float>& features) {
  consecutive_lost_samples_ = 0;

  if (!noise_estimator_->Update(features)) {
    LOG(ERROR) << "Unable to update noise estimator.";
    return false;
  }

  spectrogram_predictor_->FeedFrame(features);
  return true;
}

absl::optional<std::vector<float>> PacketLossHandler::EstimateLostFeatures(
    int num_samples) {
  if (num_samples <= 0) {
    LOG(ERROR) << "Number of samples must be positive.";
    return absl::nullopt;
  }

  consecutive_lost_samples_ += num_samples;
  if (consecutive_lost_samples_ > max_lost_samples_) {
    auto noise_estimate = noise_estimator_->NoiseEstimate();
    spectrogram_predictor_->FeedFrame(noise_estimate);
    return noise_estimate;
  }
  return spectrogram_predictor_->PredictFrame();
}

bool PacketLossHandler::is_comfort_noise() const {
  return consecutive_lost_samples_ > max_lost_samples_;
}

}  // namespace codec
}  // namespace chromemedia
