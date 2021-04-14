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

#include "noise_estimator.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numeric>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/types/optional.h"
#include "audio/dsp/signal_vector_util.h"
#include "glog/logging.h"
#include "log_mel_spectrogram_extractor_impl.h"

namespace chromemedia {
namespace codec {
namespace {

inline float Average(const std::vector<float>& vec) {
  return std::accumulate(vec.begin(), vec.end(), 0.f) / vec.size();
}

// Places the element wise min between vec1 and vec2 in assignable.
// Assignable may point to either vec1 or vec2, or may be a different vector.
void ElementWiseMin(const std::vector<float>& vec1,
                    const std::vector<float>& vec2,
                    std::vector<float>* assignable) {
  for (int i = 0; i < vec1.size(); ++i) {
    assignable->at(i) = std::min(vec1.at(i), vec2.at(i));
  }
}

// Updates the minimum value per frequency efficiently.
void UpdateMinAndTemp(uint64_t frame_num, int num_frames_per_update,
                      const std::vector<float>& smoothed_power,
                      std::vector<float>* min_power,
                      std::vector<float>* tmp_min_power) {
  if (frame_num % num_frames_per_update == 0) {
    ElementWiseMin(*tmp_min_power, smoothed_power, min_power);
    *tmp_min_power = smoothed_power;
  } else {
    ElementWiseMin(*min_power, smoothed_power, min_power);
    ElementWiseMin(*tmp_min_power, smoothed_power, tmp_min_power);
  }
}

// The smoothing factor weighs how much the smoothed power calculation should
// track the current power in a frequency band at a given frame and takes
// values on the interval (0, max_smoothing].
// Values closer to 1 indicate smoothed_power should be heavily smoothed
// (when there is noise in this frequency bin).
// Values closer to 0 indicate smoothed_power should take on the current
// power level at this frequency bin (when there is speech in this
// frequency bin).
std::vector<float> SmoothingFactor(float max_smoothing,
                                   const std::vector<float>& curr_power_db,
                                   const std::vector<float>& smoothed_power,
                                   const std::vector<float>& noise_estimate) {
  constexpr float kPowDiff = 0.3f;
  // The smoothing correction factor approaches 0 as the current power value
  // moves away from the previously calculated smoothed power, and is 1 when
  // the two are equal.
  float smoothing_correction = std::exp(-audio_dsp::Square(
      (Average(smoothed_power) - Average(curr_power_db)) / kPowDiff));

  std::vector<float> smoothing_factor(noise_estimate.size());
  for (int i = 0; i < smoothed_power.size(); ++i) {
    smoothing_factor.at(i) =
        max_smoothing * smoothing_correction *
        std::exp(-audio_dsp::Square(
            (smoothed_power.at(i) - noise_estimate.at(i)) / kPowDiff));
  }

  return smoothing_factor;
}

}  // namespace

std::unique_ptr<NoiseEstimator> NoiseEstimator::Create(
    int num_features, float num_seconds_per_frame) {
  if (num_seconds_per_frame <= 0) {
    LOG(ERROR) << "Argument num_seconds_per_frame has to be positive.";
    return nullptr;
  }

  const float kMaxSmoothingHalflifeSecs = 0.7f;
  const float kUpdateTimeSecs = 1.f;
  const float kBoundHalfLifeSecs = 1.f;

  return absl::WrapUnique(new NoiseEstimator(
      num_features, std::round(kUpdateTimeSecs / num_seconds_per_frame),
      std::pow(0.5f, num_seconds_per_frame / kMaxSmoothingHalflifeSecs),
      std::pow(0.5f, num_seconds_per_frame / kBoundHalfLifeSecs)));
}

NoiseEstimator::NoiseEstimator(int num_features, int num_frames_per_update,
                               float max_smoothing, float bound_decay_factor)
    : num_features_(num_features),
      num_frames_per_update_(num_frames_per_update),
      max_smoothing_(max_smoothing),
      bound_decay_factor_(bound_decay_factor),
      smoothed_power_(num_features),
      squared_smoothed_power_(num_features),
      tmp_min_smoothed_power_(num_features),
      noise_estimate_(num_features,
                      LogMelSpectrogramExtractorImpl::GetSilenceValue()),
      noise_bound_(num_features, 0.f) {}

// The variance of non-smoothed noise is estimated and used to calculate the
// upper bound of the noise bound.
void NoiseEstimator::ComputeBounds() {
  const float kBoundFactor = 0.9f;
  for (int i = 0; i < smoothed_power_.size(); ++i) {
    float noise_variance =
        std::max<float>(0.f, squared_smoothed_power_.at(i) -
                                 audio_dsp::Square(smoothed_power_.at(i)));
    noise_bound_.at(i) =
        kBoundFactor *
        std::sqrt(noise_variance * std::log(noise_bound_.size()));
  }
}

bool NoiseEstimator::Update(const std::vector<float>& curr_power_db) {
  if (curr_power_db.size() != num_features_) {
    return false;
  }
  if (num_frames_received_ == 0) {
    smoothed_power_ = curr_power_db;
    for (int i = 0; i < curr_power_db.size(); ++i) {
      squared_smoothed_power_.at(i) = audio_dsp::Square(curr_power_db.at(i));
    }
    tmp_min_smoothed_power_ = curr_power_db;
  }

  std::vector<float> smoothing_factor = SmoothingFactor(
      max_smoothing_, curr_power_db, smoothed_power_, noise_estimate_);

  // smoothed_power_ per frequency band = smoothing_factor * smoothed_power +
  // (1 - smoothing_factor) * curr_power_db.
  for (int i = 0; i < smoothed_power_.size(); ++i) {
    smoothed_power_.at(i) =
        smoothing_factor.at(i) * smoothed_power_.at(i) +
        (1.f - smoothing_factor.at(i)) * curr_power_db.at(i);
    squared_smoothed_power_.at(i) =
        smoothing_factor.at(i) * squared_smoothed_power_.at(i) +
        (1.f - smoothing_factor.at(i)) * audio_dsp::Square(curr_power_db.at(i));
  }

  UpdateMinAndTemp(num_frames_received_, num_frames_per_update_,
                   smoothed_power_, &noise_estimate_, &tmp_min_smoothed_power_);

  ComputeBounds();

  // Increment by 1 each time the curr_power_db is received.
  num_frames_received_ += 1;

  return true;
}

std::vector<float> NoiseEstimator::NoiseEstimate() const {
  return noise_estimate_;
}

absl::optional<bool> NoiseEstimator::IsSimilarNoise(
    const std::vector<float>& curr_power_db) {
  if (curr_power_db.size() != num_features_) {
    return absl::nullopt;
  }

  // Decide whether current frame is noise or not. A frame is considered to be
  // noise if it falls below noise_estimate_ +- noise_bound_.
  for (int i = 0; i < curr_power_db.size(); ++i) {
    if (curr_power_db.at(i) > noise_estimate_.at(i) + noise_bound_.at(i) ||
        curr_power_db.at(i) < noise_estimate_.at(i) - noise_bound_.at(i)) {
      return false;
    }
  }

  // Exponentially decay noise_bound_ if multiple frames in a row are noise.
  // This avoids getting stuck in the case where noise_bound_ is very large, as
  // the decay eventually forces Update() to recalculate the bound.
  for (auto& element : noise_bound_) {
    // x(t) = x0 * (1/2) ^ (t / t_half_life)
    //      = x(t - 1) * (1/2) ^ (1 / t_half_life)
    element *= bound_decay_factor_;
  }

  return true;
}

}  // namespace codec
}  // namespace chromemedia
