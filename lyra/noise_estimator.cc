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

#include "lyra/noise_estimator.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/types/span.h"
#include "audio/dsp/signal_vector_util.h"
#include "glog/logging.h"  // IWYU pragma: keep
#include "lyra/log_mel_spectrogram_extractor_impl.h"

namespace chromemedia {
namespace codec {
namespace {

inline float Average(const std::vector<float>& to_average) {
  return std::accumulate(to_average.begin(), to_average.end(), 0.f) /
         static_cast<float>(to_average.size());
}

// Places the element-wise min between |vector_1| and |vector_2| in
// |assignable|. Assignable may point to either |vector_1| or |vector_2|, or
// may be a different vector.
void ElementWiseMin(const std::vector<float>& vector_1,
                    const std::vector<float>& vector_2,
                    std::vector<float>* assignable) {
  std::transform(vector_1.begin(), vector_1.end(), vector_2.begin(),
                 assignable->begin(),
                 [](float a, float b) { return std::min(a, b); });
}

// Updates the minimum value per frequency efficiently.
void UpdateMinAndTemp(uint64_t num_hops_received,
                      const std::vector<float>& smoothed_power,
                      std::vector<float>* min_power,
                      std::vector<float>* tmp_min_power) {
  if (num_hops_received == 0) {
    ElementWiseMin(*tmp_min_power, smoothed_power, min_power);
    *tmp_min_power = smoothed_power;
  } else {
    ElementWiseMin(*min_power, smoothed_power, min_power);
    ElementWiseMin(*tmp_min_power, smoothed_power, tmp_min_power);
  }
}

// The smoothing factor weighs how much the smoothed power calculation should
// track the current power in a frequency band at a given hop and takes
// values on the interval (0, |max_smoothing|].
// Values closer to 1 indicate |smoothed_power| should be heavily smoothed
// (when there is noise in this frequency bin).
// Values closer to 0 indicate smoothed_power should take on the current
// power level at this frequency bin (when there is speech in this
// frequency bin).
std::vector<float> SmoothingFactor(float max_smoothing,
                                   const std::vector<float>& current_power_db,
                                   const std::vector<float>& smoothed_power,
                                   const std::vector<float>& noise_estimate) {
  constexpr float kPowDiff = 0.3f;
  // The smoothing correction factor approaches 0 as the current power value
  // moves away from the previously calculated smoothed power, and is 1 when
  // the two are equal.
  float smoothing_correction = std::exp(-audio_dsp::Square(
      (Average(smoothed_power) - Average(current_power_db)) / kPowDiff));

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
    int sample_rate_hz, int num_samples_per_hop, int num_samples_per_window,
    int num_features) {
  const float kNumSecondsPerHop =
      static_cast<float>(num_samples_per_hop) / sample_rate_hz;

  auto log_mel_spectrogram_extractor = LogMelSpectrogramExtractorImpl::Create(
      sample_rate_hz, num_samples_per_hop, num_samples_per_window,
      num_features);
  if (log_mel_spectrogram_extractor == nullptr) {
    LOG(ERROR) << "Could not create LogMelSpectrogramExtractorImpl for "
                  "NoiseEstimator.";
    return nullptr;
  }

  const float kMaxSmoothingHalflifeSecs = 0.7f;
  const float kUpdateTimeSecs = 1.f;
  const float kBoundHalfLifeSecs = 1.f;
  return absl::WrapUnique(new NoiseEstimator(
      num_samples_per_hop, std::round(kUpdateTimeSecs / kNumSecondsPerHop),
      num_features,
      std::pow(0.5f, kNumSecondsPerHop / kMaxSmoothingHalflifeSecs),
      std::pow(0.5f, kNumSecondsPerHop / kBoundHalfLifeSecs),
      std::move(log_mel_spectrogram_extractor)));
}

NoiseEstimator::NoiseEstimator(int num_samples_per_hop, int num_hops_per_update,
                               int num_features, float max_smoothing,
                               float bound_decay_factor,
                               std::unique_ptr<LogMelSpectrogramExtractorImpl>
                                   log_mel_spectrogram_extractor)
    : num_samples_per_hop_(num_samples_per_hop),
      num_hops_per_update_(num_hops_per_update),
      max_smoothing_(max_smoothing),
      bound_decay_factor_(bound_decay_factor),
      squared_smoothed_power_(num_features),
      tmp_min_smoothed_power_(num_features),
      noise_estimate_(num_features, 0.f),
      noise_bound_(num_features, 0.f),
      past_samples_hop_(num_samples_per_hop),
      is_noise_(true),
      num_hops_received_(0),
      next_sample_in_hop_(0),
      log_mel_spectrogram_extractor_(std::move(log_mel_spectrogram_extractor)) {
}

bool NoiseEstimator::ReceiveSamples(const absl::Span<const int16_t> samples) {
  if (samples.size() + next_sample_in_hop_ > num_samples_per_hop_) {
    LOG(ERROR) << "Buffer overflow in NoiseEstimator. Max sample"
               << " vector size is " << num_samples_per_hop_ << " but "
               << samples.size() << " were passed in and "
               << past_samples_hop_.size() << " were already in the buffer.";
    return false;
  }
  std::copy(samples.begin(), samples.end(),
            past_samples_hop_.begin() + next_sample_in_hop_);
  next_sample_in_hop_ += samples.size();
  // Extract a log mel spectrogram once the buffer is filled and pass it to the
  // noise estimator.
  if (next_sample_in_hop_ == num_samples_per_hop_) {
    next_sample_in_hop_ = 0;
    auto log_mel_spectrogram = log_mel_spectrogram_extractor_->Extract(
        absl::MakeConstSpan(past_samples_hop_));
    if (!log_mel_spectrogram.has_value()) {
      LOG(ERROR) << "Unable to extract features from decoded audio.";
      return false;
    }
    is_noise_ = ComputeIsNoise(log_mel_spectrogram.value());
    if (is_noise_) {
      DecayBounds();
    } else {
      UpdateNoiseEstimate(log_mel_spectrogram.value());
    }
  }
  return true;
}

void NoiseEstimator::UpdateNoiseEstimate(
    const std::vector<float>& current_power_db) {
  // Only executed once, the first time |num_samples_per_hop_| samples have been
  // passed to |ReceiveSamples|.
  if (smoothed_power_.empty()) {
    smoothed_power_ = current_power_db;
    for (int i = 0; i < current_power_db.size(); ++i) {
      squared_smoothed_power_.at(i) = audio_dsp::Square(current_power_db.at(i));
    }
    tmp_min_smoothed_power_ = current_power_db;
  }

  std::vector<float> smoothing_factor = SmoothingFactor(
      max_smoothing_, current_power_db, smoothed_power_, noise_estimate_);
  // |smoothed_power_| per frequency band =
  //     |smoothing_factor| * |smoothed_power_| +
  //     (1 - |smoothing_factor|) * |current_power_db|.
  for (int i = 0; i < smoothed_power_.size(); ++i) {
    smoothed_power_.at(i) =
        smoothing_factor.at(i) * smoothed_power_.at(i) +
        (1.f - smoothing_factor.at(i)) * current_power_db.at(i);
    squared_smoothed_power_.at(i) =
        smoothing_factor.at(i) * squared_smoothed_power_.at(i) +
        (1.f - smoothing_factor.at(i)) *
            audio_dsp::Square(current_power_db.at(i));
  }

  UpdateMinAndTemp(num_hops_received_, smoothed_power_, &noise_estimate_,
                   &tmp_min_smoothed_power_);
  ComputeBounds();

  num_hops_received_ = (num_hops_received_ + 1) % num_hops_per_update_;
}

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

bool NoiseEstimator::ComputeIsNoise(
    const std::vector<float>& current_power_db) {
  // Decide whether current hop is noise or not. A hop is considered to be
  // noise if it falls below |noise_estimate_| +- |noise_bound_|.
  for (int i = 0; i < current_power_db.size(); ++i) {
    if (std::abs(current_power_db.at(i) - noise_estimate_.at(i)) >
        noise_bound_.at(i)) {
      return false;
    }
  }
  return true;
}

void NoiseEstimator::DecayBounds() {
  // Exponentially decay |noise_bound_| if multiple hops in a row are noise.
  // This avoids getting stuck in the case where |noise_bound_| is very large,
  // as the decay eventually forces Update() to recalculate the bound.
  for (auto& element : noise_bound_) {
    // x(t) = x0 * (1/2) ^ (t / t_half_life)
    //      = x(t - 1) * (1/2) ^ (1 / t_half_life)
    element *= bound_decay_factor_;
  }
}

std::vector<float> NoiseEstimator::noise_estimate() const {
  return noise_estimate_;
}

bool NoiseEstimator::is_noise() const { return is_noise_; }

}  // namespace codec
}  // namespace chromemedia
