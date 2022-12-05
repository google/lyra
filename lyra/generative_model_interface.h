/*
 * Copyright 2021 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef LYRA_GENERATIVE_MODEL_INTERFACE_H_
#define LYRA_GENERATIVE_MODEL_INTERFACE_H_

#include <cstdint>
#include <optional>
#include <queue>
#include <vector>

#include "glog/logging.h"  // IWYU pragma: keep

namespace chromemedia {
namespace codec {

// An interface to abstract the audio generation from the model
// implementation.
class GenerativeModelInterface {
 public:
  virtual ~GenerativeModelInterface() {}

  virtual bool AddFeatures(const std::vector<float>& features) = 0;

  virtual std::optional<std::vector<int16_t>> GenerateSamples(
      int num_samples) = 0;

  virtual int num_samples_available() const = 0;
};

// Enforces that features are added and then decoded via a FIFO queue.
class GenerativeModel : public GenerativeModelInterface {
 public:
  virtual ~GenerativeModel() {}

  // Adds received features to the model.
  bool AddFeatures(const std::vector<float>& features) override final {
    if (features.size() != num_features_) {
      LOG(ERROR) << "Expecting features to be of shape " << num_features_
                 << " but were of shape " << features.size() << ".";
      return false;
    }
    features_queue_.push(features);
    return true;
  }

  // Runs the model and generates |num_samples| audio samples.
  // Returns a vector of audio samples on success. Returns a nullopt on failure.
  std::optional<std::vector<int16_t>> GenerateSamples(
      int num_samples) override final {
    if (num_samples < 0) {
      LOG(ERROR) << "Number of samples must be positive.";
      return std::nullopt;
    }
    // Do not call costly models if no samples have been requested.
    if (num_samples == 0) {
      return std::vector<int16_t>(0);
    }
    if (num_samples_available() == 0) {
      LOG(ERROR) << "Tried generating " << num_samples << " samples but only "
                 << num_samples_available() << " are available.";
      return std::nullopt;
    }
    if (next_sample_in_hop_ == 0) {
      if (!RunConditioning(features_queue_.front())) {
        return std::nullopt;
      }
    }
    const int num_samples_remaining =
        num_samples_per_hop_ - next_sample_in_hop_;
    if (num_samples > num_samples_remaining) {
      LOG(ERROR) << "Tried generating " << num_samples << " samples but only "
                 << num_samples_remaining
                 << " were available in current features.";
      return std::nullopt;
    }
    auto samples = RunModel(num_samples);
    if (samples.has_value()) {
      next_sample_in_hop_ += samples->size();
      // Cumulative samples generated are guaranteed to never straddle
      // multiples of |num_samples_per_hop_|.
      if (next_sample_in_hop_ == num_samples_per_hop_) {
        next_sample_in_hop_ = 0;
        features_queue_.pop();
      }
    }
    return samples;
  }

  int num_samples_available() const override final {
    return features_queue_.size() * num_samples_per_hop_ - next_sample_in_hop_;
  }

 protected:
  GenerativeModel(int num_samples_per_hop, int num_features)
      : num_samples_per_hop_(num_samples_per_hop),
        num_features_(num_features),
        next_sample_in_hop_(0) {
    VLOG(1) << "Number of features: " << num_features;
    VLOG(1) << "Number of samples per feature: " << num_samples_per_hop;
  }

  // Process the features on top of the queue.
  // Called from |GenerateSamples|.
  virtual bool RunConditioning(const std::vector<float>& features) = 0;

  // Generate samples from the latest set of features added by |AddFeatures|,
  // which have already been processed by |RunConditioning|.
  virtual std::optional<std::vector<int16_t>> RunModel(int num_samples) = 0;

  int next_sample_in_hop() const { return next_sample_in_hop_; }

 private:
  GenerativeModel() = delete;

  // Provide read-only access to these member variables in derived classes.
  const int num_samples_per_hop_;
  const int num_features_;
  int next_sample_in_hop_;
  std::queue<std::vector<float>> features_queue_;
};

}  // namespace codec
}  // namespace chromemedia

#endif  // LYRA_GENERATIVE_MODEL_INTERFACE_H_
