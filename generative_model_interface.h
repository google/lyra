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

#ifndef LYRA_CODEC_GENERATIVE_MODEL_INTERFACE_H_
#define LYRA_CODEC_GENERATIVE_MODEL_INTERFACE_H_

#include <cstdint>
#include <vector>

#include "absl/types/optional.h"

namespace chromemedia {
namespace codec {

// An interface to abstract the audio generation from the model
// implementation.
class GenerativeModelInterface {
 public:
  virtual ~GenerativeModelInterface() {}

  // Converts the features obtained from a packet into the format the model
  // expects.
  virtual void AddFeatures(const std::vector<float>& features) = 0;

  // Runs the model and generates |num_samples| audio samples.
  // Returns a vector of audio samples on success. Returns a nullopt on failure.
  virtual absl::optional<std::vector<int16_t>> GenerateSamples(
      int num_samples) = 0;

  // Clears any information about previous frames stored by the model.
  virtual void Reset() {}

#ifdef BENCHMARK
  // Returns amount of time elapsed between start and end of each conditioning
  // stack run.
  std::vector<int64_t> conditioning_timings_microsecs() {
    return conditioning_timings_microsecs_;
  }

  // Returns amount of time elapsed between start and end of each model run.
  std::vector<int64_t> model_timings_microsecs() {
    return model_timings_microsecs_;
  }

 protected:
  std::vector<int64_t> conditioning_timings_microsecs_;
  std::vector<int64_t> model_timings_microsecs_;
#endif  // BENCHMARK
};

}  // namespace codec
}  // namespace chromemedia

#endif  // LYRA_CODEC_GENERATIVE_MODEL_INTERFACE_H_
