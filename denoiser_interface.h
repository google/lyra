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

#ifndef LYRA_CODEC_DENOISER_INTERFACE_H_
#define LYRA_CODEC_DENOISER_INTERFACE_H_

#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"

namespace chromemedia {
namespace codec {

// An interface for an abstract denoiser implementation for the Lyra encoder.
class DenoiserInterface {
 public:
  virtual ~DenoiserInterface() {}

  // Report the number of samples per hop, or 0 if it doesn't matter.
  virtual int SamplesPerHop() const = 0;

  // Apply denoising to a frame of audio.
  virtual absl::StatusOr<std::vector<int16_t>> Denoise(
      absl::Span<const int16_t> input) = 0;
};

}  // namespace codec
}  // namespace chromemedia

#endif  // LYRA_CODEC_DENOISER_INTERFACE_H_
