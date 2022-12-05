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

#ifndef LYRA_LYRA_DECODER_INTERFACE_H_
#define LYRA_LYRA_DECODER_INTERFACE_H_

#include <cstdint>
#include <optional>
#include <vector>

#include "absl/types/span.h"

namespace chromemedia {
namespace codec {

// An interface to abstract LyraDecoder.
class LyraDecoderInterface {
 public:
  virtual ~LyraDecoderInterface() = default;

  // Parses |encoded| then prepares the decoder to decode samples from the
  // payload of |encoded|.
  // Returns false if |encoded| is invalid.
  // Returns true on success.
  virtual bool SetEncodedPacket(absl::Span<const uint8_t> encoded) = 0;

  // Decodes |num_samples|.
  // Returns nullopt on failure.
  virtual std::optional<std::vector<int16_t>> DecodeSamples(
      int num_samples) = 0;

  virtual int sample_rate_hz() const = 0;

  virtual int num_channels() const = 0;

  virtual int frame_rate() const = 0;

  virtual bool is_comfort_noise() const = 0;
};

}  // namespace codec
}  // namespace chromemedia

#endif  // LYRA_LYRA_DECODER_INTERFACE_H_
