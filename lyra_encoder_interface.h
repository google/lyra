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

#ifndef LYRA_CODEC_LYRA_ENCODER_INTERFACE_H_
#define LYRA_CODEC_LYRA_ENCODER_INTERFACE_H_

#include <cstdint>
#include <optional>
#include <vector>

#include "absl/types/span.h"

namespace chromemedia {
namespace codec {

// An interface to abstract LyraEncoder.
class LyraEncoderInterface {
 public:
  virtual ~LyraEncoderInterface() = default;

  // Returns the audio samples encoded. Returns nullptrt
  // on failure.
  virtual std::optional<std::vector<uint8_t>> Encode(
      const absl::Span<const int16_t> audio) = 0;

  virtual bool set_bitrate(int bitrate) = 0;

  virtual int sample_rate_hz() const = 0;

  virtual int num_channels() const = 0;

  virtual int bitrate() const = 0;

  virtual int frame_rate() const = 0;
};

}  // namespace codec
}  // namespace chromemedia

#endif  // LYRA_CODEC_LYRA_ENCODER_INTERFACE_H_
