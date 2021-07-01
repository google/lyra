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

#ifndef LYRA_CODEC_PACKET_INTERFACE_H_
#define LYRA_CODEC_PACKET_INTERFACE_H_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/types/span.h"

namespace chromemedia {
namespace codec {

// An interface to abstract the quantization implementation.
class PacketInterface {
 public:
  virtual ~PacketInterface() {}

  // Packs quantized bits in a string to packet bytes.
  virtual std::vector<uint8_t> PackQuantized(
      const std::string& quantized_string) = 0;

  // Unpacks an encoded packet received over the wire to quantized bits in the
  // form of a string.
  virtual std::optional<std::string> UnpackPacket(
      const absl::Span<const uint8_t> packet) = 0;

  virtual int PacketSize() const = 0;
};

}  // namespace codec
}  // namespace chromemedia

#endif  // LYRA_CODEC_PACKET_INTERFACE_H_
