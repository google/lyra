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

#ifndef LYRA_CODEC_PACKET_LOSS_HANDLER_INTERFACE_H_
#define LYRA_CODEC_PACKET_LOSS_HANDLER_INTERFACE_H_

#include <vector>

#include "absl/types/optional.h"

namespace chromemedia {
namespace codec {

// An interface to abstract the PacketLossHandler implementation.
class PacketLossHandlerInterface {
 public:
  virtual ~PacketLossHandlerInterface() {}

  // Called for every new packet received by the decoder.
  virtual bool SetReceivedFeatures(const std::vector<float>& features) = 0;

  // When a packet is not received provides a packet to be used instead.
  virtual absl::optional<std::vector<float>> EstimateLostFeatures(
      int num_samples) = 0;

  virtual bool is_comfort_noise() const = 0;
};

}  // namespace codec
}  // namespace chromemedia

#endif  // LYRA_CODEC_PACKET_LOSS_HANDLER_INTERFACE_H_
