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

#ifndef LYRA_FIXED_PACKET_LOSS_MODEL_H_
#define LYRA_FIXED_PACKET_LOSS_MODEL_H_

#include <utility>
#include <vector>

#include "lyra/packet_loss_model_interface.h"

namespace chromemedia {
namespace codec {

class FixedPacketLossModel : public PacketLossModelInterface {
 public:
  // Rounds burst durations up to align with hop boundaries.
  FixedPacketLossModel(int sample_rate_hz, int num_samples_per_hop,
                       const std::vector<float>& burst_starts_seconds,
                       const std::vector<float>& burst_durations_seconds);

  // Update the internal state according to the model parameters.
  // Returns true if the packet was received.
  // Returns false if the packet was lost.
  bool IsPacketReceived() override;

 private:
  std::vector<std::pair<int, int>> lost_packet_intervals_;
  int packet_index_;
};

}  // namespace codec
}  // namespace chromemedia

#endif  // LYRA_FIXED_PACKET_LOSS_MODEL_H_
