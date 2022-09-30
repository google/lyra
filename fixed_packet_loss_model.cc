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

#include "fixed_packet_loss_model.h"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

namespace chromemedia {
namespace codec {

FixedPacketLossModel::FixedPacketLossModel(
    int sample_rate_hz, int num_samples_per_hop,
    const std::vector<float>& burst_starts_seconds,
    const std::vector<float>& burst_durations_seconds)
    : lost_packet_intervals_(burst_starts_seconds.size()), packet_index_(0) {
  // Combine entries from |burst_starts_seconds| and |burst_durations_seconds|
  // into pairs of packet loss intervals.
  std::transform(
      burst_starts_seconds.begin(), burst_starts_seconds.end(),
      burst_durations_seconds.begin(), lost_packet_intervals_.begin(),
      [sample_rate_hz, num_samples_per_hop](float start, float duration) {
        const int start_packet = static_cast<int>(
            std::ceil(sample_rate_hz * start / num_samples_per_hop));
        const int end_packet = static_cast<int>(std::ceil(
            sample_rate_hz * (start + duration) / num_samples_per_hop));
        return std::make_pair(start_packet, end_packet);
      });
}

bool FixedPacketLossModel::IsPacketReceived() {
  // The packet is received if none of the [start, end) intervals from
  // |lost_packet_intervals_| include the current |packet_index|.
  bool is_received =
      std::none_of(lost_packet_intervals_.begin(), lost_packet_intervals_.end(),
                   [this](std::pair<int, int> start_end) {
                     return (this->packet_index_ >= start_end.first) &&
                            (this->packet_index_ < start_end.second);
                   });
  ++packet_index_;
  return is_received;
}

}  // namespace codec
}  // namespace chromemedia
