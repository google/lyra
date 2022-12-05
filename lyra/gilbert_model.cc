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

#include "lyra/gilbert_model.h"

#include <memory>
#include <random>

#include "absl/memory/memory.h"
#include "glog/logging.h"  // IWYU pragma: keep

namespace chromemedia {
namespace codec {

std::unique_ptr<GilbertModel> GilbertModel::Create(float packet_loss_rate,
                                                   float average_burst_length,
                                                   bool random_seed) {
  if (average_burst_length < 1.f) {
    LOG(ERROR) << "Average Burst Length has to be at least 1, but was "
               << average_burst_length << ".";
    return nullptr;
  }
  if (packet_loss_rate < 0.f) {
    LOG(ERROR) << "Packet Loss Rate has to be positive, but was "
               << packet_loss_rate << ".";
    return nullptr;
  }
  if (packet_loss_rate > average_burst_length / (average_burst_length + 1.f)) {
    LOG(ERROR) << "Packet Loss Rate cannot be larger than "
               << "average_burst_length/(average_burst_length+1)="
               << average_burst_length / (average_burst_length + 1.f)
               << ", but was " << packet_loss_rate << ".";
    return nullptr;
  }

  unsigned int seed = 5489u;
  if (random_seed) {
    std::random_device rd;
    seed = rd();
  }

  return absl::WrapUnique(new GilbertModel(
      packet_loss_rate / (average_burst_length * (1.f - packet_loss_rate)),
      1.f / average_burst_length, seed));
}

GilbertModel::GilbertModel(float received2lost_probability,
                           float lost2received_probability, unsigned int seed)
    : received2lost_probability_(received2lost_probability),
      lost2received_probability_(lost2received_probability),
      is_packet_received_(true),
      gen_(seed) {}

bool GilbertModel::IsPacketReceived() {
  bool current_packet_received = is_packet_received_;
  if (is_packet_received_) {
    if (prob_(gen_) < received2lost_probability_) {
      is_packet_received_ = false;
    }
  } else {
    if (prob_(gen_) < lost2received_probability_) {
      is_packet_received_ = true;
    }
  }

  return current_packet_received;
}

}  // namespace codec
}  // namespace chromemedia
