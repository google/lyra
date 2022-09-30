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

#ifndef LYRA_CODEC_GILBERT_MODEL_H_
#define LYRA_CODEC_GILBERT_MODEL_H_

#include <memory>
#include <random>

#include "packet_loss_model_interface.h"

namespace chromemedia {
namespace codec {

// Gilbert model to simulate packet loss bursts.
class GilbertModel : public PacketLossModelInterface {
 public:
  static std::unique_ptr<GilbertModel> Create(float packet_loss_rate,
                                              float average_burst_length,
                                              bool random_seed = true);

  // Update the internal state according to the corresponding probabilities.
  // Returns true if the packet was received.
  // Returns false if the packet was lost.
  bool IsPacketReceived() override;

 private:
  GilbertModel(float received2lost_probability, float lost2received_probability,
               unsigned int seed);

  const float received2lost_probability_;
  const float lost2received_probability_;

  bool is_packet_received_;

  // Not using absl::Uniform because it can't ensure the same output between
  // runs even with explicit seeding. GilbertModelTest.DistributionFollowsParams
  // would be more complex with a MockingBitGen where the results have to be set
  // manually.
  std::mt19937 gen_;
  std::uniform_real_distribution<float> prob_;
};

}  // namespace codec
}  // namespace chromemedia

#endif  // LYRA_CODEC_GILBERT_MODEL_H_
