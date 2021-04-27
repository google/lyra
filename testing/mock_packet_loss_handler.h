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

#ifndef LYRA_CODEC_TESTING_MOCK_PACKET_LOSS_HANDLER_H_
#define LYRA_CODEC_TESTING_MOCK_PACKET_LOSS_HANDLER_H_

#include <vector>

#include "absl/types/optional.h"
#include "gmock/gmock.h"
#include "packet_loss_handler_interface.h"

namespace chromemedia {
namespace codec {

class MockPacketLossHandler : public PacketLossHandlerInterface {
 public:
  ~MockPacketLossHandler() override {}

  MOCK_METHOD(absl::optional<std::vector<float>>, EstimateLostFeatures,
              (int num_samples), (override));

  MOCK_METHOD(bool, SetReceivedFeatures, (const std::vector<float>&),
              (override));

  MOCK_METHOD(bool, is_comfort_noise, (), (const, override));
};

}  // namespace codec
}  // namespace chromemedia

#endif  // LYRA_CODEC_TESTING_MOCK_PACKET_LOSS_HANDLER_H_
