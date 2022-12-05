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

#include "lyra/fixed_packet_loss_model.h"

#include <string>
#include <vector>

#include "gtest/gtest.h"

namespace chromemedia {
namespace codec {
namespace {

TEST(DeterministicPacketLossModelTest, TestPattern) {
  const int kSampleRateHz = 16000;
  const float kHopDurationSeconds = 0.02;
  const int kNumSamplesPerHop = kSampleRateHz * kHopDurationSeconds;
  // Starts are rounded to: 0.f, 0.04f, 0.12f.
  std::vector<float> start_seconds({0.f, 0.05f, 0.12f});
  // Durations are rounded to 0.02f, 0.04f, 0.02f
  std::vector<float> duration_seconds({0.02f, 0.04f, 0.01f});
  // Segements of packet loss are: [0.f, 0.02f), [0.06f, 0.1f), [0.12f, 0.14f).
  auto model = FixedPacketLossModel(kSampleRateHz, kNumSamplesPerHop,
                                    start_seconds, duration_seconds);
  EXPECT_FALSE(model.IsPacketReceived())
      << "0.00s to 0.02s packet should not be received.";
  EXPECT_TRUE(model.IsPacketReceived())
      << "0.02s to 0.04s packet should be received.";
  EXPECT_TRUE(model.IsPacketReceived())
      << "0.04s to 0.06s packet should be received.";
  EXPECT_FALSE(model.IsPacketReceived())
      << "0.06s to 0.08s packet should not be received.";
  EXPECT_FALSE(model.IsPacketReceived())
      << "0.08s to 0.10s packet should not be received.";
  EXPECT_TRUE(model.IsPacketReceived())
      << "0.10s to 0.12s packet should be received.";
  EXPECT_FALSE(model.IsPacketReceived())
      << "0.12s to 0.14s packet should not be received.";
  EXPECT_TRUE(model.IsPacketReceived())
      << "0.14s and onwards should be received.";
}

}  // namespace
}  // namespace codec
}  // namespace chromemedia
