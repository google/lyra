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

#include "gilbert_model.h"

#include <memory>
#include <numeric>
#include <vector>

#include "gtest/gtest.h"

namespace chromemedia {
namespace codec {
namespace {

TEST(GilbertModelTest, TooSmallAverageBurstLength) {
  ASSERT_EQ(nullptr, GilbertModel::Create(0.3, 0.5));
}

TEST(GilbertModelTest, NegativePacketLossRate) {
  ASSERT_EQ(nullptr, GilbertModel::Create(-0.5, 0.5));
}

TEST(GilbertModelTest, TooLargePacketLossRate) {
  ASSERT_EQ(nullptr, GilbertModel::Create(0.7, 2));
}

TEST(GilbertModelTest, DistributionFollowsParams) {
  // The tolerances have been determined empirically, since the model doesn't
  // make it easy to draw theoretical tolerances.
  const float kPacketLossRate = 0.5f;
  const float kPLRTolerance = 3e-5f;
  const float kAverageBurstLength = 2.f;
  const float kABLTolerance = 4e-4f;
  const int kNumPackets = 1000000;
  auto gilbert_model =
      GilbertModel::Create(kPacketLossRate, kAverageBurstLength, false);
  ASSERT_NE(nullptr, gilbert_model);
  int packets_received = 0;
  int burst_length = 0;
  std::vector<int> burst_lengths;

  for (int i = 0; i < kNumPackets; ++i) {
    if (gilbert_model->IsPacketReceived()) {
      ++packets_received;
      if (burst_length > 0) {
        burst_lengths.push_back(burst_length);
        burst_length = 0;
      }
    } else {
      ++burst_length;
    }
  }
  float estimated_packet_loss_rate =
      static_cast<float>(kNumPackets - packets_received) / kNumPackets;
  float estimated_average_burst_length =
      static_cast<float>(
          std::accumulate(burst_lengths.begin(), burst_lengths.end(), 0ul)) /
      burst_lengths.size();

  EXPECT_NEAR(kPacketLossRate, estimated_packet_loss_rate, kPLRTolerance);
  EXPECT_NEAR(kAverageBurstLength, estimated_average_burst_length,
              kABLTolerance);
}

}  // namespace
}  // namespace codec
}  // namespace chromemedia
