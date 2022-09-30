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

#include "packet.h"

#include <bitset>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/types/span.h"
#include "gtest/gtest.h"

namespace chromemedia {
namespace codec {
namespace {

class PacketTest : public testing::Test {
 protected:
  // Used by several tests. Unrelated to what Lyra actually uses.
  static constexpr int kNumHeaderBits = 8;
  static constexpr int kNumQuantizedBits = 104;
  static constexpr int kMaxNumPacketBits = kNumHeaderBits + kNumQuantizedBits;
  static constexpr int kPacketSize = kMaxNumPacketBits / CHAR_BIT;

  bool DoesPacketContainQuantized(const std::vector<uint8_t>& packet,
                                  const std::string& quantized_string,
                                  int num_header_bits, int num_quantized_bits) {
    std::string packet_data;
    for (size_t i = 0; i < packet.size(); i++) {
      packet_data.append(std::bitset<CHAR_BIT>(packet[i]).to_string());
    }
    packet_data = packet_data.substr(num_header_bits, packet_data.size());
    // Remove extra bits at the end of packet_data if the number of bits stored
    // in the packet is not evenly divisible by CHAR_BITS.
    packet_data = packet_data.substr(0, num_quantized_bits);
    return packet_data == quantized_string;
  }

  int ExpectedPacketSize(int num_data_bits, int header_size) {
    return static_cast<int>(
        std::ceil(static_cast<float>(num_data_bits + header_size) /
                  static_cast<float>(CHAR_BIT)));
  }
};

TEST_F(PacketTest, MaxNumPacketBitsTooLow) {
  constexpr int kNumHeaderBitsTest = 8;
  constexpr int kNumQuantizedBitsTest = 56;
  constexpr int kMaxNumPacketBitsTest =
      kNumHeaderBitsTest + kNumQuantizedBitsTest - 1;
  EXPECT_EQ(nullptr, Packet<kMaxNumPacketBitsTest>::Create(
                         kNumHeaderBitsTest, kNumQuantizedBitsTest));
}

TEST_F(PacketTest, MaxNumPacketBitsLargeEnough) {
  constexpr int kNumHeaderBitsTest = 8;
  constexpr int kNumQuantizedBitsTest = 56;
  constexpr int kMaxNumPacketBitsTest =
      kNumHeaderBitsTest + kNumQuantizedBitsTest + 1;
  EXPECT_NE(nullptr, Packet<kMaxNumPacketBitsTest>::Create(
                         kNumHeaderBitsTest, kNumQuantizedBitsTest));
}

TEST_F(PacketTest, PacketSize) {
  constexpr int kNumHeaderBitsTest = 7;
  constexpr int kNumQuantizedBitsTest = 52;
  constexpr int kMaxNumPacketBitsTest =
      kNumHeaderBitsTest + kNumQuantizedBitsTest;
  auto packet = Packet<kMaxNumPacketBitsTest>::Create(kNumHeaderBitsTest,
                                                      kNumQuantizedBitsTest);
  ASSERT_NE(packet, nullptr);
  EXPECT_EQ(packet->PacketSize(),
            ExpectedPacketSize(kNumQuantizedBitsTest, kNumHeaderBitsTest));
}

TEST_F(PacketTest, UnpackVariableHeader) {
  constexpr int kNumHeaderBitsTest = 3;
  constexpr int kNumQuantizedBitsTest = 16;
  constexpr int kMaxNumPacketBitsTest =
      kNumHeaderBitsTest + kNumQuantizedBitsTest;

  // Create packet with header set to 0s, bits set to 1s.
  std::vector<uint8_t> encoded = {
      0b00011111,
      0b11111111,
      0b11100000,
  };
  std::bitset<kNumQuantizedBitsTest> expected_bits("1111111111111111");

  auto packet = Packet<kMaxNumPacketBitsTest>::Create(kNumHeaderBitsTest,
                                                      kNumQuantizedBitsTest);
  ASSERT_NE(packet, nullptr);
  const auto unpacked = packet->UnpackPacket(absl::MakeConstSpan(encoded));
  EXPECT_EQ(expected_bits.to_string(), unpacked.value());
}

TEST_F(PacketTest, UnpackNoTrailingZeros) {
  // Create packet. (kNumHeaderBitsTest + kNumQuantizedBitsTest) % CHAR_BIT = 0.
  constexpr int kNumHeaderBitsTest = 2;
  constexpr int kNumQuantizedBitsTest = 22;
  constexpr int kMaxNumPacketBitsTest =
      kNumHeaderBitsTest + kNumQuantizedBitsTest;
  std::vector<uint8_t> encoded(3, 0b11111111);
  encoded[0] = 0b00111111;  // Set header
  std::bitset<kNumQuantizedBitsTest> expected_bits("1111111111111111111111");

  auto packet = Packet<kMaxNumPacketBitsTest>::Create(kNumHeaderBitsTest,
                                                      kNumQuantizedBitsTest);
  ASSERT_NE(packet, nullptr);
  const auto unpacked = packet->UnpackPacket(absl::MakeConstSpan(encoded));
  EXPECT_EQ(expected_bits.to_string(), unpacked.value());
}

TEST_F(PacketTest, UnpackBigHeader) {
  // Create packet with header set to 0s, bits set to 1s.
  constexpr int kNumHeaderBitsTest = 22;
  constexpr int kNumQuantizedBitsTest = 22;
  constexpr int kMaxNumPacketBitsTest =
      kNumHeaderBitsTest + kNumQuantizedBitsTest;
  std::vector<uint8_t> encoded = {
      0b00000000, 0b00000000, 0b00000011, 0b11111111, 0b11111111,
      0b11110000,  // (kNumHeaderBitsTest + kNumQuantizedBitsTest) % CHAR_BIT =
                   // 4
  };
  std::bitset<kNumQuantizedBitsTest> expected_bits("1111111111111111111111");

  auto packet = Packet<kMaxNumPacketBitsTest>::Create(kNumHeaderBitsTest,
                                                      kNumQuantizedBitsTest);
  ASSERT_NE(packet, nullptr);
  const auto unpacked = packet->UnpackPacket(absl::MakeConstSpan(encoded));
  EXPECT_EQ(expected_bits.to_string(), unpacked.value());
}

TEST_F(PacketTest, UnpackQuantizedBitsAllOnes) {
  std::vector<uint8_t> encoded(kPacketSize, 0b11111111);
  encoded[0] = 0b00000000;  // Zero out header.

  auto packet =
      Packet<kMaxNumPacketBits>::Create(kNumHeaderBits, kNumQuantizedBits);
  ASSERT_NE(packet, nullptr);
  const auto unpacked = packet->UnpackPacket(absl::MakeConstSpan(encoded));
  EXPECT_TRUE(DoesPacketContainQuantized(encoded, unpacked.value(),
                                         kNumHeaderBits, kNumQuantizedBits));
}

TEST_F(PacketTest, UnpackQuantizedBitsAlternatingBytesOfOnesAndZeros) {
  // After the header, the most significant 8 bits have the pattern 00000000,
  // then the next 8 most significant bits have the pattern 11111111. Repeat
  // this pattern for the entire packet.
  std::vector<uint8_t> encoded(kPacketSize, 0b00000000);
  const int num_header_bytes = std::ceil(static_cast<float>(kNumHeaderBits) /
                                         static_cast<float>(CHAR_BIT));
  for (size_t i = num_header_bytes + 1; i < encoded.size(); i += 2) {
    encoded[i] = 0b11111111;
  }

  auto packet =
      Packet<kMaxNumPacketBits>::Create(kNumHeaderBits, kNumQuantizedBits);
  ASSERT_NE(packet, nullptr);
  const auto unpacked = packet->UnpackPacket(absl::MakeConstSpan(encoded));
  EXPECT_TRUE(DoesPacketContainQuantized(encoded, unpacked.value(),
                                         kNumHeaderBits, kNumQuantizedBits));
}

TEST_F(PacketTest, UnpackQuantizedBitsAlternatingOnesAndZeros) {
  std::vector<uint8_t> encoded(kPacketSize, 0b10101010);
  encoded[0] = 0b00000000;  // Zero out header.

  auto packet =
      Packet<kMaxNumPacketBits>::Create(kNumHeaderBits, kNumQuantizedBits);
  ASSERT_NE(packet, nullptr);
  const auto unpacked = packet->UnpackPacket(absl::MakeConstSpan(encoded));
  EXPECT_TRUE(DoesPacketContainQuantized(encoded, unpacked.value(),
                                         kNumHeaderBits, kNumQuantizedBits));
}

TEST_F(PacketTest, InvalidPacketSize) {
  std::vector<uint8_t> invalid_packet(kPacketSize - 1, 0b11111111);
  auto packet =
      Packet<kMaxNumPacketBits>::Create(kNumHeaderBits, kNumQuantizedBits);
  ASSERT_NE(packet, nullptr);
  const auto unpacked =
      packet->UnpackPacket(absl::MakeConstSpan(invalid_packet));
  EXPECT_FALSE(unpacked.has_value());
}

TEST_F(PacketTest, PackVariableHeader) {
  std::bitset<kNumQuantizedBits> quantized(0);
  quantized.flip();
  constexpr int kNumHeaderBitsTest = 10;
  constexpr int kMaxNumPacketBitsTest = kNumHeaderBitsTest + kNumQuantizedBits;

  auto packet = Packet<kMaxNumPacketBitsTest>::Create(kNumHeaderBitsTest,
                                                      kNumQuantizedBits);
  ASSERT_NE(packet, nullptr);
  const std::vector<uint8_t> encoded =
      packet->PackQuantized(quantized.to_string());
  EXPECT_EQ(encoded.size(),
            ExpectedPacketSize(kNumQuantizedBits, kNumHeaderBitsTest));
  EXPECT_TRUE(DoesPacketContainQuantized(
      encoded, quantized.to_string(), kNumHeaderBitsTest, kNumQuantizedBits));
}

TEST_F(PacketTest, PackQuantizedBitsAllOnes) {
  std::bitset<kNumQuantizedBits> quantized(0);
  quantized.flip();

  auto packet =
      Packet<kMaxNumPacketBits>::Create(kNumHeaderBits, kNumQuantizedBits);
  ASSERT_NE(packet, nullptr);
  const std::vector<uint8_t> encoded =
      packet->PackQuantized(quantized.to_string());
  EXPECT_EQ(encoded.size(), kPacketSize);
  EXPECT_TRUE(DoesPacketContainQuantized(encoded, quantized.to_string(),
                                         kNumHeaderBits, kNumQuantizedBits));
}

TEST_F(PacketTest, PackQuantizedBitsAlternatingOnesAndZeros) {
  // The bit pattern has 1 at evenly indexed bits and 0 elsewhere.
  std::bitset<kNumQuantizedBits> quantized(0);
  for (size_t i = 0; i < quantized.size(); i += 2) {
    quantized.flip(i);
  }

  auto packet =
      Packet<kMaxNumPacketBits>::Create(kNumHeaderBits, kNumQuantizedBits);
  ASSERT_NE(packet, nullptr);
  const std::vector<uint8_t> encoded =
      packet->PackQuantized(quantized.to_string());
  EXPECT_EQ(encoded.size(), kPacketSize);
  EXPECT_TRUE(DoesPacketContainQuantized(encoded, quantized.to_string(),
                                         kNumHeaderBits, kNumQuantizedBits));
}

TEST_F(PacketTest, PackQuantizedBitsAlternateBytesOfOnesAndZeros) {
  // The most significant 8 bits have the pattern 00000000, then the next 8 most
  // significant bits have the pattern 11111111. Repeat this pattern for the
  // entire packet.
  std::bitset<kNumQuantizedBits> quantized(0);
  for (size_t bit_offset = CHAR_BIT; bit_offset < quantized.size();
       bit_offset += 2 * CHAR_BIT) {
    for (size_t bit_index = bit_offset; bit_index < bit_offset + CHAR_BIT;
         ++bit_index) {
      quantized.flip(bit_index);
    }
  }

  auto packet =
      Packet<kMaxNumPacketBits>::Create(kNumHeaderBits, kNumQuantizedBits);
  ASSERT_NE(packet, nullptr);
  const std::vector<uint8_t> encoded =
      packet->PackQuantized(quantized.to_string());
  EXPECT_EQ(encoded.size(), kPacketSize);
  EXPECT_TRUE(DoesPacketContainQuantized(encoded, quantized.to_string(),
                                         kNumHeaderBits, kNumQuantizedBits));
}

TEST_F(PacketTest, PackBigHeader) {
  // The most significant 8 bits have the pattern 00000000, then the next 8 most
  // significant bits have the pattern 11111111. Repeat this pattern for the
  // entire packet.
  std::bitset<kNumQuantizedBits> quantized(0);
  for (size_t bit_offset = CHAR_BIT; bit_offset < quantized.size();
       bit_offset += 2 * CHAR_BIT) {
    for (size_t bit_index = bit_offset; bit_index < bit_offset + CHAR_BIT;
         ++bit_index) {
      quantized.flip(bit_index);
    }
  }

  constexpr int kNumHeaderBitsTest = 100;
  constexpr int kMaxNumPacketBitsTest = kNumHeaderBitsTest + kNumQuantizedBits;

  auto packet = Packet<kMaxNumPacketBitsTest>::Create(kNumHeaderBitsTest,
                                                      kNumQuantizedBits);
  ASSERT_NE(packet, nullptr);
  const std::vector<uint8_t> encoded =
      packet->PackQuantized(quantized.to_string());
  EXPECT_EQ(encoded.size(),
            ExpectedPacketSize(kNumQuantizedBits, kNumHeaderBitsTest));
  EXPECT_TRUE(DoesPacketContainQuantized(
      encoded, quantized.to_string(), kNumHeaderBitsTest, kNumQuantizedBits));
}

}  // namespace
}  // namespace codec
}  // namespace chromemedia
