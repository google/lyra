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

#ifndef LYRA_CODEC_PACKET_H_
#define LYRA_CODEC_PACKET_H_

#include <bitset>
#include <climits>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "glog/logging.h"
#include "packet_interface.h"

namespace chromemedia {
namespace codec {

// This class provides a stateful, user-friendly way to construct and interact
// with the packet that will be sent over the wire.
template <int NumQuantizedBits, int NumHeaderBits>
class Packet : public PacketInterface {
 public:
  std::vector<uint8_t> PackQuantized(
      const std::string& quantized_string) override {
    const std::bitset<NumQuantizedBits> quantized_features(quantized_string);
    return Pack(quantized_features);
  }

  absl::optional<std::string> UnpackPacket(
      const absl::Span<const uint8_t> packet) override {
    const int expected_packet_size = static_cast<int>(std::ceil(
        static_cast<float>(NumQuantizedBits + NumHeaderBits) / CHAR_BIT));
    if (packet.length() != expected_packet_size) {
      LOG(ERROR) << "Packet of unexpected length: " << packet.length();
      return absl::nullopt;
    }
    std::bitset<NumQuantizedBits> quantized_features = UnpackFeatures(packet);
    return quantized_features.to_string();
  }

  int PacketSize() const override {
    return static_cast<int>(std::ceil(
        static_cast<float>(NumQuantizedBits + NumHeaderBits) / CHAR_BIT));
  }

 private:
  // Creates a vector of bytes containing a header of variable bits with the
  // quantized data following directly after. For example:
  //  +--------+--------+---------+
  //  |  ||    |        |  ||     |
  //  +--------+--------+---------+
  //   ^           ^           ^
  //   |           |           |
  // Header   Quantized     Extra Space
  static std::vector<uint8_t> Pack(
      const std::bitset<NumQuantizedBits>& quantized_features) {
    const int kTotalNumPacketBits = NumQuantizedBits + NumHeaderBits;
    const std::bitset<kTotalNumPacketBits> kByteMask(0b11111111);
    int right_shift_amount = static_cast<int>(kTotalNumPacketBits) - CHAR_BIT;

    std::bitset<kTotalNumPacketBits> packet_bits =
        GeneratePacketBits(quantized_features);
    std::vector<uint8_t> byte_array(
        static_cast<int>(std::ceil(static_cast<float>(kTotalNumPacketBits) /
                                   static_cast<float>(CHAR_BIT))));

    // Shift the quantized bits into their proper byte in byte_array. The
    // leftmost bits of quantized will occupy byte_array.at(NumHeaderBits) after
    // the header.
    for (auto& packet_byte : byte_array) {
      std::bitset<kTotalNumPacketBits> shifted_bits;
      if (right_shift_amount >= 0) {
        shifted_bits = packet_bits >> right_shift_amount;
      } else {
        // When the quantized bits do not take up the entire last byte, shift
        // the bits so they take up the upper part of the last byte.
        shifted_bits = packet_bits << -right_shift_amount;
      }
      // Mask off bits at index 8 or greater to avoid errors when casting.
      shifted_bits &= kByteMask;
      packet_byte = static_cast<uint8_t>(shifted_bits.to_ulong());
      right_shift_amount -= CHAR_BIT;
    }

    return byte_array;
  }

  // Unpacks the bytes from a packet into a bitset representing the indices of
  // the transformed features.
  static std::bitset<NumQuantizedBits> UnpackFeatures(
      const absl::Span<const uint8_t> encoded) {
    std::bitset<NumQuantizedBits> quantized_features(0);
    int left_shift_amount = static_cast<int>(NumQuantizedBits) +
                            static_cast<int>(NumHeaderBits) - CHAR_BIT;
    // Shift each byte from packet into its proper place in quantized_features
    // where the first bit of quantized data occurs after the header bits.
    for (uint8_t packet_byte : encoded) {
      std::bitset<NumQuantizedBits> shifted_packet_data(packet_byte);
      if (left_shift_amount >= 0) {
        shifted_packet_data <<= left_shift_amount;
      } else {
        // When the quantized bits do not take up the entire last byte, shift
        // the bits so they take up the upper part of the last byte.
        shifted_packet_data >>= -left_shift_amount;
      }
      quantized_features |= shifted_packet_data;
      left_shift_amount -= CHAR_BIT;
    }

    return quantized_features;
  }

  // Combine header and quantized features into a bitset.
  static std::bitset<NumHeaderBits + NumQuantizedBits> GeneratePacketBits(
      const std::bitset<NumQuantizedBits>& quantized_features) {
    std::bitset<NumHeaderBits + NumQuantizedBits> packet_bits = SetHeader();

    std::bitset<NumQuantizedBits + NumHeaderBits> quantized_bitset(
        quantized_features.to_string());
    packet_bits |= quantized_bitset;

    return packet_bits;
  }

  // Includes space for NumQuantBits so can be or'ed at end.
  static std::bitset<NumHeaderBits + NumQuantizedBits> SetHeader() {
    std::bitset<NumHeaderBits + NumQuantizedBits> header;
    // Must update if adding new header sections.
    int unused_bits = NumHeaderBits;
    // For each entry in the header, subtract the number of bits from the
    // unused_bits, get the bits, or it with header and the shift by the number
    // of bits of the next entry.
    header <<= (unused_bits + NumQuantizedBits);
    return header;
  }
};

}  // namespace codec
}  // namespace chromemedia

#endif  // LYRA_CODEC_PACKET_H_
