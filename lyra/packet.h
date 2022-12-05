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

#ifndef LYRA_PACKET_H_
#define LYRA_PACKET_H_

#include <bitset>
#include <climits>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/types/span.h"
#include "glog/logging.h"  // IWYU pragma: keep
#include "lyra/packet_interface.h"

namespace chromemedia {
namespace codec {

// This class provides a stateful, user-friendly way to construct and interact
// with the packet that will be sent over the wire.
template <int MaxNumPacketBits>
class Packet : public PacketInterface {
 public:
  static std::unique_ptr<Packet<MaxNumPacketBits>> Create(
      int num_header_bits, int num_quantized_bits) {
    if (num_header_bits + num_quantized_bits > MaxNumPacketBits) {
      LOG(ERROR) << "The sum of header bits (" << num_header_bits
                 << ") and quantized bits (" << num_quantized_bits
                 << ") has to be lower than the maximum packet bits ("
                 << MaxNumPacketBits << ").";
      return nullptr;
    }

    return absl::WrapUnique(
        new Packet<MaxNumPacketBits>(num_header_bits, num_quantized_bits));
  }

  std::vector<uint8_t> PackQuantized(
      const std::string& quantized_string) override {
    const std::bitset<MaxNumPacketBits> quantized_features(quantized_string);
    return Pack(quantized_features);
  }

  std::optional<std::string> UnpackPacket(
      const absl::Span<const uint8_t> packet) override {
    if (packet.length() != PacketSize()) {
      LOG(ERROR) << "Packet of unexpected length: " << packet.length();
      return std::nullopt;
    }
    std::bitset<MaxNumPacketBits> quantized_features = UnpackFeatures(packet);
    return quantized_features.to_string().substr(MaxNumPacketBits -
                                                 num_quantized_bits_);
  }

  int PacketSize() const override {
    return static_cast<int>(std::ceil(
        static_cast<float>(num_quantized_bits_ + num_header_bits_) / CHAR_BIT));
  }

 private:
  Packet(int num_header_bits, int num_quantized_bits)
      : num_header_bits_(num_header_bits),
        num_quantized_bits_(num_quantized_bits) {}

  // Creates a vector of bytes containing a header of variable bits with the
  // quantized data following directly after. For example:
  //  +--------+--------+---------+
  //  |  ||    |        |  ||     |
  //  +--------+--------+---------+
  //   ^           ^           ^
  //   |           |           |
  // Header   Quantized     Extra Space
  std::vector<uint8_t> Pack(
      const std::bitset<MaxNumPacketBits>& quantized_features) {
    const int total_num_packet_bits = num_header_bits_ + num_quantized_bits_;
    const std::bitset<MaxNumPacketBits> kByteMask(0b11111111);
    int right_shift_amount = total_num_packet_bits - CHAR_BIT;

    std::bitset<MaxNumPacketBits> packet_bits =
        GeneratePacketBits(quantized_features);
    std::vector<uint8_t> byte_array(
        static_cast<int>(std::ceil(static_cast<float>(total_num_packet_bits) /
                                   static_cast<float>(CHAR_BIT))));

    // Shift the quantized bits into their proper byte in byte_array. The
    // leftmost bits of quantized will occupy byte_array.at(num_header_bits_)
    // after the header.
    for (auto& packet_byte : byte_array) {
      std::bitset<MaxNumPacketBits> shifted_bits;
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
  std::bitset<MaxNumPacketBits> UnpackFeatures(
      const absl::Span<const uint8_t> encoded) {
    std::bitset<MaxNumPacketBits> quantized_features(0);
    int left_shift_amount = num_quantized_bits_ + num_header_bits_ - CHAR_BIT;
    // Shift each byte from packet into its proper place in quantized_features
    // where the first bit of quantized data occurs after the header bits.
    for (uint8_t packet_byte : encoded) {
      std::bitset<MaxNumPacketBits> shifted_packet_data(packet_byte);
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
  std::bitset<MaxNumPacketBits> GeneratePacketBits(
      const std::bitset<MaxNumPacketBits>& quantized_features) {
    std::bitset<MaxNumPacketBits> packet_bits = SetHeader();

    packet_bits |= quantized_features;

    return packet_bits;
  }

  // Includes space for MaxNumPacketBits so can be or'ed at end.
  std::bitset<MaxNumPacketBits> SetHeader() {
    std::bitset<MaxNumPacketBits> header;
    // Must update if adding new header sections.
    int unused_bits = num_header_bits_;
    // For each entry in the header, subtract the number of bits from the
    // unused_bits, get the bits, or it with header and the shift by the number
    // of bits of the next entry.
    header <<= (unused_bits + num_quantized_bits_);
    return header;
  }

  const int num_header_bits_;
  const int num_quantized_bits_;
};

}  // namespace codec
}  // namespace chromemedia

#endif  // LYRA_PACKET_H_
