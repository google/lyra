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

#ifndef LYRA_CODEC_TESTING_MOCK_LYRA_DECODER_H_
#define LYRA_CODEC_TESTING_MOCK_LYRA_DECODER_H_

#include <cstdint>
#include <optional>
#include <vector>

#include "absl/types/span.h"
#include "gmock/gmock.h"
#include "lyra_decoder_interface.h"

namespace chromemedia {
namespace codec {

class MockLyraDecoder : public LyraDecoderInterface {
 public:
  ~MockLyraDecoder() override {}

  MOCK_METHOD(bool, SetEncodedPacket, (absl::Span<const uint8_t>), (override));

  MOCK_METHOD(absl::optional<std::vector<int16_t>>, DecodeSamples, (int),
              (override));

  MOCK_METHOD(absl::optional<std::vector<int16_t>>, DecodePacketLoss, ());

  MOCK_METHOD(absl::optional<std::vector<int16_t>>, DecodePacketLoss, (int),
              (override));

  MOCK_METHOD(int, sample_rate_hz, (), (const, override));

  MOCK_METHOD(int, num_channels, (), (const, override));

  MOCK_METHOD(int, bitrate, (), (const, override));

  MOCK_METHOD(int, frame_rate, (), (const, override));

  MOCK_METHOD(bool, is_comfort_noise, (), (const, override));
};

}  // namespace codec
}  // namespace chromemedia

#endif  // LYRA_CODEC_TESTING_MOCK_LYRA_DECODER_H_
