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

#ifndef LYRA_LYRA_CONFIG_H_
#define LYRA_LYRA_CONFIG_H_

#include <algorithm>
#include <climits>
#include <cmath>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "glog/logging.h"  // IWYU pragma: keep
#include "include/ghc/filesystem.hpp"
#include "lyra/lyra_config.pb.h"

namespace chromemedia {
namespace codec {

// This file is reserved for non-configurable values needed by both the decoder
// and encoder. What those non-configurable values are depends on which project
// is chosen to be compiled.  As a result, a struct holding the configuration
// data is defined to ensure each new target added and each new configuration
// element is explicitly defined.

ABSL_CONST_INIT extern const int kVersionMajor;
ABSL_CONST_INIT extern const int kVersionMinor;
ABSL_CONST_INIT extern const int kVersionMicro;
ABSL_CONST_INIT extern const int kNumFeatures;
ABSL_CONST_INIT extern const int kNumMelBins;
ABSL_CONST_INIT extern const int kNumChannels;
ABSL_CONST_INIT extern const int kFrameRate;  // Frames/packets sent per second.
ABSL_CONST_INIT extern const int kOverlapFactor;
ABSL_CONST_INIT extern const int kNumHeaderBits;

inline constexpr int kSupportedSampleRates[] = {8000, 16000, 32000, 48000};
inline constexpr int kInternalSampleRateHz = 16000;

const std::vector<int>& GetSupportedQuantizedBits();

// Returns a string of form "|kVersionMajor|.|kVersionMinor|.|kVersionMicro|".
inline const std::string& GetVersionString() {
  static const std::string kVersionString = [] {
    return absl::StrCat(kVersionMajor, ".", kVersionMinor, ".", kVersionMicro);
  }();
  return kVersionString;
}

// Functions to get values depending on sample rate.
inline int GetNumSamplesPerHop(int sample_rate_hz) {
  CHECK_EQ(sample_rate_hz % kFrameRate, 0);
  return sample_rate_hz / kFrameRate;
}

inline int GetNumSamplesPerWindow(int sample_rate_hz) {
  return kOverlapFactor * GetNumSamplesPerHop(sample_rate_hz);
}

inline int GetPacketSize(int num_quantized_bits) {
  return static_cast<int>(std::ceil(
      static_cast<float>(num_quantized_bits + kNumHeaderBits) / CHAR_BIT));
}

inline int BitrateToPacketSize(int bitrate) {
  return static_cast<int>(
      std::ceil(static_cast<float>(bitrate) / (kFrameRate * CHAR_BIT)));
}

inline int GetBitrate(int num_quantized_bits) {
  return GetPacketSize(num_quantized_bits) * CHAR_BIT * kFrameRate;
}

inline bool IsSampleRateSupported(int sample_rate_hz) {
  return std::find(std::begin(kSupportedSampleRates),
                   std::end(kSupportedSampleRates),
                   sample_rate_hz) != std::end(kSupportedSampleRates);
}

inline int PacketSizeToNumQuantizedBits(int packet_size) {
  for (int num_quantized_bits : GetSupportedQuantizedBits()) {
    if (packet_size == GetPacketSize(num_quantized_bits)) {
      return num_quantized_bits;
    }
  }
  return -1;
}

inline int BitrateToNumQuantizedBits(int bitrate) {
  for (int num_quantized_bits : GetSupportedQuantizedBits()) {
    if (bitrate == GetBitrate(num_quantized_bits)) {
      return num_quantized_bits;
    }
  }
  return -1;
}

std::vector<absl::string_view> GetAssets();

inline absl::Status AreParamsSupported(
    int sample_rate_hz, int num_channels,
    const ghc::filesystem::path& model_path) {
  if (!IsSampleRateSupported(sample_rate_hz)) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Sample rate %d Hz is not supported by codec.", sample_rate_hz));
  }
  if (num_channels != kNumChannels) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Number of channels %d is not supported by codec. It needs to be %d.",
        num_channels, kNumChannels));
  }
  for (auto asset : GetAssets()) {
    std::error_code error;
    const bool exists =
        ghc::filesystem::exists(model_path / std::string(asset), error);
    if (error) {
      return absl::UnknownError(
          absl::StrFormat("Error when probing for asset %s in %s: %s", asset,
                          model_path, error.message()));
    }
    if (!exists) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Asset %s does not exist in %s.", asset, model_path));
    }
  }
  const ghc::filesystem::path lyra_config_proto_path =
      model_path / "lyra_config.binarypb";
  std::error_code error;
  const bool exists = ghc::filesystem::exists(lyra_config_proto_path, error);
  if (error) {
    return absl::UnknownError(
        absl::StrFormat("Error when probing for asset %s: %s",
                        lyra_config_proto_path.string(), error.message()));
  }
  third_party::lyra_codec::lyra::LyraConfig lyra_config;
  if (exists) {
    std::ifstream lyra_config_stream(lyra_config_proto_path.string());
    if (!lyra_config.ParseFromIstream(&lyra_config_stream)) {
      return absl::UnknownError(absl::StrFormat(
          "Error when parsing %s", lyra_config_proto_path.string()));
    }
  }
  if (lyra_config.identifier() != kVersionMinor) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Weights identifier (%d) is not compatible with code identifier (%d).",
        lyra_config.identifier(), kVersionMinor));
  }
  return absl::OkStatus();
}

}  // namespace codec
}  // namespace chromemedia

#endif  // LYRA_LYRA_CONFIG_H_
