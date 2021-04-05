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

#ifndef LYRA_CODEC_LYRA_CONFIG_H_
#define LYRA_CODEC_LYRA_CONFIG_H_

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iterator>

#include "glog/logging.h"
#include "google/protobuf/text_format.h"
#include "absl/base/attributes.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "include/ghc/filesystem.hpp"
#include "lyra_config.pb.h"

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
ABSL_CONST_INIT extern const int kNumExpectedOutputFeatures;
ABSL_CONST_INIT extern const int kNumChannels;
ABSL_CONST_INIT extern const int kFrameRate;  // Frames sent per second.
ABSL_CONST_INIT extern const int kFrameOverlapFactor;
ABSL_CONST_INIT extern const int kNumFramesPerPacket;
ABSL_CONST_INIT extern const int kPacketSize;
ABSL_CONST_INIT extern const int kBitrate;

inline constexpr int kSupportedSampleRates[] = {8000, 16000, 32000, 48000};
inline constexpr int kInternalSampleRateHz = 16000;
inline constexpr int kNumQuantizationBits = 120;

inline constexpr absl::string_view kAssets[] = {
    "lyra_16khz_ar_to_gates_bias.raw.gz",
    "lyra_16khz_ar_to_gates_mask.raw.gz",
    "lyra_16khz_ar_to_gates_weights.raw.gz",
    "lyra_16khz_conditioning_stack_0_bias.raw.gz",
    "lyra_16khz_conditioning_stack_0_mask.raw.gz",
    "lyra_16khz_conditioning_stack_0_weights.raw.gz",
    "lyra_16khz_conditioning_stack_1_bias.raw.gz",
    "lyra_16khz_conditioning_stack_1_mask.raw.gz",
    "lyra_16khz_conditioning_stack_1_weights.raw.gz",
    "lyra_16khz_conditioning_stack_2_bias.raw.gz",
    "lyra_16khz_conditioning_stack_2_mask.raw.gz",
    "lyra_16khz_conditioning_stack_2_weights.raw.gz",
    "lyra_16khz_conv1d_bias.raw.gz",
    "lyra_16khz_conv1d_mask.raw.gz",
    "lyra_16khz_conv1d_weights.raw.gz",
    "lyra_16khz_conv_cond_bias.raw.gz",
    "lyra_16khz_conv_cond_mask.raw.gz",
    "lyra_16khz_conv_cond_weights.raw.gz",
    "lyra_16khz_conv_to_gates_bias.raw.gz",
    "lyra_16khz_conv_to_gates_mask.raw.gz",
    "lyra_16khz_conv_to_gates_weights.raw.gz",
    "lyra_16khz_gru_layer_bias.raw.gz",
    "lyra_16khz_gru_layer_mask.raw.gz",
    "lyra_16khz_gru_layer_weights.raw.gz",
    "lyra_16khz_means_bias.raw.gz",
    "lyra_16khz_means_mask.raw.gz",
    "lyra_16khz_means_weights.raw.gz",
    "lyra_16khz_mix_bias.raw.gz",
    "lyra_16khz_mix_mask.raw.gz",
    "lyra_16khz_mix_weights.raw.gz",
    "lyra_16khz_proj_bias.raw.gz",
    "lyra_16khz_proj_mask.raw.gz",
    "lyra_16khz_proj_weights.raw.gz",
    "lyra_16khz_quant_codebook_dimensions.gz",
    "lyra_16khz_quant_code_vectors.gz",
    "lyra_16khz_quant_mean_vectors.gz",
    "lyra_16khz_quant_transmat.gz",
    "lyra_16khz_scales_bias.raw.gz",
    "lyra_16khz_scales_mask.raw.gz",
    "lyra_16khz_scales_weights.raw.gz",
    "lyra_16khz_transpose_0_bias.raw.gz",
    "lyra_16khz_transpose_0_mask.raw.gz",
    "lyra_16khz_transpose_0_weights.raw.gz",
    "lyra_16khz_transpose_1_bias.raw.gz",
    "lyra_16khz_transpose_1_mask.raw.gz",
    "lyra_16khz_transpose_1_weights.raw.gz",
    "lyra_16khz_transpose_2_bias.raw.gz",
    "lyra_16khz_transpose_2_mask.raw.gz",
    "lyra_16khz_transpose_2_weights.raw.gz"};
inline constexpr absl::string_view kLyraConfigProto = "lyra_config.textproto";

inline bool IsSampleRateSupported(int sample_rate_hz) {
  return std::find(std::begin(kSupportedSampleRates),
                   std::end(kSupportedSampleRates),
                   sample_rate_hz) != std::end(kSupportedSampleRates);
}

inline absl::Status AreParamsSupported(
    int sample_rate_hz, int num_channels, int bitrate,
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
  if (bitrate != kBitrate) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Bitrate %d bps is not supported by codec. It needs to be %d bps.",
        bitrate, kBitrate));
  }
  for (auto asset : kAssets) {
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
  const ghc::filesystem::path lyra_config_proto =
      model_path / std::string(kLyraConfigProto);
  std::error_code error;
  const bool exists = ghc::filesystem::exists(lyra_config_proto, error);
  if (error) {
    return absl::UnknownError(
        absl::StrFormat("Error when probing for asset %s: %s",
                        lyra_config_proto, error.message()));
  }
  third_party::lyra_codec::LyraConfig lyra_config;
  if (exists) {
    std::ifstream lyra_config_stream(lyra_config_proto.string());
    const std::string lyra_config_string{
        std::istreambuf_iterator<char>(lyra_config_stream),
        std::istreambuf_iterator<char>()};
    // Even though LyraConfig is a subclass of Message, the reinterpreting is
    // necessary for the mobile proto library.
    if (!google::protobuf::TextFormat::ParseFromString(
            lyra_config_string,
            reinterpret_cast<google::protobuf::Message*>(&lyra_config))) {
      return absl::UnknownError(absl::StrFormat(
          "Error when parsing %s: %s", lyra_config_proto, error.message()));
    }
  }
  if (lyra_config.identifier() != kVersionMinor) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Weights identifier (%d) is not compatible with code identifier (%d).",
        lyra_config.identifier(), kVersionMinor));
  }
  return absl::OkStatus();
}

// Returns a string of form "|kVersionMajor|.|kVersionMinor|.|kVersionMicro|".
const std::string& GetVersionString();

// Functions to get values depending on sample rate.
inline int GetNumSamplesPerHop(int sample_rate_hz) {
  CHECK_EQ(sample_rate_hz % kFrameRate, 0);
  return sample_rate_hz / kFrameRate;
}

inline int GetNumSamplesPerFrame(int sample_rate_hz) {
  CHECK_EQ(sample_rate_hz % kFrameRate, 0);
  return kFrameOverlapFactor * (sample_rate_hz / kFrameRate);
}

inline int GetInternalSampleRate(int external_sample_rate_hz) {
  return kInternalSampleRateHz;
}

inline int ConvertNumSamplesBetweenSampleRate(int source_num_samples,
                                              int source_sample_rate,
                                              int target_sample_rate) {
  return static_cast<int>(std::ceil(static_cast<float>(source_num_samples) *
                                    static_cast<float>(target_sample_rate) /
                                    static_cast<float>(source_sample_rate)));
}

}  // namespace codec
}  // namespace chromemedia

#endif  // LYRA_CODEC_LYRA_CONFIG_H_
