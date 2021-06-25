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

#include "lyra_config.h"

#include <climits>

#include "absl/strings/str_cat.h"

namespace chromemedia {
namespace codec {

// The Lyra version is |kVersionMajor|.|kVersionMinor|.|kVersionMicro|
// The version is not used internally, but clients may use it to configure
// behavior, such as checking for version bumps that break the bitstream.
// The major version should be bumped whenever the bitstream breaks.
const int kVersionMajor = 0;
// |kVersionMinor| needs to be increased every time a new version requires a
// simultaneous change in code and weights or if the bit stream is modified. The
// |identifier| field needs to be set in lyra_config.textproto to match this.
const int kVersionMinor = 0;
// The micro version is for other things like a release of bugfixes.
const int kVersionMicro = 2;

const int kNumFeatures = 160;
const int kNumExpectedOutputFeatures = 160;
const int kNumChannels = 1;
const int kFrameRate = 25;
const int kFrameOverlapFactor = 2;
const int kNumFramesPerPacket = 1;

// TODO(b/133794927): Calculation of kPacketSize will be determined by future
// considerations.
// LINT.IfChange
const int kPacketSize = 15;
// LINT.ThenChange(
// lyra_components.cc,
// )

const int kBitrate = kPacketSize * CHAR_BIT * kFrameRate * kNumChannels;

absl::Status AreParamsSupported(int sample_rate_hz, int num_channels,
                                int bitrate,
                                const ghc::filesystem::path& model_path) {
  constexpr absl::string_view kAssets[] = {
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
      model_path / "lyra_config.textproto";
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

const std::string& GetVersionString() {
  static const std::string kVersionString = [] {
    return absl::StrCat(kVersionMajor, ".", kVersionMinor, ".", kVersionMicro);
  }();
  return kVersionString;
}

}  // namespace codec
}  // namespace chromemedia
