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

#ifndef LYRA_CODEC_LYRA_ENCODER_H_
#define LYRA_CODEC_LYRA_ENCODER_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "audio/linear_filters/biquad_filter.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "include/ghc/filesystem.hpp"
#include "denoiser_interface.h"
#include "feature_extractor_interface.h"
#include "lyra_encoder_interface.h"
#include "noise_estimator_interface.h"
#include "packet_interface.h"
#include "resampler_interface.h"
#include "vector_quantizer_interface.h"

namespace chromemedia {
namespace codec {

/// @example lyra_integration_test.cc
/// An example of how to encode and decode an audio stream using LyraEncoder and
/// LyraDecoder can be found here.

/// Lyra encoder class.
///
/// This class encodes audio by extracting features and performing vector
/// quantization.
class LyraEncoder : public LyraEncoderInterface {
 public:
  /// Static method to create a LyraEncoder.
  ///
  /// @param sample_rate_hz Desired sample rate in Hertz. The supported sample
  ///                       rates are 8000, 16000, 32000 and 48000.
  /// @param num_channels Desired number of channels. Currently only 1 is
  ///                     supported.
  /// @param bit_rate Desired bit rate. Currently only 3000 is supported.
  /// @param enable_dtx Set to true if discontinuous transmission should be
  ///                   enabled.
  /// @param model_path Path to the model weights. The identifier in the
  ///                   lyra_config.textproto has to coincide with the
  ///                   kVersionMinor constant in lyra_config.cc.
  /// @return A unique_ptr to a LyraEncoder if all desired params are supported.
  ///         Else it returns a nullptr.
  static std::unique_ptr<LyraEncoder> Create(
      int sample_rate_hz, int num_channels, int bitrate, bool enable_dtx,
      const ghc::filesystem::path& model_path);

  /// Encodes the audio samples into a vector wrapped byte array.
  ///
  /// @param audio Span of int16-formatted samples. It is assumed to contain
  ///              40ms of data at the sample rate chosen at Create time.
  /// @return Encoded packet as a vector of bytes as long as the right amount of
  ///         data is provided. Else it returns nullopt. It also returns nullopt
  ///         if DTX is enabled and the packet is deemed to contain silence.
  absl::optional<std::vector<uint8_t>> Encode(
      const absl::Span<const int16_t> audio) override;

  /// Getter for the sample rate in Hertz.
  ///
  /// @return Sample rate in Hertz.
  int sample_rate_hz() const override;

  /// Getter for the number of channels.
  ///
  /// @return Number of channels.
  int num_channels() const override;

  /// Getter for the bitrate.
  ///
  /// @return Bitrate.
  int bitrate() const override;

  /// Getter for the frame rate.
  ///
  /// @return Frame rate.
  int frame_rate() const override;

 private:
  LyraEncoder() = delete;
  LyraEncoder(std::unique_ptr<ResamplerInterface> resampler,
              std::unique_ptr<FeatureExtractorInterface> feature_extractor,
              std::unique_ptr<NoiseEstimatorInterface> noise_estimator,
              std::unique_ptr<VectorQuantizerInterface> vector_quantizer,
              std::unique_ptr<DenoiserInterface> denoiser,
              std::unique_ptr<PacketInterface> packet, int sample_rate_hz,
              int num_channels, int bitrate, int num_frames_per_packet,
              bool enable_dtx);

  absl::optional<std::vector<uint8_t>> EncodeInternal(
      const absl::Span<const int16_t> audio, bool filter_audio);

  const std::unique_ptr<ResamplerInterface> resampler_;
  const std::unique_ptr<FeatureExtractorInterface> feature_extractor_;
  const std::unique_ptr<NoiseEstimatorInterface> noise_estimator_;
  const std::unique_ptr<VectorQuantizerInterface> vector_quantizer_;
  const std::unique_ptr<DenoiserInterface> denoiser_;
  std::unique_ptr<PacketInterface> packet_;
  const int sample_rate_hz_;
  const int num_channels_;
  const int bitrate_;
  const int num_frames_per_packet_;
  const bool enable_dtx_;
  linear_filters::BiquadFilterCascade<float> second_order_sections_filter_;
  friend class LyraEncoderPeer;
};

}  // namespace codec
}  // namespace chromemedia

#endif  // LYRA_CODEC_LYRA_ENCODER_H_
