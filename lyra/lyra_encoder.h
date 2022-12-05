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

#ifndef LYRA_LYRA_ENCODER_H_
#define LYRA_LYRA_ENCODER_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "absl/types/span.h"
#include "include/ghc/filesystem.hpp"
#include "lyra/feature_extractor_interface.h"
#include "lyra/lyra_encoder_interface.h"
#include "lyra/noise_estimator_interface.h"
#include "lyra/resampler_interface.h"
#include "lyra/vector_quantizer_interface.h"

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
  /// @param bit_rate Desired bit rate. The supported bit rates are 3200, 6000
  ///                 and 9200.
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
  ///              20ms of data at the sample rate chosen at Create time.
  /// @return Encoded packet as a vector of bytes if the correct number of
  ///              of samples are provided, otherwise it returns nullopt.
  ///              The return vector will be of length zero if discontinuous
  ///              transmission mode is enabled and the frame contains
  ///              background noise.
  std::optional<std::vector<uint8_t>> Encode(
      const absl::Span<const int16_t> audio) override;

  /// Setter for the bitrate.
  ///
  /// @param bitrate Desired bitrate in bps.
  /// @return True if the bitrate is supported and set correctly.
  bool set_bitrate(int bitrate) override;

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
              int sample_rate_hz, int num_channels, int num_quantized_bits,
              bool enable_dtx);

  const std::unique_ptr<ResamplerInterface> resampler_;
  const std::unique_ptr<FeatureExtractorInterface> feature_extractor_;
  const std::unique_ptr<NoiseEstimatorInterface> noise_estimator_;
  const std::unique_ptr<VectorQuantizerInterface> vector_quantizer_;

  const int sample_rate_hz_;
  const int num_channels_;
  int num_quantized_bits_;
  const bool enable_dtx_;
  friend class LyraEncoderPeer;
};

}  // namespace codec
}  // namespace chromemedia

#endif  // LYRA_LYRA_ENCODER_H_
