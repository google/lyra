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

#ifndef LYRA_CODEC_LYRA_DECODER_H_
#define LYRA_CODEC_LYRA_DECODER_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "include/ghc/filesystem.hpp"
#include "generative_model_interface.h"
#include "lyra_decoder_interface.h"
#include "packet_interface.h"
#include "packet_loss_handler_interface.h"
#include "resampler_interface.h"
#include "vector_quantizer_interface.h"

namespace chromemedia {
namespace codec {

/// Lyra decoder class.
///
/// This class unpacks the bit stream into features and uses a generative model
/// to decode them into audio samples.
class LyraDecoder : public LyraDecoderInterface {
 public:
  /// Static method to create a LyraDecoder.
  ///
  /// @param sample_rate_hz Desired sample rate in Hertz. The supported sample
  ///                       rates are 8000, 16000, 32000 and 48000.
  /// @param num_channels Desired number of channels. Currently only 1 is
  ///                     supported.
  /// @param bit_rate Desired bit rate. Currently only 3000 is supported.
  /// @param model_path Path to the model weights. The identifier in the
  ///                   lyra_config.textproto has to coincide with the
  ///                   |kVersionMinor| constant in lyra_config.cc.
  /// @return A unique_ptr to a |LyraDecoder| if all desired params are
  ///         supported. Else it returns a nullptr.
  static std::unique_ptr<LyraDecoder> Create(
      int sample_rate_hz, int num_channels, int bitrate,
      const ghc::filesystem::path& model_path);

  /// Parses a packet and prepares the decoder to decode samples from the
  /// payload.
  ///
  /// If estimated features were added by |DecodePacketLoss| but not fully
  /// decoded overwrites that estimated feature.
  ///
  /// @param encoded Encoded packet as a span of bytes.
  /// @return True if the provided packet is a valid Lyra packet.
  bool SetEncodedPacket(absl::Span<const uint8_t> encoded) override;

  /// Decodes audio from the most recently added packet.
  ///
  /// @param num_samples Number of samples to decode. It has to be less than the
  ///                    remaining available samples to be decoded at the sample
  ///                    rate chosen at Create time, given that each packet
  ///                    contains 40ms of data.
  /// @return Vector of int16-formatted samples as long as there are enough
  ///          remaining samples available. Else it returns nullopt.
  absl::optional<std::vector<int16_t>> DecodeSamples(int num_samples) override;

  /// Decodes audio in packet loss mode.
  ///
  /// Greedily decodes samples remaining from the last provided packet, then
  /// estimates plausible features from previous ones and uses those to decode
  /// additional samples.
  ///
  /// @param num_samples Number of samples to decode. It may be any arbitrarily
  ///                    large value, but decoding is an expensive process, so
  ///                    for real time streaming applications it is recommended
  ///                    to be equal to the size of the output audio buffer,
  ///                    usually 10ms worth of samples.
  /// @return Vector of int16-formatted samples. Returns nullopt on failure.
  absl::optional<std::vector<int16_t>> DecodePacketLoss(
      int num_samples) override;

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

  /// Checks if the decoder is in comfort noise generation mode.
  ///
  /// @return True if the decoder is in comfort noise generation mode.
  bool is_comfort_noise() const override;

 private:
  LyraDecoder() = delete;
  LyraDecoder(std::unique_ptr<GenerativeModelInterface> generative_model,
              std::unique_ptr<GenerativeModelInterface> comfort_noise_generator,
              std::unique_ptr<VectorQuantizerInterface> vector_quantizer,
              std::unique_ptr<PacketInterface> packet,
              std::unique_ptr<PacketLossHandlerInterface> packet_loss_handler,
              std::unique_ptr<ResamplerInterface> resampler, int sample_rate_hz,
              int num_channels, int bitrate, int num_frames_per_packet);

  absl::optional<std::vector<int16_t>> RunGenerativeModelForPacketLoss(
      int num_samples);

  // Runs the Comfort Noise Generator and performs any necessary overlap between
  // models.
  absl::optional<std::vector<int16_t>>
  RunComfortNoiseGeneratorWithNecessaryOverlap(
      int num_samples, bool overlap_required,
      const std::vector<float>& features,
      const std::vector<int16_t>& generative_model_frame =
          std::vector<int16_t>()) const;

  // Overlaps frames using a cos^2 window. |preceding_frame| will die down to
  // zero and |following_frame| will rise up from 0 in the resultant overlapped
  // frame. Returns a nullopt if input frames are not the same size.
  absl::optional<std::vector<int16_t>> OverlapFrames(
      const std::vector<int16_t>& preceding_frame,
      const std::vector<int16_t>& following_frame) const;

  // Used to generate the time domain samples.
  std::unique_ptr<GenerativeModelInterface> generative_model_;
  // Used to generate comfort noise.
  std::unique_ptr<GenerativeModelInterface> comfort_noise_generator_;
  // Used to get the conditioning features from the bit-stream.
  std::unique_ptr<VectorQuantizerInterface> vector_quantizer_;
  // Used to get the unpack a packet into quantized bits.
  std::unique_ptr<PacketInterface> packet_;
  // Used to fill in the blanks when a packet is lost.
  std::unique_ptr<PacketLossHandlerInterface> packet_loss_handler_;
  // Used to go from the generative model sample rate to the one expected at the
  // output.
  std::unique_ptr<ResamplerInterface> resampler_;

  const int sample_rate_hz_;
  const int num_channels_;
  const int bitrate_;
  const int num_frames_per_packet_;

  // The number of remaining samples to decode per packet expressed at the
  // frequency of |kInternalSampleRateHz|.
  int internal_num_samples_available_;
  // Prevent users from calling |DecodeSamples| without having added a real
  // encoded packet.
  bool encoded_packet_set_;
  // Used to trigger overlap when switching to or from comfort noise.
  bool prev_frame_was_comfort_noise_;
  friend class LyraDecoderPeer;
};

}  // namespace codec
}  // namespace chromemedia

#endif  // LYRA_CODEC_LYRA_DECODER_H_
