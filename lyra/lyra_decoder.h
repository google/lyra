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

#ifndef LYRA_LYRA_DECODER_H_
#define LYRA_LYRA_DECODER_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "absl/types/span.h"
#include "include/ghc/filesystem.hpp"
#include "lyra/buffered_filter_interface.h"
#include "lyra/feature_estimator_interface.h"
#include "lyra/generative_model_interface.h"
#include "lyra/lyra_decoder_interface.h"
#include "lyra/noise_estimator_interface.h"
#include "lyra/vector_quantizer_interface.h"

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
  /// @param model_path Path to the model weights. The identifier in the
  ///                   lyra_config.binarypb has to coincide with the
  ///                   |kVersionMinor| constant in lyra_config.cc.
  /// @return A unique_ptr to a |LyraDecoder| if all desired params are
  ///         supported. Else it returns a nullptr.
  static std::unique_ptr<LyraDecoder> Create(
      int sample_rate_hz, int num_channels,
      const ghc::filesystem::path& model_path);

  /// Parses a packet and prepares to decode samples from the payload.
  ///
  /// @param encoded Encoded packet as a span of bytes.
  /// @return True if the provided packet is a valid Lyra packet.
  bool SetEncodedPacket(absl::Span<const uint8_t> encoded) override;

  /// Decodes samples.
  ///
  /// If more samples are requested for decoding than are available from the
  /// payloads set by |SetEncodedPacket|, this will generate samples in packet
  /// loss mode. Packet loss mode will first attempt to conceal the lost packets
  /// and then transition to comfort noise.
  ///
  /// @param num_samples Number of samples to decode.
  ///
  /// @return Vector of int16-formatted samples, or nullopt on failure.
  std::optional<std::vector<int16_t>> DecodeSamples(int num_samples) override;

  /// Getter for the sample rate in Hertz.
  ///
  /// @return Sample rate in Hertz.
  int sample_rate_hz() const override;

  /// Getter for the number of channels.
  ///
  /// @return Number of channels.
  int num_channels() const override;

  /// Getter for the frame rate.
  ///
  /// @return Frame rate.
  int frame_rate() const override;

  /// Checks if the decoder is in comfort noise generation mode.
  ///
  /// @return True if the decoder is in comfort noise generation mode.
  bool is_comfort_noise() const override;

 private:
  // Tracks the direction we are moving along |fade_progress_|.
  enum FadeDirection {
    kFadeToCNG = 1,
    kFadeFromCNG = -1,
  };

  LyraDecoder() = delete;
  LyraDecoder(std::unique_ptr<GenerativeModelInterface> generative_model,
              std::unique_ptr<GenerativeModelInterface> comfort_noise_generator,
              std::unique_ptr<VectorQuantizerInterface> vector_quantizer,
              std::unique_ptr<NoiseEstimatorInterface> noise_estimator,
              std::unique_ptr<FeatureEstimatorInterface> feature_estimator,
              std::unique_ptr<BufferedFilterInterface> resampler,
              int external_sample_rate_hz, int num_channels);

  // Runs the while loop for generating samples at the internal sample rate.
  std::optional<std::vector<int16_t>> DecodeSamplesInternal(
      int internal_num_samples_to_generate);

  // Overlaps hops using a cos^2 window.
  // Returns true on success, false on failure.
  bool MaybeOverlapAndInsert(FadeDirection fade_direction, int fade_progress,
                             const std::vector<int16_t>& generative_model_hop,
                             const std::vector<int16_t>& comfort_noise_hop,
                             std::vector<int16_t>& result);

  // Runs the generative model and adds estimated features if needed.
  std::optional<std::vector<int16_t>> RunGenerativeModel(int num_samples);

  // Runs the comfort noise generator and adds estimated features if needed.
  std::optional<std::vector<int16_t>> RunComfortNoiseGenerator(int num_samples);

  // Generates time domain samples from conditioning features.
  std::unique_ptr<GenerativeModelInterface> generative_model_;
  // Generates comfort noise from background noise estimates.
  std::unique_ptr<GenerativeModelInterface> comfort_noise_generator_;
  // Used to get the conditioning features from the bit-stream.
  std::unique_ptr<VectorQuantizerInterface> vector_quantizer_;
  // Estimates background noise for conditioning to the
  // |comfort_noise_generator_|.
  std::unique_ptr<NoiseEstimatorInterface> noise_estimator_;
  // Used as conditioning for the |generative_model_| when more samples were
  // requested than contained in received packets.
  std::unique_ptr<FeatureEstimatorInterface> feature_estimator_;
  // Resamples from the generative model sample rate to the external sampling
  // rate.
  std::unique_ptr<BufferedFilterInterface> resampler_;

  // The packet loss state is described by the following three variables:

  // Ranges from [-num_samples_per_packet + 1, concealment_length].
  // If less than zero, abs(|concealment_progress_|) indicates how many
  // concealment samples until we will begin playing out a received packet.
  // Otherwise tracks samples since we last played out a received packet.
  int concealment_progress_;
  // Ranges from [0, fade_duration_samples_]. 0 indicates we are only generating
  // model output. |fade_duration_samples| indicates we are only generating
  // comfort noise. Values in between indicate a fade is in progress.
  int fade_progress_;
  // Indicates if we are incrementing or decrementing |fade_progress|.
  FadeDirection fade_direction_;

  const int external_sample_rate_hz_;
  const int num_channels_;

  friend class LyraDecoderPeer;
};

}  // namespace codec
}  // namespace chromemedia

#endif  // LYRA_LYRA_DECODER_H_
