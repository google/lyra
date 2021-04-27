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

#include "lyra_decoder.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "comfort_noise_generator.h"
#include "generative_model_interface.h"
#include "glog/logging.h"
#include "include/ghc/filesystem.hpp"
#include "lyra_components.h"
#include "lyra_config.h"
#include "packet_interface.h"
#include "packet_loss_handler.h"
#include "packet_loss_handler_interface.h"
#include "resampler.h"
#include "resampler_interface.h"
#include "vector_quantizer_interface.h"

namespace chromemedia {
namespace codec {

std::unique_ptr<LyraDecoder> LyraDecoder::Create(
    int sample_rate_hz, int num_channels, int bitrate,
    const ghc::filesystem::path& model_path) {
  absl::Status are_params_supported =
      AreParamsSupported(sample_rate_hz, num_channels, bitrate, model_path);
  if (!are_params_supported.ok()) {
    LOG(ERROR) << are_params_supported;
    return nullptr;
  }

  // The model is always set up for |kInternalSampleRateHz|.
  auto model = CreateGenerativeModel(GetNumSamplesPerHop(kInternalSampleRateHz),
                                     kNumExpectedOutputFeatures,
                                     kNumFramesPerPacket, model_path);
  if (model == nullptr) {
    LOG(ERROR) << "New model could not be instantiated.";
    return nullptr;
  }

  // The comfort noise generator is always set up for |kInternalSampleRateHz|.
  auto comfort_noise_generator = ComfortNoiseGenerator::Create(
      kInternalSampleRateHz, kNumExpectedOutputFeatures,
      GetNumSamplesPerFrame(kInternalSampleRateHz),
      GetNumSamplesPerHop(kInternalSampleRateHz));
  if (comfort_noise_generator == nullptr) {
    LOG(ERROR) << "Could not create Comfort Noise Generator.";
    return nullptr;
  }

  // Vector Quantizer is always set up for |kInternalSampleRateHz|.
  auto vector_quantizer =
      CreateQuantizer(kNumFramesPerPacket * kNumExpectedOutputFeatures,
                      kNumQuantizationBits, model_path);
  if (vector_quantizer == nullptr) {
    LOG(ERROR) << "Could not create Vector Quantizer.";
    return nullptr;
  }

  auto packet = CreatePacket();

  // The packet loss handler is always set up for |kInternalSampleRateHz|.
  auto packet_loss_handler = PacketLossHandler::Create(
      kInternalSampleRateHz, kNumExpectedOutputFeatures,
      static_cast<float>(GetNumSamplesPerHop(kInternalSampleRateHz)) /
          kInternalSampleRateHz);
  if (packet_loss_handler == nullptr) {
    LOG(ERROR) << "Could not create Packet Loss Handler.";
    return nullptr;
  }

  // The resampler always resamples from |kInternalSampleRateHz| to the
  // requested |sample_rate_hz|.
  auto resampler = Resampler::Create(kInternalSampleRateHz, sample_rate_hz);
  if (resampler == nullptr) {
    LOG(ERROR) << "Could not create Resampler.";
    return nullptr;
  }

  // WrapUnique is used because of private c'tor.
  return absl::WrapUnique(new LyraDecoder(
      std::move(model), std::move(comfort_noise_generator),
      std::move(vector_quantizer), std::move(packet),
      std::move(packet_loss_handler), std::move(resampler), sample_rate_hz,
      num_channels, bitrate, kNumFramesPerPacket));
}

LyraDecoder::LyraDecoder(
    std::unique_ptr<GenerativeModelInterface> generative_model,
    std::unique_ptr<GenerativeModelInterface> comfort_noise_generator,
    std::unique_ptr<VectorQuantizerInterface> vector_quantizer,
    std::unique_ptr<PacketInterface> packet,
    std::unique_ptr<PacketLossHandlerInterface> packet_loss_handler,
    std::unique_ptr<ResamplerInterface> resampler, int sample_rate_hz,
    int num_channels, int bitrate, int num_frames_per_packet)
    : generative_model_(std::move(generative_model)),
      comfort_noise_generator_(std::move(comfort_noise_generator)),
      vector_quantizer_(std::move(vector_quantizer)),
      packet_(std::move(packet)),
      packet_loss_handler_(std::move(packet_loss_handler)),
      resampler_(std::move(resampler)),
      sample_rate_hz_(sample_rate_hz),
      num_channels_(num_channels),
      bitrate_(bitrate),
      num_frames_per_packet_(num_frames_per_packet),
      internal_num_samples_available_(0),
      encoded_packet_set_(false),
      prev_frame_was_comfort_noise_(false) {}

bool LyraDecoder::SetEncodedPacket(absl::Span<const uint8_t> encoded) {
  if (encoded.size() != kPacketSize) {
    LOG(ERROR) << "The number of bytes has to equal to " << kPacketSize
               << ", but is " << encoded.size() << ".";
    return false;
  }

  const auto unpacked_or = packet_->UnpackPacket(encoded);
  if (!unpacked_or.has_value()) {
    LOG(ERROR) << "Couldn't read Lyra packet for decoding.";
    return false;
  }

  std::vector<float> concatenated_features =
      vector_quantizer_->DecodeToLossyFeatures(unpacked_or.value());
  const int num_features =
      concatenated_features.size() / num_frames_per_packet_;
  for (int i = 0; i < num_frames_per_packet_; ++i) {
    const std::vector<float> features(
        concatenated_features.begin() + num_features * i,
        concatenated_features.begin() + num_features * (i + 1));
    if (!packet_loss_handler_->SetReceivedFeatures(features)) {
      LOG(ERROR) << "Unable to update packet loss handler.";
      return false;
    }

    generative_model_->AddFeatures(features);
  }

  internal_num_samples_available_ =
      num_frames_per_packet_ * GetNumSamplesPerHop(kInternalSampleRateHz);
  encoded_packet_set_ = true;
  return true;
}

absl::optional<std::vector<int16_t>> LyraDecoder::DecodeSamples(
    int num_samples) {
  const int external_num_samples_available = ConvertNumSamplesBetweenSampleRate(
      internal_num_samples_available_, kInternalSampleRateHz, sample_rate_hz_);
  if (num_samples > external_num_samples_available) {
    LOG(ERROR) << "Requested " << num_samples
               << " samples for decoding but only "
               << external_num_samples_available
               << " remain in the current frame.";
    return absl::nullopt;
  }
  if (!encoded_packet_set_) {
    LOG(ERROR) << "Requesting normal decoding without adding "
                  "an encoded packet.";
    return absl::nullopt;
  }
  const int internal_num_samples = ConvertNumSamplesBetweenSampleRate(
      num_samples, sample_rate_hz_, kInternalSampleRateHz);
  auto audio_or = generative_model_->GenerateSamples(internal_num_samples);
  if (!audio_or.has_value()) {
    LOG(ERROR) << "Couldn't generate audio samples.";
    return absl::nullopt;
  }
  internal_num_samples_available_ -= audio_or->size();

  // Comfort noise generator should only be run during a model transition, so
  // perform this check beforehand.
  if (prev_frame_was_comfort_noise_) {
    auto estimated_features_or =
        packet_loss_handler_->EstimateLostFeatures(internal_num_samples);
    if (!estimated_features_or.has_value()) {
      LOG(ERROR) << "Unable to estimate lost features.";
      return absl::nullopt;
    }
    audio_or = RunComfortNoiseGeneratorWithNecessaryOverlap(
        internal_num_samples, true, estimated_features_or.value(),
        audio_or.value());
    if (!audio_or.has_value()) return absl::nullopt;

    // Reset CNG when going back to generative model to avoid continuity issues.
    comfort_noise_generator_->Reset();
  }
  prev_frame_was_comfort_noise_ = false;

  if (sample_rate_hz_ != kInternalSampleRateHz) {
    audio_or = resampler_->Resample(audio_or.value());
  }
  CHECK_EQ(audio_or->size(), num_samples);

  return audio_or;
}

absl::optional<std::vector<int16_t>> LyraDecoder::DecodePacketLoss(
    int num_samples) {
  const int internal_num_samples = ConvertNumSamplesBetweenSampleRate(
      num_samples, sample_rate_hz_, kInternalSampleRateHz);
  auto audio_or = RunGenerativeModelForPacketLoss(internal_num_samples);

  if (!audio_or.has_value()) {
    LOG(ERROR) << "Couldn't generate audio samples.";
    return absl::nullopt;
  }
  if (sample_rate_hz_ != kInternalSampleRateHz) {
    audio_or = resampler_->Resample(audio_or.value());
  }

  // Possibly truncate some extra samples in the end.
  audio_or->resize(num_samples);
  return audio_or;
}

absl::optional<std::vector<int16_t>>
LyraDecoder::RunGenerativeModelForPacketLoss(int num_samples) {
  const auto estimated_features_or =
      packet_loss_handler_->EstimateLostFeatures(num_samples);
  if (!estimated_features_or.has_value()) {
    LOG(ERROR) << "Unable to estimate lost features.";
    return absl::nullopt;
  }

  // Do not perform overlap if both previous and current frames were produced
  // by the comfort noise generator.
  const bool current_frame_is_comfort_noise =
      packet_loss_handler_->is_comfort_noise();
  if (prev_frame_was_comfort_noise_ && current_frame_is_comfort_noise) {
    prev_frame_was_comfort_noise_ = true;
    return RunComfortNoiseGeneratorWithNecessaryOverlap(
        num_samples, false, estimated_features_or.value());
  }

  std::vector<int16_t> result;
  result.reserve(num_samples);

  // Generate samples to fill |result| with desired number of samples. Add
  // estimated features when the previous packet has been fully decoded.
  int num_samples_to_decode;
  while (result.size() < num_samples) {
    const int remaining_num_samples =
        num_samples - static_cast<int>(result.size());
    if (internal_num_samples_available_ == 0) {
      // The previous sample generation used up the features added, add a new
      // one.
      generative_model_->AddFeatures(estimated_features_or.value());
      internal_num_samples_available_ =
          GetNumSamplesPerHop(kInternalSampleRateHz);
      encoded_packet_set_ = false;
    }
    num_samples_to_decode =
        std::min(remaining_num_samples, internal_num_samples_available_);
    const auto audio_or =
        generative_model_->GenerateSamples(num_samples_to_decode);
    if (!audio_or.has_value()) {
      LOG(ERROR) << "Model could not be run on features.";
      return absl::nullopt;
    }
    result.insert(result.end(), audio_or->begin(), audio_or->end());

    CHECK_LE(audio_or->size(), internal_num_samples_available_);
    internal_num_samples_available_ -= audio_or->size();
  }
  CHECK_EQ(result.size(), num_samples);

  // Implies a transition between models, which requires overlap.
  if (current_frame_is_comfort_noise) {
    result = RunComfortNoiseGeneratorWithNecessaryOverlap(
                 num_samples, true, estimated_features_or.value(), result)
                 .value();
  }
  prev_frame_was_comfort_noise_ = current_frame_is_comfort_noise;

  return result;
}

absl::optional<std::vector<int16_t>>
LyraDecoder::RunComfortNoiseGeneratorWithNecessaryOverlap(
    int num_samples, bool overlap_required, const std::vector<float>& features,
    const std::vector<int16_t>& generative_model_frame) const {
  comfort_noise_generator_->AddFeatures(features);
  const auto comfort_noise_or =
      comfort_noise_generator_->GenerateSamples(num_samples);
  if (!comfort_noise_or.has_value()) {
    LOG(ERROR) << "Comfort noise generator could not be run on features.";
    return absl::nullopt;
  }
  CHECK_EQ(comfort_noise_or->size(), num_samples);

  if (overlap_required) {
    // If overlap is required, a model transition is guaranteed. The direction
    // of such transition can be deduced by looking at which model produced the
    // previous frame.
    absl::optional<std::vector<int16_t>> overlapped_frame_or;
    if (prev_frame_was_comfort_noise_) {
      // Transition from CNG to generative model.
      overlapped_frame_or =
          OverlapFrames(comfort_noise_or.value(), generative_model_frame);
    } else {
      // Transition from generative model to CNG.
      overlapped_frame_or =
          OverlapFrames(generative_model_frame, comfort_noise_or.value());
    }
    return overlapped_frame_or;
  } else {
    return comfort_noise_or;
  }
}

absl::optional<std::vector<int16_t>> LyraDecoder::OverlapFrames(
    const std::vector<int16_t>& preceding_frame,
    const std::vector<int16_t>& following_frame) const {
  if (preceding_frame.size() != following_frame.size()) {
    LOG(ERROR) << "Overlapped frame could not be computed because frame sizes "
                  "differed. Preceding frame size was "
               << preceding_frame.size() << " and following frame size was "
               << following_frame.size() << ".";
    return absl::nullopt;
  }

  const int kFrameSize = preceding_frame.size();
  std::vector<int16_t> overlapped_frame(kFrameSize);
  for (int i = 0; i < kFrameSize; ++i) {
    const float overlap_weight = (1.f + std::cos(i * M_PI / kFrameSize)) / 2.f;
    overlapped_frame[i] = preceding_frame[i] * overlap_weight +
                          following_frame[i] * (1.f - overlap_weight);
  }

  return overlapped_frame;
}

int LyraDecoder::sample_rate_hz() const { return sample_rate_hz_; }

int LyraDecoder::num_channels() const { return num_channels_; }

int LyraDecoder::bitrate() const { return bitrate_; }

int LyraDecoder::frame_rate() const { return kFrameRate; }

bool LyraDecoder::is_comfort_noise() const {
  return packet_loss_handler_->is_comfort_noise();
}

}  // namespace codec
}  // namespace chromemedia
