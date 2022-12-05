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

#include "lyra/lyra_decoder.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "glog/logging.h"  // IWYU pragma: keep
#include "include/ghc/filesystem.hpp"
#include "lyra/buffered_resampler.h"
#include "lyra/comfort_noise_generator.h"
#include "lyra/lyra_components.h"
#include "lyra/lyra_config.h"
#include "lyra/noise_estimator.h"

namespace chromemedia {
namespace codec {
namespace {

// Duration of pure packet loss concealment.
inline int GetConcealmentDurationSamples() {
  static constexpr float kConcealmentDurationSeconds = 0.08;
  static constexpr int kConcealmentDurationSamples =
      kConcealmentDurationSeconds * kInternalSampleRateHz;
  CHECK_EQ(
      kConcealmentDurationSamples % GetNumSamplesPerHop(kInternalSampleRateHz),
      0);
  return kConcealmentDurationSamples;
}

// Duration it takes to fade from concealment to comfort noise, and from
// comfort noise to received packets.
inline int GetFadeDurationSamples() {
  static constexpr float kFadeDurationSeconds = 0.04;
  static constexpr int kFadeDurationSamples =
      kFadeDurationSeconds * kInternalSampleRateHz;
  CHECK_EQ(kFadeDurationSamples % GetNumSamplesPerHop(kInternalSampleRateHz),
           0);
  return kFadeDurationSamples;
}

// Reconciles the number of samples requested with the number we should
// decode within one iteration of the |DecodeSamples| while loop.
int GetNumSamplesToGenerate(int num_samples_requested,
                            int samples_generated_so_far,
                            int concealment_progress,
                            int model_samples_available,
                            int cng_samples_available) {
  int samples_remaining_packet;
  if (concealment_progress < 0) {
    // Finish playing out the remainder of the last fake packet.
    samples_remaining_packet = std::abs(concealment_progress);
  } else if (concealment_progress < GetConcealmentDurationSamples()) {
    // If we have not yet maxed out concealment progress, the
    // |generative_model_| will be used.
    samples_remaining_packet =
        model_samples_available % GetNumSamplesPerHop(kInternalSampleRateHz);
  } else {
    // Otherwise the  |comfort_noise_generator_| is guaranteed to be used.
    samples_remaining_packet = cng_samples_available;
  }
  // If we are out of samples, assume we can always add estimated features.
  if (samples_remaining_packet == 0) {
    samples_remaining_packet = GetNumSamplesPerHop(kInternalSampleRateHz);
  }
  // Take the min between the next packet boundary and the remaining number of
  // samples requested.
  return std::min(num_samples_requested - samples_generated_so_far,
                  samples_remaining_packet);
}

}  // namespace

std::unique_ptr<LyraDecoder> LyraDecoder::Create(
    int sample_rate_hz, int num_channels,
    const ghc::filesystem::path& model_path) {
  absl::Status are_params_supported =
      AreParamsSupported(sample_rate_hz, num_channels, model_path);
  if (!are_params_supported.ok()) {
    LOG(ERROR) << are_params_supported;
    return nullptr;
  }
  const int kNumSamplesPerHop = GetNumSamplesPerHop(kInternalSampleRateHz);
  const int kNumSamplesPerWindow =
      GetNumSamplesPerWindow(kInternalSampleRateHz);

  // The resampler always resamples from |kInternalSampleRateHz| to the
  // requested |sample_rate_hz|.
  auto resampler =
      BufferedResampler::Create(kInternalSampleRateHz, sample_rate_hz);
  if (resampler == nullptr) {
    LOG(ERROR) << "Could not create Buffered Resampler.";
    return nullptr;
  }
  // All internal components operate at |kInternalSampleRateHz|.
  auto model = CreateGenerativeModel(kNumFeatures, model_path);
  if (model == nullptr) {
    LOG(ERROR) << "New model could not be instantiated.";
    return nullptr;
  }
  auto comfort_noise_generator =
      ComfortNoiseGenerator::Create(kInternalSampleRateHz, kNumSamplesPerHop,
                                    kNumSamplesPerWindow, kNumMelBins);
  if (comfort_noise_generator == nullptr) {
    LOG(ERROR) << "Could not create Comfort Noise Generator.";
    return nullptr;
  }
  auto noise_estimator =
      NoiseEstimator::Create(kInternalSampleRateHz, kNumSamplesPerHop,
                             kNumSamplesPerWindow, kNumMelBins);
  if (noise_estimator == nullptr) {
    LOG(ERROR) << "Could not create Noise Estimator.";
    return nullptr;
  }
  auto vector_quantizer = CreateQuantizer(model_path);
  if (vector_quantizer == nullptr) {
    LOG(ERROR) << "Could not create Vector Quantizer.";
    return nullptr;
  }
  auto feature_estimator = CreateFeatureEstimator(kNumFeatures);

  // WrapUnique is used because of private c'tor.
  return absl::WrapUnique(
      new LyraDecoder(std::move(model), std::move(comfort_noise_generator),
                      std::move(vector_quantizer), std::move(noise_estimator),
                      std::move(feature_estimator), std::move(resampler),
                      /*external_sample_rate_hz=*/sample_rate_hz,
                      /*num_channels=*/num_channels));
}

LyraDecoder::LyraDecoder(
    std::unique_ptr<GenerativeModelInterface> generative_model,
    std::unique_ptr<GenerativeModelInterface> comfort_noise_generator,
    std::unique_ptr<VectorQuantizerInterface> vector_quantizer,
    std::unique_ptr<NoiseEstimatorInterface> noise_estimator,
    std::unique_ptr<FeatureEstimatorInterface> feature_estimator,
    std::unique_ptr<BufferedFilterInterface> resampler,
    int external_sample_rate_hz, int num_channels)
    : generative_model_(std::move(generative_model)),
      comfort_noise_generator_(std::move(comfort_noise_generator)),
      vector_quantizer_(std::move(vector_quantizer)),
      noise_estimator_(std::move(noise_estimator)),
      feature_estimator_(std::move(feature_estimator)),
      resampler_(std::move(resampler)),
      concealment_progress_(0),
      fade_progress_(0),
      fade_direction_(FadeDirection::kFadeFromCNG),
      external_sample_rate_hz_(external_sample_rate_hz),
      num_channels_(num_channels) {}

bool LyraDecoder::SetEncodedPacket(absl::Span<const uint8_t> encoded) {
  const int num_quantized_bits = PacketSizeToNumQuantizedBits(encoded.size());
  if (num_quantized_bits < 0) {
    LOG(ERROR) << "The packet size (" << encoded.size()
               << " bytes) is not supported.";
    return false;
  }
  auto packet = CreatePacket(kNumHeaderBits, num_quantized_bits);
  const auto unpacked = packet->UnpackPacket(encoded);
  if (!unpacked.has_value()) {
    LOG(ERROR) << "Could not read Lyra packet for decoding.";
    return false;
  }

  // Finish playing out any concealment or comfort noise packets before
  // moving on to the packet we are receiving.
  if (concealment_progress_ == GetConcealmentDurationSamples()) {
    concealment_progress_ = -comfort_noise_generator_->num_samples_available();
  } else if (concealment_progress_ > 0) {
    concealment_progress_ = -generative_model_->num_samples_available();
  }
  // No need to update |concealment_progress| if it is already less than
  // or equal to zero. If equal to zero we were decoding samples normally.
  // If less than zero we received than one packet while still decoding
  // concealment or comfort noise.

  auto features = vector_quantizer_->DecodeToLossyFeatures(unpacked.value());
  if (!features.has_value()) {
    LOG(ERROR) << "Could not decode to lossy features.";
    return false;
  }
  if (!generative_model_->AddFeatures(features.value())) {
    LOG(ERROR) << "Could not add received features to generative model.";
    return false;
  }
  feature_estimator_->Update(features.value());
  return true;
}

std::optional<std::vector<int16_t>> LyraDecoder::DecodeSamples(
    int num_samples) {
  std::function<std::optional<std::vector<int16_t>>(int)> decode_function =
      [this](int internal_num_samples_to_generate)
      -> std::optional<std::vector<int16_t>> {
    return DecodeSamplesInternal(internal_num_samples_to_generate);
  };
  auto external_samples =
      resampler_->FilterAndBuffer(decode_function, num_samples);

  if (!external_samples.has_value()) {
    LOG(ERROR) << "Could not decode samples.";
    return std::nullopt;
  }
  return external_samples;
}

std::optional<std::vector<int16_t>> LyraDecoder::DecodeSamplesInternal(
    int internal_num_samples_to_generate) {
  std::vector<int16_t> result;
  result.reserve(internal_num_samples_to_generate);
  while (result.size() < internal_num_samples_to_generate) {
    // Aligns the number of samples requested with the number of samples per
    // packet.
    // |GetFadeDurationSamples()| and |GetConcealmentDurationSamples()| are also
    // multiples of the number of samples per packet, so
    // |num_samples_to_generate| will be aligned with fade and concealment
    // progress as well.
    const int num_samples_to_generate = GetNumSamplesToGenerate(
        /*num_samples_requested=*/internal_num_samples_to_generate,
        /*samples_generated_so_far=*/result.size(),
        /*concealment_progress=*/concealment_progress_,
        /*model_samples_available=*/
        generative_model_->num_samples_available(),
        /*cng_samples_available=*/
        comfort_noise_generator_->num_samples_available());

    // Check if we are decoding from a received packet;
    const bool is_packet_received =
        generative_model_->num_samples_available() > 0 &&
        concealment_progress_ == 0;

    if (is_packet_received) {
      // Decoding from a received packet triggers comfort noise, if there is
      // any, to fade out.
      fade_direction_ = kFadeFromCNG;
    } else if (concealment_progress_ == GetConcealmentDurationSamples()) {
      // Comfort noise begins fading in again once we have lost
      // |GetConcealmentDurationSamples()| samples in a row.
      fade_direction_ = kFadeToCNG;
    } else {
      // We are not decoding from a received packet and have not yet started
      // playing out pure comfort noise.
      concealment_progress_ += num_samples_to_generate;
    }

    int cng_samples_to_generate = num_samples_to_generate;
    int generative_samples_to_generate = num_samples_to_generate;
    int next_fade_progress =
        fade_progress_ + fade_direction_ * num_samples_to_generate;
    if (fade_direction_ == kFadeToCNG &&
        fade_progress_ == GetFadeDurationSamples()) {
      // |fade_progress_| maxes out at |GetFadeDurationSamples()|. Once here
      // we only generate comfort noise until |fade_direction_| is reversed.
      next_fade_progress = GetFadeDurationSamples();
      generative_samples_to_generate = 0;
    } else if (fade_direction_ == kFadeFromCNG && fade_progress_ == 0) {
      // |fade_progress_| has a minimum at 0. Once here we only produce
      // generative model output until |fade_direction_| is reversed.
      next_fade_progress = 0;
      cng_samples_to_generate = 0;
    }

    auto audio = RunGenerativeModel(generative_samples_to_generate);
    if (!audio.has_value()) {
      LOG(ERROR) << "Model could not be run on features.";
      return std::nullopt;
    }
    auto comfort_noise = RunComfortNoiseGenerator(cng_samples_to_generate);
    if (!comfort_noise.has_value()) {
      LOG(ERROR) << "Could not generate comfort noise.";
      return std::nullopt;
    }

    // Perform any necessary overlap and insert into |result|.
    if (!MaybeOverlapAndInsert(fade_direction_, fade_progress_, audio.value(),
                               comfort_noise.value(), result)) {
      LOG(ERROR) << "Could not overlap comfort noise.";
      return std::nullopt;
    }

    fade_progress_ = next_fade_progress;

    // Only update |noise_estimator_| if we are dealing with received packets.
    // Do not update with concealment.
    if (is_packet_received) {
      if (!noise_estimator_->ReceiveSamples(audio.value())) {
        LOG(ERROR) << "Could not update noise estimator on decoder output.";
        return std::nullopt;
      }
    }
  }
  CHECK_EQ(result.size(), internal_num_samples_to_generate);
  return result;
}

std::optional<std::vector<int16_t>> LyraDecoder::RunGenerativeModel(
    int num_samples) {
  if (num_samples > 0 && generative_model_->num_samples_available() == 0) {
    if (!generative_model_->AddFeatures(feature_estimator_->Estimate())) {
      LOG(ERROR) << "Could not add estimated features to generative model.";
      return std::nullopt;
    }
  }
  return generative_model_->GenerateSamples(num_samples);
}

std::optional<std::vector<int16_t>> LyraDecoder::RunComfortNoiseGenerator(
    int num_samples) {
  if (num_samples > 0 &&
      comfort_noise_generator_->num_samples_available() == 0) {
    if (!comfort_noise_generator_->AddFeatures(
            noise_estimator_->noise_estimate())) {
      LOG(ERROR)
          << "Could not add noise estimate features to comfort noise generator";
      return std::nullopt;
    }
  }
  return comfort_noise_generator_->GenerateSamples(num_samples);
}

bool LyraDecoder::MaybeOverlapAndInsert(
    FadeDirection fade_direction, int fade_progress,
    const std::vector<int16_t>& generative_model_hop,
    const std::vector<int16_t>& comfort_noise_hop,
    std::vector<int16_t>& result) {
  if (comfort_noise_hop.empty()) {
    result.insert(result.end(), generative_model_hop.begin(),
                  generative_model_hop.end());
    return true;
  }
  if (generative_model_hop.empty()) {
    result.insert(result.end(), comfort_noise_hop.begin(),
                  comfort_noise_hop.end());
    return true;
  }
  if (generative_model_hop.size() != comfort_noise_hop.size()) {
    LOG(ERROR) << "Overlapped hop could not be computed because hop sizes "
                  "differed. Generative model hop size was "
               << generative_model_hop.size() << " and comfort noise hop size "
               << " was " << comfort_noise_hop.size() << ".";
    return false;
  }

  for (int i = 0; i < generative_model_hop.size(); ++i) {
    const float overlap_weight =
        (1.f + std::cos(fade_progress * M_PI / GetFadeDurationSamples())) / 2.f;
    result.push_back(generative_model_hop.at(i) * overlap_weight +
                     comfort_noise_hop.at(i) * (1.f - overlap_weight));
    fade_progress += fade_direction;
  }
  return true;
}

int LyraDecoder::sample_rate_hz() const { return external_sample_rate_hz_; }

int LyraDecoder::num_channels() const { return num_channels_; }

int LyraDecoder::frame_rate() const { return kFrameRate; }

bool LyraDecoder::is_comfort_noise() const {
  return fade_progress_ == GetFadeDurationSamples();
}

}  // namespace codec
}  // namespace chromemedia
