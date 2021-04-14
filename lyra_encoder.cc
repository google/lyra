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

#include "lyra_encoder.h"

#include <algorithm>
#include <bitset>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "audio/linear_filters/biquad_filter.h"
#include "audio/linear_filters/biquad_filter_coefficients.h"
#include "denoiser_interface.h"
#include "dsp_util.h"
#include "feature_extractor_interface.h"
#include "glog/logging.h"
#include "include/ghc/filesystem.hpp"
#include "lyra_components.h"
#include "lyra_config.h"
#include "noise_estimator.h"
#include "noise_estimator_interface.h"
#include "packet.h"
#include "packet_interface.h"
#include "resampler.h"
#include "resampler_interface.h"
#include "vector_quantizer_interface.h"

namespace chromemedia {
namespace codec {

std::unique_ptr<LyraEncoder> LyraEncoder::Create(
    int sample_rate_hz, int num_channels, int bitrate, bool enable_dtx,
    const ghc::filesystem::path& model_path) {
  absl::Status are_params_supported =
      AreParamsSupported(sample_rate_hz, num_channels, bitrate, model_path);
  if (!are_params_supported.ok()) {
    LOG(ERROR) << are_params_supported;
    return nullptr;
  }

  std::unique_ptr<Resampler> resampler = nullptr;
  if (kInternalSampleRateHz != sample_rate_hz) {
    resampler = Resampler::Create(sample_rate_hz, kInternalSampleRateHz);
    if (resampler == nullptr) {
      LOG(ERROR) << "Could not create Resampler.";
      return nullptr;
    }
  }

  const int internal_samples_per_hop =
      GetNumSamplesPerHop(kInternalSampleRateHz);
  auto feature_extractor = CreateFeatureExtractor(
      kInternalSampleRateHz, kNumFeatures, internal_samples_per_hop,
      GetNumSamplesPerFrame(kInternalSampleRateHz));
  if (feature_extractor == nullptr) {
    LOG(ERROR) << "Could not create Features Extractor.";
    return nullptr;
  }

  auto vector_quantizer =
      CreateQuantizer(kNumFramesPerPacket * kNumExpectedOutputFeatures,
                      kNumQuantizationBits, model_path);
  if (vector_quantizer == nullptr) {
    LOG(ERROR) << "Could not create Vector Quantizer.";
    return nullptr;
  }

  auto packet = CreatePacket();

  std::unique_ptr<NoiseEstimatorInterface> noise_estimator =
      NoiseEstimator::Create(
          kNumFeatures,
          static_cast<float>(internal_samples_per_hop) / kInternalSampleRateHz);
  if (noise_estimator == nullptr) {
    LOG(ERROR) << "Could not create Noise Estimator.";
    return nullptr;
  }

  // Default to the internal frame hop size.
  auto denoiser = CreateDenoiser(model_path);
  if (!denoiser.ok()) {
    LOG(ERROR) << "Failed to create denoiser.";
    return nullptr;
  }
  if (denoiser.value() != nullptr) {
    const int denoiser_samples_per_hop =
        denoiser.value()->SamplesPerHop() == 0
            ? internal_samples_per_hop
            : denoiser.value()->SamplesPerHop();
    if (internal_samples_per_hop % denoiser_samples_per_hop != 0) {
      LOG(ERROR) << "Denoiser hop size must divide encoder hop size.";
      return nullptr;
    }
  }

  // WrapUnique is used because of private c'tor.
  return absl::WrapUnique(new LyraEncoder(
      std::move(resampler), std::move(feature_extractor),
      std::move(noise_estimator), std::move(vector_quantizer),
      std::move(denoiser.value()), std::move(packet), sample_rate_hz,
      num_channels, bitrate, kNumFramesPerPacket, enable_dtx));
}

LyraEncoder::LyraEncoder(
    std::unique_ptr<ResamplerInterface> resampler,
    std::unique_ptr<FeatureExtractorInterface> feature_extractor,
    std::unique_ptr<NoiseEstimatorInterface> noise_estimator,
    std::unique_ptr<VectorQuantizerInterface> vector_quantizer,
    std::unique_ptr<DenoiserInterface> denoiser,
    std::unique_ptr<PacketInterface> packet, int sample_rate_hz,
    int num_channels, int bitrate, int num_frames_per_packet, bool enable_dtx)
    : resampler_(std::move(resampler)),
      feature_extractor_(std::move(feature_extractor)),
      noise_estimator_(std::move(noise_estimator)),
      vector_quantizer_(std::move(vector_quantizer)),
      denoiser_(std::move(denoiser)),
      packet_(std::move(packet)),
      sample_rate_hz_(sample_rate_hz),
      num_channels_(num_channels),
      bitrate_(bitrate),
      num_frames_per_packet_(num_frames_per_packet),
      enable_dtx_(enable_dtx) {
  // This filter has a -60 dB response for frequencies below 60 Hz for 16 kHz
  // sample rate or 30 Hz for 8 kHz sample rate. For sample rates of 32 kHz and
  // 48 kHz, the audio is resampled to 16 kHz before filtering, so the cutoff
  // will be 60 Hz too.
  // TODO(b/143491858): Remove this filtering once we find a vector quantizer
  // that is robust to DC.
  second_order_sections_filter_.Init(
      1, linear_filters::BiquadFilterCascadeCoefficients({
             {{0.99860809, -1.99666786, 0.99860809},
              {1.0, -1.99658432, 0.99729972}},
             {{0.99597739, -1.99145467, 0.99597739},
              {1.0, -1.99137134, 0.99203811}},
             {{0.99353280, -1.98665193, 0.99353280},
              {1.0, -1.98656881, 0.98714873}},
             {{0.99137777, -1.98245157, 0.99137777},
              {1.0, -1.98236863, 0.98283848}},
             {{0.98960226, -1.97901469, 0.98960226},
              {1.0, -1.97893189, 0.97928731}},
             {{0.98827957, -1.97646836, 0.98827957},
              {1.0, -1.97638567, 0.97664182}},
             {{0.98746381, -1.97490392, 0.98746381},
              {1.0, -1.9748213, 0.97501025}},
             {{0.99357343, -0.99357343, 0.0}, {1.0, -0.98714687, 0.0}},
         }));
}

absl::optional<std::vector<uint8_t>> LyraEncoder::Encode(
    const absl::Span<const int16_t> audio) {
  return EncodeInternal(audio, true);
}

absl::optional<std::vector<uint8_t>> LyraEncoder::EncodeInternal(
    const absl::Span<const int16_t> audio, bool filter_audio) {
  absl::Span<const int16_t> audio_for_encoding = audio;

  // Space to store resampled and/or filtered samples.
  std::vector<int16_t> processed;
  if (kInternalSampleRateHz != sample_rate_hz_) {
    processed = resampler_->Resample(audio);
    audio_for_encoding = absl::MakeConstSpan(processed);
  }

  const int internal_samples_per_hop =
      GetNumSamplesPerHop(kInternalSampleRateHz);
  if (audio_for_encoding.size() !=
      num_frames_per_packet_ * internal_samples_per_hop) {
    LOG(ERROR) << "The number of audio samples has to be exactly "
               << num_frames_per_packet_ * GetNumSamplesPerHop(sample_rate_hz_)
               << ", but is " << audio.size() << ".";
    return absl::nullopt;
  }

  std::vector<int16_t> denoised_audio;
  if (denoiser_ != nullptr) {
    denoised_audio.reserve(audio_for_encoding.size());
    for (int t = 0; t < audio_for_encoding.size();
         t += denoiser_->SamplesPerHop()) {
      auto denoised_frame = denoiser_->Denoise(
          audio_for_encoding.subspan(t, denoiser_->SamplesPerHop()));
      if (!denoised_frame.ok()) {
        LOG(ERROR) << "Denoising failed.";
        return absl::nullopt;
      }
      denoised_audio.insert(denoised_audio.end(),
                            denoised_frame.value().begin(),
                            denoised_frame.value().end());
    }
    audio_for_encoding = absl::MakeConstSpan(denoised_audio);
  }

  if (filter_audio) {
    // High-pass filter before encoding.
    std::vector<float> pre_filtered_floats(audio_for_encoding.begin(),
                                           audio_for_encoding.end());
    std::vector<float> filtered_floats;
    second_order_sections_filter_.ProcessBlock(pre_filtered_floats,
                                               &filtered_floats);
    processed.resize(filtered_floats.size());
    std::transform(filtered_floats.begin(), filtered_floats.end(),
                   processed.begin(), ClipToInt16);
    audio_for_encoding = absl::MakeConstSpan(processed);
  }

  // We send an empty packet only if all constituent frames are noise similar
  // to the previous ones.
  int num_similar_noise_frames = 0;
  std::vector<float> concatenated_features;
  for (int i = 0; i < num_frames_per_packet_; ++i) {
    auto features_or = feature_extractor_->Extract(audio_for_encoding.subspan(
        internal_samples_per_hop * i, internal_samples_per_hop));
    if (!features_or.has_value()) {
      LOG(ERROR) << "Unable to extract features from audio frame.";
      return absl::nullopt;
    }
    const std::vector<float>& features = features_or.value();

    if (enable_dtx_) {
      auto is_similar_noise = noise_estimator_->IsSimilarNoise(features);
      if (!is_similar_noise.has_value()) {
        LOG(ERROR) << "Unable to check noise estimation.";
        return absl::nullopt;
      }

      if (is_similar_noise.value()) {
        num_similar_noise_frames++;
      } else {
        if (!noise_estimator_->Update(features)) {
          LOG(ERROR) << "Unable to update noise estimator.";
          return absl::nullopt;
        }
      }
    }

    if (concatenated_features.empty()) {
      concatenated_features.resize(num_frames_per_packet_ * features.size());
    }
    std::copy(features.begin(), features.end(),
              concatenated_features.begin() + i * features.size());
  }

  if (num_similar_noise_frames == num_frames_per_packet_) {
    Packet<0, 0> empty_packet;
    return empty_packet.PackQuantized(std::bitset<0>{}.to_string());
  }

  auto quantized_features_or =
      vector_quantizer_->Quantize(concatenated_features);
  if (!quantized_features_or.has_value()) {
    LOG(ERROR) << "Unable to quantize features.";
    return absl::nullopt;
  }
  return packet_->PackQuantized(quantized_features_or.value());
}

int LyraEncoder::sample_rate_hz() const { return sample_rate_hz_; }

int LyraEncoder::num_channels() const { return num_channels_; }

int LyraEncoder::bitrate() const { return bitrate_; }

int LyraEncoder::frame_rate() const { return kFrameRate; }
}  // namespace codec
}  // namespace chromemedia
