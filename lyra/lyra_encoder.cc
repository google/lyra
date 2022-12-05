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

#include "lyra/lyra_encoder.h"

#include <bitset>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "glog/logging.h"  // IWYU pragma: keep
#include "include/ghc/filesystem.hpp"
#include "lyra/feature_extractor_interface.h"
#include "lyra/lyra_components.h"
#include "lyra/lyra_config.h"
#include "lyra/noise_estimator.h"
#include "lyra/noise_estimator_interface.h"
#include "lyra/packet.h"
#include "lyra/packet_interface.h"
#include "lyra/resampler.h"
#include "lyra/resampler_interface.h"
#include "lyra/vector_quantizer_interface.h"

namespace chromemedia {
namespace codec {

std::unique_ptr<LyraEncoder> LyraEncoder::Create(
    int sample_rate_hz, int num_channels, int bitrate, bool enable_dtx,
    const ghc::filesystem::path& model_path) {
  absl::Status are_params_supported =
      AreParamsSupported(sample_rate_hz, num_channels, model_path);
  if (!are_params_supported.ok()) {
    LOG(ERROR) << are_params_supported;
    return nullptr;
  }
  const int num_quantized_bits = BitrateToNumQuantizedBits(bitrate);
  if (num_quantized_bits < 0) {
    LOG(ERROR) << "Bitrate " << bitrate << " bps is not supported by codec.";
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

  auto feature_extractor = CreateFeatureExtractor(model_path);
  if (feature_extractor == nullptr) {
    LOG(ERROR) << "Could not create Features Extractor.";
    return nullptr;
  }

  auto vector_quantizer = CreateQuantizer(model_path);
  if (vector_quantizer == nullptr) {
    LOG(ERROR) << "Could not create Vector Quantizer.";
    return nullptr;
  }

  std::unique_ptr<NoiseEstimatorInterface> noise_estimator = nullptr;
  if (enable_dtx) {
    noise_estimator = NoiseEstimator::Create(
        sample_rate_hz, GetNumSamplesPerHop(kInternalSampleRateHz),
        GetNumSamplesPerWindow(kInternalSampleRateHz), kNumMelBins);
    if (noise_estimator == nullptr) {
      LOG(ERROR) << "Could not create Noise Estimator.";
      return nullptr;
    }
  }

  // WrapUnique is used because of private c'tor.
  return absl::WrapUnique(new LyraEncoder(
      std::move(resampler), std::move(feature_extractor),
      std::move(noise_estimator), std::move(vector_quantizer), sample_rate_hz,
      num_channels, num_quantized_bits, enable_dtx));
}

LyraEncoder::LyraEncoder(
    std::unique_ptr<ResamplerInterface> resampler,
    std::unique_ptr<FeatureExtractorInterface> feature_extractor,
    std::unique_ptr<NoiseEstimatorInterface> noise_estimator,
    std::unique_ptr<VectorQuantizerInterface> vector_quantizer,
    int sample_rate_hz, int num_channels, int num_quantized_bits,
    bool enable_dtx)
    : resampler_(std::move(resampler)),
      feature_extractor_(std::move(feature_extractor)),
      noise_estimator_(std::move(noise_estimator)),
      vector_quantizer_(std::move(vector_quantizer)),
      sample_rate_hz_(sample_rate_hz),
      num_channels_(num_channels),
      num_quantized_bits_(num_quantized_bits),
      enable_dtx_(enable_dtx) {}

std::optional<std::vector<uint8_t>> LyraEncoder::Encode(
    const absl::Span<const int16_t> audio) {
  absl::Span<const int16_t> audio_for_encoding = audio;

  // Space to store resampled and/or filtered samples.
  std::vector<int16_t> processed;
  if (kInternalSampleRateHz != sample_rate_hz_) {
    processed = resampler_->Resample(audio);
    audio_for_encoding = absl::MakeConstSpan(processed);
  }

  if (audio_for_encoding.size() != GetNumSamplesPerHop(kInternalSampleRateHz)) {
    LOG(ERROR) << "The number of audio samples has to be exactly "
               << GetNumSamplesPerHop(sample_rate_hz_) << ", but is "
               << audio.size() << ".";
    return std::nullopt;
  }

  if (enable_dtx_) {
    if (!noise_estimator_->ReceiveSamples(audio_for_encoding)) {
      LOG(ERROR) << "Unable to update encoder noise estimator.";
      return std::nullopt;
    }
    // We send an empty packet only if this hop is just noise.
    if (noise_estimator_->is_noise()) {
      auto empty_packet = Packet<0>::Create(0, 0);
      return empty_packet->PackQuantized(std::bitset<0>{}.to_string());
    }
  }

  auto features = feature_extractor_->Extract(audio_for_encoding);
  if (!features.has_value()) {
    LOG(ERROR) << "Unable to extract features from audio hop.";
    return std::nullopt;
  }
  auto quantized_features =
      vector_quantizer_->Quantize(features.value(), num_quantized_bits_);
  if (!quantized_features.has_value()) {
    LOG(ERROR) << "Unable to quantize features.";
    return std::nullopt;
  }
  auto packet = CreatePacket(kNumHeaderBits, num_quantized_bits_);
  return packet->PackQuantized(quantized_features.value());
}

bool LyraEncoder::set_bitrate(int bitrate) {
  const int num_quantized_bits = BitrateToNumQuantizedBits(bitrate);
  if (num_quantized_bits < 0) {
    LOG(ERROR) << "Bitrate " << bitrate << " bps is not supported by codec.";
    return false;
  }
  num_quantized_bits_ = num_quantized_bits;
  return true;
}

int LyraEncoder::sample_rate_hz() const { return sample_rate_hz_; }

int LyraEncoder::num_channels() const { return num_channels_; }

int LyraEncoder::bitrate() const { return GetBitrate(num_quantized_bits_); }

int LyraEncoder::frame_rate() const { return kFrameRate; }
}  // namespace codec
}  // namespace chromemedia
