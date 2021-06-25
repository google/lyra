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

#include "lyra_components.h"

#include <cstdint>
#include <memory>
#include <vector>

#include "Eigen/Core"
#include "absl/memory/memory.h"
#include "feature_extractor_interface.h"
#include "generative_model_interface.h"
#include "log_mel_spectrogram_extractor_impl.h"
#include "packet.h"
#include "packet_interface.h"
#include "vector_quantizer_impl.h"
#include "vector_quantizer_interface.h"
#include "wavegru_model_impl.h"

namespace chromemedia {
namespace codec {
namespace {

// LINT.IfChange
constexpr int kNumQuantizedBits = 120;
constexpr int kNumHeaderBits = 0;
// LINT.ThenChange(
// lyra_config.cc,
// )

}  // namespace

std::unique_ptr<VectorQuantizerInterface> CreateQuantizer(
    int num_output_features, int num_bits,
    const ghc::filesystem::path& model_path) {
  return VectorQuantizerImpl::Create(num_output_features, num_bits, model_path);
}

std::unique_ptr<VectorQuantizerInterface> CreateQuantizer(
    int num_features, int num_bits, const Eigen::RowVectorXf& mean_vector,
    const Eigen::MatrixXf& transformation_matrix,
    const std::vector<float>& code_vectors,
    const std::vector<int16_t>& codebook_dimensions) {
  return VectorQuantizerImpl::Create(num_features, num_bits, mean_vector,
                                     transformation_matrix, code_vectors,
                                     codebook_dimensions);
}

std::unique_ptr<GenerativeModelInterface> CreateGenerativeModel(
    int num_samples_per_hop, int num_output_features, int num_frames_per_packet,
    const ghc::filesystem::path& model_path) {
  return WavegruModelImpl::Create(
      num_samples_per_hop, num_output_features, num_frames_per_packet,
      LogMelSpectrogramExtractorImpl::GetSilenceValue(), model_path);
}

std::unique_ptr<FeatureExtractorInterface> CreateFeatureExtractor(
    int sample_rate_hz, int num_features, int num_samples_per_hop,
    int num_samples_per_frame) {
  return LogMelSpectrogramExtractorImpl::Create(
      sample_rate_hz, num_features, num_samples_per_hop, num_samples_per_frame);
}

std::unique_ptr<PacketInterface> CreatePacket() {
  return absl::make_unique<Packet<kNumQuantizedBits, kNumHeaderBits>>();
}

absl::StatusOr<std::unique_ptr<DenoiserInterface>> CreateDenoiser(
    const ghc::filesystem::path& model_path) {
  return nullptr;
}

}  // namespace codec
}  // namespace chromemedia
