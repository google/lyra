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

#include "lyra/lyra_components.h"

#include <memory>

#include "lyra/feature_extractor_interface.h"
#include "lyra/generative_model_interface.h"
#include "lyra/lyra_gan_model.h"
#include "lyra/packet.h"
#include "lyra/packet_interface.h"
#include "lyra/residual_vector_quantizer.h"
#include "lyra/soundstream_encoder.h"
#include "lyra/vector_quantizer_interface.h"
#include "lyra/zero_feature_estimator.h"

namespace chromemedia {
namespace codec {
namespace {

//  LINT.IfChange
constexpr int kMaxNumPacketBits = 184;
// LINT.ThenChange(
// lyra_config.cc,
// residual_vector_quantizer.h,
// )

}  // namespace

std::unique_ptr<VectorQuantizerInterface> CreateQuantizer(
    const ghc::filesystem::path& model_path) {
  return ResidualVectorQuantizer::Create(model_path);
}

std::unique_ptr<GenerativeModelInterface> CreateGenerativeModel(
    int num_output_features, const ghc::filesystem::path& model_path) {
  return LyraGanModel::Create(model_path, num_output_features);
}

std::unique_ptr<FeatureExtractorInterface> CreateFeatureExtractor(
    const ghc::filesystem::path& model_path) {
  return SoundStreamEncoder::Create(model_path);
}

std::unique_ptr<PacketInterface> CreatePacket(int num_header_bits,
                                              int num_quantized_bits) {
  return Packet<kMaxNumPacketBits>::Create(num_header_bits, num_quantized_bits);
}

std::unique_ptr<FeatureEstimatorInterface> CreateFeatureEstimator(
    int num_features) {
  return std::make_unique<ZeroFeatureEstimator>(num_features);
}

}  // namespace codec
}  // namespace chromemedia
