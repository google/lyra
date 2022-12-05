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

#ifndef LYRA_LYRA_COMPONENTS_H_
#define LYRA_LYRA_COMPONENTS_H_

#include <memory>

#include "include/ghc/filesystem.hpp"
#include "lyra/feature_estimator_interface.h"
#include "lyra/feature_extractor_interface.h"
#include "lyra/generative_model_interface.h"
#include "lyra/packet_interface.h"
#include "lyra/vector_quantizer_interface.h"

namespace chromemedia {
namespace codec {

std::unique_ptr<VectorQuantizerInterface> CreateQuantizer(
    const ghc::filesystem::path& model_path);

std::unique_ptr<GenerativeModelInterface> CreateGenerativeModel(
    int num_output_features, const ghc::filesystem::path& model_path);

std::unique_ptr<FeatureExtractorInterface> CreateFeatureExtractor(
    const ghc::filesystem::path& model_path);

std::unique_ptr<PacketInterface> CreatePacket(int num_header_bits,
                                              int num_quantized_bits);

std::unique_ptr<FeatureEstimatorInterface> CreateFeatureEstimator(
    int num_features);

}  // namespace codec
}  // namespace chromemedia

#endif  // LYRA_LYRA_COMPONENTS_H_
