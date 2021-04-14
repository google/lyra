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

#ifndef LYRA_CODEC_ENCODER_MAIN_LIB_H_
#define LYRA_CODEC_ENCODER_MAIN_LIB_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "include/ghc/filesystem.hpp"

namespace chromemedia {
namespace codec {

// Encodes a vector of wav_data into encoded_features.
// Uses the quant files located under |model_path|.
bool EncodeWav(const std::vector<int16_t>& wav_data, int num_channels,
               int sample_rate_hz, bool enable_preprocessing, bool enable_dtx,
               const ghc::filesystem::path& model_path,
               std::vector<uint8_t>* encoded_features);

// Encodes a wav file into an encoded feature file. Encodes num_samples from the
// file at |wav_path| and writes the encoded features out to |output_path|.
// Uses the quant files located under |model_path|.
bool EncodeFile(const ghc::filesystem::path& wav_path,
                const ghc::filesystem::path& output_path,
                bool enable_preprocessing, bool enable_dtx,
                const ghc::filesystem::path& model_path);

}  // namespace codec
}  // namespace chromemedia

#endif  // LYRA_CODEC_ENCODER_MAIN_LIB_H_
