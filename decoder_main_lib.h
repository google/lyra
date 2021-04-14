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

#ifndef LYRA_CODEC_DECODER_MAIN_LIB_H_
#define LYRA_CODEC_DECODER_MAIN_LIB_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "include/ghc/filesystem.hpp"
#include "lyra_decoder.h"

namespace chromemedia {
namespace codec {

// Decodes a vector of bytes into wav data.
bool DecodeFeatures(const std::vector<uint8_t>& packet_stream,
                    float packet_loss_rate, float average_burst_length,
                    LyraDecoder* decoder, std::vector<int16_t>* decoded_audio);

// Decodes an encoded features file into a wav file.
// Uses the model and quant files located under |model_path|.
// Given the file /tmp/lyra/file1.lyra exists and is a valid encoded file. For:
// |encoded_path| = "/tmp/lyra/file1.lyra"
// |output_path| = "/tmp/lyra/file1_decoded.lyra"
// Then successful decoding will write out the file
// /tmp/lyra/encoded/file1_decoded.wav
bool DecodeFile(const ghc::filesystem::path& encoded_path,
                const ghc::filesystem::path& output_path, int sample_rate_hz,
                float packet_loss_rate, float average_burst_length,
                const ghc::filesystem::path& model_path);

}  // namespace codec
}  // namespace chromemedia

#endif  // LYRA_CODEC_DECODER_MAIN_LIB_H_
