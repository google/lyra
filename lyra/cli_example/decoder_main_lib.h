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

#ifndef LYRA_CLI_EXAMPLE_DECODER_MAIN_LIB_H_
#define LYRA_CLI_EXAMPLE_DECODER_MAIN_LIB_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/random/bit_gen_ref.h"
#include "absl/strings/string_view.h"
#include "include/ghc/filesystem.hpp"
#include "lyra/lyra_decoder.h"
#include "lyra/packet_loss_model_interface.h"

namespace chromemedia {
namespace codec {

// Used for custom command line flag in decoder_main.
struct PacketLossPattern {
  explicit PacketLossPattern(const std::vector<float>& starts,
                             const std::vector<float>& durations)
      : starts_(starts), durations_(durations) {}

  std::vector<float> starts_;
  std::vector<float> durations_;
};

std::string AbslUnparseFlag(chromemedia::codec::PacketLossPattern pattern);

bool AbslParseFlag(absl::string_view text,
                   chromemedia::codec::PacketLossPattern* p,
                   std::string* error);

// Decodes a vector of bytes into wav data.
// If |packet_loss_model| is nullptr no packets will be lost.
bool DecodeFeatures(const std::vector<uint8_t>& packet_stream, int packet_size,
                    bool randomize_num_samples_requested, absl::BitGenRef gen,
                    LyraDecoder* decoder,
                    PacketLossModelInterface* packet_loss_model,
                    std::vector<int16_t>* decoded_audio);

// Decodes an encoded features file into a wav file.
// Uses the model and quant files located under |model_path|.
// Given the file /tmp/lyra/file1.lyra exists and is a valid encoded file. For:
// |encoded_path| = "/tmp/lyra/file1.lyra"
// |output_path| = "/tmp/lyra/file1_decoded.lyra"
// Then successful decoding will write out the file
// /tmp/lyra/encoded/file1_decoded.wav
bool DecodeFile(const ghc::filesystem::path& encoded_path,
                const ghc::filesystem::path& output_path, int sample_rate_hz,
                int bitrate, bool randomize_num_samples_requested,
                float packet_loss_rate, float average_burst_length,
                const PacketLossPattern& fixed_packet_loss_pattern,
                const ghc::filesystem::path& model_path);

}  // namespace codec
}  // namespace chromemedia

#endif  // LYRA_CLI_EXAMPLE_DECODER_MAIN_LIB_H_
