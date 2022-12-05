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

#include "lyra/cli_example/encoder_main_lib.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <memory>
#include <optional>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "glog/logging.h"  // IWYU pragma: keep
#include "include/ghc/filesystem.hpp"
#include "lyra/lyra_config.h"
#include "lyra/lyra_encoder.h"
#include "lyra/no_op_preprocessor.h"
#include "lyra/wav_utils.h"

namespace chromemedia {
namespace codec {

// Packets are appended to encoded_features. The oldest packet is encoded
// starting at index 0.
bool EncodeWav(const std::vector<int16_t>& wav_data, int num_channels,
               int sample_rate_hz, int bitrate, bool enable_preprocessing,
               bool enable_dtx, const ghc::filesystem::path& model_path,
               std::vector<uint8_t>* encoded_features) {
  auto encoder = LyraEncoder::Create(/*sample_rate_hz=*/sample_rate_hz,
                                     /*num_channels=*/num_channels,
                                     /*bitrate=*/bitrate,
                                     /*enable_dtx=*/enable_dtx,
                                     /*model_path=*/model_path);
  if (encoder == nullptr) {
    LOG(ERROR) << "Could not create lyra encoder.";
    return false;
  }

  std::unique_ptr<PreprocessorInterface> preprocessor;
  if (enable_preprocessing) {
    preprocessor = std::make_unique<NoOpPreprocessor>();
  }

  const auto benchmark_start = absl::Now();

  std::vector<int16_t> processed_data(wav_data);
  if (enable_preprocessing) {
    processed_data = preprocessor->Process(
        absl::MakeConstSpan(wav_data.data(), wav_data.size()), sample_rate_hz);
  }

  const int num_samples_per_packet = sample_rate_hz / encoder->frame_rate();
  // Iterate over the wav data until the end of the vector.
  for (int wav_iterator = 0;
       wav_iterator + num_samples_per_packet <= processed_data.size();
       wav_iterator += num_samples_per_packet) {
    // Move audio samples from the large in memory wav file frame by frame to
    // the encoder.
    auto encoded = encoder->Encode(absl::MakeConstSpan(
        &processed_data.at(wav_iterator), num_samples_per_packet));
    if (!encoded.has_value()) {
      LOG(ERROR) << "Unable to encode features starting at samples at byte "
                 << wav_iterator << ".";
      return false;
    }

    // Append the encoded audio frames to the encoded_features accumulator
    // vector.
    encoded_features->insert(encoded_features->end(), encoded.value().begin(),
                             encoded.value().end());
  }
  const auto elapsed = absl::Now() - benchmark_start;
  LOG(INFO) << "Elapsed seconds : " << absl::ToInt64Seconds(elapsed);
  LOG(INFO) << "Samples per second : "
            << wav_data.size() / absl::ToDoubleSeconds(elapsed);

  return true;
}

bool EncodeFile(const ghc::filesystem::path& wav_path,
                const ghc::filesystem::path& output_path, int bitrate,
                bool enable_preprocessing, bool enable_dtx,
                const ghc::filesystem::path& model_path) {
  // Reads the entire wav file into memory.
  absl::StatusOr<ReadWavResult> read_wav_result =
      Read16BitWavFileToVector(wav_path.string());

  if (!read_wav_result.ok()) {
    LOG(ERROR) << read_wav_result.status();
    return false;
  }

  // Keep an accumulator vector of all the encoded features to write to file.
  std::vector<uint8_t> encoded_features;
  if (!EncodeWav(read_wav_result->samples, read_wav_result->num_channels,
                 read_wav_result->sample_rate_hz, bitrate, enable_preprocessing,
                 enable_dtx, model_path, &encoded_features)) {
    LOG(ERROR) << "Unable to encode features for file " << wav_path;
    return false;
  }

  std::ofstream output_stream(output_path.string(),
                              std::ios_base::binary | std::ios_base::trunc);
  if (!output_stream.is_open()) {
    LOG(ERROR) << "Could not open output file " << output_path;
    return false;
  }
  absl::string_view encoded_data(
      reinterpret_cast<const char*>(encoded_features.data()),
      encoded_features.size());
  std::ostreambuf_iterator<char> output_iterator(output_stream);
  std::copy(encoded_data.begin(), encoded_data.end(), output_iterator);

  return true;
}

}  // namespace codec
}  // namespace chromemedia
