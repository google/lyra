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

#ifndef LYRA_CODEC_WAV_UTILS_H_
#define LYRA_CODEC_WAV_UTILS_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace chromemedia::codec {

struct ReadWavResult {
  const std::vector<int16_t> samples;
  const int num_channels;
  const int sample_rate_hz;
};

// Reads a 16 bit .wav file into a vector.
// Writes the found parsed values into `num_channels` and `sample_rate_hz`.
// Returns an StatusOr<ReadWavResult> which contains samples and metadata
// when the file is successfully read.
absl::StatusOr<ReadWavResult> Read16BitWavFileToVector(
    const std::string& file_name);

// Writes a 16 bit .wav file.
// For a multichannel file, (when `num_channels` > 1), `samples` should be
// interleaved.
absl::Status Write16BitWavFileFromVector(const std::string& file_name,
                                         int num_channels, int sample_rate_hz,
                                         const std::vector<int16_t>& samples);

}  // namespace chromemedia::codec

#endif  // LYRA_CODEC_WAV_UTILS_H_
