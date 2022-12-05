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

#include "lyra/wav_utils.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "audio/dsp/portable/read_wav_file.h"
#include "audio/dsp/portable/write_wav_file.h"

namespace chromemedia::codec {

absl::StatusOr<ReadWavResult> Read16BitWavFileToVector(
    const std::string& file_name) {
  int num_channels;
  int sample_rate_hz;
  size_t read_num_samples;
  int16_t* out_buffer = Read16BitWavFile(file_name.c_str(), &read_num_samples,
                                         &num_channels, &sample_rate_hz);

  if (out_buffer == nullptr) {
    return absl::AbortedError(
        absl::StrCat("Failed to read from wav at path: ", file_name));
  }
  std::vector<int16_t> samples(out_buffer, out_buffer + read_num_samples);
  free(out_buffer);

  return ReadWavResult{samples, num_channels, sample_rate_hz};
}

absl::Status Write16BitWavFileFromVector(const std::string& file_name,
                                         int num_channels, int sample_rate_hz,
                                         const std::vector<int16_t>& samples) {
  int status = WriteWavFile(file_name.c_str(), samples.data(), samples.size(),
                            sample_rate_hz, num_channels);

  // WriteWavFile has a success code of 1.
  if (status == 1) {
    return absl::OkStatus();
  }
  return absl::AbortedError(
      absl::StrCat("Failed to write to wav file at: ", file_name));
}

}  // namespace chromemedia::codec
