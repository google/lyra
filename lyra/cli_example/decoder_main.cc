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

#include <string>
#include <system_error>  // NOLINT(build/c++11)
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "absl/strings/string_view.h"
#include "glog/logging.h"  // IWYU pragma: keep
#include "include/ghc/filesystem.hpp"
#include "lyra/architecture_utils.h"
#include "lyra/cli_example/decoder_main_lib.h"

ABSL_FLAG(std::string, encoded_path, "",
          "Complete path to the file containing the encoded features.");
ABSL_FLAG(std::string, output_dir, "",
          "The complete output dir for the wav to be written out. "
          "Recursively creates dir if it does not exist. Will "
          "overwrite existing files.");
ABSL_FLAG(std::string, output_suffix, "_decoded",
          "A prefix for each of the output .wav files.");
ABSL_FLAG(int, sample_rate_hz, 16000, "Desired output sample rate in Hertz.");
ABSL_FLAG(int, bitrate, 3200,
          "The bitrate in bps at which the file has been quantized.");
ABSL_FLAG(bool, randomize_num_samples_requested, false,
          "If true, requests a random number of samples for decoding within "
          "each hop. If false, requests only one whole hop at a time.");
ABSL_FLAG(double, packet_loss_rate, 0.0,
          "Percentage of packets that are lost.");
ABSL_FLAG(double, average_burst_length, 1.0,
          "Average length of periods where packets are lost.");
ABSL_FLAG(chromemedia::codec::PacketLossPattern, fixed_packet_loss_pattern,
          chromemedia::codec::PacketLossPattern({}, {}),
          "Two lists representing the start, duration in seconds of "
          "packet loss patterns. The exact start and duration of packet loss "
          "bursts will be rounded up to the nearest packet duration boundary. "
          "If this flag contains a nonzero number of values we ignore "
          "|packet_loss_rate| and |average_burst_length|.");
ABSL_FLAG(std::string, model_path, "lyra/model_coeffs",
          "Path to directory containing TFLite files. For mobile this is the "
          "absolute path, like "
          "'/data/local/tmp/lyra/model_coeffs/'."
          " For desktop this is the path relative to the binary.");

int main(int argc, char** argv) {
  absl::SetProgramUsageMessage(argv[0]);
  absl::ParseCommandLine(argc, argv);

  const ghc::filesystem::path encoded_path(absl::GetFlag(FLAGS_encoded_path));
  const ghc::filesystem::path output_dir(absl::GetFlag(FLAGS_output_dir));
  const std::string output_suffix = absl::GetFlag(FLAGS_output_suffix);
  const int sample_rate_hz = absl::GetFlag(FLAGS_sample_rate_hz);
  const int bitrate = absl::GetFlag(FLAGS_bitrate);
  const bool randomize_num_samples_requested =
      absl::GetFlag(FLAGS_randomize_num_samples_requested);
  const float packet_loss_rate = absl::GetFlag(FLAGS_packet_loss_rate);
  const float average_burst_length = absl::GetFlag(FLAGS_average_burst_length);
  const chromemedia::codec::PacketLossPattern fixed_packet_loss_pattern =
      absl::GetFlag(FLAGS_fixed_packet_loss_pattern);
  const ghc::filesystem::path model_path =
      chromemedia::codec::GetCompleteArchitecturePath(
          absl::GetFlag(FLAGS_model_path));
  if (!fixed_packet_loss_pattern.starts_.empty()) {
    LOG(INFO) << "Using fixed packet loss pattern instead of gilbert model.";
  }
  if (encoded_path.empty()) {
    LOG(ERROR) << "Flag --encoded_path not set.";
    return -1;
  }
  if (output_dir.empty()) {
    LOG(ERROR) << "Flag --output_dir not set.";
    return -1;
  }

  std::error_code error_code;
  if (!ghc::filesystem::is_directory(output_dir, error_code)) {
    LOG(INFO) << "Creating non existent output dir " << output_dir;
    if (!ghc::filesystem::create_directories(output_dir, error_code)) {
      LOG(ERROR) << "Tried creating output dir " << output_dir
                 << " but failed.";
      return -1;
    }
  }
  auto base_name = encoded_path.stem();
  const auto output_path = ghc::filesystem::path(output_dir) /
                           encoded_path.stem().concat(output_suffix + ".wav");

  if (!chromemedia::codec::DecodeFile(encoded_path, output_path, sample_rate_hz,
                                      bitrate, randomize_num_samples_requested,
                                      packet_loss_rate, average_burst_length,
                                      fixed_packet_loss_pattern, model_path)) {
    LOG(ERROR) << "Could not decode " << encoded_path;
    return -1;
  }
  return 0;
}
