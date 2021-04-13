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

#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "glog/logging.h"
#include "absl/flags/flag.h"
#include "absl/strings/string_view.h"
#include "include/ghc/filesystem.hpp"
#include "architecture_utils.h"
#include "encoder_main_lib.h"

ABSL_FLAG(std::string, input_path, "",
          "Complete path to the WAV file to be encoded.");
ABSL_FLAG(std::string, output_dir, "",
          "The dir for the encoded file to be written out. Recursively "
          "creates dir if it does not exist. Output files use the same "
          "name as the wav file they come from with a '.lyra' postfix. Will "
          "overwrite existing files.");
ABSL_FLAG(bool, enable_preprocessing, false,
          "If enabled runs the input signal through the preprocessing "
          "module before encoding.");
ABSL_FLAG(bool, enable_dtx, false,
          "Enables discontinuous transmission (DTX). DTX does not send packets "
          "when noise is detected.");
ABSL_FLAG(
    std::string, model_path, "wavegru",
    "Path to directory containing quantization files. For mobile "
    "this is the absolute path, like '/sdcard/wavegru/'. For desktop this is "
    "the path relative to the binary.");

int main(int argc, char** argv) {
  absl::SetProgramUsageMessage(argv[0]);
  absl::ParseCommandLine(argc, argv);

  const ghc::filesystem::path input_path(absl::GetFlag(FLAGS_input_path));
  const ghc::filesystem::path output_dir(absl::GetFlag(FLAGS_output_dir));
  const ghc::filesystem::path model_path =
      chromemedia::codec::GetCompleteArchitecturePath(
          absl::GetFlag(FLAGS_model_path));
  const bool enable_preprocessing =
      absl::GetFlag(FLAGS_enable_preprocessing);
  const bool enable_dtx = absl::GetFlag(FLAGS_enable_dtx);

  if (input_path.empty()) {
    LOG(ERROR) << "Flag --input_path not set.";
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
  const auto output_path =
      ghc::filesystem::path(output_dir) / input_path.stem().concat(".lyra");

  if (!chromemedia::codec::EncodeFile(input_path, output_path,
                                      enable_preprocessing, enable_dtx,
                                      model_path)) {
    LOG(ERROR) << "Failed to encode " << input_path;
    return -1;
  }
  return 0;
}
