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

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "benchmark_decode_lib.h"

ABSL_FLAG(int, num_cond_vectors, 2000,
          "The number of conditioning vectors to feed to the conditioning "
          "stack / network. "
          "Equivalent to the number of calls to Precompute and Run.");

ABSL_FLAG(
    std::string, model_path, "wavegru",
    "Path to directory containing model weights and quant files. For mobile "
    "this is the absolute path, like '/sdcard/wavegru/'. For desktop this is "
    "the path relative to the binary.");

int main(int argc, char** argv) {
  absl::SetProgramUsageMessage(argv[0]);
  absl::ParseCommandLine(argc, argv);

  return chromemedia::codec::benchmark_decode(
      absl::GetFlag(FLAGS_num_cond_vectors), absl::GetFlag(FLAGS_model_path));
}
