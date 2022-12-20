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

#include "lyra/lyra_config.h"

#include <vector>

#include "absl/strings/string_view.h"

namespace chromemedia {
namespace codec {

// The Lyra version is |kVersionMajor|.|kVersionMinor|.|kVersionMicro|
// The version is not used internally, but clients may use it to configure
// behavior, such as checking for version bumps that break the bitstream.
// The major version should be bumped for major architectural changes.
const int kVersionMajor = 1;
// The minor version needs to be increased every time a new version requires a
// simultaneous change in code and weights or if the bit stream is modified. The
// |identifier| field needs to be set in lyra_config.textproto to match this.
const int kVersionMinor = 3;
// The micro version is for other things like a release of bugfixes.
const int kVersionMicro = 2;

const int kNumFeatures = 64;
const int kNumMelBins = 160;
const int kNumChannels = 1;
const int kOverlapFactor = 2;

// LINT.IfChange
const int kNumHeaderBits = 0;
const int kFrameRate = 50;
const std::vector<int>& GetSupportedQuantizedBits() {
  static const std::vector<int>* const supported_quantization_bits =
      new std::vector<int>{64, 120, 184};
  return *supported_quantization_bits;
}
// LINT.ThenChange(
// lyra_components.cc,
// lyra_encoder.h,
// residual_vector_quantizer.h,
// )

std::vector<absl::string_view> GetAssets() {
  return std::vector<absl::string_view>{"quantizer.tflite", "lyragan.tflite",
                                        "soundstream_encoder.tflite"};
}

}  // namespace codec
}  // namespace chromemedia
