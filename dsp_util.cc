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

#include "dsp_util.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>

#include "audio/dsp/signal_vector_util.h"
#include "glog/logging.h"
#include "absl/types/span.h"

namespace chromemedia {
namespace codec {

absl::optional<float> LogSpectralDistance(
    const absl::Span<const float> first_log_spectrum,
    const absl::Span<const float> second_log_spectrum) {
  const int num_features = first_log_spectrum.size();
  if (num_features != second_log_spectrum.size()) {
    LOG(ERROR) << "Spectrum sizes are not equal.";
    return absl::nullopt;
  }
  float log_spectral_distance = 0.f;
  for (int i = 0; i < num_features; ++i) {
    log_spectral_distance +=
        audio_dsp::Square(first_log_spectrum[i] - second_log_spectrum[i]);
  }
  return 10 * std::sqrt(log_spectral_distance / num_features);
}

int16_t ClipToInt16(float value) {
  value =
      std::max(value, static_cast<float>(std::numeric_limits<int16_t>::min()));
  return std::min(value,
                  static_cast<float>(std::numeric_limits<int16_t>::max()));
}

}  // namespace codec
}  // namespace chromemedia
