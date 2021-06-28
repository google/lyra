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

#include "absl/types/span.h"
#include "audio/dsp/signal_vector_util.h"
#include "glog/logging.h"

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

int16_t UnitFloatToInt16Scalar(float unit_float) {
  // First, scale unit_float linearly to int16 ranges.
  // The unary negation is used here to scale by the negative min int16_t value,
  // which has a greater absolute value than the max.
  float int16_range_float =
      unit_float * (-std::numeric_limits<int16_t>().min());
  // If unit_float was outside the [-1, 1), clip to the min/max value.
  return ClipToInt16(int16_range_float);
}

std::vector<int16_t> UnitFloatToInt16(absl::Span<const float> input) {
  std::vector<int16_t> output;
  output.reserve(input.size());
  std::transform(input.begin(), input.end(), std::back_inserter(output),
                 UnitFloatToInt16Scalar);
  return output;
}

std::vector<float> Int16ToUnitFloat(absl::Span<const int16_t> input) {
  std::vector<float> output(input.size());
  std::transform(input.begin(), input.end(), output.begin(), [](int16_t x) {
    return -static_cast<float>(x) / std::numeric_limits<int16_t>().min();
  });
  return output;
}

}  // namespace codec
}  // namespace chromemedia
