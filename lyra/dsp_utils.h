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

#ifndef LYRA_DSP_UTILS_H_
#define LYRA_DSP_UTILS_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <limits>
#include <optional>
#include <type_traits>
#include <vector>

#include "absl/types/span.h"

namespace chromemedia {
namespace codec {

// The log-spectral distance (LSD) is a distance measure (expressed in dB)
// between two spectra.
std::optional<float> LogSpectralDistance(
    const absl::Span<const float> first_log_spectrum,
    const absl::Span<const float> second_log_spectrum);

// Given the source and target sample rate, this method converts the number of
// samples from the former to the latter.
inline int ConvertNumSamplesBetweenSampleRate(int source_num_samples,
                                              int source_sample_rate,
                                              int target_sample_rate) {
  return static_cast<int>(std::ceil(static_cast<float>(source_num_samples) *
                                    static_cast<float>(target_sample_rate) /
                                    static_cast<float>(source_sample_rate)));
}

// Clip values above max value or below min value for int16_t.
// The quantization scheme uses native c rounding (non-centered, decimal
// truncation)
template <typename T>
typename std::enable_if<std::is_floating_point<T>::value, int16_t>::type
ClipToInt16Scalar(T unit_value) {
  unit_value =
      std::max(unit_value, static_cast<T>(std::numeric_limits<int16_t>::min()));
  return std::min(unit_value,
                  static_cast<T>(std::numeric_limits<int16_t>::max()));
}

// Clips a vector of unit-floats or unit-doubles to a vector of 16-bit
// integers. Does not perform scaling.
template <typename T>
typename std::enable_if<std::is_floating_point<T>::value,
                        std::vector<int16_t>>::type
ClipToInt16(const absl::Span<const T> input) {
  std::vector<int16_t> output;
  output.reserve(input.size());
  std::transform(input.begin(), input.end(), std::back_inserter(output),
                 ClipToInt16Scalar<T>);
  return output;
}

// Converts from a unit-float or unit-double to a 16-bit integer.
// If |value| is in the [-1, 1) interval it will scale linearly to the
// int16_t limits.  Values outside the interval are clipped to the limits.
// The clipping, rounding, and quantization follows ClipToInt16Scalar().
template <typename T>
typename std::enable_if<std::is_floating_point<T>::value, int16_t>::type
UnitToInt16Scalar(T value) {
  // First, scale |value| linearly to int16 ranges.
  // The unary negation is used here to scale by the negative min int16_t value,
  // which has a greater absolute value than the max.
  T int16_range_value = value * (-std::numeric_limits<int16_t>().min());
  // If value was outside the [-1, 1), clip to the min/max value.
  return ClipToInt16Scalar(int16_range_value);
}

// Converts from a vector of unit-floats or unit-doubles to a vector of 16-bit
// integers.
template <typename T>
typename std::enable_if<std::is_floating_point<T>::value,
                        std::vector<int16_t>>::type
UnitToInt16(const absl::Span<const T> input) {
  std::vector<int16_t> output;
  output.reserve(input.size());
  std::transform(input.begin(), input.end(), std::back_inserter(output),
                 UnitToInt16Scalar<T>);
  return output;
}

// Converts from a 16-bit integers to a unit-floats or unit-doubles.
template <typename T>
typename std::enable_if<std::is_floating_point<T>::value, T>::type
Int16ToUnitScalar(int16_t integer) {
  return -static_cast<T>(integer) / std::numeric_limits<int16_t>().min();
}

// Converts from a vector of 16-bit integers to a vector of unit-floats or
// unit-doubles.
template <typename T>
typename std::enable_if<std::is_floating_point<T>::value, std::vector<T>>::type
Int16ToUnit(const absl::Span<const int16_t> input) {
  std::vector<T> output;
  output.reserve(input.size());
  std::transform(input.begin(), input.end(), std::back_inserter(output),
                 Int16ToUnitScalar<T>);
  return output;
}

}  // namespace codec
}  // namespace chromemedia

#endif  // LYRA_DSP_UTILS_H_
