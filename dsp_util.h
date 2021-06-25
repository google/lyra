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

#ifndef LYRA_CODEC_DSP_UTIL_H_
#define LYRA_CODEC_DSP_UTIL_H_

#include <cstdint>

#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "sparse_matmul/sparse_matmul.h"

namespace chromemedia {
namespace codec {

// The log-spectral distance (LSD) is a distance measure (expressed in dB)
// between two spectra.
absl::optional<float> LogSpectralDistance(
    const absl::Span<const float> first_log_spectrum,
    const absl::Span<const float> second_log_spectrum);

// Clip values above max value or below min value for int16_t.
// The quantization scheme uses native c rounding (non-centered, decimal
// truncation)
int16_t ClipToInt16(float value);

// Converts from a unit-float to a 16-bit integer.
// If |unit_float| is in the [-1, 1) interval it will scale linearly to the
// int16_t limits.  Values outside the interval are clipped to the limits.
// The clipping, rounding, and quantization follows ClipToInt16().
int16_t UnitFloatToInt16Scalar(float unit_float);

// Converts from a Span of unit-floats to a vector of 16-bit integers.
std::vector<int16_t> UnitFloatToInt16(absl::Span<const float> input);

// Converts from a Span of 16-bit integers to a vector of unit-floats.
std::vector<float> Int16ToUnitFloat(absl::Span<const int16_t> input);

#if defined __aarch64__

// We do not provide fixed16 to fixed32 casting as there is no use case so far.
template <typename InputType, typename OutputType>
struct ShouldEnableGenericCast
    : std::integral_constant<
          bool, (!(csrblocksparse::IsFixedType<InputType>::value) ||
                 !(csrblocksparse::IsFixedType<OutputType>::value) ||
                 (csrblocksparse::IsFixed16Type<InputType>::value &&
                  csrblocksparse::IsFixed32Type<OutputType>::value))> {};

template <typename InputType, typename OutputType>
typename std::enable_if<csrblocksparse::IsFixed16Type<InputType>::value &&
                        csrblocksparse::IsFixed16Type<OutputType>::value>::type
CastVector(int start, int end, const InputType* input, OutputType* output) {
  constexpr int kShiftAmount =
      OutputType::kExponentBits - InputType::kExponentBits;
  for (int i = start; i < end; i += 8) {
    int16x8_t input_int16 =
        vld1q_s16(reinterpret_cast<const int16_t*>(input + i));
    int16x8_t output_int16;
    if constexpr (kShiftAmount > 0) {
      output_int16 = vshrq_n_s16(input_int16, kShiftAmount);
    } else if constexpr (kShiftAmount < 0) {
      output_int16 = vqshlq_n_s16(input_int16, -kShiftAmount);
    } else {
      output_int16 = input_int16;
    }
    vst1q_s16(reinterpret_cast<int16_t*>(output + i), output_int16);
  }
}

template <typename InputType, typename OutputType>
typename std::enable_if<csrblocksparse::IsFixed32Type<InputType>::value &&
                        csrblocksparse::IsFixed16Type<OutputType>::value>::type
CastVector(int start, int end, const InputType* input, OutputType* output) {
  constexpr int kShiftAmount =
      16 + OutputType::kExponentBits - InputType::kExponentBits;
  for (int i = start; i < end; i += 4) {
    int32x4_t input_int32 =
        vld1q_s32(reinterpret_cast<const int32_t*>(input + i));
    int16x4_t output_int16;
    if constexpr (kShiftAmount < 0) {
      output_int16 = vqmovn_s32(input_int32);
      output_int16 = vqshl_n_s16(output_int16, -kShiftAmount);
    } else if constexpr (kShiftAmount == 0) {
      output_int16 = vqmovn_s32(input_int32);
    } else if constexpr (kShiftAmount <= 16) {
      output_int16 = vqshrn_n_s32(input_int32, kShiftAmount);
    } else {
      // Perform two stages of shifting because there is no one intrinsic that
      // can shift more than 16 bits at once.
      output_int16 = vqshrn_n_s32(input_int32, 16);
      output_int16 = vshr_n_s16(output_int16, kShiftAmount - 16);
    }
    vst1_s16(reinterpret_cast<int16_t*>(output + i), output_int16);
  }
}

template <typename InputType, typename OutputType>
typename std::enable_if<csrblocksparse::IsFixed32Type<InputType>::value &&
                        csrblocksparse::IsFixed32Type<OutputType>::value>::type
CastVector(int start, int end, const InputType* input, OutputType* output) {
  constexpr int kShiftAmount =
      OutputType::kExponentBits - InputType::kExponentBits;
  for (int i = start; i < end; i += 4) {
    int32x4_t input_int32 =
        vld1q_s32(reinterpret_cast<const int32_t*>(input + i));
    int32x4_t output_int32;
    if constexpr (kShiftAmount > 0) {
      output_int32 = vshrq_n_s32(input_int32, kShiftAmount);
    } else if constexpr (kShiftAmount < 0) {
      output_int32 = vqshlq_n_s32(input_int32, -kShiftAmount);
    } else {
      output_int32 = input_int32;
    }
    vst1q_s32(reinterpret_cast<int32_t*>(output + i), output_int32);
  }
}

#else  // defined __aarch64__

template <typename InputType, typename OutputType>
struct ShouldEnableGenericCast : std::true_type {};

#endif  // defined __aarch64__

template <typename InputType, typename OutputType>
typename std::enable_if<
    ShouldEnableGenericCast<InputType, OutputType>::value>::type
CastVector(int start, int end, const InputType* input, OutputType* output) {
  std::transform(input + start, input + end, output + start, [](InputType x) {
    return static_cast<OutputType>(static_cast<float>(x));
  });
}

}  // namespace codec
}  // namespace chromemedia

#endif  // LYRA_CODEC_DSP_UTIL_H_
