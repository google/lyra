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

#ifndef LYRA_CODEC_SPARSE_MATMUL_NUMERICS_FIXED_TYPES_H_
#define LYRA_CODEC_SPARSE_MATMUL_NUMERICS_FIXED_TYPES_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <type_traits>

#include "glog/logging.h"

namespace csrblocksparse {

// Useful for meta-programming and determining if a type is a fixed point type
class fixed_type {};
class fixed16_type : fixed_type {};
class fixed32_type : fixed_type {};

// Storage class for 16-bit fixed point values, not meant to be used directly
// for computation. Used for storage and converting to/from float32.
// N = 16 - 1 - |ExponentBits|.
// range = [-2^|ExponentBits|, 2^|ExponentBits|), increment = 2^-N.
template <int ExponentBits>
class fixed16 : fixed16_type {
  static_assert(ExponentBits >= 0 && ExponentBits < 16,
                "ExponentBits must be in"
                " the interval [0, 15]");

 public:
  static constexpr int kExponentBits = ExponentBits;
  static constexpr int kMantissaBits = 16 - ExponentBits - 1;

  fixed16() = default;
  explicit fixed16(float x) : val_(float_to_fixed16(x)) {}
  explicit fixed16(int16_t x) : val_(x) {}

  explicit operator float() const { return fixed16_to_float(val_); }

  int raw_val() const { return val_; }

 private:
  inline float fixed16_to_float(int16_t x) const {
    return static_cast<float>(x) / (1 << kMantissaBits);
  }

  // Conversion clips to the representable range.
  inline int16_t float_to_fixed16(float x) const {
    float fval = std::round(x * static_cast<float>(1 << kMantissaBits));
    const float max_bound = std::numeric_limits<int16_t>::max();
    const float min_bound = std::numeric_limits<int16_t>::min();
    auto val =
        static_cast<int16_t>(std::max(std::min(fval, max_bound), min_bound));
    LOG_IF(INFO, fval > max_bound || fval < min_bound)
        << "Conversion clipping: " << x << " to " << fixed16_to_float(val);
    return val;
  }

  int16_t val_;
};

// Storage class for 32-bit fixed point values, not meant to be used directly
// for computation. Used for storage and converting to/from float32.
// N = 32 - 1 - |ExponentBits|.
// range = [-2^|ExponentBits|, 2^|ExponentBits|), increment = 2^-N.
template <int ExponentBits>
class fixed32 : fixed32_type {
  static_assert(ExponentBits >= 0 && ExponentBits < 32,
                "ExponentBits must be in"
                " the interval [0, 31]");

 public:
  static constexpr int kExponentBits = ExponentBits;
  static constexpr int kMantissaBits = 32 - ExponentBits - 1;

  fixed32() = default;
  explicit fixed32(float x) : val_(float_to_fixed32(x)) {}
  explicit fixed32(int32_t x) : val_(x) {}

  explicit operator float() const { return fixed32_to_float(val_); }

  int raw_val() const { return val_; }

 private:
  inline float fixed32_to_float(int32_t x) const {
    return static_cast<float>(x) / (1LL << kMantissaBits);
  }

  // Conversion clips to the representable range.
  inline int32_t float_to_fixed32(float x) const {
    float fval = std::round(x * static_cast<float>(1LL << kMantissaBits));
    const int32_t max_bound = std::numeric_limits<int32_t>::max();
    const int32_t min_bound = std::numeric_limits<int32_t>::min();
    int32_t val = fval >= static_cast<float>(max_bound)
                      ? max_bound
                      : (fval < static_cast<float>(min_bound)
                             ? min_bound
                             : static_cast<int32_t>(fval));

    LOG_IF(INFO, fval >= max_bound || fval < min_bound)
        << "Conversion clipping: " << x << " to " << fixed32_to_float(val);
    return val;
  }

  int32_t val_;
};

template <typename T>
struct IsFixed16Type
    : std::integral_constant<bool, std::is_base_of<fixed16_type, T>::value> {};

template <typename T>
struct IsFixed32Type
    : std::integral_constant<bool, std::is_base_of<fixed32_type, T>::value> {};

template <typename T>
struct IsFixedType : std::integral_constant<bool, IsFixed16Type<T>::value ||
                                                      IsFixed32Type<T>::value> {
};

}  // namespace csrblocksparse

#endif  // LYRA_CODEC_SPARSE_MATMUL_NUMERICS_FIXED_TYPES_H_
