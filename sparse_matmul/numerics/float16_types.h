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

#ifndef LYRA_CODEC_SPARSE_MATMUL_NUMERICS_FLOAT16_TYPES_H_
#define LYRA_CODEC_SPARSE_MATMUL_NUMERICS_FLOAT16_TYPES_H_

#include <cstdint>
#include <cstring>
#include <type_traits>

namespace csrblocksparse {

// Storage class for fp16 values, not meant to be used directly for computation.
// Used for converting to/from float32.
class fp16 {
 public:
  fp16() = default;
  explicit fp16(float x) : val_(float_to_fp16(x)) {}
  explicit fp16(uint16_t x) : val_(x) {}
  static constexpr int kMantissaBits = 11;

  explicit operator float() const { return fp16_to_float(val_); }

 private:
  inline float fp16_to_float(uint16_t as_int) const {
#if defined __aarch64__
    float x;
    float* x_ptr = &x;
    asm volatile(
        "dup v0.8h, %w[as_int]\n"
        "fcvtl v1.4s, v0.4h\n "
        "st1 {v1.s}[0], [%[x_ptr]]\n"
        :  // outputs
        :  // inputs
        [x_ptr] "r"(x_ptr),
        [as_int] "r"(as_int)
        :  // clobbers
        "cc", "memory", "v0", "v1");
    return x;
#else
    unsigned int sign_bit = (as_int & 0x8000) << 16;
    unsigned int exponent = as_int & 0x7c00;

    unsigned int mantissa;
    if (exponent == 0)
      mantissa = 0;
    else
      mantissa = ((as_int & 0x7fff) << 13) + 0x38000000;
    mantissa |= sign_bit;

    float x;
    memcpy(&x, &mantissa, sizeof(int));
    return x;
#endif  // defined __aarch64__
  }

  inline uint16_t float_to_fp16(float x) const {
#if defined __aarch64__
    uint16_t as_int;
    uint16_t* as_int_ptr = &as_int;
    asm volatile(
        "dup v0.4s, %w[x]\n"
        "fcvtn v1.4h, v0.4s\n"
        "st1 {v1.h}[0], [%[as_int_ptr]]\n"
        :  // outputs
        :  // inputs
        [as_int_ptr] "r"(as_int_ptr),
        [x] "r"(x)
        :  // clobbers
        "cc", "memory", "v0", "v1");
    return as_int;
#else
    unsigned int x_int;
    memcpy(&x_int, &x, sizeof(int));

    unsigned int sign_bit = (x_int & 0x80000000) >> 16;
    unsigned int exponent = x_int & 0x7f800000;

    unsigned int mantissa;
    if (exponent < 0x38800000) {  // exponent too small or denormal
      mantissa = 0;
    } else if (exponent > 0x8e000000) {
      mantissa = 0x7bff;  // exponent too big, inf
    } else {
      mantissa = ((x_int & 0x7fffffff) >> 13) - 0x1c000;
    }

    mantissa |= sign_bit;

    return static_cast<uint16_t>(mantissa & 0xFFFF);
#endif
  }

  uint16_t val_;
};

// Storage class for bfloat16 values, not meant to be used directly for
// computation.  Used for converting to/from float32.
class bfloat16 {
 public:
  bfloat16() = default;
  explicit bfloat16(float x) : val_(float_to_bfloat16(x)) {}
  explicit bfloat16(uint16_t x) : val_(x) {}
  static constexpr int kMantissaBits = 7;

  explicit operator float() const { return bfloat16_to_float(val_); }

 private:
  inline uint16_t float_to_bfloat16(float x) const {
    uint32_t as_int;
    std::memcpy(&as_int, &x, sizeof(float));
    return as_int >> 16;
  }

  inline float bfloat16_to_float(uint32_t as_int) const {
    as_int <<= 16;
    float x;
    std::memcpy(&x, &as_int, sizeof(float));
    return x;
  }

  uint16_t val_;
};

template <typename T>
struct IsCustomFloatType
    : std::integral_constant<bool, std::is_same<T, bfloat16>::value ||
                                       std::is_same<T, fp16>::value> {};
template <typename T>
struct IsAnyFloatType
    : std::integral_constant<bool, std::is_floating_point<T>::value ||
                                       IsCustomFloatType<T>::value> {};

}  // namespace csrblocksparse

#endif  // LYRA_CODEC_SPARSE_MATMUL_NUMERICS_FLOAT16_TYPES_H_
