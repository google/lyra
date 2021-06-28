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

#ifndef LYRA_CODEC_SPARSE_MATMUL_NUMERICS_TYPE_UTILS_H_
#define LYRA_CODEC_SPARSE_MATMUL_NUMERICS_TYPE_UTILS_H_

// A collection of useful utilities for determining types based on other types.

#include <type_traits>

#include "sparse_matmul/numerics/fixed_types.h"
#include "sparse_matmul/numerics/float16_types.h"

namespace csrblocksparse {

// Basic idea is that any two float types yield a float, fixed16 types
// yield a fixed32 with the exponent bits summed.  Other options are not
// allowed.
template <typename LhsType, typename RhsType, class Enable = void>
struct TypeOfProduct {};

template <typename LhsType, typename RhsType>
struct TypeOfProduct<
    LhsType, RhsType,
    typename std::enable_if<IsAnyFloatType<LhsType>::value &&
                            IsAnyFloatType<RhsType>::value>::type> {
  using type = float;
};

template <typename LhsType, typename RhsType>
struct TypeOfProduct<
    LhsType, RhsType,
    typename std::enable_if<IsFixed16Type<LhsType>::value &&
                            IsFixed16Type<RhsType>::value>::type> {
  static_assert(LhsType::kMantissaBits + RhsType::kMantissaBits < 31,
                "Sum of mantissa bits must not exceed 31.");
  using type = fixed32<31 - LhsType::kMantissaBits - RhsType::kMantissaBits>;
};

// Given a weight type T, determine what the RhsType should be for that type.
// bfloat16 / fp16 -> float; fixed16 = fixed16
template <typename T, class Enable = void>
struct RhsTypeIs {
  using type = float;
};

template <typename T>
struct RhsTypeIs<T, typename std::enable_if<IsFixed16Type<T>::value>::type> {
  using type = T;
};

template <typename T, class Enable = void>
struct MantissaBitsOf {
  // Although int types have zero mantissa bits, use 1 to avoid division by 0.
  static constexpr int value = 1;
};

template <typename T>
struct MantissaBitsOf<
    T, typename std::enable_if<IsFixedType<T>::value ||
                               IsCustomFloatType<T>::value>::type> {
 public:
  static constexpr int value = T::kMantissaBits;
};

template <typename T>
struct MantissaBitsOf<
    T, typename std::enable_if<std::is_floating_point<T>::value>::type> {
 public:
  // Ignoring the fact that doubles have more mantissa bits.
  static constexpr int value = 24;
};

}  // namespace csrblocksparse

#endif  // LYRA_CODEC_SPARSE_MATMUL_NUMERICS_TYPE_UTILS_H_
