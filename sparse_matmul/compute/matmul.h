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

#ifndef LYRA_CODEC_SPARSE_MATMUL_COMPUTE_MATMUL_H_
#define LYRA_CODEC_SPARSE_MATMUL_COMPUTE_MATMUL_H_

#include <cstdint>
#include <vector>

#include "absl/time/time.h"
#include "sparse_matmul/compute/matmul_fixed_avx2.h"
#include "sparse_matmul/compute/matmul_generic.h"
#include "sparse_matmul/numerics/fixed_types.h"
#include "sparse_matmul/numerics/type_utils.h"
#if defined(__x86_64__) || defined(__i386__) || defined(_WIN32)
#include <cpuid.h>
#endif

namespace csrblocksparse {

// The number of elements in a block.
constexpr int kBlockSize = 4;

// Base class for Matmul containing the members that are non type-specicfic.
class MatmulBase {
 public:
  // Constructor initializes the flags that determine which implementation to
  // use at run-time, constrained by both compiler flags and cpuid.
  MatmulBase() {
#if defined(__x86_64__) || defined(__i386__) || defined(_WIN32)
    // Code tested to work on Linux systems and multiple Android emulators.
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx) != 0) {
      using_avx_ = (ecx & bit_AVX) != 0;
      if (using_avx_) {
        __get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx);
        using_avx2_ = (ebx & bit_AVX2) != 0;
        using_avx512_ = (ebx & bit_AVX512F) != 0 && (ebx & bit_AVX512DQ) &&
                        (ebx & bit_AVX512BW) != 0;
        VLOG(2) << "avx2 flag=" << using_avx2_ << " 512=" << using_avx512_;
      } else {
        LOG(ERROR) << "AVX not found at all!";
      }
    }
#else
    using_aarch64_ = true;
#endif
  }

 protected:
  // Flags that define what (runtime) architectures are available. Flags that
  // are set are limited by both the compiler flags and runtime environment.
  bool using_avx512_ = false;
  bool using_avx2_ = false;
  bool using_avx_ = false;
  bool using_aarch64_ = false;
};

// The master template is really a catch-all for the unimplmented cases to
// report an error.
template <typename WeightType, typename RhsType>
class Matmul : public MatmulBase {
 public:
  // Sparse inputs, outputs replicated strided for each thread.
  template <typename OutType>
  void MatVec4x4(const WeightType* weights, const RhsType* rhs,
                 const typename TypeOfProduct<WeightType, RhsType>::type* bias,
                 const int32_t* nnz_per_row, const int16_t* rhs_indices,
                 int start_row, int end_row, bool relu, int replicas,
                 int stride, OutType* output) {
    // The specializations should take care of every real case.
    CHECK(false) << "Unsupported combination of types used!";
  }
  template <typename OutType>
  void MatVec8x4(const WeightType* weights, const RhsType* rhs,
                 const typename TypeOfProduct<WeightType, RhsType>::type* bias,
                 const int32_t* nnz_per_row, const int16_t* rhs_indices,
                 int start_row, int end_row, bool relu, int replicas,
                 int stride, OutType* output) {
    // The specializations should take care of every real case.
    CHECK(false) << "Unsupported combination of types used!";
  }
};

// Full specialization for float.
template <>
class Matmul<float, float> : public MatmulBase {
 public:
  void MatVec4x4(const float* weights, const float* rhs, const float* bias,
                 const int32_t* nnz_per_row, const int16_t* rhs_indices,
                 int start_row, int end_row, bool relu, int replicas,
                 int stride, float* output) {
    detail::MatVecFloatGeneric(weights, rhs, bias, nnz_per_row, rhs_indices,
                               start_row, end_row, /*block_height=*/4,
                               /*block_width=*/4, relu, replicas, stride,
                               output);
  }
  void MatVec8x4(const float* weights, const float* rhs, const float* bias,
                 const int32_t* nnz_per_row, const int16_t* rhs_indices,
                 int start_row, int end_row, bool relu, int replicas,
                 int stride, float* output) {
    detail::MatVecFloatGeneric(weights, rhs, bias, nnz_per_row, rhs_indices,
                               start_row, end_row, /*block_height=*/8,
                               /*block_width=*/4, relu, replicas, stride,
                               output);
  }
};

// Partial specialization for fixed types. Covers fixed16xfixed16 = OutType,
// where OutType should be fixed16 or fixed32. The mantissa bits don't have
// to match.
template <int WeightBits, int RhsBits>
class Matmul<fixed16<WeightBits>, fixed16<RhsBits>> : public MatmulBase {
 public:
  using WeightType = fixed16<WeightBits>;
  using RhsType = fixed16<RhsBits>;

  template <typename OutType>
  void MatVec4x4(const int16_t* weights, const int16_t* rhs,
                 const int32_t* bias, const int32_t* nnz_per_row,
                 const int16_t* rhs_indices, int start_row, int end_row,
                 bool relu, int replicas, int stride, OutType* output) {
    constexpr int kShiftAmount =
        TypeOfProduct<WeightType, RhsType>::type::kMantissaBits -
        OutType::kMantissaBits;
    static_assert(kShiftAmount >= 0,
                  "OutType must not have more mantissa bits than inputs");
#if defined __AVX2__
    CHECK(using_avx2_) << "Compiled for AVX2, but cpu flag not set!";
    if (sizeof(*output) == 4) {
      int32_t* out32 = reinterpret_cast<int32_t*>(output);
      detail::MatVec4x4FixedAVX2(weights, rhs, bias, nnz_per_row, rhs_indices,
                                 start_row, end_row, relu, kShiftAmount,
                                 replicas, stride, out32);
    } else {
      int16_t* out16 = reinterpret_cast<int16_t*>(output);
      detail::MatVec4x4FixedAVX2(weights, rhs, bias, nnz_per_row, rhs_indices,
                                 start_row, end_row, relu, kShiftAmount,
                                 replicas, stride, out16);
    }
#elif defined __aarch64__
    if (using_aarch64_) {
      LOG(FATAL) << "Fixed16 MatVec4x4 not yet implemented!";
    }

#else
    detail::MatVecFixedGeneric(weights, rhs, bias, nnz_per_row, rhs_indices,
                               start_row, end_row, /*block_height=*/4,
                               /*block_width=*/4, relu, sizeof(*output),
                               kShiftAmount, replicas, stride, output);
#endif  // __AVX2__
  }

  template <typename OutType>
  void MatVec8x4(const int16_t* weights, const int16_t* rhs,
                 const int32_t* bias, const int32_t* nnz_per_row,
                 const int16_t* rhs_indices, int start_row, int end_row,
                 bool relu, int replicas, int stride, OutType* output) {
    constexpr int kShiftAmount =
        TypeOfProduct<WeightType, RhsType>::type::kMantissaBits -
        OutType::kMantissaBits;
    static_assert(kShiftAmount >= 0,
                  "OutType must not have more mantissa bits than inputs");
#if defined __AVX2__
    CHECK(replicas == 1 && sizeof(*output) == 4)
        << "Only replicas == 1 and fixed32 output are implemented for AVX2!";
    CHECK(using_avx2_) << "Compiled for AVX2, but cpu flag not set!";
    int32_t* out32 = reinterpret_cast<int32_t*>(output);
    detail::MatVec8x4FixedAVX2(weights, rhs, bias, nnz_per_row, rhs_indices,
                               start_row, end_row, relu, kShiftAmount, out32);
#elif defined __aarch64__
    if (using_aarch64_) {
      LOG(FATAL) << "Fixed16 MatVec8x4 not yet implemented!";
    }
#else
    detail::MatVecFixedGeneric(weights, rhs, bias, nnz_per_row, rhs_indices,
                               start_row, end_row, /*block_height=*/8,
                               /*block_width=*/4, relu, sizeof(*output),
                               kShiftAmount, replicas, stride, output);
#endif  // __AVX2__
  }
};

}  // namespace csrblocksparse

#endif  // LYRA_CODEC_SPARSE_MATMUL_COMPUTE_MATMUL_H_
