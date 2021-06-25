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

#if defined __aarch64__
#include <arm_neon.h>
#endif
#if defined __AVX__ || defined __AVX2__
#include <immintrin.h>
#endif

#include <stdio.h>

#include <array>
#include <random>
#include <tuple>
#include <vector>

#include "gtest/gtest.h"
#include "sparse_matmul/numerics/fast_transcendentals.h"
#include "sparse_matmul/numerics/test_utils.h"

namespace csrblocksparse {

const float kExpFixedRelTolerance = .084f;

#ifdef SIGMOID_AS_TANH
#if defined FAST_TRANSCENDENTALS && defined ACCURATE_TRANSCENDENTAL_APPROX
const float kSigmoidRelTolerance = .093f;  // 9.3% relative
const float kSigmoidAbsTolerance = .0005f;
const float kSigmoidFixedRelTolerance = .093f;
const float kSigmoidFixedAbsTolerance = .0005f;
#elif defined FAST_TRANSCENDENTALS
const float kSigmoidRelTolerance = .09f;  // 9.0% relative
const float kSigmoidAbsTolerance = .003f;
const float kSigmoidFixedRelTolerance = .09f;
const float kSigmoidFixedAbsTolerance = .003f;
#endif
#elif defined FAST_TRANSCENDENTALS and defined ACCURATE_TRANSCENDENTAL_APPROX
const float kSigmoidRelTolerance = .102f;  // 10.2% relative
const float kSigmoidAbsTolerance = .0003f;
const float kSigmoidFixedRelTolerance = .102f;
const float kSigmoidFixedAbsTolerance = .0003f;
#elif defined FAST_TRANSCENDENTALS
const float kSigmoidRelTolerance = .09f;  // 9.0% relative
const float kSigmoidAbsTolerance = .006f;
const float kSigmoidFixedRelTolerance = .09f;
const float kSigmoidFixedAbsTolerance = .006f;
#else
const float kSigmoidRelTolerance = .0001f;
const float kSigmoidAbsTolerance = 1e-5f;
const float kSigmoidFixedRelTolerance = .001f;
const float kSigmoidFixedAbsTolerance = .001f;
#endif

#if (defined FAST_TRANSCENDENTALS && defined ACCURATE_TRANSCENDENTAL_APPROX || \
     defined FASTER_TRANSCENDENTALS)
const float kExpRelTolerance = .03f;    // 3% relative
const float kTanhRelTolerance = .006f;  // .6% relative
const float kTanhAbsTolerance = .0003f;
#elif defined FAST_TRANSCENDENTALS
const float kExpRelTolerance = .03f;    // 3% relative
const float kTanhRelTolerance = .091f;  // .91% relative
const float kTanhAbsTolerance = .00525f;
#else
const float kExpRelTolerance = .0001f;
const float kTanhRelTolerance = .0001f;
const float kTanhAbsTolerance = 1e-5f;
#endif

constexpr float kQuarticFloatExpRelTolerance = 8e-6f;
constexpr float kQuarticFloatExpTolerance = 9e-6f;
constexpr float kQuarticExpRelTolerance = 3e-5f;
constexpr float kQuarticExpTolerance = 6e-5f;
constexpr float kCubicExpRelTolerance = 6e-4f;
constexpr float kCubicExpTolerance = 2e-3f;
constexpr float kQuarticFloatTanhRelTolerance = 3e-5f;
constexpr float kQuarticFloatTanhTolerance = 3e-6f;
constexpr float kCubicTanhRelTolerance = 3e-3f;
constexpr float kCubicTanhTolerance = 3e-4f;
constexpr float kQuarticSigmoidRelTolerance = 3e-5f;
constexpr float kQuarticSigmoidTolerance = 7e-6f;
constexpr float kCubicSigmoidRelTolerance = 6e-4f;
constexpr float kCubicSigmoidTolerance = 2e-4f;
#ifdef __AVX2__
constexpr float kQuarticTanhRelTolerance = 1e-4f;
constexpr float kQuarticTanhTolerance = 2e-5f;
constexpr float kQuarticFloatSigmoidRelTolerance = 4e-6f;
constexpr float kQuarticFloatSigmoidTolerance = 1e-6f;
#endif  // __AVX2__

TEST(Transcendentals, Exp) {
  // 132 - 127 = 5, we check between -63.99... and 63.99...
  const int maxExponent = 132;
  const int minExponent = 0;
  float max_error = 0.f;
  constexpr int kExponentBits = 7;
  for (int s = 0; s < 2; ++s) {
    for (int e = minExponent; e < maxExponent; ++e) {
      // Don't check every mantissa for speed reasons.
      for (int m = 0; m < (1 << 23); m += (1 << 10)) {
        uint32_t int_val = s << 31 | e << 23 | m;
        float x;
        memcpy(&x, &int_val, sizeof(float));

        float exact_exp = expf(x);
        float approx_exp = csrblocksparse::fast_exp(x);
        float approx_exp_fixed = csrblocksparse::fast_exp<kExponentBits>(
            csrblocksparse::fixed32<kExponentBits>(x));

        float rel_diff = RelDiff(exact_exp, approx_exp);
        float rel_diff_fixed = RelDiff(exact_exp, approx_exp_fixed);
        max_error = std::max(max_error, rel_diff);
        EXPECT_LT(rel_diff, kExpRelTolerance)
            << exact_exp << " " << approx_exp << " " << x;
        EXPECT_LT(rel_diff_fixed, kExpRelTolerance)
            << exact_exp << " " << approx_exp << " " << x;
      }
    }
  }
}

TEST(Transcendentals, FixedExp) {
  const int maxExponent = 132;
  const int minExponent = 120;
  float max_error = 0.f;
  float max_abs_error = 0.f;
  for (int s = 0; s < 2; ++s) {
    for (int e = minExponent; e < maxExponent; ++e) {
      // Don't check every mantissa for speed reasons.
      for (int m = 0; m < (1 << 23); m += (1 << 10)) {
        uint32_t int_val = s << 31 | e << 23 | m;
        float x;
        memcpy(&x, &int_val, sizeof(float));

        float exact_exp = expf(x);
        float approx_exp =
            csrblocksparse::fast_exp_fixed(csrblocksparse::fixed32<16>(x));

        float rel_diff = RelDiff(exact_exp, approx_exp);
        float abs_diff = std::abs(exact_exp - approx_exp);
        max_error = std::max(max_error, rel_diff);
        max_abs_error = std::max(max_abs_error, abs_diff);
        EXPECT_LT(rel_diff, kExpFixedRelTolerance)
            << exact_exp << " " << approx_exp << " " << x;
      }
    }
  }
  LOG(INFO) << "Max relative exp error = " << max_error
            << ", abs=" << max_abs_error;
}

template <TranscendentalMode kOrder>
void TestExp(float abs_tolerance, float rel_tolerance) {
  constexpr int kMaxInput = 80 << 16;
  constexpr int kMinInput = -(80 << 16);
  constexpr int kExponentBits = 15;
  float max_error = 0.f;
  float max_abs_error = 0.f;
  for (int i = kMinInput; i <= kMaxInput; ++i) {
    csrblocksparse::fixed32<kExponentBits> fixed_int(i);
    float x = static_cast<float>(fixed_int);
    float exact_exp = expf(x);
    float approx_exp = fixed32_exp<kExponentBits, kOrder>(fixed_int);
    float diff = exact_exp - approx_exp;
    float abs_diff = std::abs(diff);
    float rel_diff = RelDiff(exact_exp, approx_exp);
    max_error = std::max(max_error, rel_diff);
    if (x <= 1.0f) {
      ASSERT_LT(abs_diff, abs_tolerance)
          << "x=" << x << ", target=" << exact_exp << ", aprx=" << approx_exp;
      max_abs_error = std::max(max_abs_error, abs_diff);
    }
    ASSERT_LT(rel_diff, rel_tolerance)
        << "x=" << x << ", target=" << exact_exp << ", aprx=" << approx_exp;
  }
  LOG(INFO) << "Max relative error = " << max_error
            << ", abs=" << max_abs_error;
}

TEST(Transcendentals, QuarticExp) {
  TestExp<TM_ORDER4_16BIT>(kQuarticFloatExpTolerance,
                           kQuarticFloatExpRelTolerance);
}

TEST(Transcendentals, CubicExp) {
  TestExp<TM_ORDER3_16BIT>(kCubicExpTolerance, kCubicExpRelTolerance);
}

template <TranscendentalMode kOrder>
void TestTanh(float abs_tolerance, float rel_tolerance) {
  constexpr int kMaxInput = (40 << 16);
  constexpr int kMinInput = -(40 << 16);
  constexpr int kExponentBits = 15;
  float max_error = 0.f;
  float max_abs_error = 0.f;
  for (int i = kMinInput; i <= kMaxInput; ++i) {
    csrblocksparse::fixed32<kExponentBits> fixed_int(i);
    float x = static_cast<float>(fixed_int);
    float exact_tanh = tanh(x);
    float approx_tanh = fixed32_tanh<kExponentBits, kOrder>(fixed_int);
    float diff = exact_tanh - approx_tanh;
    float abs_diff = std::abs(diff);
    float rel_diff = RelDiff(exact_tanh, approx_tanh);
    ASSERT_LT(abs_diff, abs_tolerance)
        << "x=" << x << ", target=" << exact_tanh << ", aprx=" << approx_tanh;
    max_abs_error = std::max(max_abs_error, abs_diff);
    max_error = std::max(max_error, rel_diff);
    ASSERT_LT(rel_diff, rel_tolerance)
        << "x=" << x << ", target=" << exact_tanh << ", aprx=" << approx_tanh;
  }
  LOG(INFO) << "Max relative error = " << max_error
            << ", abs=" << max_abs_error;
}

TEST(Transcendentals, QuarticTanh) {
  TestTanh<TM_ORDER4_16BIT>(kQuarticFloatTanhTolerance,
                            kQuarticFloatTanhRelTolerance);
}

TEST(Transcendentals, CubicTanh) {
  TestTanh<TM_ORDER3_16BIT>(kCubicTanhTolerance, kCubicTanhRelTolerance);
}

template <TranscendentalMode kOrder>
void TestSigmoid(float abs_tolerance, float rel_tolerance) {
  constexpr int kMaxInput = 80 << 16;
  constexpr int kMinInput = -(80 << 16);
  constexpr int kExponentBits = 15;
  float max_error = 0.f;
  float max_abs_error = 0.f;
  for (int i = kMinInput; i <= kMaxInput; ++i) {
    csrblocksparse::fixed32<kExponentBits> fixed_int(i);
    float x = static_cast<float>(fixed_int);
    float exact_sigmoid = 1.0f / (1.0f + exp(-x));
    float approx_sigmoid = fixed32_sigmoid<kExponentBits, kOrder>(fixed_int);
    float diff = exact_sigmoid - approx_sigmoid;
    float abs_diff = std::abs(diff);
    float rel_diff = RelDiff(exact_sigmoid, approx_sigmoid);
    max_error = std::max(max_error, rel_diff);
    ASSERT_LT(abs_diff, abs_tolerance)
        << "x=" << x << ", target=" << exact_sigmoid
        << ", aprx=" << approx_sigmoid;
    max_abs_error = std::max(max_abs_error, abs_diff);
    ASSERT_LT(rel_diff, rel_tolerance)
        << "x=" << x << ", target=" << exact_sigmoid
        << ", aprx=" << approx_sigmoid;
  }
  LOG(INFO) << "Max relative sigmoid error = " << max_error
            << ", abs=" << max_abs_error;
}

TEST(Transcendentals, QuarticSigmoidExp) {
  TestSigmoid<TM_ORDER4_16BIT>(kQuarticSigmoidTolerance,
                               kQuarticSigmoidRelTolerance);
}

TEST(Transcendentals, CubicSigmoidExp) {
  TestSigmoid<TM_ORDER3_16BIT>(kCubicSigmoidTolerance,
                               kCubicSigmoidRelTolerance);
}

TEST(Transcendentals, Sigmoid) {
  // 132 - 127 = 5, we check between -63.99... and 63.99...
  const int maxExponent = 132;
  const int minExponent = 0;
  // The mantissa bits must not exceed 23, so min exponent bits here is:
  // 31 - 23 = 8.
  constexpr int kExponentBits = 9;
  float max_error = 0.f;
  float max_abs_error = 0.f;
#if defined __aarch64__
  float max_vector_error = 0.f;
  float max_vector_abs_error = 0.f;
#endif
  for (int s = 0; s < 2; ++s) {
    for (int e = minExponent; e < maxExponent; ++e) {
      // Don't check every mantissa for speed reasons.
      for (int m = 0; m < (1 << 23); m += (1 << 10)) {
        uint32_t int_val = s << 31 | e << 23 | m;
        float x;
        memcpy(&x, &int_val, sizeof(float));

        float exact_sigmoid = 1. / (1. + expf(-x));
        float approx_sigmoid = csrblocksparse::fast_sigmoid(x);
        float approx_sigmoid_fixed =
            csrblocksparse::fast_sigmoid<kExponentBits>(
                csrblocksparse::fixed32<kExponentBits>(x));

        float rel_diff = RelDiff(exact_sigmoid, approx_sigmoid);
        float abs_diff = std::abs(exact_sigmoid - approx_sigmoid);
        float rel_diff_fixed = RelDiff(exact_sigmoid, approx_sigmoid_fixed);
        max_error = std::max(max_error, rel_diff);
        max_abs_error = std::max(max_abs_error, abs_diff);
        EXPECT_LT(rel_diff, kSigmoidRelTolerance)
            << exact_sigmoid << " " << approx_sigmoid << " " << x;
        EXPECT_NEAR(approx_sigmoid, exact_sigmoid, kSigmoidAbsTolerance) << x;

        EXPECT_LT(rel_diff_fixed, kSigmoidFixedRelTolerance)
            << exact_sigmoid << " " << approx_sigmoid_fixed << " " << x;
        EXPECT_NEAR(approx_sigmoid_fixed, exact_sigmoid,
                    kSigmoidFixedAbsTolerance)
            << x;
#if defined __aarch64__
        constexpr int kSIMD_WIDTH = 4;
        float approx_results[kSIMD_WIDTH];
        int32x4_t input =
            vdupq_n_s32(csrblocksparse::fixed32<kExponentBits>(x).raw_val());
        float32x4_t result = csrblocksparse::fast_sigmoid<kExponentBits>(input);
        vst1q_f32(approx_results, result);

        for (int i = 0; i < kSIMD_WIDTH; ++i) {
          float rel_diff = RelDiff(exact_sigmoid, approx_results[i]);
          float abs_diff = std::abs(exact_sigmoid - approx_results[i]);
          max_vector_error = std::max(max_vector_error, rel_diff);
          max_vector_abs_error = std::max(max_vector_abs_error, abs_diff);
          EXPECT_LT(rel_diff, kSigmoidRelTolerance)
              << exact_sigmoid << " " << approx_sigmoid << " " << x;
          EXPECT_NEAR(approx_sigmoid, exact_sigmoid, kSigmoidAbsTolerance) << x;
        }
#endif
      }
    }
  }
  LOG(INFO) << "Max relative error in float sigmoid=" << max_error;
  LOG(INFO) << "Max abs error in float sigmoid=" << max_abs_error;
#if defined __aarch64__
  LOG(INFO) << "Max relative vector error fixed sigmoid=" << max_vector_error;
  LOG(INFO) << "Max abs vector error fixed sigmoid=" << max_vector_abs_error;
#endif
}

TEST(Transcendentals, Tanh) {
  // 132 - 127 = 5, we check between -63.99... and 63.99...
  const int maxExponent = 132;
  const int minExponent = 0;
  float max_error = 0.f;
  float max_abs_error = 0.f;
  for (int s = 0; s < 2; ++s) {
    for (int e = minExponent; e < maxExponent; ++e) {
      // Don't check every mantissa for speed reasons.
      for (int m = 0; m < (1 << 23); m += (1 << 10)) {
        uint32_t int_val = s << 31 | e << 23 | m;
        float x;
        memcpy(&x, &int_val, sizeof(float));

        float exact_tanh = tanhf(x);
        float approx_tanh = csrblocksparse::fast_tanh(x);

        float rel_diff = RelDiff(exact_tanh, approx_tanh);
        float abs_diff = std::abs(exact_tanh - approx_tanh);
        max_error = std::max(rel_diff, max_error);
        max_abs_error = std::max(abs_diff, max_abs_error);

        EXPECT_LT(rel_diff, kTanhRelTolerance)
            << exact_tanh << " " << approx_tanh << " " << x;
        EXPECT_NEAR(approx_tanh, exact_tanh, kTanhAbsTolerance) << x;
      }
    }
  }
  LOG(INFO) << "Max relative error in float tanh=" << max_error;
  LOG(INFO) << "Max abs error in float tanh=" << max_abs_error;

  // tanh behavior is not identical across all lanes, so need to test
  // with some values in the linear region and some not.
#if defined __aarch64__
  float vals[4] = {-1.f, -.1f, .1f, 1.f};
  float exact_results[4];
  float approx_results[4];
  max_error = 0.f;
  max_abs_error = 0.f;

  float32x4_t input = vld1q_f32(vals);
  float32x4_t result = csrblocksparse::fast_tanh(input);
  vst1q_f32(approx_results, result);

  for (int i = 0; i < 4; ++i) {
    exact_results[i] = tanh(vals[i]);
    float rel_diff = RelDiff(exact_results[i], approx_results[i]);
    float abs_diff = std::abs(exact_results[i] - approx_results[i]);
    max_error = std::max(rel_diff, max_error);
    max_abs_error = std::max(abs_diff, max_abs_error);

    EXPECT_LT(rel_diff, kTanhRelTolerance)
        << exact_results[i] << " " << approx_results[i] << " " << vals[i];
    EXPECT_NEAR(approx_results[i], exact_results[i], kTanhAbsTolerance)
        << vals[i];
  }
  LOG(INFO) << "Max relative vector error in float tanh=" << max_error;
  LOG(INFO) << "Max abs vector error in float tanh=" << max_abs_error;
#endif
}

#if defined __AVX2__

constexpr int kSIMDSize = 8;
constexpr int kNumExpBitsIn = 10;
constexpr int kNumExpBitsOut = 5;

TEST(Transcendentals, TanhLut) {
  // Test every value in (-1, 1) for round-trip exactness.
  constexpr int kNumMantissaBitsIn = fixed32<kNumExpBitsIn>::kMantissaBits;
  constexpr int kNumMantissaBitsOut = fixed16<kNumExpBitsOut>::kMantissaBits;
  const int32_t* tanh_table = TanhTable(kNumMantissaBitsOut);
  float in_factor = static_cast<float>(1 << kNumMantissaBitsIn);
  float out_factor = static_cast<float>(1 << kNumMantissaBitsOut);
  for (int i = 1 - (1 << kNumMantissaBitsOut);
       i + kSIMDSize < (1 << kNumMantissaBitsOut); i += kSIMDSize) {
    int32_t inputs[kSIMDSize];
    int32_t outputs[kSIMDSize];
    int32_t target_outputs[kSIMDSize];
    for (int j = 0; j < kSIMDSize; ++j) {
      float target_tanh = (i + j) / out_factor;
      float x = atanhf(static_cast<float>(target_tanh));
      inputs[j] = static_cast<int>(x * in_factor);
      target_outputs[j] = i + j;
    }
    __m256i x_in = _mm256_loadu_si256(reinterpret_cast<__m256i*>(inputs));
    __m256i output =
        fixed32_tanh_fixed16<kNumMantissaBitsIn, kNumMantissaBitsOut>(
            tanh_table, x_in);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(outputs), output);
    for (int j = 0; j < kSIMDSize; ++j) {
      EXPECT_EQ(target_outputs[j], outputs[j]);
    }
  }
}

TEST(Transcendentals, SigmoidLut) {
  // Test every value in (-1, 1) for round-trip exactness.
  constexpr int kNumMantissaBitsIn = fixed32<kNumExpBitsIn>::kMantissaBits;
  constexpr int kNumMantissaBitsOut = fixed16<kNumExpBitsOut>::kMantissaBits;
  const int32_t* sigmoid_table = SigmoidTable(kNumMantissaBitsOut);
  float in_factor = static_cast<float>(1 << kNumMantissaBitsIn);
  float out_factor = static_cast<float>(1 << kNumMantissaBitsOut);
  for (int i = 1; i + kSIMDSize < (1 << kNumMantissaBitsOut); i += kSIMDSize) {
    int32_t inputs[kSIMDSize];
    int32_t outputs[kSIMDSize];
    int32_t target_outputs[kSIMDSize];
    for (int j = 0; j < kSIMDSize; ++j) {
      float target_sigmoid = (i + j) / out_factor;
      float x = 2.0f * atanhf(2.0f * static_cast<float>(target_sigmoid) - 1.0f);
      inputs[j] = static_cast<int>(x * in_factor);
      target_outputs[j] = i + j;
    }
    __m256i x_in = _mm256_loadu_si256(reinterpret_cast<__m256i*>(inputs));
    __m256i output =
        fixed32_sigmoid_fixed16<kNumMantissaBitsIn, kNumMantissaBitsOut>(
            sigmoid_table, x_in);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(outputs), output);
    for (int j = 0; j < kSIMDSize; ++j) {
      EXPECT_EQ(target_outputs[j], outputs[j]);
    }
  }
}

template <TranscendentalMode kOrder>
static void TestExpAVX2(float abs_tolerance, float rel_tolerance) {
  constexpr int kMantissaBits = 20;
  // Test every value in [-80, 80] and report the max error.
  constexpr int kMinInput = -(80 << kMantissaBits);
  constexpr int kMaxInput = 80 << kMantissaBits;
  constexpr int kNumInputs = kMaxInput - kMinInput;
  std::vector<int32> inputs(kNumInputs);
  std::vector<float> outputs(kNumInputs);
  std::vector<float> target_outputs(kNumInputs);
  for (int i = 0; i < inputs.size(); ++i) {
    csrblocksparse::fixed32<31 - kMantissaBits> fixed_int(i + kMinInput);
    float x = static_cast<float>(fixed_int);
    inputs[i] = fixed_int.raw_val();
    target_outputs[i] = expf(x);
  }
  absl::Time t_start = absl::Now();
  for (int i = 0; i + kSIMDSize * 2 <= kNumInputs; i += kSIMDSize * 2) {
    __m256i x0 =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(inputs.data() + i));
    __m256i x1 = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(inputs.data() + i + kSIMDSize));
    __m256 y0, y1;
    fixed32_exp_float<kMantissaBits, kOrder>(x0, x1, y0, y1);
    _mm256_storeu_ps(outputs.data() + i, y0);
    _mm256_storeu_ps(outputs.data() + i + kSIMDSize, y1);
  }
  LOG(INFO) << "Time=" << absl::ToDoubleMilliseconds(absl::Now() - t_start);
  float max_error = 0.f;
  float max_abs_error = 0.f;
  for (int i = 0; i < kNumInputs; ++i) {
    float diff = target_outputs[i] - outputs[i];
    float abs_diff = std::abs(diff);
    csrblocksparse::fixed32<31 - kMantissaBits> fixed_int(i + kMinInput);
    float x = static_cast<float>(fixed_int);
    float rel_diff = RelDiff(target_outputs[i], outputs[i]);
    max_error = std::max(max_error, rel_diff);
    if (x <= 1.0f) {
      ASSERT_LT(abs_diff, abs_tolerance)
          << "x=" << x << ", target=" << target_outputs[i]
          << ", result= " << outputs[i] << ", i=" << i;
      max_abs_error = std::max(max_abs_error, abs_diff);
    }
    ASSERT_LT(rel_diff, rel_tolerance)
        << "x=" << x << ", target=" << target_outputs[i]
        << ", result= " << outputs[i] << ", i=" << i;
  }
  LOG(INFO) << "Max relative error = " << max_error
            << ", abs=" << max_abs_error;
}

TEST(Transcendentals, QuarticFloatExpAVX2) {
  TestExpAVX2<TM_ORDER4_FLOAT>(kQuarticFloatExpTolerance,
                               kQuarticFloatExpRelTolerance);
}

TEST(Transcendentals, QuarticExpAVX2) {
  TestExpAVX2<TM_ORDER4_16BIT>(kQuarticExpTolerance, kQuarticExpRelTolerance);
}

TEST(Transcendentals, CubicExpAVX2) {
  TestExpAVX2<TM_ORDER3_16BIT>(kCubicExpTolerance, kCubicExpRelTolerance);
}

template <TranscendentalMode kOrder>
void TestTanhAVX2Float(float abs_tolerance, float rel_tolerance) {
  constexpr int kMantissaBits = 16;
  // Test every value in [-10, 10] and report the max error.
  constexpr int kMinInput = -(10 << kMantissaBits);
  constexpr int kMaxInput = 10 << kMantissaBits;
  constexpr int kNumInputs = kMaxInput - kMinInput;
  float max_error = 0.f;
  float max_abs_error = 0.f;
  std::vector<float> inputs(kNumInputs);
  std::vector<float> outputs(kNumInputs);
  std::vector<float> target_outputs(kNumInputs);
  for (int i = 0; i < inputs.size(); ++i) {
    csrblocksparse::fixed32<31 - kMantissaBits> fixed_int(i + kMinInput);
    float x = static_cast<float>(fixed_int);
    float exact = tanh(x);
    inputs[i] = static_cast<float>(fixed_int.raw_val());
    target_outputs[i] = exact;
  }
  absl::Time t_start = absl::Now();
  for (int i = 0; i + kSIMDSize * 2 <= inputs.size(); i += kSIMDSize * 2) {
    __m256 x0 = _mm256_loadu_ps(inputs.data() + i);
    __m256 x1 = _mm256_loadu_ps(inputs.data() + kSIMDSize + i);
    __m256 y0, y1;
    float_tanh_float<kMantissaBits, kOrder>(x0, x1, y0, y1);
    _mm256_storeu_ps(outputs.data() + i, y0);
    _mm256_storeu_ps(outputs.data() + i + kSIMDSize, y1);
  }
  LOG(INFO) << "Time=" << absl::ToDoubleMilliseconds(absl::Now() - t_start);
  float worst_abs_x = 0.0f, worst_rel_x = 0.0f;
  for (int i = 0; i < inputs.size(); ++i) {
    float diff = target_outputs[i] - outputs[i];
    float abs_diff = std::abs(diff);
    csrblocksparse::fixed32<31 - kMantissaBits> fixed_int(i + kMinInput);
    float x = static_cast<float>(fixed_int);
    ASSERT_LT(abs_diff, abs_tolerance)
        << "x=" << x << ", target=" << target_outputs[i]
        << ", aprx=" << outputs[i];
    if (abs_diff > max_abs_error) worst_abs_x = x;
    max_abs_error = std::max(max_abs_error, abs_diff);
    float rel_diff = 0.0f;
    rel_diff = RelDiff(target_outputs[i], outputs[i]);
    if (rel_diff > max_error) worst_rel_x = x;
    max_error = std::max(max_error, rel_diff);
    ASSERT_LT(rel_diff, rel_tolerance)
        << "x=" << x << ", target=" << target_outputs[i]
        << ", aprx=" << outputs[i];
  }
  LOG(INFO) << "Max relative error = " << max_error
            << ", abs=" << max_abs_error;
  LOG(INFO) << "Worst rel x = " << worst_rel_x << ", abs=" << worst_abs_x;
}

TEST(Transcendentals, QuarticTanhFloatAVX2Float) {
  TestTanhAVX2Float<TM_ORDER4_FLOAT>(kQuarticFloatTanhTolerance,
                                     kQuarticFloatTanhRelTolerance);
}

TEST(Transcendentals, QuarticTanhAVX2Float) {
  TestTanhAVX2Float<TM_ORDER4_16BIT>(kQuarticTanhTolerance,
                                     kQuarticTanhRelTolerance);
}

TEST(Transcendentals, CubicTanhAVX2Float) {
  TestTanhAVX2Float<TM_ORDER3_16BIT>(kCubicTanhTolerance,
                                     kCubicTanhRelTolerance);
}

template <TranscendentalMode kOrder>
void TestSigmoidAVX2Float(float abs_tolerance, float rel_tolerance) {
  constexpr int kMantissaBits = 20;
  // Test every value in [-20, 20] and report the max error.
  constexpr int kMaxInput = 20 << kMantissaBits;
  constexpr int kMinInput = -(20 << kMantissaBits);
  float max_error = 0.f;
  float max_abs_error = 0.f;
  std::vector<int32> inputs(kMaxInput - kMinInput);
  std::vector<float> outputs(kMaxInput - kMinInput);
  std::vector<float> target_outputs(kMaxInput - kMinInput);
  for (int i = 0; i < inputs.size(); ++i) {
    csrblocksparse::fixed32<31 - kMantissaBits> fixed_int(i + kMinInput);
    float x = static_cast<float>(fixed_int);
    float exact = 1.0f / (1.0f + expf(-x));
    inputs[i] = fixed_int.raw_val();
    target_outputs[i] = exact;
  }
  absl::Time t_start = absl::Now();
  for (int i = 0; i + kSIMDSize * 2 <= inputs.size(); i += kSIMDSize * 2) {
    __m256i x0 =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(inputs.data() + i));
    __m256i x1 = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(inputs.data() + i + kSIMDSize));
    __m256 y0 = _mm256_cvtepi32_ps(x0);
    __m256 y1 = _mm256_cvtepi32_ps(x1);
    float_sigmoid_float<kMantissaBits, kOrder>(y0, y1);
    _mm256_storeu_ps(outputs.data() + i, y0);
    _mm256_storeu_ps(outputs.data() + i + kSIMDSize, y1);
  }
  LOG(INFO) << "Time=" << absl::ToDoubleMilliseconds(absl::Now() - t_start);
  for (int i = 0; i < inputs.size(); ++i) {
    float diff = target_outputs[i] - outputs[i];
    float abs_diff = std::abs(diff);
    csrblocksparse::fixed32<31 - kMantissaBits> fixed_int(i + kMinInput);
    float x = static_cast<float>(fixed_int);
    float rel_diff = RelDiff(target_outputs[i], outputs[i]);
    max_error = std::max(max_error, rel_diff);
    ASSERT_LT(abs_diff, abs_tolerance)
        << "x=" << x << ", target=" << target_outputs[i]
        << ", aprx=" << outputs[i];
    max_abs_error = std::max(max_abs_error, abs_diff);
    ASSERT_LT(rel_diff, rel_tolerance)
        << "x=" << x << ", target=" << target_outputs[i]
        << ", aprx=" << outputs[i];
  }
  LOG(INFO) << "Max relative error = " << max_error
            << ", abs=" << max_abs_error;
}

TEST(Transcendentals, QuarticSigmoidFloatAVX2Float) {
  TestSigmoidAVX2Float<TM_ORDER4_FLOAT>(kQuarticFloatSigmoidTolerance,
                                        kQuarticFloatSigmoidRelTolerance);
}

TEST(Transcendentals, QuarticSigmoidAVX2Float) {
  TestSigmoidAVX2Float<TM_ORDER4_16BIT>(kQuarticSigmoidTolerance,
                                        kQuarticSigmoidRelTolerance);
}

TEST(Transcendentals, CubicSigmoidAVX2Float) {
  TestSigmoidAVX2Float<TM_ORDER3_16BIT>(kCubicSigmoidTolerance,
                                        kCubicSigmoidRelTolerance);
}
#endif  // __AVX2__

}  // namespace csrblocksparse
