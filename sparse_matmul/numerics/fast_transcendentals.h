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

#ifndef LYRA_CODEC_SPARSE_MATMUL_NUMERICS_FAST_TRANSCENDENTALS_H_
#define LYRA_CODEC_SPARSE_MATMUL_NUMERICS_FAST_TRANSCENDENTALS_H_

#include <cstdint>
#if defined __ARM_NEON || defined __aarch64__
#include <arm_neon.h>
#else
#include <algorithm>
#endif
#if defined __AVX__ || defined __AVX2__
#include <immintrin.h>
#endif
#include <math.h>

#include "sparse_matmul/numerics/fixed_types.h"
#include "sparse_matmul/numerics/type_utils.h"

namespace csrblocksparse {

// The input to exp is clipped to bounds that prevent overflow/underflow in a
// 32 bit float representation. e^80 ~ 6e34, which is close to maxfloat.
constexpr float kMaxExpInput = 80.f;
constexpr int kMaxExpInputInt = static_cast<int>(kMaxExpInput);
constexpr float kMinExpInput = -80.f;
// tanh(9) ~ 0.99999997, which cannot be resolved from 1 in a float32.
constexpr float kMaxTanhInput = 9.f;
constexpr float kMinTanhInput = -9.f;
// sigmoid(18) ~ 0.999999985, which cannot be resolved from 1 in a float32.
constexpr float kMaxSigmoidInput = 18.f;
constexpr float kMinSigmoidInput = -18.f;
// kAConstant ~= 2^23 / ln 2
constexpr uint32_t kAConstant = 0x4b38aa3b;
// kBConstant ~= (127 << 23) - 366000
constexpr uint32_t kBConstant = 0x4e7de9a9;
// Coefficients of the rational approximation to tanh.
// Coefficients of the numerator polynomial (odd).
constexpr float kTanhAlpha1 = 4.89352455891786e-03;
constexpr float kTanhAlpha3 = 6.37261928875436e-04;
constexpr float kTanhAlpha5 = 1.48572235717979e-05;
constexpr float kTanhAlpha7 = 5.12229709037114e-08;
constexpr float kTanhAlpha9 = -8.60467152213735e-11;
constexpr float kTanhAlpha11 = 2.00018790482477e-13;
constexpr float kTanhAlpha13 = -2.76076847742355e-16;
// The monomial coefficients of the denominator polynomial (even).
constexpr float kTanhBeta0 = 4.89352518554385e-03;
constexpr float kTanhBeta2 = 2.26843463243900e-03;
constexpr float kTanhBeta4 = 1.18534705686654e-04;
constexpr float kTanhBeta6 = 1.19825839466702e-06;

// Coefficients of the rational approximation to sigmoid.
// Coefficients of the numerator polynomial (odd).
constexpr float kSigmoidAlpha1 = 2.48287947061529e-01;
constexpr float kSigmoidAlpha3 = 8.51377133304701e-03;
constexpr float kSigmoidAlpha5 = 6.08574864600143e-05;
constexpr float kSigmoidAlpha7 = 1.15627324459942e-07;
constexpr float kSigmoidAlpha9 = 4.37031012579801e-11;

// The monomial coefficients of the denominator polynomial (even).
constexpr float kSigmoidBeta0 = 9.93151921023180e-01;
constexpr float kSigmoidBeta2 = 1.16817656904453e-01;
constexpr float kSigmoidBeta4 = 1.70198817374094e-03;
constexpr float kSigmoidBeta6 = 6.29106785017040e-06;
constexpr float kSigmoidBeta8 = 5.76102136993427e-09;
constexpr float kSigmoidBeta10 = 6.10247389755681e-13;

// x is the first term of the Taylor series approximation of tanh near 0 and
// because the leading error term of tanh(x) - x is O(x^3), it is good for a
// wide interval, use it in this region where the other approximation is
// inaccurate. tanh(x) = x - x^3 / 3 + 2x^5 / 15 - 17x^7 / 315 + ...
// Similarly for sigmoid where the first term is .25x
constexpr float kTanhLinearRegion = .15f;
constexpr float kSigmoidLinearRegion = .75f;

// Maximum shift factor for 1/log 2 to keep it inside int32.
constexpr int kMaxLog2Shift = 30;
static const int kLogFactor = static_cast<int>((1 << kMaxLog2Shift) / log(2.f));
static const float kOneOverLog2 = 1.0f / log(2.f);
// Number of real mantissa bits in IEEE float32.
constexpr int kFloatMantissaBits = 23;
// Offset to correct the exponent value in the resulting float.
constexpr int kFloatExponentOffset = 127 << kFloatMantissaBits;
// Mask for mantissa.
constexpr int kFloatMantissaMask = (1 << kFloatMantissaBits) - 1;
// Mask for exponent;
constexpr int kFloatExponentMask = (-1) ^ kFloatMantissaMask;

// ========== COMMON DOCUMENTATION FOR THE FLOATING EXPONENT TRICK ============
// Summary: Use the exponent-mantissa representation of a floating point number
// to give exponentiation of 2 for free. If we desire f(z) = e^z = 2^(x+n), (for
// some fixed-point z expressed as an integer with imaginary binary point within
// it) then we have to compute x+n = z / ln 2 and then splitting x+n into
// n = int(x+n) and x = fract(x+n) in [0, 1), we can use n and 2^x as the
// exponent and mantissa of a floating point number, and that float is equal to
// e^z. For original reference see:
// http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.9.4508&rep=rep1&type=pdf
// Important detail:
// IEEE floats are stored normalized, ie 1.bbbbbbb... x 2^exponent. The leading
// 1 bit is not actually stored, (as it is always 1), providing an extra bit of
// precision.
// Since 2^0=1 and 2^1=2, we can treat the problem as 2^x = 1 + u and we thus
// need a mapping x in [0, 1) -> u in [0, 1) and the 1 + is provided by the
// representation.
// In the original paper cited above, the mapping is u = x - c, where c is set
// to minimize the average error. The function to compute exp(x) this way is
// incredibly simple and computationally cheap, but not very accurate.
// Fortunately, the problem has been reduced to u = 2^x - 1 over [0, 1) for
// which it is far easier to construct accurate approximations with small
// polynomials than a full range exp(x), and this is what the cubic and quartic
// versions below do. An important feature of these functions is that they
// constrain the solution to be exact at 0 and 1 so there is continuity at each
// integer boundary where we wrap from 1 to 0 and increment the power of 2.

// Coefficients for quartic representation of 2^x - 1 for x on [0,1).
// The quartic representation is 2^x - 1 ~ x - x(1-x)(ax^2 + bx + c), hence the
// coefficients of a quadratic are all that is required.
// Coefficients came from numerical experiments.
constexpr float kExpQuarticFactor2 = 0.0135302434f;
constexpr float kExpQuarticFactor1 = 0.0656107542f;
constexpr float kExpQuarticFactor0 = 0.306963906f;
// Coefficients for cubic representation of 2^x - 1 for x on [0,1]
// The cubic representation is 2^x - 1 ~ x - x(1-x)(mx + c), hence the
// coefficients of a linear function are all that is required.
// Coefficients came from numerical experiments.
constexpr float kExpCubicFactor1 = 0.0780252018f;
constexpr float kExpCubicFactor0 = 0.304684167f;
// Coefficients are optimized to minimize the absolute error on
// tanh = (e^2x - 1) / (e^2x + 1) instead of on pure e^x.

// Enum that determines how a transcendental is computed.
enum TranscendentalMode {
  // Cubic using 16 bit integer arithmetic.
  TM_ORDER3_16BIT,
  // Quartic using 16 bit integer arithmetic.
  TM_ORDER4_16BIT,
  // Quartic using 32 bit float arithmetic.
  TM_ORDER4_FLOAT,
};

inline int FloatAsInt16(float x) {
  return static_cast<int>(x * (1 << 15) + 0.5f);
}

inline int FloatAsInt32(float x) {
  return static_cast<int>(x * (1 << 30) + 0.5f);
}

#if defined __ARM_NEON || defined __aarch64__

constexpr int kMaxSigmoidInputInt = static_cast<int>(kMaxSigmoidInput);

// Computes and returns 2^(x>>23) ie 2^u where x = u << 23 bits.
// Uses the quartic floating point exponent trick, see COMMON DOCUMENTATION FOR
// THE FLOATING EXPONENT TRICK above for details.
// Returns the true value, ie not scaled.
inline float32x4_t float32_pow2(float32x4_t x) {
  // The input is already shifted left by 23 bits, so when we convert to int,
  // the bottom 23 bits are the fractional part, and the top bits are the
  // integer part. We want to compute a function of the fractional part, so
  // we will mask it off and manipulate it.
  int32x4_t exp_int_x = vcvtq_s32_f32(x);
  // Mask to allow conversion of just the fractional part of x to fixed16<0>.
  int32x4_t mantissa_mask16 = vdupq_n_s32(0x7fff00);
  // Mask to allow conversion of just the fractional part of x to fixed32<1>.
  int32x4_t mantissa_mask32 = vdupq_n_s32(0x7fffff);
  // Narrowing shift to convert to fixed16<0>.
  int16x4_t x_16 = vshrn_n_s32(vandq_s32(mantissa_mask16, exp_int_x), 8);
  // Shift to convert to fixed32<1>.
  int32x4_t x_32 = vshlq_n_s32(vandq_s32(mantissa_mask32, exp_int_x), 7);
  // Compute the polynomial x(x - 1)(ax^2 + bx + c) of the fractional part.
  // Ordering these lines carefully makes it faster, as some of the multiply
  // operations can pipeline instead of waiting for the previous result.
  int32x4_t x_squared = vmull_s16(x_16, x_16);
  int16x4_t b = vdup_n_s16(FloatAsInt16(kExpQuarticFactor1));
  int32x4_t c = vdupq_n_s32(FloatAsInt32(kExpQuarticFactor0));
  int32x4_t bx_plus_c = vmlal_s16(c, b, x_16);
  int16x4_t a = vdup_n_s16(FloatAsInt16(kExpQuarticFactor2));
  // Finish the quadratic: result = ax^2 + bx + c.
  int32x4_t result = vmlal_s16(bx_plus_c, a, vshrn_n_s32(x_squared, 15));
  int32x4_t x_squared_minus_x = vsubq_s32(x_squared, x_32);

  // Multiply by x^2 - x.
  result = vqrdmulhq_s32(result, x_squared_minus_x);
  // Shift back to mantissa position. vqrdmulhq_s32 took 2x 30-mantissa bit
  // inputs, made 60-mantissa bit result, doubled it to 61 bits, then discarded
  // the bottom 32 making 29, so shift right 6 to get 23.
  result = vshrq_n_s32(result, 6);
  // Add the constant to normalize the exponent for IEEE format.
  int32x4_t exp_offset = vdupq_n_s32(kFloatExponentOffset);
  exp_int_x = vaddq_s32(exp_int_x, exp_offset);
  exp_int_x = vaddq_s32(exp_int_x, result);
  // Cast back to float, as we just computed the exponent and mantissa and
  // assembled them in IEEE format.
  return vreinterpretq_f32_s32(exp_int_x);
}

// Scaled float to float exp approximation, using a quartic refinement of
// the exponent trick. See COMMON DOCUMENTATION FOR THE FLOATING EXPONENT TRICK
// above for details. Input is a fixed32<31 - mantissa_bits> that has been
// converted to a float without any further shifting. MUST HAVE ALREADY BEEN
// CLIPPED to a suitable range for exp!
// Returns a vector of standard unscaled floats.
inline float32x4_t fixed32_exp_float_preclipped(const int mantissa_bits,
                                                float32x4_t x) {
  // Divide by log 2 to convert problem to 2^x, and scale to match the
  // mantissa bits required by IEEE floats.
  // This is the shift of the FP mantissa relative to the input mantissa.
  const int kXShift = kFloatMantissaBits - mantissa_bits;
  const float kLogFactor = static_cast<float>(1 << kXShift);
  float32x4_t factor = vdupq_n_f32(kLogFactor * kOneOverLog2);
  float32x4_t y = vmulq_f32(x, factor);
  // Now compute 2^x.
  return float32_pow2(y);
}

// uses trick that 2^x can be computed by shifting integer into the
// exponent, see the following reference for a derivation using double:
// goo.gl/aUVTK3
// Input x is clamped to [-64, 64], even infinity and NaN.
// Accurate to within 3% relative across the entire range.
// Fully pipelined throughput is about 10 cycles per fast_exp call.
inline float32x4_t fast_exp(float32x4_t x) {
#if defined FAST_TRANSCENDENTALS && __ARM_ARCH >= 800
  // Uses vcvtnq_s32_f32, not available on ARM v7 NEON.

  // Load A and B, which are defined as integers into float registers.
  float32x4_t A = vreinterpretq_f32_u32(vdupq_n_u32(kAConstant));
  float32x4_t res = vreinterpretq_f32_u32(vdupq_n_u32(kBConstant));

  // Make sure x within the allowed range.
  x = vminq_f32(x, vdupq_n_f32(kMaxExpInput));
  x = vmaxq_f32(x, vdupq_n_f32(kMinExpInput));

  // res = A * x + B.
  // This shifts x into the exponent field and adds the bias.
  res = vmlaq_f32(res, A, x);

  // Convert back to an integer, this is what uses the floating point
  // unit to compute 2^x.
  int32x4_t x_int = vcvtnq_s32_f32(res);

  return vreinterpretq_f32_s32(x_int);
#else
  float32x4_t return_val = vdupq_n_f32(0.f);

  float exponent = expf(vgetq_lane_f32(x, 0));
  return_val = vld1q_lane_f32(&exponent, return_val, 0);

  exponent = expf(vgetq_lane_f32(x, 1));
  return_val = vld1q_lane_f32(&exponent, return_val, 1);
  exponent = expf(vgetq_lane_f32(x, 2));
  return_val = vld1q_lane_f32(&exponent, return_val, 2);
  exponent = expf(vgetq_lane_f32(x, 3));
  return_val = vld1q_lane_f32(&exponent, return_val, 3);

  return return_val;
#endif  // FAST_TRANSCENDENTALS
}

// This version does a conversion of the input to floating point, then calls
// the floating point fast_exp function.  There is another version
// fast_exp_fixed, that never does a conversion and is less accurate, but much
// faster.
template <int ExponentBits>
inline float32x4_t fast_exp(int32x4_t x) {
  return fast_exp(vcvtq_n_f32_s32(x, 31 - ExponentBits));
}

// Performs an exp estimate without doing any floating point operations. The
// result is a floating point number.  See scalar version for an explanation.
template <int ExponentBits>
inline float32x4_t fast_exp_fixed(int32x4_t x) {
  static_assert(ExponentBits > 8, "Must have more than 8 ExponentBits");
  constexpr int kA = 1.4426950408889634 * (1 << (ExponentBits - 8));
  constexpr int kB = (127 << 23) - 366000;

  constexpr int maxInput = 80 << (31 - ExponentBits);
  constexpr int minInput = -maxInput;

  int32x4_t A = vdupq_n_s32(kA);
  int32x4_t res = vdupq_n_s32(kB);

  // Make sure x within the allowed range.
  x = vminq_s32(x, vdupq_n_s32(maxInput));
  x = vmaxq_s32(x, vdupq_n_s32(minInput));

  // res = A * x + B.
  // This shifts x into the exponent field and adds the bias.
  res = vmlaq_s32(res, A, x);

  return vreinterpretq_f32_s32(res);
}

// fast_exp_norange_check uses vcvtnq_s32_f32, not available on ARM v7 NEON.
#if __ARM_ARCH >= 800
namespace detail {
// tanh can do range check once.
// Input x is clamped to [-64, 64], even infinity and NaN.
inline float32x4_t fast_exp_norange_check(float32x4_t x) {
  float32x4_t A = vreinterpretq_f32_u32(vdupq_n_u32(kAConstant));
  float32x4_t res = vreinterpretq_f32_u32(vdupq_n_u32(kBConstant));

  res = vmlaq_f32(res, A, x);

  int32x4_t x_int = vcvtnq_s32_f32(res);

  return vreinterpretq_f32_s32(x_int);
}

}  // namespace detail
#endif  // __ARM_ARCH >= 800

// Clips float input to [-kLimit,kLimit].
inline float32x4_t ClipToFloatBounds(const float kLimit, const float32x4_t x) {
  // Clip to the input bounds for this approximation.
  float32x4_t clip_limit = vdupq_n_f32(kLimit);
  float32x4_t clipped_x = vminq_f32(x, clip_limit);
  clip_limit = vnegq_f32(clip_limit);
  return vmaxq_f32(clipped_x, clip_limit);
}

inline float32x4_t float_tanh_float(const float32x4_t& x) {
  float32x4_t clipped_x = ClipToFloatBounds(kMaxTanhInput, x);
  // Divide by log 2 to convert problem to 2^x, double (as we need exp(2x)) and
  // scale to the mantissa bits required by float32_pow2 all in one multiply.
  // Add one to double the input.
  const float kLogFactor = static_cast<float>(1 << (kFloatMantissaBits + 1));
  float32x4_t factor = vdupq_n_f32(kLogFactor * kOneOverLog2);
  clipped_x = vmulq_f32(clipped_x, factor);
  // Now compute 2^x.
  float32x4_t exp_result = float32_pow2(clipped_x);
  // Now compute tanh using (e^2x - 1) / (e^2x + 1).
  float32x4_t one = vdupq_n_f32(1.0f);
  float32x4_t numerator = vsubq_f32(exp_result, one);
  float32x4_t denominator = vaddq_f32(exp_result, one);
  float32x4_t recp = vrecpeq_f32(denominator);
  // Newton-Raphson iteration, accuracy is important for audio quality
  recp = vmulq_f32(recp, vrecpsq_f32(recp, denominator));
  recp = vmulq_f32(recp, numerator);
  // Compute 3rd-order Taylor tanh ~ x - x^3/3 for high accuracy and thus low
  // relative error close to 0.
  float32x4_t third = vdupq_n_f32(1.0f / 3.0f);
  float32x4_t taylor = vmulq_f32(x, x);
  taylor = vmulq_f32(taylor, x);
  taylor = vmulq_f32(taylor, third);
  taylor = vsubq_f32(x, taylor);
  // Test |x| <= 1/9, roughly where the errors cross over, without needing yet
  // another constant.
  float32x4_t ninth = vmulq_f32(third, third);
  uint32x4_t cmp_results = vcaleq_f32(x, ninth);
  return vbslq_f32(cmp_results, taylor, recp);
}

// Calculates (exp(x) - exp(-x)) / (exp(x) + exp(-x)).
// Input x is clamped to [-9, 9], even infinity and NaN.
// See test program for bounds.  Throughput of FAST is 334 Mega/sec,
// throughput of accurate is 232 Mega/sec.
inline float32x4_t fast_tanh(float32x4_t x) {
#if defined FASTER_TRANSCENDENTALS
  return float_tanh_float(x);
#elif defined ACCURATE_TRANSCENDENTAL_APPROX && defined FAST_TRANSCENDENTALS
  x = vminq_f32(x, vdupq_n_f32(kMaxTanhInput));
  x = vmaxq_f32(x, vdupq_n_f32(kMinTanhInput));

  // The monomial coefficients of the numerator polynomial (odd).
  const float32x4_t alpha_1 = vdupq_n_f32(kTanhAlpha1);
  const float32x4_t alpha_3 = vdupq_n_f32(kTanhAlpha3);
  const float32x4_t alpha_5 = vdupq_n_f32(kTanhAlpha5);
  const float32x4_t alpha_7 = vdupq_n_f32(kTanhAlpha7);
  const float32x4_t alpha_9 = vdupq_n_f32(kTanhAlpha9);
  const float32x4_t alpha_11 = vdupq_n_f32(kTanhAlpha11);
  const float32x4_t alpha_13 = vdupq_n_f32(kTanhAlpha13);

  // The monomial coefficients of the denominator polynomial (even).
  const float32x4_t beta_0 = vdupq_n_f32(kTanhBeta0);
  const float32x4_t beta_2 = vdupq_n_f32(kTanhBeta2);
  const float32x4_t beta_4 = vdupq_n_f32(kTanhBeta4);
  const float32x4_t beta_6 = vdupq_n_f32(kTanhBeta6);

  // Since the polynomials are odd/even, we need x^2.
  const float32x4_t x2 = vmulq_f32(x, x);

  // Evaluate the numerator polynomial |p|.
  float32x4_t p = vmlaq_f32(alpha_11, x2, alpha_13);
  p = vmlaq_f32(alpha_9, x2, p);
  p = vmlaq_f32(alpha_7, x2, p);
  p = vmlaq_f32(alpha_5, x2, p);
  p = vmlaq_f32(alpha_3, x2, p);
  p = vmlaq_f32(alpha_1, x2, p);
  p = vmulq_f32(x, p);

  // Evaluate the denominator polynomial p.
  float32x4_t q = vmlaq_f32(beta_4, x2, beta_6);
  q = vmlaq_f32(beta_2, x2, q);
  q = vmlaq_f32(beta_0, x2, q);

  // Divide the numerator by the denominator.
  float32x4_t recp = vrecpeq_f32(q);
  recp = vmulq_f32(recp, vrecpsq_f32(recp, q));
  return vmulq_f32(p, recp);
#elif defined FAST_TRANSCENDENTALS && __ARM_ARCH >= 800
  // Uses vcvtnq_s32_f32, not available on ARM v7 NEON.

  x = vminq_f32(x, vdupq_n_f32(kMaxTanhInput));
  x = vmaxq_f32(x, vdupq_n_f32(kMinTanhInput));
  float32x4_t exp_est = detail::fast_exp_norange_check(x);
  float32x4_t neg_exp_est = detail::fast_exp_norange_check(-x);

  // If we're in the linear region.
  // caleq = compare absolute <=
  uint32x4_t cmp_results = vcaleq_f32(x, vdupq_n_f32(kTanhLinearRegion));

  float32x4_t diff = vsubq_f32(exp_est, neg_exp_est);
  float32x4_t sum = vaddq_f32(exp_est, neg_exp_est);
  float32x4_t recp = vrecpeq_f32(sum);
  recp = vmulq_f32(recp, vrecpsq_f32(recp, sum));
  float32x4_t tanh_estimate = vmulq_f32(diff, recp);

  // Based on comparison, possibly copy x through instead of calculated value.
  // TODO(b/191497441): Is the compiler generating VBIT or VBSL ? VBIT is one
  // cycle and VBSL is two... documentation suggests it can do either.
  return vbslq_f32(cmp_results, x, tanh_estimate);
#else
  float32x4_t return_val = vdupq_n_f32(0.f);

  float tanh_value = tanhf(vgetq_lane_f32(x, 0));
  return_val = vld1q_lane_f32(&tanh_value, return_val, 0);
  tanh_value = tanhf(vgetq_lane_f32(x, 1));
  return_val = vld1q_lane_f32(&tanh_value, return_val, 1);
  tanh_value = tanhf(vgetq_lane_f32(x, 2));
  return_val = vld1q_lane_f32(&tanh_value, return_val, 2);
  tanh_value = tanhf(vgetq_lane_f32(x, 3));
  return_val = vld1q_lane_f32(&tanh_value, return_val, 3);

  return return_val;
#endif  // FAST_TRANSCENDENTALS
}

// Input x is clamped to [-18, 18], even infinity and NaN.
// See tests for error bounds.  Using SIGMOID_AS_TANH with
// ACCURATE_TRANSCENDENTAL_APPROX is both faster and more accurate.  Using
// SIGMOID_AS_TANH with just FAST is slower, but more accurate.
// SIGMOID_AS_TANH, ACCURATE is 205 Mega/sec
// SIGMOID_AS_TANH, FAST is 290 Mega/sec
// FAST is 340 Mega/sec
inline float32x4_t fast_sigmoid(float32x4_t x) {
#ifdef SIGMOID_AS_TANH
  float32x4_t half = vdupq_n_f32(0.5f);
  return vmlaq_f32(half, half, fast_tanh(vmulq_f32(half, x)));
#else  // SIGMOID_AS_TANH
#if defined FAST_TRANSCENDENTALS && defined ACCURATE_TRANSCENDENTAL_APPROX
  x = vminq_f32(x, vdupq_n_f32(kMaxSigmoidInput));
  x = vmaxq_f32(x, vdupq_n_f32(kMinSigmoidInput));

  // The monomial coefficients of the numerator polynomial (odd).
  const float32x4_t alpha_1 = vdupq_n_f32(kSigmoidAlpha1);
  const float32x4_t alpha_3 = vdupq_n_f32(kSigmoidAlpha3);
  const float32x4_t alpha_5 = vdupq_n_f32(kSigmoidAlpha5);
  const float32x4_t alpha_7 = vdupq_n_f32(kSigmoidAlpha7);
  const float32x4_t alpha_9 = vdupq_n_f32(kSigmoidAlpha9);

  // The monomial coefficients of the denominator polynomial (even).
  const float32x4_t beta_0 = vdupq_n_f32(kSigmoidBeta0);
  const float32x4_t beta_2 = vdupq_n_f32(kSigmoidBeta2);
  const float32x4_t beta_4 = vdupq_n_f32(kSigmoidBeta4);
  const float32x4_t beta_6 = vdupq_n_f32(kSigmoidBeta6);
  const float32x4_t beta_8 = vdupq_n_f32(kSigmoidBeta8);
  const float32x4_t beta_10 = vdupq_n_f32(kSigmoidBeta10);

  // Since the polynomials are odd/even, we need x^2.
  const float32x4_t x2 = vmulq_f32(x, x);

  // Evaluate the numerator polynomial p.
  float32x4_t p = vmlaq_f32(alpha_7, x2, alpha_9);
  p = vmlaq_f32(alpha_5, x2, p);
  p = vmlaq_f32(alpha_3, x2, p);
  p = vmlaq_f32(alpha_1, x2, p);
  p = vmulq_f32(x, p);

  // Evaluate the denominator polynomial p.
  float32x4_t q = vmlaq_f32(beta_8, x2, beta_10);
  q = vmlaq_f32(beta_6, x2, q);
  q = vmlaq_f32(beta_4, x2, q);
  q = vmlaq_f32(beta_2, x2, q);
  q = vmlaq_f32(beta_0, x2, q);

  // Divide the numerator by the denominator.
  float32x4_t recp = vrecpeq_f32(q);
  recp = vmulq_f32(recp, vrecpsq_f32(recp, q));
  return vmlaq_f32(vdupq_n_f32(0.5f), p, recp);
#elif defined FAST_TRANSCENDENTALS
  float32x4_t denom = vaddq_f32(fast_exp(vnegq_f32(x)), vdupq_n_f32(1.f));

  float32x4_t recp = vrecpeq_f32(denom);
  // Newton-Raphson iteration, accuracy is important for audio quality.
  recp = vmulq_f32(recp, vrecpsq_f32(recp, denom));
  float32x4_t half = vdupq_n_f32(0.5f);
  float32x4_t quarter = vdupq_n_f32(0.245f);
  float32x4_t linear_approx = vmlaq_f32(half, quarter, x);
  uint32x4_t cmp_results = vcaleq_f32(x, vdupq_n_f32(kSigmoidLinearRegion));

  return vbslq_f32(cmp_results, linear_approx, recp);
#else
  float32x4_t return_val = vdupq_n_f32(0.f);

  float result = 1.f / (1.f + expf(-vgetq_lane_f32(x, 0)));
  return_val = vld1q_lane_f32(&result, return_val, 0);
  result = 1.f / (1.f + expf(-vgetq_lane_f32(x, 1)));
  return_val = vld1q_lane_f32(&result, return_val, 1);
  result = 1.f / (1.f + expf(-vgetq_lane_f32(x, 2)));
  return_val = vld1q_lane_f32(&result, return_val, 2);
  result = 1.f / (1.f + expf(-vgetq_lane_f32(x, 3)));
  return_val = vld1q_lane_f32(&result, return_val, 3);

  return return_val;
#endif  // FAST_TRANSCENDENTALS
#endif  // SIGMOID_AS_TANH
}

// Scalar implementations, mainly useful for testing.
inline float fast_exp(float x) {
  return vgetq_lane_f32(fast_exp(vdupq_n_f32(x)), 0);
}

template <int ExponentBits>
inline float fast_exp(fixed32<ExponentBits> x) {
  return vgetq_lane_f32(fast_exp<ExponentBits>(vdupq_n_s32(x.raw_val())), 0);
}

// Returns the exponent of a fixed point number in floating point without ever
// doing any conversions.  Less accurate than the version that does conversions,
// but still accurate to within 4% relative for x < 16.
template <int ExponentBits>
inline float fast_exp_fixed(fixed32<ExponentBits> x) {
  return vgetq_lane_f32(fast_exp_fixed<ExponentBits>(vdupq_n_s32(x.raw_val())),
                        0);
}

inline float fast_sigmoid(float x) {
  return vgetq_lane_f32(fast_sigmoid(vdupq_n_f32(x)), 0);
}

inline float fast_tanh(float x) {
  return vgetq_lane_f32(fast_tanh(vdupq_n_f32(x)), 0);
}

// Clips integer input to [-|kLimit|, |kLimit|].
// Input: register containins 4x fixed32 with mantissa_bits.
// Output: register containing 4x fixed32 limited to
// [-|kLimit| << |mantissa_bits|, |kLimit| << |mantissa_bits|].
template <int kLimit>
inline int32x4_t ClipToBounds(const int mantissa_bits, const int32x4_t x) {
  // Clip to the input bounds for this approximation.
  int32x4_t clip_limit = vdupq_n_s32(-(kLimit << mantissa_bits));
  int32x4_t clipped_x = vmaxq_s32(x, clip_limit);
  clip_limit = vnegq_s32(clip_limit);
  return vminq_s32(clipped_x, clip_limit);
}

// Fixed32 sigmoid approximation via a quadratic refinement of the exponent
// trick.
// Input: Register containing 4x fixed32 with |mantissa_bits|.
// Output: Register containing 4x float results.
inline float32x4_t fixed32_sigmoid_float(const int mantissa_bits,
                                         const int32x4_t x) {
  int32x4_t input = vnegq_s32(x);
  float32x4_t y =
      vcvtq_f32_s32(ClipToBounds<kMaxSigmoidInputInt>(mantissa_bits, input));
  y = fixed32_exp_float_preclipped(mantissa_bits, y);
  float32x4_t one = vdupq_n_f32(1.0f);
  // Approximate reciprocal is not accurate enough - use full division.
  float32x4_t denom = vaddq_f32(y, one);
  float32x4_t recp = vrecpeq_f32(denom);
  // Newton-Raphson iteration, accuracy is important for audio quality
  recp = vmulq_f32(recp, vrecpsq_f32(recp, denom));
  return recp;
}

template <int ExponentBits>
inline float32x4_t fast_sigmoid(int32x4_t x) {
#if defined FASTER_TRANSCENDENTALS
  // Computation will fail to produce the right result if the input mantissa
  // bits exceeds the number in a float.
  static_assert(kFloatMantissaBits >= fixed32<ExponentBits>::kMantissaBits,
                "Mantissa bits must be at most 23!");
  return fixed32_sigmoid_float(fixed32<ExponentBits>::kMantissaBits, x);
#else
  return fast_sigmoid(vcvtq_n_f32_s32(x, fixed32<ExponentBits>::kMantissaBits));
#endif  // FASTER_TRANSCENDENTALS
}

template <int ExponentBits>
inline float fast_sigmoid(fixed32<ExponentBits> x) {
  return vgetq_lane_f32(fast_sigmoid<ExponentBits>(vdupq_n_s32(x.raw_val())),
                        0);
}

#else  // defined __ARM_NEON || defined __aarch64__

inline float fast_exp(float x) {
#ifdef FAST_TRANSCENDENTALS
  if (isnan(x)) return 0.0f;
  x = std::max(std::min(x, kMaxExpInput), kMinExpInput);
  float AConstant, BConstant;
  memcpy(&AConstant, &kAConstant, sizeof(int));
  memcpy(&BConstant, &kBConstant, sizeof(int));
  float y = x * AConstant + BConstant;
  int x_int = static_cast<int>(y);
  float ret;
  memcpy(&ret, &x_int, sizeof(float));
  return ret;
#else
  return expf(x);
#endif  // FAST_TRANSCENDENTALS
}

template <int ExponentBits>
inline float fast_exp(fixed32<ExponentBits> x) {
  return fast_exp(static_cast<float>(x));
}

template <int ExponentBits>
inline float fast_exp_fixed(fixed32<ExponentBits> x) {
  static_assert(ExponentBits > 8, "Must have more than 8 ExponentBits");
  int matched_decimal =
      std::max(std::min(x.raw_val(), (80 << (31 - ExponentBits))),
               -(80 << (31 - ExponentBits)));
  // Convert 1 / log(2) to 16-bit fixed point with 1 exponent bit
  // (1 / log(2)) * (1 << 14), but then right shift by the appropriate amount to
  // line the decimal point up with the 32-bit float representation.
  // (MantissaBits of x) + (MantissaBits of constant) = 23
  // 23 - (MantissaBits of x) = MantissaBits of constant
  // 23 - (31 - ExponentBits of x) = ...
  // (ExponentBits of x - 8) = MantissaBits of constant
  const int16_t A = (1.f / logf(2.f)) * (1 << (ExponentBits - 8));
  // Same rationale as for floating point versions, bias exponent, subtract
  // 366000 to reduce error by centering approximation, instead of being
  // one-sided.
  const int B = (127 << 23) - 366000;
  matched_decimal = A * matched_decimal + B;
  float ret_val;
  memcpy(&ret_val, &matched_decimal, sizeof(float));
  return ret_val;
}

inline float fast_tanh(float x) {
#if defined FAST_TRANSCENDENTALS && defined ACCURATE_TRANSCENDENTAL_APPROX
  // Doesn't do anything fancy, just a 13/6-degree rational interpolant which
  // is accurate up to a couple of ulp in the range [-9, 9], outside of which
  // fl(tanh(x)) = +/-1.
  x = std::max(std::min(x, kMaxTanhInput), kMinTanhInput);

  // Since the polynomials are odd/even, we need x^2.
  float x2 = x * x;

  // Evaluate numerator.
  float p = kTanhAlpha11 + x2 * kTanhAlpha13;
  p = kTanhAlpha9 + x2 * p;
  p = kTanhAlpha7 + x2 * p;
  p = kTanhAlpha5 + x2 * p;
  p = kTanhAlpha3 + x2 * p;
  p = kTanhAlpha1 + x2 * p;
  p = x * p;

  // Evaluate denominator.
  float q = kTanhBeta4 + x2 * kTanhBeta6;
  q = kTanhBeta2 + x2 * q;
  q = kTanhBeta0 + x2 * q;

  return p / q;
#elif defined FAST_TRANSCENDENTALS
  if (std::abs(x) < kTanhLinearRegion) {
    return x;
  } else {
    x = std::max(std::min(x, kMaxTanhInput), kMinTanhInput);
    float positive = fast_exp(x);
    float negative = fast_exp(-x);
    return (positive - negative) / (positive + negative);
  }
#else
  return tanhf(x);
#endif  // FAST_TRANSCENDENTALS
}

inline float fast_sigmoid(float x) {
#ifdef SIGMOID_AS_TANH
  return .5f * fast_tanh(.5f * x) + .5f;
#else
#if defined FAST_TRANSCENDENTALS && defined ACCURATE_TRANSCENDENTAL_APPROX
  // Doesn't do anything fancy, just a 9/10-degree rational interpolant which
  // interpolates 1/(1+exp(-x)) - 0.5 up to a couple of ulp in the range
  // [-18, 18], outside of which the fl(sigmoid(x)) = {0|1}. The shifted
  // sigmoid is interpolated because it was easier to make the fit converge.
  // See GenericPacketMath.h* in the open source Eigen library.
  x = std::max(std::min(x, kMaxSigmoidInput), kMinSigmoidInput);

  // Since the polynomials are odd/even, we need x^2.
  float x2 = x * x;

  // Evaluate numerator.
  float p = kSigmoidAlpha7 + x2 * kSigmoidAlpha9;
  p = kSigmoidAlpha5 + x2 * p;
  p = kSigmoidAlpha3 + x2 * p;
  p = kSigmoidAlpha1 + x2 * p;
  p = x * p;

  // Evaluate denominator.
  float q = kSigmoidBeta8 + x2 * kSigmoidBeta10;
  q = kSigmoidBeta6 + x2 * q;
  q = kSigmoidBeta4 + x2 * q;
  q = kSigmoidBeta2 + x2 * q;
  q = kSigmoidBeta0 + x2 * q;

  return p / q + 0.5f;
#elif defined FAST_TRANSCENDENTALS
  if (std::abs(x) < kSigmoidLinearRegion) {
    return .245 * x + .5;
  } else {
    return 1.f / (1.f + fast_exp(-x));
  }
#else
  return 1.f / (1.f + expf(-x));
#endif  // FAST_TRANSCENDENTALS
#endif  // SIGMOID_AS_TANH
}

template <int ExponentBits>
inline float fast_sigmoid(fixed32<ExponentBits> x) {
  return fast_sigmoid(static_cast<float>(x));
}

#endif  // defined __aarch64__

// Number of exponent bits to use for tanh.
static constexpr int kNumTanhExpBits = 3;
// Number of exponent bits to use for sigmoid.
static constexpr int kNumSigmoidExpBits = 4;
// Number of extra bits to shift sigmoid, due to its low gradient.
static constexpr int kNumExtraSigmoidShiftBits = 1;

// Returns (and builds if not done yet) a static data table (that is never
// deleted, as per the style guide) that implements tanh on fixed32 input,
// returning another fixed32 with the given number of mantissa bits (which is
// assumed to be less than the input mantissa bits).
// NOTE that this function is intended to be used only with fixed16 outputs that
// are sign-extended to 32 bits for convenience, and will return a nullptr
// if asked for more than |kMaxMantissaBits| of precision in the output table.
const int* TanhTable(int num_mantissa_bits_out);
// As TanhTable, but for Sigmoid.
const int* SigmoidTable(int num_mantissa_bits_out);

// Scalar/generic function to compute and return the fast approximation to exp
// via a polynomial refinement of the floating point exponent trick.
// TM_ORDER4_16BIT:Max relative error < 5e-6, absolute error < 1e-5 for x < 1.
// TM_ORDER3_16BIT:Max relative error < 1.1e-4, absolute error < 3e-4 for x
// < 1.
template <int kExponentBits, TranscendentalMode kOrder = TM_ORDER4_16BIT>
float fixed32_exp(fixed32<kExponentBits> x) {
  constexpr int kMantissaBits = MantissaBitsOf<fixed32<kExponentBits>>::value;
  // Clip x to min/max exp input to avoid infinities.
  int64_t clipped_x =
      std::max(std::min(x.raw_val(), kMaxExpInputInt << kMantissaBits),
               -(kMaxExpInputInt << kMantissaBits));
  // First convert problem from e^x to 2^x by multiplying by 1/log(2).
  // To maximize precision, log_factor is shifted left the maximum amount to
  // keep within int32, and we shift x left a further amount such that the
  // binary point of the product sits in the correct place in the top 32 bits of
  // the result to be used directly as a float. We can't do that directly, as x
  // would overflow, so we have to shift by 1 bit less and shift the result by
  // 1 bit less to match.
  constexpr int kXShift =
      kFloatMantissaBits + 31 - kMaxLog2Shift - kMantissaBits;
  static_assert(kXShift >= 0,
                "Mantissa bits > kFloatMantissaBits + 31 - kMaxLog2Shift");
  clipped_x <<= kXShift;
  int float_as_int = (kLogFactor * clipped_x >> 31) + kFloatExponentOffset;
  // Separate the resulting fixed-point into integer and fractional parts.
  int int_part = float_as_int & kFloatExponentMask;
  int float_part = float_as_int & kFloatMantissaMask;
  float fraction = static_cast<float>(float_part) / (1 << kFloatMantissaBits);
  // Compute the mantissa = 2^fraction using:
  // fraction - fraction*(1-fraction)*(polynomial of fraction)
  // This guarantees exactness at 0 and 1, providing continuity of the error at
  // integer boundaries.
  float mantissa;
  if (kOrder == TM_ORDER4_16BIT || kOrder == TM_ORDER4_FLOAT) {
    mantissa = (kExpQuarticFactor2 * fraction + kExpQuarticFactor1) * fraction +
               kExpQuarticFactor0;
  } else if (kOrder == TM_ORDER3_16BIT) {
    mantissa = kExpCubicFactor1 * fraction + kExpCubicFactor0;
  }
  mantissa = fraction - fraction * (1.0f - fraction) * mantissa;
  // Since the function above guarantees to stay within [0, 1), we could do all
  // the above in fixed point if necessary, in which case, we can just stuff
  // the bottom kFloatMantissaBits in with the exponent and we are done.
  // In the floating point world, it is simpler to just multiply them together.
  float result;
  memcpy(&result, &int_part, sizeof(float));
  return result * (1.0f + mantissa);
}

// Computes and returns tanh(x) fixed32->float using a polynomial refinement of
// the floating point exponent trick.
// kOrder=4: Absolute error < 1.8e-6. Relative error < 1.2e-4 for |x| > 0.01.
// kOrder=3: Absolute error < 6e-5. Relative error < 3e-3 for |x| > 0.01
template <int kExponentBits, TranscendentalMode kOrder = TM_ORDER4_16BIT>
float fixed32_tanh(fixed32<kExponentBits> x) {
  float float_x = static_cast<float>(x);
  if (std::abs(float_x) < 1.0f / 9.0f) {
    return float_x * (1 - float_x * float_x / 3.0f);
  }
  x = static_cast<fixed32<kExponentBits>>(x.raw_val() * 2);
  float exp_2x = fixed32_exp<kExponentBits, kOrder>(x);
  return (exp_2x - 1.0f) / (exp_2x + 1.0f);
}

// Computes and returns sigmoid(x) fixed32->float using a polynomial refinement
// of the floating point exponent trick.
// TM_ORDER4_16BIT: Absolute error < 9e-7, relative < 4e-6.
// TM_ORDER3_16BIT: Absolute error < 3e-5, relative < 1.1e-4.
template <int kExponentBits, TranscendentalMode kOrder = TM_ORDER4_16BIT>
float fixed32_sigmoid(fixed32<kExponentBits> x) {
  x = static_cast<fixed32<kExponentBits>>(-x.raw_val());
  float exp_x = fixed32_exp<kExponentBits, kOrder>(x);
  return 1.0f / (exp_x + 1.0f);
}

#if defined __AVX2__

// Inline function to access an int32 data table by shifting |x| right by
// |kNumShiftBits|, and adding |kTableOffset| to the result. |x| contains 8
// indices and 8 results are returned. The data table is of size
// |kTableOffset| * 2 + 1.
template <int kNumShiftBits, int kTableOffset>
inline __m256i index_data_table(const int32_t* data_table, const __m256i& x) {
  // Shift right with rounding to match input and output precision.
  __m256i shifted = _mm256_set1_epi32(1 << (kNumShiftBits - 1));
  shifted = _mm256_add_epi32(x, shifted);
  shifted = _mm256_srai_epi32(shifted, kNumShiftBits);
  // Add the offset.
  __m256i addend = _mm256_set1_epi32(kTableOffset);
  shifted = _mm256_add_epi32(shifted, addend);
  // And clamp to the indices of the LUT.
  addend = _mm256_add_epi32(addend, addend);
  shifted = _mm256_min_epi32(shifted, addend);
  shifted = _mm256_max_epi32(shifted, _mm256_setzero_si256());
  // Lookup the results in the table.
  return _mm256_i32gather_epi32(data_table, shifted, 4);
}

// Fixed32 to fixed16-in-an-int32 tanh LUT function.
// Input: register containins 8x fixed32 with |NumInputMantissaBits|.
// Output: a register containing 8x fixed16 with |NumOutputMantissaBits|, but
// note that they are sign-extended to 32 bits and are therefore basically the
// same as fixed32 with |NumOutputMantissaBits|.
template <int NumInputMantissaBits, int NumOutputMantissaBits>
inline __m256i fixed32_tanh_fixed16(const int* tanh_table, const __m256i& x) {
  // Lose the unnecessary input precision.
  constexpr int kNumShiftBits = NumInputMantissaBits - NumOutputMantissaBits;
  constexpr int kTableOffset = 1 << (NumOutputMantissaBits + kNumTanhExpBits);
  return index_data_table<kNumShiftBits, kTableOffset>(tanh_table, x);
}

// Fixed32 to fixed16-in-an-int32 sigmoid LUT function.
// Input: register containins 8x fixed32 with |NumInputMantissaBits|.
// Output: a register containing 8x fixed16 with |NumOutputMantissaBits|, but
// note  that they are sign-extended to 32 bits and are therefore basically the
// same as fixed32 with |NumOutputMantissaBits|.
template <int NumInputMantissaBits, int NumOutputMantissaBits>
inline __m256i fixed32_sigmoid_fixed16(const int* sigmoid_table,
                                       const __m256i& x) {
  // Lose the unnecessary input precision.
  constexpr int kNumShiftBits =
      kNumExtraSigmoidShiftBits + NumInputMantissaBits - NumOutputMantissaBits;
  constexpr int kTableOffset = 1
                               << (NumOutputMantissaBits + kNumSigmoidExpBits -
                                   kNumExtraSigmoidShiftBits);
  return index_data_table<kNumShiftBits, kTableOffset>(sigmoid_table, x);
}

// Convert 2x registers of 8x float32 into 1 register of 16x16 bit fixed int,
// assuming that the floats are already scaled up.
inline __m256i PackFloatsToFixed16(const __m256& x0, const __m256& x1) {
  __m256i int0 = _mm256_cvtps_epi32(x0);
  __m256i int1 = _mm256_cvtps_epi32(x1);
  int0 = _mm256_packs_epi32(int0, int1);
  // Swap the middle 64 bit elements so the results are in the right order.
  return _mm256_permute4x64_epi64(int0, 0xd8);
}

// Clips integer input to [-|kLimit|, |kLimit|].
// Input: register containins 8x fixed32 with |mantissa_bits|.
// Output: register containing 8x fixed32 limited to
// [-|kLimit| << |mantissa_bits|, |kLimit| << |mantissa_bits|].
template <int kLimit>
inline __m256i ClipToBounds(const int mantissa_bits, const __m256i& x) {
  // Clip to the input bounds for this approximation.
  __m256i clip_limit = _mm256_set1_epi32(-(kLimit << mantissa_bits));
  __m256i clipped_x = _mm256_max_epi32(x, clip_limit);
  // This quickly negates the limit without having to load another constant.
  clip_limit = _mm256_sign_epi32(clip_limit, clip_limit);
  return _mm256_min_epi32(clipped_x, clip_limit);
}

// Clips float input to [-|kLimit|, |kLimit|].
// Input: register containins 8x float.
// Output: register containing 8x float limited to [-|kLimit|, |kLimit|].
inline __m256 ClipToFloatBounds(const float kLimit, const __m256& x) {
  __m256 clip_limit = _mm256_set1_ps(kLimit);
  __m256 clipped_x = _mm256_min_ps(x, clip_limit);
  clip_limit = _mm256_set1_ps(-kLimit);
  return _mm256_max_ps(clipped_x, clip_limit);
}

// Float to float power of 2 approximation, using a quartic refinement of
// the exponent trick. For TM_ORDER4_16BIT and TM_ORDER3_16BIT, implementation
// is entirely in integer, using 16x16=16 multiplication, using AVX2, which
// enables 16 elements to be computed in parallel, hence the double register
// input/output args.
// The price paid for this speed is an increase in error over the (scalar) int32
// example implementations above by a variable factor of 4-10.
// For the TM_ORDER4_FLOAT case, the computation is all done in float, solving
// this lower precision problem.
// NOTE: The input must have already been clipped to prevent overflow, which
// sets the practical limit to +/-126 << kFloatMantissaBits.
// NOTE: The input is a scaled float, as if converted raw from int, and the
// scale factor is fixed at kFloatMantissaBits!
// Input: 2x register containining 8x float * 1 << kFloatMantissaBits.
// Output: 2x register containing 8x float.
// TM_ORDER4_FLOAT: Max relative error < 8e-6, absolute error < 9e-6 for x < 1.
// TM_ORDER4_16BIT: Max relative error < 3e-5, absolute error < 6e-5 for x < 1.
// TM_ORDER3_16BIT: Max relative error < 6e-4, absolute error < 2e-3 for x < 1.
template <TranscendentalMode kOrder = TM_ORDER4_16BIT>
inline void float32_pow2(__m256& x0, __m256& x1) {
  // Convert straight to int.
  __m256i exp_int_x0 = _mm256_cvtps_epi32(x0);
  __m256i exp_int_x1 = _mm256_cvtps_epi32(x1);
  __m256i result_x0, result_x1;

  static_assert(kOrder == TM_ORDER4_FLOAT || kOrder == TM_ORDER4_16BIT ||
                    kOrder == TM_ORDER3_16BIT,
                "Invalid order.");

  if (kOrder == TM_ORDER4_FLOAT) {
    __m256i mantissa_mask = _mm256_set1_epi32(0x7fffff);
    __m256 float_factor =
        _mm256_set1_ps(1.0f / static_cast<float>(1 << kFloatMantissaBits));
    __m256i fract0 = _mm256_and_si256(mantissa_mask, exp_int_x0);
    __m256i fract1 = _mm256_and_si256(mantissa_mask, exp_int_x1);
    __m256 float0 = _mm256_mul_ps(_mm256_cvtepi32_ps(fract0), float_factor);
    __m256 float1 = _mm256_mul_ps(_mm256_cvtepi32_ps(fract1), float_factor);
    // Compute the polynomial of the fractional part.
    // Ordering these lines carefully makes it faster, as some of the multiply
    // operations can pipeline instead of waiting for the previous result.
    __m256 x_squared0 = _mm256_mul_ps(float0, float0);
    __m256 x_squared1 = _mm256_mul_ps(float1, float1);
    __m256 b = _mm256_set1_ps(kExpQuarticFactor1);
    __m256 b_x0 = _mm256_mul_ps(b, float0);
    __m256 b_x1 = _mm256_mul_ps(b, float1);
    __m256 a = _mm256_set1_ps(kExpQuarticFactor2);
    __m256 a_x_squared0 = _mm256_mul_ps(a, x_squared0);
    __m256 a_x_squared1 = _mm256_mul_ps(a, x_squared1);
    __m256 x_squared_minus_x0 = _mm256_sub_ps(x_squared0, float0);
    __m256 x_squared_minus_x1 = _mm256_sub_ps(x_squared1, float1);
    __m256 c = _mm256_set1_ps(kExpQuarticFactor0);
    b_x0 = _mm256_add_ps(b_x0, c);
    b_x1 = _mm256_add_ps(b_x1, c);
    float_factor = _mm256_set1_ps(static_cast<float>(1 << kFloatMantissaBits));
    a_x_squared0 = _mm256_add_ps(a_x_squared0, b_x0);
    a_x_squared1 = _mm256_add_ps(a_x_squared1, b_x1);
    a_x_squared0 = _mm256_mul_ps(a_x_squared0, x_squared_minus_x0);
    a_x_squared1 = _mm256_mul_ps(a_x_squared1, x_squared_minus_x1);
    result_x0 = _mm256_cvtps_epi32(_mm256_mul_ps(a_x_squared0, float_factor));
    result_x1 = _mm256_cvtps_epi32(_mm256_mul_ps(a_x_squared1, float_factor));
  } else {
    // Combine the fractional part of both inputs into a single register.
    // The representation is fixed16<0>, ie 15 mantissa bits.
    __m256i mantissa_mask = _mm256_set1_epi32(0x7fff00);
    __m256i x_01 =
        _mm256_srli_epi32(_mm256_and_si256(mantissa_mask, exp_int_x0), 8);
    x_01 = _mm256_or_si256(
        x_01,
        _mm256_slli_epi32(_mm256_and_si256(mantissa_mask, exp_int_x1), 8));
    // Compute the polynomial of the fractional part.
    // Ordering these lines carefully makes it faster, as some of the multiply
    // operations can pipeline instead of waiting for the previous result.
    __m256i x_squared = _mm256_mulhrs_epi16(x_01, x_01);
    __m256i result, x_squared_minus_x;
    if (kOrder == TM_ORDER4_16BIT) {
      __m256i b = _mm256_set1_epi16(FloatAsInt16(kExpQuarticFactor1));
      __m256i b_x = _mm256_mulhrs_epi16(b, x_01);
      __m256i a = _mm256_set1_epi16(FloatAsInt16(kExpQuarticFactor2));
      __m256i a_x_squared = _mm256_mulhrs_epi16(a, x_squared);
      x_squared_minus_x = _mm256_sub_epi16(x_squared, x_01);
      // LOG(INFO) << "x_squared_minus_x=" <<
      // static_cast<int16>(_mm256_extract_epi16(x_squared_minus_x, 0)) /
      // 32768.0f;
      __m256i c = _mm256_set1_epi16(FloatAsInt16(kExpQuarticFactor0));
      b_x = _mm256_add_epi16(b_x, c);
      // LOG(INFO) << "bx+c=" << static_cast<int16>(_mm256_extract_epi16(b_x,
      // 0)) / 32768.0f;
      result = _mm256_add_epi16(a_x_squared, b_x);
    } else {  // kOrder = TM_ORDER3_16BIT
      __m256i a = _mm256_set1_epi16(FloatAsInt16(kExpCubicFactor1));
      __m256i b = _mm256_set1_epi16(FloatAsInt16(kExpQuarticFactor0));
      __m256i a_x = _mm256_mulhrs_epi16(a, x_01);
      x_squared_minus_x = _mm256_sub_epi16(x_squared, x_01);
      result = _mm256_add_epi16(a_x, b);
    }
    result = _mm256_mulhrs_epi16(result, x_squared_minus_x);
    // Extract 16x16-bit results back to the separate sets of 8x32.
    result_x0 = _mm256_slli_epi32(result, 16);
    result_x0 = _mm256_srai_epi32(result_x0, 8);
    result_x1 = _mm256_srai_epi32(result, 16);
    result_x1 = _mm256_slli_epi32(result_x1, 8);
  }
  // Add the constant to normalize the exponent.
  __m256i exp_offset = _mm256_set1_epi32(kFloatExponentOffset);
  exp_int_x0 = _mm256_add_epi32(exp_int_x0, exp_offset);
  exp_int_x0 = _mm256_add_epi32(exp_int_x0, result_x0);
  exp_int_x1 = _mm256_add_epi32(exp_int_x1, exp_offset);
  exp_int_x1 = _mm256_add_epi32(exp_int_x1, result_x1);
  // Cast back to float, as we just computed the exponent and mantissa and
  // assembled them in IEEE format.
  x0 = _mm256_castsi256_ps(exp_int_x0);
  x1 = _mm256_castsi256_ps(exp_int_x1);
}

// Fixed32 to to float exp approximation, using a quartic/cubic refinement of
// the exponent trick. Implementation is entirely in integer, using 16x16=16
// multiplication, using AVX2, which enables 16 elements to be computed in
// parallel, hence the double register input/output args.
// The price paid for this speed is an increase in error over the (scalar) int32
// example implementations above by a variable factor of 4-10.
// The TM_ORDER4_FLOAT version uses floats and improves the precision.
// Input: 2x registers containins 8x fixed32 with kMantissaBits.
// Output: 2x registers containing 8x float32.
// TM_ORDER4_FLOAT: Max relative error < 8e-6, absolute error < 9e-6 for x < 1.
// TM_ORDER4_16BIT: Max relative error < 3e-5, absolute error < 6e-5 for x < 1.
// TM_ORDER3_16BIT: Max relative error < 6e-4, absolute error < 2e-3 for x < 1.
template <int kInputMantissaBits, TranscendentalMode kOrder = TM_ORDER4_16BIT>
inline void float_exp_float_preclipped(__m256& y0, __m256& y1) {
  // Divide by log 2 to convert problem to 2^x, and scale to match the
  // mantissa bits required by IEEE floats. Without a _mm256_mulhrs_epi32, it is
  // much easier to do this in float, even with the double conversion, as 16 bit
  // is not precise enough here.
  // This is the shift of the FP mantissa relative to the input mantissa.
  constexpr int kXShift = kFloatMantissaBits - kInputMantissaBits;
  constexpr float kLogFactor = static_cast<float>(1 << kXShift);
  __m256 factor = _mm256_set1_ps(kLogFactor * kOneOverLog2);
  y0 = _mm256_mul_ps(y0, factor);
  y1 = _mm256_mul_ps(y1, factor);
  // Now compute 2^x.
  float32_pow2<kOrder>(y0, y1);
}
template <int kInputMantissaBits, TranscendentalMode kOrder = TM_ORDER4_16BIT>
inline void fixed32_exp_float(const __m256i& x0, const __m256i& x1, __m256& y0,
                              __m256& y1) {
  // Clip to acceptable bounds to prevent overflow, and convert to float.
  y0 =
      _mm256_cvtepi32_ps(ClipToBounds<kMaxExpInputInt>(kInputMantissaBits, x0));
  y1 =
      _mm256_cvtepi32_ps(ClipToBounds<kMaxExpInputInt>(kInputMantissaBits, x1));
  float_exp_float_preclipped<kInputMantissaBits, kOrder>(y0, y1);
}

// Float->float tanh approximation via the exponent trick.
// Note that the input is scaled floats, as if converted raw from fixed16/32.
// Input: 2x registers containing 8x float scaled by input_mantissa_bits.
// Output: two registers containing 8x float.
// TM_ORDER4_FLOAT: Max relative error < 2.1e-5, absolute error < 2.3e-6.
// TM_ORDER4_16BIT: Max relative error < 1e-4, absolute error < 1.3e-5.
// TM_ORDER3_16BIT: Max relative error < 2.1e-3, absolute error < 3e-4.
template <int kInputMantissaBits, TranscendentalMode kOrder = TM_ORDER4_FLOAT>
inline void float_tanh_float(const __m256& x0, const __m256& x1, __m256& y0,
                             __m256& y1) {
  // Divide by log 2 to convert problem to 2^x, double (as we need exp(2x)) and
  // scale to the mantissa bits required by float32_pow2 all in one multiply.
  // This is the shift of the FP mantissa relative to the input mantissa.
  // Add one to double the input.
  const float kLogFactor =
      static_cast<float>(1 << (kFloatMantissaBits - kInputMantissaBits + 1));
  __m256 factor = _mm256_set1_ps(kLogFactor * kOneOverLog2);
  // Clip to suitable input bounds for tanh.
  __m256 clip_limit = _mm256_set1_ps(kMaxTanhInput * (1 << kInputMantissaBits));
  __m256 clip0 = _mm256_min_ps(x0, clip_limit);
  __m256 clip1 = _mm256_min_ps(x1, clip_limit);
  clip_limit = _mm256_set1_ps(-kMaxTanhInput * (1 << kInputMantissaBits));
  clip0 = _mm256_max_ps(clip0, clip_limit);
  clip1 = _mm256_max_ps(clip1, clip_limit);
  __m256 exp0 = _mm256_mul_ps(clip0, factor);
  __m256 exp1 = _mm256_mul_ps(clip1, factor);
  // Now compute 2^x.
  float32_pow2<kOrder>(exp0, exp1);
  // Now compute tanh using (e^2x - 1) / (e^2x + 1).
  __m256 one = _mm256_set1_ps(1.0f);
  __m256 numerator = _mm256_sub_ps(exp0, one);
  __m256 denominator = _mm256_add_ps(exp0, one);
  // Approximate reciprocal is not accurate enough - use full division.
  exp0 = _mm256_div_ps(numerator, denominator);
  numerator = _mm256_sub_ps(exp1, one);
  denominator = _mm256_add_ps(exp1, one);
  exp1 = _mm256_div_ps(numerator, denominator);
  // Compute 3rd-order Taylor tanh ~ x - x^3/3 for high accuracy and thus low
  // relative error close to 0.
  // Normalize the inputs back to proper floats.
  factor = _mm256_set1_ps(1.0f / (1 << kInputMantissaBits));
  clip0 = _mm256_mul_ps(clip0, factor);
  clip1 = _mm256_mul_ps(clip1, factor);
  __m256 third = _mm256_set1_ps(-1.0f / 3.0f);
  __m256 taylor0 = _mm256_mul_ps(clip0, clip0);
  __m256 taylor1 = _mm256_mul_ps(clip1, clip1);
  taylor0 = _mm256_mul_ps(taylor0, clip0);
  taylor1 = _mm256_mul_ps(taylor1, clip1);
  // TODO(b/191497441): The next two pairs of instructions could be combined to
  // _mm256_fmadd_ps, but requires -mfma compilation option, eg:
  // taylor0 = _mm256_fmadd_ps(taylor0, third, clip0);
  taylor0 = _mm256_mul_ps(taylor0, third);
  taylor1 = _mm256_mul_ps(taylor1, third);
  taylor0 = _mm256_add_ps(clip0, taylor0);
  taylor1 = _mm256_add_ps(clip1, taylor1);
  // Test |x| <= 1/9, roughly where the errors cross over, without needing yet
  // another constant.
  third = _mm256_mul_ps(third, third);
  __m256 neg_zero = _mm256_set1_ps(-0.0f);
  clip0 = _mm256_andnot_ps(neg_zero, clip0);
  clip1 = _mm256_andnot_ps(neg_zero, clip1);
  __m256 cmp_results0 = _mm256_cmp_ps(clip0, third, _CMP_LE_OQ);
  __m256 cmp_results1 = _mm256_cmp_ps(clip1, third, _CMP_LE_OQ);
  y0 = _mm256_blendv_ps(exp0, taylor0, cmp_results0);
  y1 = _mm256_blendv_ps(exp1, taylor1, cmp_results1);
}

// Fixed32 sigmoid approximation via the AVX2 implementation of the exponent
// trick.
// Input: 2x registers containins 8x float containing converted fixed32 scaled
// with kInputMantissaBits.
// Output: 2x registers containing 8x float.
// TM_ORDER4_FLOAT: Max relative error < 4e-6, absolute error < 1e-6.
// TM_ORDER4_16BIT: Max relative error < 3e-5, absolute error < 7e-6.
// TM_ORDER3_16BIT: Max relative error < 5.4e-4, absolute error < 1.4e-4.
template <int kInputMantissaBits, TranscendentalMode kOrder = TM_ORDER4_FLOAT>
inline void float_sigmoid_float(__m256& y0, __m256& y1) {
  constexpr float kInputFactor = static_cast<float>(1 << kInputMantissaBits);
  // Negate the inputs.
  __m256 minus_zero = _mm256_set1_ps(-0.0f);
  y0 = _mm256_xor_ps(y0, minus_zero);
  y1 = _mm256_xor_ps(y1, minus_zero);
  y0 = ClipToFloatBounds(kMaxSigmoidInput * kInputFactor, y0);
  y1 = ClipToFloatBounds(kMaxSigmoidInput * kInputFactor, y1);
  float_exp_float_preclipped<kInputMantissaBits, kOrder>(y0, y1);
  __m256 one = _mm256_set1_ps(1.0f);
  // Approximate reciprocal is not accurate enough - use full division.
  y0 = _mm256_div_ps(one, _mm256_add_ps(y0, one));
  y1 = _mm256_div_ps(one, _mm256_add_ps(y1, one));
}

#endif  // defined __AVX2__

}  // namespace csrblocksparse

#endif  // LYRA_CODEC_SPARSE_MATMUL_NUMERICS_FAST_TRANSCENDENTALS_H_
