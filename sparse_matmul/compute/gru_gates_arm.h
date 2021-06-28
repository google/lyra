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

#ifndef LYRA_CODEC_SPARSE_MATMUL_COMPUTE_GRU_GATES_ARM_H_
#define LYRA_CODEC_SPARSE_MATMUL_COMPUTE_GRU_GATES_ARM_H_

#if defined __ARM_NEON || defined __aarch64__
#include <arm_neon.h>
#endif
#include <cstdint>

#include "sparse_matmul/compute/ar_inputs.h"
#include "sparse_matmul/numerics/fast_transcendentals.h"

namespace csrblocksparse {

static constexpr int kNeonSIMDWidth = 4;

// ------ Scalar calculation --------
// See "Efficient Neural Audio Synthesis" for a description of the calculation.
// https://arxiv.org/abs/1802.08435
//
// NOTE:
// |sample| = (|coarse_at_sminus1|, |fine_at_sminus1|,
//             |coarse_at_sminus1|, |fine_at_sminus1|)
// |w_sample| = (|coarse_at_s|, |coarse_at_s|, |coarse_at_s|, |coarse_at_s|)
//
// CHEATSHEET:
// vld1q_f32 = load 4 32-bit floats
// vmulq_f32(a, b) : return a * b;
// vaddq_f32(a, b) : return a + b;
// vmlaq_f32(c, a, b) : return c + a * b;
// vpaddq_f32(a, b) : return (a0 + a1, a2 + a3, b0 + b1, b2 + b3)
// vsubq_f32(a, b) : return a - b;
// vst1q_f32 = store 4 32-bit floats
#if defined __ARM_NEON || defined __aarch64__

#if !defined __aarch64__
// Backport of vpaddq_f32 to ARM32.
inline float32x4_t vpaddq_f32(float32x4_t a, float32x4_t b) {
  float32x2_t a10 = vget_low_f32(a);
  float32x2_t a32 = vget_high_f32(a);
  float32x2_t b10 = vget_low_f32(b);
  float32x2_t b32 = vget_high_f32(b);
  return vcombine_f32(vpadd_f32(a10, a32), vpadd_f32(b10, b32));
}
#endif

template <ARInputsMode kInputsMode, bool SplitGates>
void GoThroughGatesFloat(int start, int end, const float* qr_ptr,
                         const float* gru_gates_ptr,
                         const float* gru_gates_other_ptr,
                         const float* conditioning_ptr, float* gru_h_ptr,
                         const float* w_hat, int proj_size,
                         const float* coarse_at_sminus1,
                         const float* fine_at_sminus1,
                         const float* coarse_at_s) {
  // Increment all the pointers to save on pointer arithmetic in the loop.
  conditioning_ptr += start;
  gru_h_ptr += start;
  gru_gates_ptr += start;
  if (SplitGates) {
    DCHECK_NE(gru_gates_other_ptr, nullptr);
    gru_gates_other_ptr += start;
  }
  if (kInputsMode != ARInputsMode::k0ARInputs) {
    DCHECK_NE(qr_ptr, nullptr);
    qr_ptr += 2 * start;
    DCHECK_NE(coarse_at_sminus1, nullptr);
    DCHECK_NE(fine_at_sminus1, nullptr);
    if (kInputsMode == ARInputsMode::k3ARInputs) {
      DCHECK_NE(w_hat, nullptr);
      DCHECK_NE(coarse_at_s, nullptr);
      w_hat += start;
    }
  }
  for (int i = start; i < end; i += kNeonSIMDWidth) {
    float32x4_t reset = vld1q_f32(gru_gates_ptr);
    float32x4_t update = vld1q_f32(gru_gates_ptr + proj_size);
    float32x4_t cell = vld1q_f32(gru_gates_ptr + 2 * proj_size);
    float32x4_t qr_cell;
    if (SplitGates) {
      reset = vaddq_f32(reset, vld1q_f32(gru_gates_other_ptr));
      update = vaddq_f32(update, vld1q_f32(gru_gates_other_ptr + proj_size));
      cell = vaddq_f32(cell, vld1q_f32(gru_gates_other_ptr + 2 * proj_size));
    }
    if (kInputsMode != ARInputsMode::k0ARInputs) {
      // Setup the sample vector.
      float32x4_t sample = vdupq_n_f32(*coarse_at_sminus1);
      sample = vsetq_lane_f32(*fine_at_sminus1, sample, 1);
      sample = vsetq_lane_f32(*fine_at_sminus1, sample, 3);

      // All auto types are float32x4_t, auto used to fit statements on one line
      // for readability. Do two rows of QR at once.
      auto qr_reset_0 = vmulq_f32(vld1q_f32(qr_ptr), sample);
      auto qr_reset_1 = vmulq_f32(vld1q_f32(qr_ptr + 4), sample);
      auto qr_reset = vpaddq_f32(qr_reset_0, qr_reset_1);

      auto qr_update_0 = vmulq_f32(vld1q_f32(qr_ptr + 2 * proj_size), sample);
      auto qr_update_1 =
          vmulq_f32(vld1q_f32(qr_ptr + 4 + 2 * proj_size), sample);
      auto qr_update = vpaddq_f32(qr_update_0, qr_update_1);

      auto qr_cell_0 = vmulq_f32(vld1q_f32(qr_ptr + 4 * proj_size), sample);
      auto qr_cell_1 = vmulq_f32(vld1q_f32(qr_ptr + 4 + 4 * proj_size), sample);
      qr_cell = vpaddq_f32(qr_cell_0, qr_cell_1);

      if (kInputsMode == ARInputsMode::k3ARInputs) {
        float32x4_t w_sample = vdupq_n_f32(*coarse_at_s);
        qr_reset = vmlaq_f32(qr_reset, vld1q_f32(w_hat), w_sample);
        qr_update =
            vmlaq_f32(qr_update, vld1q_f32(w_hat + proj_size), w_sample);
        qr_cell =
            vmlaq_f32(qr_cell, vld1q_f32(w_hat + 2 * proj_size), w_sample);
      }
      reset = vaddq_f32(reset, qr_reset);
      update = vaddq_f32(update, qr_update);
    }
    auto reset_conditioning = vld1q_f32(conditioning_ptr);
    auto update_conditioning = vld1q_f32(conditioning_ptr + proj_size);
    auto cell_conditioning = vld1q_f32(conditioning_ptr + 2 * proj_size);

    reset = fast_sigmoid(vaddq_f32(reset, reset_conditioning));
    update = fast_sigmoid(vaddq_f32(update, update_conditioning));
    if (kInputsMode == ARInputsMode::k0ARInputs) {
      cell = vmulq_f32(reset, cell);
    } else {
      cell = vmlaq_f32(qr_cell, reset, cell);
    }
    auto hbar = fast_tanh(vaddq_f32(cell, cell_conditioning));

    auto prev_h = vld1q_f32(gru_h_ptr);
    auto diff = vsubq_f32(prev_h, hbar);
    auto new_h = vmlaq_f32(hbar, diff, update);

    vst1q_f32(gru_h_ptr, new_h);
    // Increment all the pointers.
    conditioning_ptr += kNeonSIMDWidth;
    gru_h_ptr += kNeonSIMDWidth;
    gru_gates_ptr += kNeonSIMDWidth;
    if (SplitGates) gru_gates_other_ptr += kNeonSIMDWidth;
    if (kInputsMode != ARInputsMode::k0ARInputs) {
      qr_ptr += 2 * kNeonSIMDWidth;
      if (kInputsMode == ARInputsMode::k3ARInputs) w_hat += kNeonSIMDWidth;
    }
  }
}

// This version should only be used if all of the 32-bit fixed point
// representations have the same number of mantissa bits.
// |ar_at_sminus1| packs sample 0 and 1 into a pair because the QR weights are
// formatted with the weights interleaved for sample 0 and 1. The two samples
// represent coarse and fine for WaveRNN.
template <typename GRUStateType, typename GRUMatMulOutType,
          ARInputsMode kInputsMode, bool SplitGates>
void GoThroughGatesFixed(int start, int end, const float* qr_ptr,
                         const int32_t* gru_gates_ptr,
                         const int32_t* gru_gates_other_ptr,
                         const int32_t* conditioning_ptr, int16_t* gru_h_ptr,
                         const float* w_hat, int proj_size,
                         const std::pair<float, float>* ar_at_sminus1,
                         const float* coarse_at_s) {
  // Increment all the pointers to save on pointer arithmetic in the loop.
  conditioning_ptr += start;
  gru_h_ptr += start;
  gru_gates_ptr += start;
  if (SplitGates) {
    DCHECK_NE(gru_gates_other_ptr, nullptr);
    gru_gates_other_ptr += start;
  }
  float32x4_t sample01;
  float32x4_t w_sample;
  if (kInputsMode != ARInputsMode::k0ARInputs) {
    DCHECK_NE(qr_ptr, nullptr);
    qr_ptr += 2 * start;
    DCHECK_NE(ar_at_sminus1, nullptr);
    sample01 = vdupq_n_f32(ar_at_sminus1->first);
    sample01 = vsetq_lane_f32(ar_at_sminus1->second, sample01, 1);
    sample01 = vsetq_lane_f32(ar_at_sminus1->second, sample01, 3);
    if (kInputsMode == ARInputsMode::k3ARInputs) {
      DCHECK_NE(w_hat, nullptr);
      DCHECK_NE(coarse_at_s, nullptr);
      w_hat += start;
      w_sample = vdupq_n_f32(*coarse_at_s);
    }
  }
  for (int i = start; i < end; i += kNeonSIMDWidth) {
    auto reset = vld1q_s32(gru_gates_ptr);
    auto update = vld1q_s32(gru_gates_ptr + proj_size);
    // vcvtq_n_f32_s32 = convert 32-bit fixed point to fp32
    auto cell_int = vld1q_s32(gru_gates_ptr + 2 * proj_size);
    if (SplitGates) {
      reset = vaddq_s32(reset, vld1q_s32(gru_gates_other_ptr));
      update = vaddq_s32(update, vld1q_s32(gru_gates_other_ptr + proj_size));
      cell_int =
          vaddq_s32(cell_int, vld1q_s32(gru_gates_other_ptr + 2 * proj_size));
    }
    float32x4_t cell =
        vcvtq_n_f32_s32(cell_int, GRUMatMulOutType::kMantissaBits);
    float32x4_t qr_cell;
    if (kInputsMode != ARInputsMode::k0ARInputs) {
      // Do two rows of QR at once.
      float32x4_t qr_reset_0 = vmulq_f32(vld1q_f32(qr_ptr), sample01);
      float32x4_t qr_reset_1 = vmulq_f32(vld1q_f32(qr_ptr + 4), sample01);
      float32x4_t qr_reset = vpaddq_f32(qr_reset_0, qr_reset_1);

      float32x4_t qr_update_0 =
          vmulq_f32(vld1q_f32(qr_ptr + 2 * proj_size), sample01);
      float32x4_t qr_update_1 =
          vmulq_f32(vld1q_f32(qr_ptr + 4 + 2 * proj_size), sample01);
      float32x4_t qr_update = vpaddq_f32(qr_update_0, qr_update_1);

      float32x4_t qr_cell_0 =
          vmulq_f32(vld1q_f32(qr_ptr + 4 * proj_size), sample01);
      float32x4_t qr_cell_1 =
          vmulq_f32(vld1q_f32(qr_ptr + 4 + 4 * proj_size), sample01);
      qr_cell = vpaddq_f32(qr_cell_0, qr_cell_1);
      if (kInputsMode == ARInputsMode::k3ARInputs) {
        float32x4_t w_sample = vdupq_n_f32(*coarse_at_s);
        qr_reset = vmlaq_f32(qr_reset, vld1q_f32(w_hat), w_sample);
        qr_update =
            vmlaq_f32(qr_update, vld1q_f32(w_hat + proj_size), w_sample);
        qr_cell =
            vmlaq_f32(qr_cell, vld1q_f32(w_hat + 2 * proj_size), w_sample);
      }
      reset = vaddq_s32(
          reset, vcvtq_n_s32_f32(qr_reset, GRUMatMulOutType::kMantissaBits));
      update = vaddq_s32(
          update, vcvtq_n_s32_f32(qr_update, GRUMatMulOutType::kMantissaBits));
    }

    auto reset_conditioning = vld1q_s32(conditioning_ptr);
    auto update_conditioning = vld1q_s32(conditioning_ptr + proj_size);
    float32x4_t cell_conditioning =
        vcvtq_n_f32_s32(vld1q_s32(conditioning_ptr + 2 * proj_size),
                        GRUMatMulOutType::kMantissaBits);

    float32x4_t reset_f32 = fast_sigmoid<GRUMatMulOutType::kExponentBits>(
        vaddq_s32(reset, reset_conditioning));
    float32x4_t update_f32 = fast_sigmoid<GRUMatMulOutType::kExponentBits>(
        vaddq_s32(update, update_conditioning));
    if (kInputsMode == ARInputsMode::k0ARInputs) {
      cell = vmulq_f32(reset_f32, cell);
    } else {
      cell = vmlaq_f32(qr_cell, reset_f32, cell);
    }
    float32x4_t hbar = fast_tanh(vaddq_f32(cell, cell_conditioning));

    float32x4_t prev_h = vcvtq_n_f32_s32(vmovl_s16(vld1_s16(gru_h_ptr)),
                                         GRUStateType::kMantissaBits);
    float32x4_t diff = vsubq_f32(prev_h, hbar);
    float32x4_t new_h = vmlaq_f32(hbar, diff, update_f32);

    // vcvtq_n_s32_f32 = convert fp32 to signed 32-bit fixed point
    // vqrshrn_n_s32 = saturating, rounding, narrowing right shift - used to
    // convert a 32-bit fixed point value to a 16-bit fixed point value
    vst1_s16(gru_h_ptr,
             vqrshrn_n_s32(
                 vcvtq_n_s32_f32(new_h, GRUStateType::kMantissaBits + 16), 16));
    // Increment all the pointers.
    conditioning_ptr += kNeonSIMDWidth;
    gru_h_ptr += kNeonSIMDWidth;
    gru_gates_ptr += kNeonSIMDWidth;
    if (SplitGates) gru_gates_other_ptr += kNeonSIMDWidth;
    if (kInputsMode != ARInputsMode::k0ARInputs) {
      qr_ptr += 2 * kNeonSIMDWidth;
      if (kInputsMode == ARInputsMode::k3ARInputs) w_hat += kNeonSIMDWidth;
    }
  }
}
#endif  // defined __ARM_NEON || defined __aarch64__

}  // namespace csrblocksparse

#endif  // LYRA_CODEC_SPARSE_MATMUL_COMPUTE_GRU_GATES_ARM_H_
