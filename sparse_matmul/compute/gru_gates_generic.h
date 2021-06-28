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

#ifndef LYRA_CODEC_SPARSE_MATMUL_COMPUTE_GRU_GATES_GENERIC_H_
#define LYRA_CODEC_SPARSE_MATMUL_COMPUTE_GRU_GATES_GENERIC_H_

#include "sparse_matmul/compute/ar_inputs.h"
#include "sparse_matmul/numerics/fast_transcendentals.h"

namespace csrblocksparse {

constexpr int kGenericSIMDWidth = 4;

// TODO(b/188702959): Rename arguments to match gru_gates.h.
template <typename GRUStateType, typename GRUMatMulOutType, typename QR_W_Type,
          typename SampleType, ARInputsMode kInputsMode,
          bool SplitGates = false>
void GoThroughGates(int start, int end, const QR_W_Type* qr_ptr,
                    const GRUMatMulOutType* gru_gates_ptr,
                    const GRUMatMulOutType* gru_gates_other_ptr,
                    const GRUMatMulOutType* conditioning_ptr,
                    GRUStateType* gru_h_ptr, const QR_W_Type* w_hat,
                    int proj_size, const SampleType* coarse_at_sminus1,
                    const SampleType* fine_at_sminus1,
                    const SampleType* coarse_at_s = nullptr) {
  float qr_cell = 0.0f, reset, update, cell;
  for (int i = start; i < end; ++i) {
    if (kInputsMode == ARInputsMode::k0ARInputs) {
      reset = static_cast<float>(gru_gates_ptr[i]);
      update = static_cast<float>(gru_gates_ptr[proj_size + i]);
    } else {
      float qr_c_reset = static_cast<float>(qr_ptr[2 * i + 0]);
      float qr_f_reset = static_cast<float>(qr_ptr[2 * i + 1]);
      float qr_c_update = static_cast<float>(qr_ptr[2 * proj_size + 2 * i + 0]);
      float qr_f_update = static_cast<float>(qr_ptr[2 * proj_size + 2 * i + 1]);
      float qr_c_cell = static_cast<float>(qr_ptr[4 * proj_size + 2 * i + 0]);
      float qr_f_cell = static_cast<float>(qr_ptr[4 * proj_size + 2 * i + 1]);
      float w_hat_i_reset = 0.0f;
      float w_hat_i_update = 0.0f;
      float w_hat_i_cell = 0.0f;
      if (kInputsMode == ARInputsMode::k3ARInputs) {
        w_hat_i_reset = static_cast<float>(w_hat[i]);
        w_hat_i_update = static_cast<float>(w_hat[proj_size + i]);
        w_hat_i_cell = static_cast<float>(w_hat[2 * proj_size + i]);
      }
      float coarse = static_cast<float>(coarse_at_sminus1[0]);
      float fine = static_cast<float>(fine_at_sminus1[0]);
      reset = qr_c_reset * coarse + qr_f_reset * fine;
      update = qr_c_update * coarse + qr_f_update * fine;
      qr_cell = qr_c_cell * coarse + qr_f_cell * fine;
      if (kInputsMode == ARInputsMode::k3ARInputs) {
        float coarse = static_cast<float>(coarse_at_s[0]);
        reset += w_hat_i_reset * coarse;
        update += w_hat_i_update * coarse;
        qr_cell += w_hat_i_cell * coarse;
      }
      reset += static_cast<float>(gru_gates_ptr[i]);
      update += static_cast<float>(gru_gates_ptr[proj_size + i]);
    }
    cell = static_cast<float>(gru_gates_ptr[2 * proj_size + i]);
    if (SplitGates) {
      reset += static_cast<float>(gru_gates_other_ptr[i]);
      update += static_cast<float>(gru_gates_other_ptr[proj_size + i]);
      cell += static_cast<float>(gru_gates_other_ptr[2 * proj_size + i]);
    }
    float reset_conditioning = static_cast<float>(conditioning_ptr[i]);
    float update_conditioning =
        static_cast<float>(conditioning_ptr[proj_size + i]);
    float cell_conditioning =
        static_cast<float>(conditioning_ptr[2 * proj_size + i]);
    reset = fast_sigmoid(reset + reset_conditioning);
    update = fast_sigmoid(update + update_conditioning);
    float hbar = fast_tanh(qr_cell + reset * cell + cell_conditioning);
    int h_index = i;
    float prev_h = static_cast<float>(gru_h_ptr[h_index]);
    float diff = prev_h - hbar;
    float new_h = hbar + diff * update;
    gru_h_ptr[h_index] = static_cast<GRUStateType>(new_h);
  }
}

}  // namespace csrblocksparse

#endif  // LYRA_CODEC_SPARSE_MATMUL_COMPUTE_GRU_GATES_GENERIC_H_
