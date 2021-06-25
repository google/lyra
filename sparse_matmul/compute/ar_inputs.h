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

#ifndef LYRA_CODEC_SPARSE_MATMUL_COMPUTE_AR_INPUTS_H_
#define LYRA_CODEC_SPARSE_MATMUL_COMPUTE_AR_INPUTS_H_

namespace csrblocksparse {

// Possible numbers of Autoregressive inputs.
// TODO(b/188702959): Generalize to any non-negative integer value?
enum class ARInputsMode {
  // There are no autoregressive inputs. Inputs to the GRU gates are strictly
  // from the gate-recurrent matmul and other unrelated inputs.
  k0ARInputs,
  // Two autoregressive inputs, such as coarse and fine for WaveRNN.
  k2ARInputs,
  // Three autoregressive inputs, such as prev coarse and fine plus current
  // coarse for WaveRNN.
  k3ARInputs,
};

}  // namespace csrblocksparse

#endif  // LYRA_CODEC_SPARSE_MATMUL_COMPUTE_AR_INPUTS_H_
