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

#ifndef LYRA_CODEC_SPARSE_MATMUL_SPARSE_MATMUL_H_
#define LYRA_CODEC_SPARSE_MATMUL_SPARSE_MATMUL_H_

// IWYU pragma: begin_exports
#include "sparse_matmul/compute/gru_gates.h"
#include "sparse_matmul/layers/csr_blocksparse_matrix.h"
#include "sparse_matmul/layers/masked_sparse_matrix.h"
#include "sparse_matmul/layers/sparse_linear_layer.h"
#include "sparse_matmul/layers/utils.h"
#include "sparse_matmul/numerics/fast_transcendentals.h"
#include "sparse_matmul/numerics/fixed_types.h"
#include "sparse_matmul/numerics/float16_types.h"
#include "sparse_matmul/numerics/type_utils.h"
#include "sparse_matmul/os/coop_threads.h"
#include "sparse_matmul/vector/cache_aligned_vector.h"
// IWYU pragma: end_exports

#endif  // LYRA_CODEC_SPARSE_MATMUL_SPARSE_MATMUL_H_
