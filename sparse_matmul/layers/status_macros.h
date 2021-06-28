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

#ifndef THIRD_PARTY_LYRA_CODEC_SPARSE_MATMUL_LAYERS_STATUS_MACROS_H_
#define THIRD_PARTY_LYRA_CODEC_SPARSE_MATMUL_LAYERS_STATUS_MACROS_H_

#include "absl/status/status.h"
#include "absl/status/statusor.h"

#define SPARSE_MATMUL_RETURN_IF_ERROR(expr) \
  do {                                      \
    const absl::Status _status = (expr);    \
    if (!_status.ok()) return _status;      \
  } while (0)
template <typename T>
absl::Status DoAssignOrReturn(T& lhs, absl::StatusOr<T> result) {
  if (result.ok()) {
    lhs = result.value();
  }
  return result.status();
}

#endif  // THIRD_PARTY_LYRA_CODEC_SPARSE_MATMUL_LAYERS_STATUS_MACROS_H_
