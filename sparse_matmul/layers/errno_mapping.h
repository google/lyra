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

#ifndef THIRD_PARTY_LYRA_CODEC_SPARSE_MATMUL_LAYERS_ERRNO_MAPPING_H_
#define THIRD_PARTY_LYRA_CODEC_SPARSE_MATMUL_LAYERS_ERRNO_MAPPING_H_

#include "absl/status/status.h"
#include "absl/strings/string_view.h"

namespace csrblocksparse {

// Converts |error_number| value to absl::Status.
absl::Status ErrnoToCanonicalStatus(int error_number,
                                    absl::string_view message);

}  // namespace csrblocksparse

#endif  // THIRD_PARTY_LYRA_CODEC_SPARSE_MATMUL_LAYERS_ERRNO_MAPPING_H_
