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

#include <memory>

#include "benchmark/benchmark.h"
#include "sparse_matmul/vector/cache_aligned_vector.h"

// A simple benchmark for CacheAlignedVector.
//
// Running on x86:
// As written, it's not representative of x86 performance since ReducingSample
// is used on x86 and not Sample.
//
// Running on arm64:
// bazel build -c opt --dynamic_mode=off --copt=-gmlt \
//   --copt=-DUSE_FIXED32 --config=android_arm64 \
//   sparse_matmul/vector:cachealignedvector_benchmark
namespace csrblocksparse {

#ifdef USE_BFLOAT16
using ComputeType = csrblocksparse::bfloat16;
#elif defined USE_FIXED32
using ComputeType = csrblocksparse::fixed32<11>;  // kGruMatMulOutBits
#else
using ComputeType = float;
#endif  // USE_BFLOAT16

#if defined(USE_FIXED32) && defined(__aarch64__)
using ScratchType = int;
#else
using ScratchType = float;
#endif  // defined(USE_FIXED32) && defined(__aarch64__)

void BM_Sample(benchmark::State& state) {
  constexpr int kVectorSize = 16384;  // A large vector.
  std::minstd_rand generator;

  CacheAlignedVector<ComputeType> values(kVectorSize);
  CacheAlignedVector<ScratchType> scratch(kVectorSize);
  values.FillRandom();

  for (auto _ : state) {
    values.Sample(/*temperature=*/0.98f, &generator, &scratch);
  }
}
BENCHMARK(BM_Sample);

}  // namespace csrblocksparse
