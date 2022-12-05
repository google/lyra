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

#ifndef LYRA_LYRA_BENCHMARK_LIB_H_
#define LYRA_LYRA_BENCHMARK_LIB_H_

#include <cstdint>
#include <string>

namespace chromemedia {
namespace codec {

struct TimingStats {
  int64_t max_microsecs;
  int64_t mean_microsecs;
  int64_t min_microsecs;
  int64_t num_calls;
  float standard_deviation;
};

int lyra_benchmark(int num_cond_vectors, const std::string& model_base_path,
                   bool benchmark_feature_extraction, bool benchmark_quantizer,
                   bool benchmark_generative_model);

}  // namespace codec
}  // namespace chromemedia

#endif  // LYRA_LYRA_BENCHMARK_LIB_H_
