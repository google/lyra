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

#ifndef LYRA_CODEC_BENCHMARK_DECODE_LIB_H_
#define LYRA_CODEC_BENCHMARK_DECODE_LIB_H_

#include <vector>

#include "absl/base/attributes.h"
#include "absl/strings/string_view.h"

namespace chromemedia {
namespace codec {

struct TimingStats {
  int64_t max_microsecs;
  int64_t mean_microsecs;
  int64_t min_microsecs;
  int64_t num_calls;
  float standard_deviation;
};

// Given an array of ints, computes the max, min, mean, and standard deviation.
// Returns the results as a string.
ABSL_ATTRIBUTE_UNUSED TimingStats
GetTimingStats(const std::vector<int64_t>& timings_microsecs);

// Prints stats and writes CSV for the runtime information in |timings| to file
// under /{sdcard,tmp}/benchmark/lyra/.
ABSL_ATTRIBUTE_UNUSED void PrintStatsAndWriteCSV(
    const std::vector<int64_t>& timings, const absl::string_view title);

int benchmark_decode(const int num_cond_vectors,
                     const std::string& model_base_path);

}  // namespace codec
}  // namespace chromemedia

#endif  // LYRA_CODEC_BENCHMARK_DECODE_LIB_H_
