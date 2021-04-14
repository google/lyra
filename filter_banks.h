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

#ifndef LYRA_CODEC_FILTER_BANKS_H_
#define LYRA_CODEC_FILTER_BANKS_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/types/span.h"
#include "filter_banks_interface.h"
#include "quadrature_mirror_filter.h"

namespace chromemedia {
namespace codec {

// Filter bank to split a signal into multiple bands sampled at sub-Nyquist.
class SplitFilter {
 public:
  // Create a SplitFilter. Return nullptr if num_bands isn't a power of 2.
  static std::unique_ptr<SplitFilter> Create(int num_bands);

  // Split signal into multiple bands sampled at sub-Nyquist.
  // The size of the signal has to be divisible by num_bands.
  std::vector<std::vector<int16_t>> Split(absl::Span<const int16_t> signal);

  int num_bands() const { return num_bands_; }

 private:
  explicit SplitFilter(int num_bands);

  const int num_bands_;
  std::vector<std::vector<SplitQuadratureMirrorFilter<int16_t>>>
      filters_per_level_;
};

// Filter bank to merge multiple bands sampled at sub-Nyquist into signal.
class MergeFilter : public MergeFilterInterface {
 public:
  // Create a MergeFilter. Return nullptr if num_bands isn't a power of 2.
  static std::unique_ptr<MergeFilter> Create(int num_bands);

  // Merge multiple bands sampled at sub-Nyquist into signal.
  // The size of the bands have to coincide.
  std::vector<int16_t> Merge(
      const std::vector<std::vector<int16_t>>& bands) override;

 private:
  explicit MergeFilter(int num_bands);

  std::vector<std::vector<MergeQuadratureMirrorFilter<int16_t>>>
      filters_per_level_;
};

}  // namespace codec
}  // namespace chromemedia

#endif  // LYRA_CODEC_FILTER_BANKS_H_
