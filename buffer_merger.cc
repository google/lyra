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

#include "buffer_merger.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "filter_banks.h"
#include "filter_banks_interface.h"
#include "glog/logging.h"

namespace chromemedia {
namespace codec {

std::unique_ptr<BufferMerger> BufferMerger::Create(int num_bands) {
  auto merge_filter = MergeFilter::Create(num_bands);
  if (merge_filter == nullptr) {
    LOG(ERROR) << "Cannot create a MergeFilter with " << num_bands << " bands.";
    return nullptr;
  }

  return absl::WrapUnique(new BufferMerger(std::move(merge_filter)));
}

BufferMerger::BufferMerger(std::unique_ptr<MergeFilterInterface> merge_filter)
    : merge_filter_(std::move(merge_filter)),
      num_bands_(merge_filter_->num_bands()) {
  leftover_samples_.reserve(num_bands_ - 1);
}

int BufferMerger::GetNumSamplesToGenerate(int num_samples) const {
  if (num_samples < leftover_samples_.size()) {
    return 0;
  }

  const int num_samples_to_generate_per_band = static_cast<int>(
      std::ceil(static_cast<float>(num_samples - leftover_samples_.size()) /
                static_cast<float>(num_bands_)));
  return num_samples_to_generate_per_band * num_bands_;
}

std::vector<int16_t> BufferMerger::BufferAndMerge(
    const std::function<const std::vector<std::vector<int16_t>>&(int)>&
        sample_generator,
    int num_samples) {
  int num_samples_to_generate = GetNumSamplesToGenerate(num_samples);

  // 1. If we have any leftover samples from last time we must use them.
  std::vector<int16_t> samples(num_samples);
  const int num_leftover_used = UseLeftoverSamples(num_samples, &samples);

  // 2. Generate samples using |sample_generator|.
  const std::vector<std::vector<int16_t>>& new_split_samples =
      sample_generator(num_samples_to_generate);

  // 3. Merge the buffer of split samples if needed to produce new samples.
  const std::vector<int16_t> new_samples = MergeSamples(new_split_samples);
  CHECK_EQ(new_samples.size(), num_samples_to_generate);

  // 4. Copy the new samples to output and the leftover buffers.
  CopyNewSamples(new_samples, num_samples, num_leftover_used, &samples);
  return samples;
}

int BufferMerger::UseLeftoverSamples(int num_samples,
                                     std::vector<int16_t>* samples) {
  const int num_leftover_used =
      std::min(static_cast<int>(leftover_samples_.size()), num_samples);
  std::move(leftover_samples_.begin(),
            leftover_samples_.begin() + num_leftover_used, samples->begin());
  std::move(leftover_samples_.begin() + num_leftover_used,
            leftover_samples_.end(), leftover_samples_.begin());
  leftover_samples_.resize(leftover_samples_.size() - num_leftover_used);
  return num_leftover_used;
}

std::vector<int16_t> BufferMerger::MergeSamples(
    const std::vector<std::vector<int16_t>>& new_split_samples) {
  // If there is only one band, no need to merge.
  if (num_bands_ == 1) {
    return new_split_samples.at(0);
  }
  // Otherwise merge the split samples.
  return merge_filter_->Merge(new_split_samples);
}

void BufferMerger::CopyNewSamples(const std::vector<int16_t>& new_samples,
                                  int num_samples, int num_leftover_used,
                                  std::vector<int16_t>* samples) {
  // Copy the needed samples to the destination, which already has some
  // leftover samples from the last run.
  const int num_samples_to_copy = num_samples - num_leftover_used;
  CHECK_GE(new_samples.size(), num_samples_to_copy);
  std::copy(new_samples.begin(), new_samples.begin() + num_samples_to_copy,
            samples->begin() + num_leftover_used);

  // Store the rest in the |leftover_samples_|.
  leftover_samples_.insert(leftover_samples_.end(),
                           new_samples.begin() + num_samples_to_copy,
                           new_samples.end());
}

}  // namespace codec
}  // namespace chromemedia
