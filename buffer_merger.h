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

#ifndef LYRA_CODEC_BUFFER_MERGER_H_
#define LYRA_CODEC_BUFFER_MERGER_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "filter_banks_interface.h"

namespace chromemedia {
namespace codec {

// A class to buffer samples in split bands and merge them to produce an
// arbitrary number of samples. This is because the model can only produce
// time domain samples in multiples of |num_bands|.
class BufferMerger {
 public:
  static std::unique_ptr<BufferMerger> Create(int num_bands);

  // Buffer the newly generated split samples and merge them to produce
  // |num_samples| samples.
  std::vector<int16_t> BufferAndMerge(
      const std::function<const std::vector<std::vector<int16_t>>&(int)>&
          sample_generator,
      int num_samples);

  void Reset() { leftover_samples_.clear(); }

 private:
  explicit BufferMerger(std::unique_ptr<MergeFilterInterface> merge_filter);

  // Helper function to inform the generative model how many samples need to
  // be generated if a total of |num_samples| is requested upstream. Computed
  // based on the number of leftover samples from previous calls and number of
  // bands.
  int GetNumSamplesToGenerate(int num_samples) const;

  // Use at most |num_samples| from |leftover_samples_| to fill the beginning
  // of |samples|.
  int UseLeftoverSamples(int num_samples, std::vector<int16_t>* samples);

  std::vector<int16_t> MergeSamples(
      const std::vector<std::vector<int16_t>>& new_split_samples);

  void CopyNewSamples(const std::vector<int16_t>& new_samples, int num_samples,
                      int num_leftover_used, std::vector<int16_t>* samples);

  std::unique_ptr<MergeFilterInterface> merge_filter_;
  const int num_bands_;
  // Buffer of (at most |num_bands_ - 1|) leftover samples from the last run.
  std::vector<int16_t> leftover_samples_;
  friend class BufferMergerPeer;
};

}  // namespace codec
}  // namespace chromemedia

#endif  // LYRA_CODEC_BUFFER_MERGER_H_
