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

#ifndef LYRA_CODEC_FILTER_BANKS_INTERFACE_H_
#define LYRA_CODEC_FILTER_BANKS_INTERFACE_H_

#include <cstdint>
#include <vector>

namespace chromemedia {
namespace codec {

// An interface to abstract the merging of split band signals.
class MergeFilterInterface {
 public:
  explicit MergeFilterInterface(int num_bands) : num_bands_(num_bands) {}
  virtual ~MergeFilterInterface() {}

  // Merge multiple bands sampled at sub-Nyquist into signal.
  virtual std::vector<int16_t> Merge(
      const std::vector<std::vector<int16_t>>& bands) = 0;

  int num_bands() const { return num_bands_; }

 protected:
  const int num_bands_;
};

}  // namespace codec
}  // namespace chromemedia

#endif  // LYRA_CODEC_FILTER_BANKS_INTERFACE_H_
