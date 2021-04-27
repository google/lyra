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

#ifndef LYRA_CODEC_TESTING_MOCK_FILTER_BANKS_H_
#define LYRA_CODEC_TESTING_MOCK_FILTER_BANKS_H_

#include <cstdint>
#include <vector>

#include "filter_banks_interface.h"
#include "gmock/gmock.h"

namespace chromemedia {
namespace codec {

class MockMergeFilter : public MergeFilterInterface {
 public:
  explicit MockMergeFilter(int num_bands) : MergeFilterInterface(num_bands) {}
  ~MockMergeFilter() override {}

  MOCK_METHOD(std::vector<int16_t>, Merge,
              (const std::vector<std::vector<int16_t>>& bands), (override));
};

}  // namespace codec
}  // namespace chromemedia

#endif  // LYRA_CODEC_TESTING_MOCK_FILTER_BANKS_H_
