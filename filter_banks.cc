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

#include "filter_banks.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/types/span.h"
#include "filter_banks_interface.h"
#include "glog/logging.h"
#include "quadrature_mirror_filter.h"

namespace chromemedia {
namespace codec {
namespace {

bool IsPowerOfTwo(int x) { return x && !(x & (x - 1)); }

int IntLogTwo(int x) {
  int log_two = 0;
  while (x >>= 1) {
    ++log_two;
  }
  return log_two;
}

}  // namespace

std::unique_ptr<SplitFilter> SplitFilter::Create(int num_bands) {
  if (!IsPowerOfTwo(num_bands)) {
    LOG(ERROR) << "Number of bands has to be a power of 2, but was "
               << num_bands << ".";
    return nullptr;
  }
  return absl::WrapUnique(new SplitFilter(num_bands));
}

SplitFilter::SplitFilter(int num_bands) : num_bands_(num_bands) {
  const int num_levels = IntLogTwo(num_bands);
  filters_per_level_.reserve(num_levels);
  for (int level = 0; level < num_levels; ++level) {
    // Number of filters per level goes up as power of 2, starting at 1.
    filters_per_level_.push_back(
        std::vector<SplitQuadratureMirrorFilter<int16_t>>(1 << level));
  }
}

std::vector<std::vector<int16_t>> SplitFilter::Split(
    absl::Span<const int16_t> signal) {
  CHECK_EQ(signal.size() % num_bands_, 0)
      << "The number of samples has to be divisible by " << num_bands_
      << ", but was " << signal.size() << ".";
  std::vector<std::vector<int16_t>> old_bands;
  old_bands.push_back(std::vector<int16_t>(signal.begin(), signal.end()));
  for (std::vector<SplitQuadratureMirrorFilter<int16_t>>& filters :
       filters_per_level_) {
    std::vector<std::vector<int16_t>> new_bands;
    for (int filter = 0; filter < filters.size(); ++filter) {
      Bands<int16_t> bands = filters.at(filter).Split(old_bands.at(filter));
      // Because of the mirroring characteristic of aliasing, odd bands are
      // reversed.
      if (filter % 2 == 0) {
        new_bands.push_back(std::move(bands.low_band));
        new_bands.push_back(std::move(bands.high_band));
      } else {
        new_bands.push_back(std::move(bands.high_band));
        new_bands.push_back(std::move(bands.low_band));
      }
    }
    old_bands = new_bands;
  }
  return old_bands;
}

std::unique_ptr<MergeFilter> MergeFilter::Create(int num_bands) {
  if (!IsPowerOfTwo(num_bands)) {
    LOG(ERROR) << "Number of bands has to be a power of 2, but was "
               << num_bands << ".";
    return nullptr;
  }
  return absl::WrapUnique(new MergeFilter(num_bands));
}

MergeFilter::MergeFilter(int num_bands) : MergeFilterInterface(num_bands) {
  const int num_levels = IntLogTwo(num_bands);
  filters_per_level_.reserve(num_levels);
  for (int level = 0; level < num_levels; ++level) {
    // Number of filters per level goes down as power of 2, ending at 1.
    filters_per_level_.push_back(
        std::vector<MergeQuadratureMirrorFilter<int16_t>>(num_bands >>
                                                          (level + 1)));
  }
}

std::vector<int16_t> MergeFilter::Merge(
    const std::vector<std::vector<int16_t>>& bands) {
  CHECK_EQ(bands.size(), num_bands_)
      << "The number of bands has to be " << num_bands_ << ", but was "
      << bands.size() << ".";
  const int band_size = bands.at(0).size();
  for (int band = 0; band < bands.size(); ++band) {
    CHECK_EQ(bands.at(band).size(), band_size)
        << "The number of samples of all bands has to be the same, but was "
        << band_size << " for the first band and " << bands.at(band).size()
        << " for the band number " << band + 1 << ".";
  }
  std::vector<std::vector<int16_t>> old_bands(bands);
  for (std::vector<MergeQuadratureMirrorFilter<int16_t>>& filters :
       filters_per_level_) {
    std::vector<std::vector<int16_t>> new_bands;
    for (int filter = 0; filter < filters.size(); ++filter) {
      // Because of the mirroring characteristic of aliasing, odd bands are
      // reversed.
      const Bands<int16_t> bands(old_bands.at(2 * filter + filter % 2),
                                 old_bands.at(2 * filter + 1 - filter % 2));
      new_bands.push_back(filters.at(filter).Merge(bands));
    }
    old_bands = new_bands;
  }
  return old_bands.at(0);
}

}  // namespace codec
}  // namespace chromemedia
