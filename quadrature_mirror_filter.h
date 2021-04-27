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

#ifndef LYRA_CODEC_QUADRATURE_MIRROR_FILTER_H_
#define LYRA_CODEC_QUADRATURE_MIRROR_FILTER_H_

#include <cstdint>
#include <vector>

#include "absl/types/span.h"
#include "audio/linear_filters/biquad_filter.h"

namespace chromemedia {
namespace codec {

// Struct to hold the split signals.
template <typename T>
struct Bands {
  explicit Bands(int num_samples_per_band)
      : num_samples_per_band(num_samples_per_band),
        low_band(num_samples_per_band),
        high_band(num_samples_per_band) {}
  Bands(const std::vector<T>& low, const std::vector<T>& high)
      : num_samples_per_band(low.size()), low_band(low), high_band(high) {}
  int num_samples_per_band;
  std::vector<T> low_band;
  std::vector<T> high_band;
};

// Quadrature mirror filter bank to split a signal into 2 bands sampled at
// sub-Nyquist.
template <typename T>
class SplitQuadratureMirrorFilter {
 public:
  SplitQuadratureMirrorFilter();

  // Split signal into low and high bands sampled at sub-Nyquist.
  // Signal size has to be even.
  Bands<T> Split(absl::Span<const T> signal);

 private:
  // Two all-pass filters with a relative phase difference that allows the
  // splitting of the low and high bands as the sum and difference respectively.
  linear_filters::BiquadFilterCascade<float> all_pass_1_;
  linear_filters::BiquadFilterCascade<float> all_pass_2_;
};

// Quadrature mirror filter bank to merge 2 bands sampled at sub-Nyquist into a
// signal.
template <typename T>
class MergeQuadratureMirrorFilter {
 public:
  MergeQuadratureMirrorFilter();

  // Merge the low and high bands sampled at sub-Nyquist into signal.
  // The low and high band have to have the same size as num_samples_per_band.
  std::vector<T> Merge(const Bands<T>& bands);

 private:
  // Two all-pass filters with a relative phase difference that allows the
  // merging of the sum and difference of the low and high bands.
  linear_filters::BiquadFilterCascade<float> all_pass_1_;
  linear_filters::BiquadFilterCascade<float> all_pass_2_;
};

extern template struct Bands<int16_t>;
extern template class SplitQuadratureMirrorFilter<int16_t>;
extern template class MergeQuadratureMirrorFilter<int16_t>;

extern template struct Bands<float>;
extern template class SplitQuadratureMirrorFilter<float>;
extern template class MergeQuadratureMirrorFilter<float>;

}  // namespace codec
}  // namespace chromemedia

#endif  // LYRA_CODEC_QUADRATURE_MIRROR_FILTER_H_
