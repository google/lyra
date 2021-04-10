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

#include "quadrature_mirror_filter.h"

#include <cstdint>
#include <vector>

#include "absl/types/span.h"
#include "audio/linear_filters/biquad_filter.h"
#include "audio/linear_filters/biquad_filter_coefficients.h"
#include "dsp_util.h"
#include "glog/logging.h"

namespace chromemedia {
namespace codec {
namespace {

// Zeros of the 2 all-pass filters that provide the right relative phase
// difference that allows the splitting and merging of the low and high bands.
constexpr float zeros_1[] = {0.3255157470703125f, 0.748626708984375f,
                             0.961456298828125f};
constexpr float zeros_2[] = {0.097930908203125f, 0.564300537109375f,
                             0.8737335205078125f};

linear_filters::BiquadFilterCascadeCoefficients AllPassCoefficients(
    absl::Span<const float> zeros) {
  std::vector<linear_filters::BiquadFilterCoefficients> coefficients;
  coefficients.reserve(zeros.size());
  for (float zero : zeros) {
    coefficients.push_back(linear_filters::BiquadFilterCoefficients(
        {zero, 1.f, 0.f}, {1.f, zero, 0.f}));
  }
  return linear_filters::BiquadFilterCascadeCoefficients(coefficients);
}

}  // namespace

template <typename T>
SplitQuadratureMirrorFilter<T>::SplitQuadratureMirrorFilter() {
  all_pass_1_.Init(/*num_channels=*/1, AllPassCoefficients(zeros_1));
  all_pass_2_.Init(/*num_channels=*/1, AllPassCoefficients(zeros_2));
}

template <typename T>
Bands<T> SplitQuadratureMirrorFilter<T>::Split(absl::Span<const T> signal) {
  CHECK_EQ(signal.size() % 2, 0)
      << "The number of samples has to be even, but was " << signal.size()
      << ".";
  float all_pass_out_1, all_pass_out_2;
  Bands<T> bands(signal.size() / 2);
  for (int i = 0; i < bands.num_samples_per_band; ++i) {
    all_pass_1_.ProcessSample(static_cast<float>(signal.at(2 * i)),
                              &all_pass_out_1);
    all_pass_2_.ProcessSample(static_cast<float>(signal.at(2 * i + 1)),
                              &all_pass_out_2);
    if (std::is_same<T, int16_t>::value) {
      bands.low_band.at(i) = ClipToInt16((all_pass_out_1 + all_pass_out_2) / 2);
      bands.high_band.at(i) =
          ClipToInt16((all_pass_out_1 - all_pass_out_2) / 2);
    } else {
      bands.low_band.at(i) = (all_pass_out_1 + all_pass_out_2) / 2;
      bands.high_band.at(i) = (all_pass_out_1 - all_pass_out_2) / 2;
    }
  }
  return bands;
}

template <typename T>
MergeQuadratureMirrorFilter<T>::MergeQuadratureMirrorFilter() {
  all_pass_1_.Init(/*num_channels=*/1, AllPassCoefficients(zeros_1));
  all_pass_2_.Init(/*num_channels=*/1, AllPassCoefficients(zeros_2));
}

template <typename T>
std::vector<T> MergeQuadratureMirrorFilter<T>::Merge(const Bands<T>& bands) {
  CHECK_EQ(bands.low_band.size(), bands.num_samples_per_band)
      << "The number of samples of all bands has to be "
      << bands.num_samples_per_band << ", but was " << bands.low_band.size()
      << " for the low band.";
  CHECK_EQ(bands.high_band.size(), bands.num_samples_per_band)
      << "The number of samples of all bands has to be "
      << bands.num_samples_per_band << ", but was " << bands.high_band.size()
      << " for the high band.";
  float all_pass_out_1, all_pass_out_2;
  std::vector<T> merged_signal;
  merged_signal.reserve(2 * bands.num_samples_per_band);
  for (int i = 0; i < bands.num_samples_per_band; ++i) {
    all_pass_1_.ProcessSample(
        static_cast<float>(bands.low_band.at(i) - bands.high_band.at(i)),
        &all_pass_out_1);
    all_pass_2_.ProcessSample(
        static_cast<float>(bands.low_band.at(i) + bands.high_band.at(i)),
        &all_pass_out_2);
    if (std::is_same<T, int16_t>::value) {
      merged_signal.push_back(ClipToInt16(all_pass_out_2));
      merged_signal.push_back(ClipToInt16(all_pass_out_1));
    } else {
      merged_signal.push_back(all_pass_out_2);
      merged_signal.push_back(all_pass_out_1);
    }
  }
  return merged_signal;
}

template struct Bands<int16_t>;
template class SplitQuadratureMirrorFilter<int16_t>;
template class MergeQuadratureMirrorFilter<int16_t>;

template struct Bands<float>;
template class SplitQuadratureMirrorFilter<float>;
template class MergeQuadratureMirrorFilter<float>;

}  // namespace codec
}  // namespace chromemedia
