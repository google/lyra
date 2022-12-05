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

#include "lyra/resampler.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/types/span.h"
#include "audio/dsp/resampler_q.h"
#include "glog/logging.h"  // IWYU pragma: keep
#include "lyra/dsp_utils.h"

namespace chromemedia {
namespace codec {
std::unique_ptr<Resampler> Resampler::Create(int input_sample_rate_hz,
                                             int target_sample_rate_hz) {
  audio_dsp::QResamplerParams params;
  // Set kernel radius to 17 input samples. Since |ResetFullyPrimed()| is used
  // below, the resampler has a delay of 2 * 17 input samples, or about 2 ms
  // at 16 kHz input sample rate.
  params.filter_radius_factor =
      17.f * std::min(1.f, static_cast<float>(target_sample_rate_hz) /
                               input_sample_rate_hz);
  audio_dsp::QResampler<float> dsp_resampler(
      static_cast<float>(input_sample_rate_hz),
      static_cast<float>(target_sample_rate_hz), /*num_channels=*/1, params);
  if (!dsp_resampler.Valid()) {
    LOG(ERROR) << "Error creating QResampler.";
    return nullptr;
  }
  return absl::WrapUnique(new Resampler(dsp_resampler, input_sample_rate_hz,
                                        target_sample_rate_hz));
}

Resampler::~Resampler() {}

Resampler::Resampler(audio_dsp::QResampler<float> dsp_resampler,
                     int input_sample_rate_hz, int target_sample_rate_hz)
    : input_sample_rate_hz_(input_sample_rate_hz),
      target_sample_rate_hz_(target_sample_rate_hz),
      resampler_(std::move(dsp_resampler)) {
  resampler_.ResetFullyPrimed();
}

std::vector<int16_t> Resampler::Resample(absl::Span<const int16_t> audio) {
  std::vector<float> input_floats(audio.begin(), audio.end());
  std::vector<float> output_floats;
  resampler_.ProcessSamples(input_floats, &output_floats);
  return ClipToInt16(absl::MakeConstSpan(output_floats));
}

void Resampler::Reset() { resampler_.ResetFullyPrimed(); }

int Resampler::input_sample_rate_hz() const { return input_sample_rate_hz_; }

int Resampler::target_sample_rate_hz() const { return target_sample_rate_hz_; }

int Resampler::samples_until_steady_state() const {
  // Convert the reset delay described by |QResamplerParams| in |Create| from
  // the input target rate |factor_numerator| to the target rate
  // |factor_denominator|.
  const float kResampleRatio =
      static_cast<float>(resampler_.factor_denominator()) /
      static_cast<float>(resampler_.factor_numerator());
  return static_cast<int>(2.f * resampler_.radius() * kResampleRatio);
}

}  // namespace codec
}  // namespace chromemedia
