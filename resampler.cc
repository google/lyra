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

#include "resampler.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/types/span.h"
#include "audio/dsp/resampler_q.h"
#include "dsp_util.h"
#include "glog/logging.h"

namespace chromemedia {
namespace codec {
std::unique_ptr<Resampler> Resampler::Create(double input_sample_rate_hz,
                                             double target_sample_rate_hz) {
  audio_dsp::QResamplerParams params;
  // Set kernel radius to 17 input samples. Since `ResetFullyPrimed()` is used
  // below, the resampler has a delay of 2 * 17 input samples, or about 2 ms
  // at 16 kHz input sample rate.
  params.filter_radius_factor =
      17.0 * std::min(1.0, target_sample_rate_hz / input_sample_rate_hz);
  audio_dsp::QResampler<float> dsp_resampler(
      input_sample_rate_hz, target_sample_rate_hz, /*num_channels=*/1, params);
  if (!dsp_resampler.Valid()) {
    LOG(ERROR) << "Error creating QResampler.";
    return nullptr;
  }
  return absl::WrapUnique(new Resampler(dsp_resampler));
}

Resampler::~Resampler() {}

Resampler::Resampler(audio_dsp::QResampler<float> dsp_resampler)
    : resampler_(std::move(dsp_resampler)) {
  resampler_.ResetFullyPrimed();
}

std::vector<int16_t> Resampler::Resample(absl::Span<const int16_t> audio) {
  std::vector<float> input_floats(audio.begin(), audio.end());
  std::vector<float> output_floats;
  resampler_.ProcessSamples(input_floats, &output_floats);
  std::vector<int16_t> output(output_floats.size());
  std::transform(output_floats.begin(), output_floats.end(), output.begin(),
                 ClipToInt16);
  return output;
}

void Resampler::Reset() { resampler_.ResetFullyPrimed(); }

}  // namespace codec
}  // namespace chromemedia
