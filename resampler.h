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

#ifndef LYRA_CODEC_RESAMPLER_H_
#define LYRA_CODEC_RESAMPLER_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/types/span.h"
#include "audio/dsp/resampler_q.h"
#include "resampler_interface.h"

namespace chromemedia {
namespace codec {

// This class wraps a resampler that can either upsample or downsample audio.
class Resampler : public ResamplerInterface {
 public:
  ~Resampler() override;

  static std::unique_ptr<Resampler> Create(int input_sample_rate_hz,
                                           int target_sample_rate_hz);

  // Resamples audio at |input_sample_rate_hz| to |target_sample_rate_hz|.
  std::vector<int16_t> Resample(absl::Span<const int16_t> audio) override;

  void Reset() override;

  int input_sample_rate_hz() const override;

  int target_sample_rate_hz() const override;

  int samples_until_steady_state() const override;

 private:
  const int input_sample_rate_hz_;
  const int target_sample_rate_hz_;

  explicit Resampler(audio_dsp::QResampler<float> dsp_resampler,
                     int input_sample_rate_hz, int target_sample_rate_hz);
  audio_dsp::QResampler<float> resampler_;
};

}  // namespace codec
}  // namespace chromemedia
#endif  // LYRA_CODEC_RESAMPLER_H_
