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

#ifndef LYRA_BUFFERED_RESAMPLER_H_
#define LYRA_BUFFERED_RESAMPLER_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

#include "lyra/buffered_filter_interface.h"
#include "lyra/resampler_interface.h"

namespace chromemedia {
namespace codec {

// A class to buffer samples at the external sample rate and reuse them to
// produce an arbitrary number of samples. Time domain samples can only be
// generated in multiples of |external_sample_rate| / |internal_sample_rate|,
// if this ratio is greater than 1.
class BufferedResampler : public BufferedFilterInterface {
 public:
  static std::unique_ptr<BufferedResampler> Create(int internal_sample_rate,
                                                   int external_sample_rate);

  // Buffer the newly generated samples and resample them to produce
  // |num_external_samples_requested| samples.
  std::optional<std::vector<int16_t>> FilterAndBuffer(
      const std::function<std::optional<std::vector<int16_t>>(int)>&
          sample_generator,
      int num_external_samples_requested) override;

 private:
  explicit BufferedResampler(std::unique_ptr<ResamplerInterface> resampler);

  // Helper function to inform the generative model how many samples need to
  // be generated if a total of |num_external_samples_requested| are requested
  // upstream. Computed based on the number of leftover samples from previous
  // calls and the external to internal resample ratio.
  int GetInternalNumSamplesToGenerate(int num_external_samples_requested) const;

  // Use at most |num_external_samples_requested| from |leftover_samples_| to
  // fill the beginning of |samples|.
  int UseLeftoverSamples(int num_external_samples_requested,
                         std::vector<int16_t>* samples);

  std::vector<int16_t> Resample(const std::vector<int16_t>& internal_samples);

  void CopyNewSamples(const std::vector<int16_t>& external_samples,
                      int num_external_samples_requested, int num_leftover_used,
                      std::vector<int16_t>* samples);

  // If the resample ratio is greater than 1, buffer at most
  // |external_sample_rate| / |internal_sample_rate_hz|/ - 1 leftover samples
  // from the last run. Otherwise this is unused.
  std::vector<int16_t> leftover_samples_;

  std::unique_ptr<ResamplerInterface> resampler_;

  friend class BufferedResamplerPeer;
};

}  // namespace codec
}  // namespace chromemedia

#endif  // LYRA_BUFFERED_RESAMPLER_H_
