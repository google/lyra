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

#ifndef LYRA_TESTING_MOCK_RESAMPLER_H_
#define LYRA_TESTING_MOCK_RESAMPLER_H_

#include <cstdint>
#include <vector>

#include "absl/types/span.h"
#include "gmock/gmock.h"
#include "lyra/resampler.h"
#include "lyra/resampler_interface.h"

namespace chromemedia {
namespace codec {

class MockResampler : public ResamplerInterface {
 public:
  MockResampler(int input_sample_rate_hz, int target_sample_rate_hz)
      : resampler_(
            Resampler::Create(input_sample_rate_hz, target_sample_rate_hz)) {
    ON_CALL(*this, Resample)
        .WillByDefault([this](absl::Span<const int16_t> audio) {
          return resampler_->Resample(audio);
        });
    ON_CALL(*this, Reset).WillByDefault([this]() {
      return resampler_->Reset();
    });
    ON_CALL(*this, target_sample_rate_hz).WillByDefault([this]() {
      return resampler_->target_sample_rate_hz();
    });
    ON_CALL(*this, input_sample_rate_hz).WillByDefault([this]() {
      return resampler_->input_sample_rate_hz();
    });
    ON_CALL(*this, samples_until_steady_state).WillByDefault([this]() {
      return resampler_->samples_until_steady_state();
    });
  }

  ~MockResampler() override {}

  MOCK_METHOD(std::vector<int16_t>, Resample, (absl::Span<const int16_t> audio),
              (override));

  MOCK_METHOD(void, Reset, (), (override));

  MOCK_METHOD(int, input_sample_rate_hz, (), (const override));

  MOCK_METHOD(int, target_sample_rate_hz, (), (const override));

  MOCK_METHOD(int, samples_until_steady_state, (), (const override));

 private:
  std::unique_ptr<Resampler> resampler_;
};

}  // namespace codec
}  // namespace chromemedia

#endif  // LYRA_TESTING_MOCK_RESAMPLER_H_
