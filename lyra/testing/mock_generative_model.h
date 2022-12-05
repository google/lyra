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

#ifndef LYRA_TESTING_MOCK_GENERATIVE_MODEL_H_
#define LYRA_TESTING_MOCK_GENERATIVE_MODEL_H_

#include <cstdint>
#include <optional>
#include <vector>

#include "gmock/gmock.h"
#include "lyra/generative_model_interface.h"

namespace chromemedia {
namespace codec {

// Implements shell functions for running the conditioning and model.
// Uses the state management of |GenerativeModel| for handling
// |num_samples_available()|, |AddFeatures()|, and |GenerateSamples()|.
class FakeGenerativeModel : public GenerativeModel {
 public:
  ~FakeGenerativeModel() override {}

  FakeGenerativeModel(int16_t sample_value, int num_samples_per_hop,
                      int num_features)
      : GenerativeModel(num_samples_per_hop, num_features),
        sample_value_(sample_value) {}

 protected:
  bool RunConditioning(const std::vector<float>& features) override {
    return true;
  }

  std::optional<std::vector<int16_t>> RunModel(int num_samples) override {
    return std::vector<int16_t>(num_samples, sample_value_);
  }

 private:
  const int sample_value_;
};

// Ensures no more samples may be requested than are available.
// Returns a constant valued array as samples.
class MockGenerativeModel : public GenerativeModelInterface {
 public:
  ~MockGenerativeModel() override {}

  MockGenerativeModel(int16_t sample_value, int num_samples_per_hop,
                      int num_features)
      : fake_generative_model_(sample_value, num_samples_per_hop,
                               num_features) {
    ON_CALL(*this, AddFeatures)
        .WillByDefault([this](const std::vector<float>& features) {
          return fake_generative_model_.AddFeatures(features);
        });
    ON_CALL(*this, GenerateSamples).WillByDefault([this](int num_samples) {
      return fake_generative_model_.GenerateSamples(num_samples);
    });
    ON_CALL(*this, num_samples_available).WillByDefault([this]() {
      return fake_generative_model_.num_samples_available();
    });
  }

  MOCK_METHOD(bool, AddFeatures, (const std::vector<float>& features),
              (override));
  MOCK_METHOD(std::optional<std::vector<int16_t>>, GenerateSamples,
              (int num_samples), (override));
  MOCK_METHOD(int, num_samples_available, (), (const override));

 private:
  MockGenerativeModel() = delete;
  FakeGenerativeModel fake_generative_model_;
};

}  // namespace codec
}  // namespace chromemedia

#endif  // LYRA_TESTING_MOCK_GENERATIVE_MODEL_H_
