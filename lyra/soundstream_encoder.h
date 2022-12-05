/*
 * Copyright 2022 Google LLC
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

#ifndef LYRA_SOUNDSTREAM_ENCODER_H_
#define LYRA_SOUNDSTREAM_ENCODER_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "absl/types/span.h"
#include "include/ghc/filesystem.hpp"
#include "lyra/feature_extractor_interface.h"
#include "lyra/tflite_model_wrapper.h"

namespace chromemedia {
namespace codec {

// This class wraps a SoundStream encoder TFLite model to extract learned
// features.
class SoundStreamEncoder : public FeatureExtractorInterface {
 public:
  // Returns a nullptr on failure.
  static std::unique_ptr<SoundStreamEncoder> Create(
      const ghc::filesystem::path& model_path);

  ~SoundStreamEncoder() override {}

  // Extracts features from the audio. On failure returns a nullopt.
  std::optional<std::vector<float>> Extract(
      const absl::Span<const int16_t> audio) override;

 private:
  explicit SoundStreamEncoder(std::unique_ptr<TfLiteModelWrapper> model);

  const std::unique_ptr<TfLiteModelWrapper> model_;
  const int num_features_;
};

}  // namespace codec
}  // namespace chromemedia

#endif  // LYRA_SOUNDSTREAM_ENCODER_H_
