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

#include "lyra/soundstream_encoder.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/types/span.h"
#include "glog/logging.h"  // IWYU pragma: keep
#include "include/ghc/filesystem.hpp"
#include "lyra/dsp_utils.h"
#include "lyra/tflite_model_wrapper.h"

namespace chromemedia {
namespace codec {

std::unique_ptr<SoundStreamEncoder> SoundStreamEncoder::Create(
    const ghc::filesystem::path& model_path) {
  auto model =
      TfLiteModelWrapper::Create(model_path / "soundstream_encoder.tflite",
                                 /*use_xnn=*/true, /*int8_quantized=*/true);
  if (model == nullptr) {
    LOG(ERROR) << "Unable to create SoundStream encoder TFLite model wrapper.";
    return nullptr;
  }
  return absl::WrapUnique(new SoundStreamEncoder(std::move(model)));
}

SoundStreamEncoder::SoundStreamEncoder(
    std::unique_ptr<TfLiteModelWrapper> model)
    : model_(std::move(model)),
      num_features_(model_->get_output_tensor<float>(0).size()) {}

std::optional<std::vector<float>> SoundStreamEncoder::Extract(
    const absl::Span<const int16_t> audio) {
  absl::Span<float> input = model_->get_input_tensor<float>(0);
  std::transform(audio.begin(), audio.end(), input.begin(),
                 Int16ToUnitScalar<float>);
  if (!model_->Invoke()) {
    LOG(ERROR) << "Unable to invoke SoundStream encoder TFLite model wrapper.";
    return std::nullopt;
  }
  absl::Span<const float> output = model_->get_output_tensor<float>(0);
  return std::vector<float>(output.begin(), output.end());
}

}  // namespace codec
}  // namespace chromemedia
