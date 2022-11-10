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

#include "lyra_gan_model.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/types/span.h"
#include "dsp_utils.h"
#include "glog/logging.h"  // IWYU pragma: keep
#include "tflite_model_wrapper.h"

namespace chromemedia {
namespace codec {

std::unique_ptr<LyraGanModel> LyraGanModel::Create(
    const ghc::filesystem::path& model_path, int num_features) {
  auto model =
      TfLiteModelWrapper::Create(model_path / "lyragan.tflite",
                                 /*use_xnn=*/true, /*int8_quantized=*/true);
  if (model == nullptr) {
    LOG(ERROR) << "Unable to create LyraGAN TFLite model wrapper.";
    return nullptr;
  }
  return absl::WrapUnique(new LyraGanModel(std::move(model), num_features));
}

LyraGanModel::LyraGanModel(std::unique_ptr<TfLiteModelWrapper> model,
                           int num_features)
    : GenerativeModel(model->get_output_tensor<float>(0).size(), num_features),
      model_(std::move(model)) {}

bool LyraGanModel::RunConditioning(const std::vector<float>& features) {
  absl::Span<float> input = model_->get_input_tensor<float>(0);
  std::copy(features.begin(), features.end(), input.begin());
  model_->Invoke();
  return true;
}

std::optional<std::vector<int16_t>> LyraGanModel::RunModel(int num_samples) {
  return UnitToInt16(absl::MakeConstSpan(
      &model_->get_output_tensor<float>(0).at(next_sample_in_hop()),
      num_samples));
}

}  // namespace codec
}  // namespace chromemedia
