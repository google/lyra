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

#ifndef LYRA_RESIDUAL_VECTOR_QUANTIZER_H_
#define LYRA_RESIDUAL_VECTOR_QUANTIZER_H_

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "include/ghc/filesystem.hpp"
#include "lyra/tflite_model_wrapper.h"
#include "lyra/vector_quantizer_interface.h"

namespace chromemedia {
namespace codec {

// This class wraps a Residual Vector Quantizer TFLite model to quantize and
// decode back to lossy features.
class ResidualVectorQuantizer : public VectorQuantizerInterface {
 public:
  // Returns nullptr if the TFLite model can't be built or allocated.
  static std::unique_ptr<ResidualVectorQuantizer> Create(
      const ghc::filesystem::path& model_path);

  // Quantizes the features using vector quantization.
  std::optional<std::string> Quantize(const std::vector<float>& features,
                                      int num_bits) const override;

  // Unpacks the string of bits into features.
  std::optional<std::vector<float>> DecodeToLossyFeatures(
      const std::string& quantized_features) const override;

 private:
  // LINT.IfChange
  static constexpr int kMaxNumQuantizedBits = 184;
  // LINT.ThenChange(
  // lyra_components.cc,
  // lyra_config.cc,
  // )

  explicit ResidualVectorQuantizer(
      std::unique_ptr<TfLiteModelWrapper> quantizer_model);

  const std::unique_ptr<TfLiteModelWrapper> quantizer_model_;
  tflite::SignatureRunner* encode_runner_;
  tflite::SignatureRunner* decode_runner_;
  const int bits_per_quantizer_;
};

}  // namespace codec
}  // namespace chromemedia

#endif  // LYRA_RESIDUAL_VECTOR_QUANTIZER_H_
