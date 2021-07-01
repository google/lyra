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

#ifndef LYRA_CODEC_VECTOR_QUANTIZER_INTERFACE_H_
#define LYRA_CODEC_VECTOR_QUANTIZER_INTERFACE_H_

#include <optional>
#include <string>
#include <vector>

namespace chromemedia {
namespace codec {

// An interface to abstract the quantization implementation.
class VectorQuantizerInterface {
 public:
  virtual ~VectorQuantizerInterface() {}

  // Converts the features into a bitset representing the indices of code
  // vectors closest to the klt transform of features.
  virtual std::optional<std::string> Quantize(
      const std::vector<float>& features, int num_bits) const = 0;

  // Converts quantized bits back into lossy features in the log mel
  // spectrogram domain.
  virtual std::optional<std::vector<float>> DecodeToLossyFeatures(
      const std::string& quantized_features) const = 0;
};

}  // namespace codec
}  // namespace chromemedia

#endif  // LYRA_CODEC_VECTOR_QUANTIZER_INTERFACE_H_
