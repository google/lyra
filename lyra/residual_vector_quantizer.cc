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

#include "lyra/residual_vector_quantizer.h"

#include <algorithm>
#include <bitset>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "glog/logging.h"  // IWYU pragma: keep
#include "include/ghc/filesystem.hpp"
#include "lyra/tflite_model_wrapper.h"

namespace chromemedia {
namespace codec {

std::unique_ptr<ResidualVectorQuantizer> ResidualVectorQuantizer::Create(
    const ghc::filesystem::path& model_path) {
  auto quantizer_model =
      TfLiteModelWrapper::Create(model_path / "quantizer.tflite",
                                 /*use_xnn=*/false, /*int8_quantized=*/false);
  if (quantizer_model == nullptr) {
    LOG(ERROR) << "Unable to create the quantizer TfLite model wrapper.";
    return nullptr;
  }
  tflite::SignatureRunner* encode_runner =
      quantizer_model->GetSignatureRunner("encode");
  if (encode_runner == nullptr) {
    LOG(ERROR) << "The quantizer TFLite model has no encode signature";
    return nullptr;
  }
  if (encode_runner->AllocateTensors() != kTfLiteOk) {
    LOG(ERROR) << "Could not allocate encode runner TFLite tensors.";
    return nullptr;
  }
  tflite::SignatureRunner* decode_runner =
      quantizer_model->GetSignatureRunner("decode");
  if (decode_runner == nullptr) {
    LOG(ERROR) << "The quantizer TFLite interpreter has no decode signature";
    return nullptr;
  }
  if (decode_runner->AllocateTensors() != kTfLiteOk) {
    LOG(ERROR) << "Could not allocate decode runner TFLite tensors.";
    return nullptr;
  }
  return absl::WrapUnique(
      new ResidualVectorQuantizer(std::move(quantizer_model)));
}

ResidualVectorQuantizer::ResidualVectorQuantizer(
    std::unique_ptr<TfLiteModelWrapper> quantizer_model)
    : quantizer_model_(std::move(quantizer_model)),
      encode_runner_(quantizer_model_->GetSignatureRunner("encode")),
      decode_runner_(quantizer_model_->GetSignatureRunner("decode")),
      bits_per_quantizer_(
          encode_runner_->output_tensor("output_1")->data.i32[0]) {}

std::optional<std::string> ResidualVectorQuantizer::Quantize(
    const std::vector<float>& features, int num_bits) const {
  if (num_bits > kMaxNumQuantizedBits) {
    LOG(ERROR) << "The number of bits cannot exceed maximum ("
               << kMaxNumQuantizedBits << ").";
    return std::nullopt;
  }
  if (num_bits % bits_per_quantizer_ != 0) {
    LOG(ERROR) << "The number of bits (" << num_bits
               << ") has to be divisible by the number of bits per quantizer ("
               << bits_per_quantizer_ << ").";
    return std::nullopt;
  }
  const int required_quantizers = num_bits / bits_per_quantizer_;
  encode_runner_->input_tensor("num_quantizers")->data.i32[0] =
      required_quantizers;
  std::copy(features.begin(), features.end(),
            encode_runner_->input_tensor("input_frames")->data.f);
  if (encode_runner_->Invoke() != kTfLiteOk) {
    LOG(ERROR) << "Unable to invoke the quantize runner.";
    return std::nullopt;
  }
  const int32_t* nearest_neighbors =
      encode_runner_->output_tensor("output_0")->data.i32;
  std::bitset<kMaxNumQuantizedBits> quantized_bits = 0;
  for (int i = 0; i < required_quantizers; ++i) {
    // First cast the current quantizer bits into a bitset that can contain all,
    // then shift it to the desired position and add it to the bitset.
    // The first quantizer is positioned in the most significant bits.
    quantized_bits |= std::bitset<quantized_bits.size()>(nearest_neighbors[i])
                      << ((required_quantizers - i - 1) * bits_per_quantizer_);
  }
  return quantized_bits.to_string().substr(kMaxNumQuantizedBits - num_bits);
}

std::optional<std::vector<float>>
ResidualVectorQuantizer::DecodeToLossyFeatures(
    const std::string& quantized_features) const {
  const int num_bits = quantized_features.size();
  if (num_bits > kMaxNumQuantizedBits) {
    LOG(ERROR) << "The number of bits cannot exceed maximum ("
               << kMaxNumQuantizedBits << ").";
    return std::nullopt;
  }
  if (num_bits % bits_per_quantizer_ != 0) {
    LOG(ERROR) << "The number of bits (" << num_bits
               << ") has to be divisible by the number of bits per quantizer ("
               << bits_per_quantizer_ << ").";
    return std::nullopt;
  }
  const int required_quantizers = num_bits / bits_per_quantizer_;
  const int max_num_quantizers = kMaxNumQuantizedBits / bits_per_quantizer_;
  if (decode_runner_->ResizeInputTensor(
          "encoding_indices", {max_num_quantizers, 1, 1}) != kTfLiteOk) {
    LOG(ERROR)
        << "Failed to resize the indices tensor to the required number of "
        << "quantizers (" << max_num_quantizers << ").";
    return std::nullopt;
  }
  if (decode_runner_->AllocateTensors() != kTfLiteOk) {
    LOG(ERROR) << "Unable to allocate tensors.";
    return std::nullopt;
  }
  const std::bitset<kMaxNumQuantizedBits> quantized_bits(quantized_features);
  const std::bitset<kMaxNumQuantizedBits> quantizer_mask(
      (1 << bits_per_quantizer_) - 1);
  int32_t* indices = decode_runner_->input_tensor("encoding_indices")->data.i32;
  for (int i = 0; i < required_quantizers; ++i) {
    // First shift the desired quantizer bits into the least significant
    // section, then mask out any more significant bits from other quantizers
    // and finally cast to int32.
    // The first quantizer is expected to be in the most significant bits.
    indices[i] = static_cast<int32_t>(
        ((quantized_bits >>
          ((required_quantizers - i - 1) * bits_per_quantizer_)) &
         quantizer_mask)
            .to_ulong());
  }
  for (int j = required_quantizers; j < max_num_quantizers; ++j) {
    indices[j] = -1;
  }

  if (decode_runner_->Invoke() != kTfLiteOk) {
    LOG(ERROR) << "Unable to invoke the decode runner.";
    return std::nullopt;
  }
  const TfLiteTensor* features_tensor =
      decode_runner_->output_tensor("output_0");
  const float* features = features_tensor->data.f;
  const int num_features = features_tensor->bytes / sizeof(features[0]);
  return std::vector<float>(features, features + num_features);
}

}  // namespace codec
}  // namespace chromemedia
