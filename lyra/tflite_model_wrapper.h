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

#ifndef LYRA_TFLITE_MODEL_WRAPPER_H_
#define LYRA_TFLITE_MODEL_WRAPPER_H_

#include <functional>
#include <memory>

#include "absl/types/span.h"
#include "include/ghc/filesystem.hpp"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/signature_runner.h"

namespace chromemedia {
namespace codec {

class TfLiteModelWrapper {
 public:
  static std::unique_ptr<TfLiteModelWrapper> Create(
      const ghc::filesystem::path& model_file, bool use_xnn,
      bool int8_quantized);

  bool Invoke();

  tflite::SignatureRunner* GetSignatureRunner(const char* signature);

  bool ResetVariableTensors();

  int num_input_tensors();

  int num_output_tensors();

  template <class T>
  absl::Span<T> get_input_tensor(int index) {
    return absl::Span<T>(interpreter_->typed_input_tensor<T>(index),
                         interpreter_->input_tensor(index)->bytes / sizeof(T));
  }

  template <class T>
  absl::Span<const T> get_output_tensor(int index) {
    return absl::Span<const T>(
        interpreter_->typed_output_tensor<T>(index),
        interpreter_->output_tensor(index)->bytes / sizeof(T));
  }

 private:
  TfLiteModelWrapper(std::unique_ptr<tflite::FlatBufferModel> model,
                     std::unique_ptr<tflite::Interpreter> interpreter);

  std::unique_ptr<tflite::FlatBufferModel> model_;
  std::unique_ptr<tflite::Interpreter> interpreter_;
};

}  // namespace codec
}  // namespace chromemedia

#endif  // LYRA_TFLITE_MODEL_WRAPPER_H_
