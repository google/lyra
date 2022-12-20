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

#include "lyra/tflite_model_wrapper.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "glog/logging.h"  // IWYU pragma: keep
#include "include/ghc/filesystem.hpp"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/signature_runner.h"

namespace chromemedia {
namespace codec {

std::unique_ptr<TfLiteModelWrapper> TfLiteModelWrapper::Create(
    const ghc::filesystem::path& model_file, bool use_xnn,
    bool int8_quantized) {
  auto model = tflite::FlatBufferModel::BuildFromFile(model_file.c_str());
  if (model == nullptr) {
    LOG(ERROR) << "Could not build TFLite FlatBufferModel for file: "
               << model_file;
    return nullptr;
  }

  // Disable any default delegate and explicitly control which delegate
  // to use below.
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;

  auto builder = tflite::InterpreterBuilder(*model, resolver);
  if (builder.SetNumThreads(1) != 0) {
    LOG(ERROR) << "Failed to SetNumThreads in TFLite interpreter.";
    return nullptr;
  }

  std::unique_ptr<tflite::Interpreter> interpreter;
  if (builder(&interpreter) != kTfLiteOk) {
    LOG(ERROR) << "Could not build TFLite Interpreter for file: " << model_file;
    return nullptr;
  }

  // Start of XNNPack delegate creation.
  if (use_xnn) {
    // Enable XXNPack.
    auto options = TfLiteXNNPackDelegateOptionsDefault();
    // TODO(b/219786261) Remove once XNNPACK is enabled by default.
    options.flags |= TFLITE_XNNPACK_DELEGATE_FLAG_QU8;
    options.num_threads = 1;
    auto delegate =
        std::unique_ptr<TfLiteDelegate, std::function<void(TfLiteDelegate*)> >(
            TfLiteXNNPackDelegateCreate(&options),
            &TfLiteXNNPackDelegateDelete);
    // Allow dynamic tensors.
    // TODO(b/204470960): Remove this flag once the bug is fixed.
    delegate->flags |= kTfLiteDelegateFlagsAllowDynamicTensors;

    auto status = interpreter->ModifyGraphWithDelegate(std::move(delegate));
    if (status == kTfLiteDelegateError) {
      LOG(WARNING) << "Failed to set delegate; continuing without.";
    } else if (status != kTfLiteOk) {
      LOG(ERROR) << "Failed to set delegate, and cannot continue.";
      return nullptr;
    }
  }
  // End of XNNPack delegate creation.

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    LOG(ERROR) << "Could not allocate quantize TFLite tensors for file: "
               << model_file;
    return nullptr;
  }

  return absl::WrapUnique(
      new TfLiteModelWrapper(std::move(model), std::move(interpreter)));
}

TfLiteModelWrapper::TfLiteModelWrapper(
    std::unique_ptr<tflite::FlatBufferModel> model,
    std::unique_ptr<tflite::Interpreter> interpreter)
    : model_(std::move(model)), interpreter_(std::move(interpreter)) {}

bool TfLiteModelWrapper::Invoke() {
  return interpreter_->Invoke() == kTfLiteOk;
}

tflite::SignatureRunner* TfLiteModelWrapper::GetSignatureRunner(
    const char* signature) {
  return interpreter_->GetSignatureRunner(signature);
}

bool TfLiteModelWrapper::ResetVariableTensors() {
  return interpreter_->ResetVariableTensors() == kTfLiteOk;
}

int TfLiteModelWrapper::num_input_tensors() {
  return interpreter_->inputs().size();
}

int TfLiteModelWrapper::num_output_tensors() {
  return interpreter_->outputs().size();
}

}  // namespace codec
}  // namespace chromemedia
