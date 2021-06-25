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

#ifndef LYRA_CODEC_WAVEGRU_MODEL_IMPL_H_
#define LYRA_CODEC_WAVEGRU_MODEL_IMPL_H_

#include <cstdint>
#include <memory>
#include <string>
#include <thread>  // NOLINT
#include <vector>

#include "absl/types/optional.h"
#include "buffer_merger.h"
#include "causal_convolutional_conditioning.h"
#include "generative_model_interface.h"
#include "include/ghc/filesystem.hpp"
#include "lyra_types.h"
#include "lyra_wavegru.h"
#include "sparse_matmul/sparse_matmul.h"

namespace chromemedia {
namespace codec {

// Wraps a custom Wavegru C++ implementation.
class WavegruModelImpl : public GenerativeModelInterface {
 public:
  // Returns a nullptr on failure.
  static std::unique_ptr<WavegruModelImpl> Create(
      int num_samples_per_hop, int num_features, int num_frames_per_packet,
      float silence_value, const ghc::filesystem::path& model_path);

  ~WavegruModelImpl() override;

  void AddFeatures(const std::vector<float>& features) override;

  absl::optional<std::vector<int16_t>> GenerateSamples(
      int num_samples) override;

 private:
#ifdef USE_FIXED16
  using ComputeType = csrblocksparse::fixed16_type;
#elif USE_BFLOAT16
  using ComputeType = csrblocksparse::bfloat16;
#else
  using ComputeType = float;
#endif  // USE_FIXED16
  using ConditioningType =
      CausalConvolutionalConditioning<ConditioningTypes<ComputeType>>;

  WavegruModelImpl() = delete;
  WavegruModelImpl(const std::string& model_path,
                   const std::string& model_prefix, int num_threads,
                   int num_features, int num_cond_hiddens,
                   int num_samples_per_hop, int num_frames_per_packet,
                   float silence_value,
                   std::unique_ptr<LyraWavegru<ComputeType>> wavegru,
                   std::unique_ptr<BufferMerger> buffer_merger);

  const int num_threads_;
  const int num_samples_per_hop_;

  // The direct output samples from the model in the split domain.
  std::vector<std::vector<int16_t>> model_split_samples_;
  std::vector<std::unique_ptr<std::thread>> background_threads_;

  std::unique_ptr<LyraWavegru<ComputeType>> wavegru_;
  std::unique_ptr<ConditioningType> conditioning_;
  std::unique_ptr<BufferMerger> buffer_merger_;
};

}  // namespace codec
}  // namespace chromemedia

#endif  // LYRA_CODEC_WAVEGRU_MODEL_IMPL_H_
