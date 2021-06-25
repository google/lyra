// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "wavegru_model_impl.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "buffer_merger.h"
#include "causal_convolutional_conditioning.h"
#include "glog/logging.h"
#include "include/ghc/filesystem.hpp"
#include "lyra_wavegru.h"
#include "sparse_matmul/sparse_matmul.h"
// IWYU pragma: no_include "speech/greco3/core/thread.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/types/optional.h"

#ifdef BENCHMARK
#include "absl/time/time.h"
#endif  // BENCHMARK

namespace chromemedia {
namespace codec {

std::unique_ptr<WavegruModelImpl> WavegruModelImpl::Create(
    int num_samples_per_hop, int num_features, int num_frames_per_packet,
    float silence_value, const ghc::filesystem::path& model_path) {
  const int kNumThreads = 1;
  const int kNumCondHiddens = 512;
  const std::string kModelPrefix = "lyra_16khz";

  auto wavegru = LyraWavegru<ComputeType>::Create(
      kNumThreads, std::string(model_path), kModelPrefix);
  if (wavegru == nullptr) {
    LOG(ERROR) << "Could not create wavegru.";
    return nullptr;
  }

  auto merge_filter = BufferMerger::Create(wavegru->num_split_bands());
  if (merge_filter == nullptr) {
    LOG(ERROR) << "Could not create merge filter.";
    return nullptr;
  }
  // WrapUnique is used because of private c'tor.
  return absl::WrapUnique(new WavegruModelImpl(
      std::string(model_path), kModelPrefix, kNumThreads, num_features,
      kNumCondHiddens, num_samples_per_hop, num_frames_per_packet,
      silence_value, std::move(wavegru), std::move(merge_filter)));
}

WavegruModelImpl::WavegruModelImpl(
    const std::string& model_path, const std::string& model_prefix,
    int num_threads, int num_features, int num_cond_hiddens,
    int num_samples_per_hop, int num_frames_per_packet, float silence_value,
    std::unique_ptr<LyraWavegru<ComputeType>> wavegru,
    std::unique_ptr<BufferMerger> buffer_merger)
    : num_threads_(num_threads),
      num_samples_per_hop_(num_samples_per_hop),
      model_split_samples_(wavegru->num_split_bands()),
      wavegru_(std::move(wavegru)),
      buffer_merger_(std::move(buffer_merger)) {
  // The number of samples generated per band is based on the model, not
  // requested sampling rate. If the requested sample rate is less than the
  // model sample rate we just merge less bands.
  for (auto& band : model_split_samples_) {
    band.reserve(num_samples_per_hop_ / wavegru_->num_split_bands());
  }
  background_threads_.reserve(num_threads - 1);
  LOG(INFO) << "Feature size: " << num_features;
  LOG(INFO) << "Number of samples per hop: " << num_samples_per_hop_;

  conditioning_ = absl::make_unique<ConditioningType>(
      num_features, num_cond_hiddens, wavegru_->num_gru_hiddens(),
      num_samples_per_hop_, num_frames_per_packet,
      /*num_threads=*/1, silence_value, model_path, model_prefix);
}

WavegruModelImpl::~WavegruModelImpl() {
  wavegru_->TerminateThreads();
  for (const auto& thread : background_threads_) {
    thread->join();
  }
}

void WavegruModelImpl::AddFeatures(const std::vector<float>& features) {
  const int kNumFrames = 1;
  csrblocksparse::FatCacheAlignedVector<float> input(features.size(),
                                                     kNumFrames);
  std::copy(features.begin(), features.end(), input.data());
  wavegru_->ResetConditioningStart();

#ifdef BENCHMARK
  const int64_t conditioning_start_microsecs = absl::ToUnixMicros(absl::Now());
#endif  // BENCHMARK
  buffer_merger_->Reset();
  conditioning_->Precompute(input, /*num_threads=*/1);
#ifdef BENCHMARK
  conditioning_timings_microsecs_.push_back(absl::ToUnixMicros(absl::Now()) -
                                            conditioning_start_microsecs);
#endif  // BENCHMARK
}

absl::optional<std::vector<int16_t>> WavegruModelImpl::GenerateSamples(
    int num_samples) {
  // Launch background threads on the first packet.
  if (background_threads_.empty() && num_threads_ > 1) {
    // |tid| = 0 is reserved for the main thread which will be returned to the
    // caller.
    LOG(INFO) << "Starting up background threads for wavegru.";
    for (int tid = 1; tid < num_threads_; ++tid) {
      auto f = [&, tid]() {
        wavegru_->SampleThreaded(tid, conditioning_.get(),
                                 &model_split_samples_, 0);
      };
      background_threads_.emplace_back(absl::make_unique<std::thread>(f));
    }
  }

  const int kLocalTid = 0;
#ifdef BENCHMARK
  const int64_t wavegru_start_microsecs = absl::ToUnixMicros(absl::Now());
#endif  // BENCHMARK

  // Without specifying the |-> const std::vector<std::vector<int16_t>>&| the
  // return type of the lambda is inferred and will be a copy of
  // |model_split_samples_|. The std::function will then incorrectly return a
  // reference to this local variable which is destroyed as soon as the
  // std::function call operator is returned.
  std::function<const std::vector<std::vector<int16_t>>&(int)>
      sample_generator = [&](int num_samples_to_generate)
      -> const std::vector<std::vector<int16_t>>& {
    const int num_samples_to_generate_per_band =
        num_samples_to_generate / wavegru_->num_split_bands();
    for (auto& band : model_split_samples_) {
      band.resize(num_samples_to_generate_per_band);
    }

    // The background threads will wait at the beginning of their sample
    // generation loops until the main thread executes this function.
    int num_samples_generated = wavegru_->SampleThreaded(
        kLocalTid, conditioning_.get(), &model_split_samples_,
        num_samples_to_generate);
    CHECK_EQ(num_samples_generated, num_samples_to_generate)
        << "Model did not generate the right number of samples.";
    return model_split_samples_;
  };

  // Only ask the buffer merger for the min of the number of requested samples
  // and the number we actually generated, because the model may have run out of
  // conditioning but the BufferAndMerge retains state until Reset() is called.
  auto samples = buffer_merger_->BufferAndMerge(sample_generator, num_samples);
#ifdef BENCHMARK
  model_timings_microsecs_.push_back(absl::ToUnixMicros(absl::Now()) -
                                     wavegru_start_microsecs);
#endif  // BENCHMARK
  return samples;
}

}  // namespace codec
}  // namespace chromemedia
