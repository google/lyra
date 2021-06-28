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

#ifndef LYRA_CODEC_LYRA_WAVEGRU_H_
#define LYRA_CODEC_LYRA_WAVEGRU_H_

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/time/clock.h"  // for SleepFor
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "causal_convolutional_conditioning.h"
#include "dsp_util.h"
#include "glog/logging.h"
#include "include/ghc/filesystem.hpp"
#include "layer_wrappers_lib.h"
#include "lyra_types.h"
#include "project_and_sample.h"
#include "sparse_matmul/sparse_matmul.h"

namespace chromemedia {
namespace codec {

template <typename WeightTypeKind>
class LyraWavegru {
 public:
  using Types = WavegruTypes<WeightTypeKind>;
  using ArWeightType = typename Types::ArWeightType;
  using ArRhsType = typename Types::ArRhsType;
  using ArOutputType = typename Types::ArOutputType;
  using GruWeightType = typename Types::GruWeightType;
  using GruStateType = typename Types::GruStateType;
  using GruRhsType = typename Types::GruRhsType;
  using DiskWeightType = typename Types::DiskWeightType;
  using ScratchType = typename Types::ScratchType;

  using ArLayerType =
      LayerWrapper<ArWeightType, ArRhsType, ArOutputType, DiskWeightType>;
  using GruLayerType =
      LayerWrapper<GruWeightType, GruStateType, GruRhsType, DiskWeightType>;

  using ConditioningType =
      CausalConvolutionalConditioning<ConditioningTypes<WeightTypeKind>>;

  // TODO(b/161747203): Use LayerWrapper for the project and sample layer.
  using ProjectAndSampleType =
      ProjectAndSample<ProjectAndSampleTypes<WeightTypeKind>>;

  static std::unique_ptr<LyraWavegru<WeightTypeKind>> Create(
      int num_threads, const ghc::filesystem::path& path,
      const std::string& prefix) {
#if defined __aarch64__
    LOG(INFO)
        << "lyra_wavegru running fast multiplication kernels for aarch64.";
#elif defined __AVX__
    LOG(INFO) << "lyra_wavegru running fast multiplication kernels for AVX.";
#else   // defined __AVX__
    LOG(WARNING) << "lyra_wavegru running in slow generic mode.";
#endif  // defined __aarch64__

    LayerParams ar_to_gates_params{.num_input_channels = kNumSplitBands,
                                   .num_filters = 3 * kNumGruHiddens,
                                   .length = 1,
                                   .kernel_size = 1,
                                   .dilation = 1,
                                   .stride = 1,
                                   .relu = false,
                                   .skip_connection = false,
                                   .type = LayerType::kConv1D,
                                   .num_threads = num_threads,
                                   .per_column_barrier = false,
                                   .from =
                                       LayerParams::FromDisk{
                                           .path = path.string(),
                                           .zipped = true,
                                       },
                                   .prefix = prefix + "_ar_to_gates_"};
    auto ar_to_gates_layer = ArLayerType::Create(ar_to_gates_params);
    if (ar_to_gates_layer == nullptr) {
      return nullptr;
    }

    LayerParams gru_params{.num_input_channels = kNumGruHiddens,
                           .num_filters = 3 * kNumGruHiddens,
                           .length = 1,
                           .kernel_size = 1,
                           .dilation = 1,
                           .stride = 1,
                           .relu = false,
                           .skip_connection = false,
                           .type = LayerType::kConv1D,
                           .num_threads = num_threads,
                           .per_column_barrier = false,
                           .from =
                               LayerParams::FromDisk{
                                   .path = path.string(),
                                   .zipped = true,
                               },
                           .prefix = prefix + "_gru_layer_"};
    auto gru_layer = GruLayerType::Create(gru_params);
    if (gru_layer == nullptr) {
      return nullptr;
    }

    auto project_and_sample_layer = absl::make_unique<ProjectAndSampleType>();
    project_and_sample_layer->LoadRaw(path, prefix + "_", /*zipped=*/true);
    if (project_and_sample_layer->PrepareForThreads(num_threads) !=
        num_threads) {
      LOG(ERROR) << "Could not prepare project_and_sample for " << num_threads
                 << " threads.";
      return nullptr;
    }
    return absl::WrapUnique(new LyraWavegru<WeightTypeKind>(
        num_threads, std::move(ar_to_gates_layer), std::move(gru_layer),
        std::move(project_and_sample_layer)));
  }

  // The |num_samples_to_generate| is only used by the main thread, which
  // will store this number in |num_samples_to_generate_|. All background
  // threads will load from |num_samples_to_generate_| for each iteration of the
  // while loop.
  int SampleThreaded(int tid, ConditioningType* conditioning,
                     std::vector<std::vector<int16_t>>* split_band_samples,
                     int num_samples_to_generate) {
    int num_samples_generated = 0;
    // Thread with |tid| = 0 will break out of this while loop after 1
    // iteration. All other threads will be in this while loop until
    // |terminate_threads_| is set to true.
    while (!terminate_threads_.load()) {
      // |main_thread_ready_| is used to let background threads do a non-busy
      // wait while the main thread is returned to the caller to do extra work
      // outside such as packet handling.
      if (tid == 0) {
        num_samples_to_generate_.store(num_samples_to_generate);
      } else {
        // Background threads will effectively non-busy wait here until
        // |num_samples_to_generate_| is set to a value greater than 0.
        if (num_samples_to_generate_.load() <= 0) {
          absl::SleepFor(absl::Microseconds(100));
          continue;
        }
      }
      num_samples_generated = SamplingBody(
          spin_barrier_.get(), tid, conditioning, split_band_samples, nullptr);

      // Synchronize all threads before the main thread updates
      // |num_samples_to_generate_|, otherwise the background threads may not
      // have the chance to run SamplingBody().
      spin_barrier_->barrier();

      // Synchronize all threads after the main thread updates
      // |num_samples_to_generate_|, otherwise on the final packet the
      // background threads may bypass the
      // (num_samples_to_generate_.load() <= 0) check and
      // become stuck at the first spin barrier in |SamplingBody| which will
      // never be matched by the main thread.
      if (tid == 0) {
        conditioning_start_.store(conditioning_start_.load() +
                                  num_samples_generated);
        num_samples_to_generate_.store(0);
      }
      spin_barrier_->barrier();
      if (tid == 0) {
        break;
      }
    }
    return num_samples_generated;
  }

  // Causes all threads with |tid| != 0 to break out of their |SamplingBody|
  // loop.
  void TerminateThreads() { terminate_threads_.store(true); }

  void ResetConditioningStart() { conditioning_start_.store(0); }

  int num_gru_hiddens() const { return kNumGruHiddens; }

  int num_split_bands() const { return kNumSplitBands; }

 private:
  static constexpr int kNumGruHiddens = 1024;
  static constexpr int kNumSplitBands = 4;

  LyraWavegru() = delete;

  LyraWavegru(int num_threads, std::unique_ptr<ArLayerType> ar_to_gates_layer,
              std::unique_ptr<GruLayerType> gru_layer,
              std::unique_ptr<ProjectAndSampleType> project_and_sample_layer)
      : num_threads_(num_threads),
        ar_to_gates_layer_(std::move(ar_to_gates_layer)),
        gru_layer_(std::move(gru_layer)),
        project_and_sample_layer_(std::move(project_and_sample_layer)),
        sample_at_s_(kNumSplitBands),
        terminate_threads_(false),
        num_samples_to_generate_(0),
        conditioning_start_(0) {
    InitLoadedLayers();
    InitializeGenerators();
    spin_barrier_ =
        absl::make_unique<csrblocksparse::SpinBarrier>(num_threads_);
  }

  void InitLoadedLayers() {
    LOG(INFO) << "Model size: " << ModelSize() << " bytes";
    // Working space for activations.
    ar_output_buffer_ = csrblocksparse::CacheAlignedVector<ArOutputType>(
        ar_to_gates_layer_->rows());
    ar_output_buffer_.FillZero();
    ar_and_cond_to_gates_buffer_ =
        csrblocksparse::CacheAlignedVector<GruRhsType>(
            ar_to_gates_layer_->rows());
    ar_and_cond_to_gates_buffer_.FillZero();
    gru_gates_buffer_ =
        csrblocksparse::CacheAlignedVector<GruRhsType>(gru_layer_->rows());
    gru_gates_buffer_.FillZero();
  }

  std::size_t ModelSize() const {
    return gru_layer_->bytes() + project_and_sample_layer_->ModelSize() +
           ar_to_gates_layer_->bytes();
  }

  int SamplingBody(
      csrblocksparse::SpinBarrier* spin_barrier, int tid,
      ConditioningType* conditioning,
      std::vector<std::vector<int16_t>>* split_band_samples,
      const std::function<void(int16_t*, int, int, int)>& /*unused*/) {
    CHECK_EQ(kNumSplitBands, split_band_samples->size());
    const int conditioning_start = conditioning_start_.load();
    const int num_samples_to_generate =
        std::min(num_samples_to_generate_.load(),
                 conditioning->num_samples() - conditioning_start);
    // We can only generate samples in multiples of |kNumSplitBands|.
    CHECK_EQ(num_samples_to_generate % kNumSplitBands, 0);
    CHECK_GE(num_samples_to_generate, 0);

    // This is a scratch space, whose size should be multiple of 8.
    csrblocksparse::CacheAlignedVector<ScratchType> sample_tmp(
        project_and_sample_layer_->expanded_mixes_size());

    sample_tmp.FillZero();

    std::minstd_rand* thread_local_gen = &thread_local_gens_[tid];

    int start, end;
    std::tie(start, end) = ComputeStartAndEnd(tid, kNumGruHiddens);

    for (int s = 0; s < num_samples_to_generate; s += kNumSplitBands) {
      // Bring the AR sample(s) up to 3 * kNumGruHiddens.
      ar_to_gates_layer_->Run(tid, spin_barrier,
                              ar_output_buffer_.AsMutableView());

      // Sum the conditioning and autoregressive output.
      SumConditioningAndAutoregressive(
          conditioning->AtStep(conditioning_start + s), 3 * start, 3 * end,
          spin_barrier);

      // Pass through the GRU layer.
      gru_layer_->Run(tid, spin_barrier, gru_gates_buffer_.AsMutableView());
      gru_gates_
          .template GruWithARInput<csrblocksparse::ARInputsMode::k0ARInputs>(
              start, end, /*state_size=*/kNumGruHiddens,
              /*gru_recurrent_ptr=*/gru_gates_buffer_.data(),
              /*input_ptr=*/ar_and_cond_to_gates_buffer_.data(),
              /*gru_state_ptr=*/gru_layer_->InputViewToUpdate().data());
      spin_barrier->barrier();

      // Project and sample.
      project_and_sample_layer_->GetSamples(
          gru_layer_->InputViewToUpdate(), tid, thread_local_gen, &sample_tmp,
          kNumSplitBands, sample_at_s_.data());

      if (tid == 0) {
        // Loop back the samples as the input of |ar_to_gates_layer_| for the
        // next step.
        auto sample_at_sminus1 = ar_to_gates_layer_->InputViewToUpdate();
        for (int i = 0; i < kNumSplitBands; ++i) {
          sample_at_sminus1[i] =
              static_cast<ArRhsType>(SampleToFloat(sample_at_s_.at(i)));
          split_band_samples->at(i).at(s / kNumSplitBands) = sample_at_s_.at(i);
        }
      }
      spin_barrier->barrier();
    }  // end of for (int s = 0; ...).
    return num_samples_to_generate;
  }

  // Computes the intervals of gru gates to be computed by the given tid.
  std::tuple<int, int> ComputeStartAndEnd(int tid, int state_size) const {
    int factor = gru_gates_.kSIMDWidth;
    factor *= state_size / (factor * num_threads_);
    return std::make_tuple(factor * tid, tid == num_threads_ - 1
                                             ? state_size
                                             : factor * (tid + 1));
  }

  // The range [-32768, 32767] is mapped to floating point by x / 32768.0f
  // resulting in a range of [-1.f, 1.f).
  static float SampleToFloat(int sample) {
    return static_cast<float>(sample) / 32768.0f;
  }

  // Initializes the thread local sampling generators, one per thread of
  // |num_threads_|. This method is called once during construction.
  void InitializeGenerators() {
    // All threads see the same sequence and make the same sampling decisions.
    thread_local_gens_ = std::vector<std::minstd_rand>(num_threads_);
    // Discard the first 10 samples for each generator, to get them into a good
    // state for sampling.
    for (int i = 0; i < num_threads_; ++i) {
      thread_local_gens_[i].discard(10);
    }
  }

  void SumConditioningAndAutoregressive(
      const absl::Span<GruRhsType> conditioning_span, int sum_start,
      int sum_end, csrblocksparse::SpinBarrier* spin_barrier) {
    CastVector(sum_start, sum_end, ar_output_buffer_.data(),
               ar_and_cond_to_gates_buffer_.data());
    csrblocksparse::detail::SumVectors(sum_start, sum_end,
                                       conditioning_span.data(),
                                       ar_and_cond_to_gates_buffer_.data(),
                                       ar_and_cond_to_gates_buffer_.data());
    spin_barrier->barrier();
  }

  const int num_threads_;

  // Random generators for each thread.
  std::vector<std::minstd_rand> thread_local_gens_;

  // Layers.
  // The layer that transforms the AR input to the input of GRU gates is just a
  // column vector with no bias (the combined bias is handled in the
  // conditioning stack).
  std::unique_ptr<ArLayerType> ar_to_gates_layer_;
  std::unique_ptr<GruLayerType> gru_layer_;

  // TODO(b/161747203): Use LayerWrapper for the project and sample layer.
  std::unique_ptr<ProjectAndSampleType> project_and_sample_layer_;
  csrblocksparse::GruGates<GruStateType, GruRhsType, ArRhsType> gru_gates_;

  // Buffers.
  csrblocksparse::CacheAlignedVector<ArOutputType> ar_output_buffer_;
  csrblocksparse::CacheAlignedVector<GruRhsType> ar_and_cond_to_gates_buffer_;
  csrblocksparse::CacheAlignedVector<GruRhsType> gru_gates_buffer_;
  std::vector<int> sample_at_s_;

  std::atomic<bool> terminate_threads_;

  // To support generating any number of samples, the main thread is responsible
  // for setting the number (which will be read by children threads), as well
  // as tracking the position to read next from the conditioning vector.
  std::atomic<int> num_samples_to_generate_;
  std::atomic<int> conditioning_start_;

  std::unique_ptr<csrblocksparse::SpinBarrier> spin_barrier_;
};

}  // namespace codec
}  // namespace chromemedia

#endif  // LYRA_CODEC_LYRA_WAVEGRU_H_
