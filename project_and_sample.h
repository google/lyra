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

#ifndef LYRA_CODEC_PROJECT_AND_SAMPLE_H_
#define LYRA_CODEC_PROJECT_AND_SAMPLE_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "glog/logging.h"
#include "lyra_types.h"
#include "sparse_matmul/sparse_matmul.h"

namespace chromemedia {
namespace codec {

// Class to encapsulate the projection/mixture_of_logistics/softmax/sampling
// layers that are used in Wavegru. This enables improvements to be made in a
// single place to benefit all models.
// Layers included are:
// projection: a 1x1 convolution with RELU non-linearity.
// mixture_of_logistics: linear layers outputting mixes, scales, and means for
// each logistics distribution in a mixture.
// softmax: of size 2^num_bits.
// Multinomial random sampling from the probability distribution modelled by the
// softmax.
// These layers don't parallelize well as the inter-thread communication
// typically exceeds the saving resulting from computing with more parallelism.
// As the projection gets faster with more threads, so the mixture_of_logistics
// and softmax typically get slower. Some hand-crafted optimization is necessary
// here, and this factored class is the place it is going to be.
template <typename Types>
class ProjectAndSample {
 public:
  using DiskWeightType = typename Types::DiskWeightType;
  using ScratchType = typename Types::ScratchType;
  using ProjWeightType = typename Types::ProjWeightType;
  using ProjRhsType = typename Types::ProjRhsType;
  using ProjMatMulOutType = typename Types::ProjMatMulOutType;
  using ScaleWeightType = typename Types::ScaleWeightType;
  using MeanWeightType = typename Types::MeanWeightType;
  using MixWeightType = typename Types::MixWeightType;
  using ScaleMatMulOutType =
      typename csrblocksparse::TypeOfProduct<ScaleWeightType,
                                             ProjMatMulOutType>::type;
  using MeanMatMulOutType =
      typename csrblocksparse::TypeOfProduct<MeanWeightType,
                                             ProjMatMulOutType>::type;
  using MixMatMulOutType =
      typename csrblocksparse::TypeOfProduct<MixWeightType,
                                             ProjMatMulOutType>::type;

  explicit ProjectAndSample(float probability_offset = 1e-5f,
                            float temperature = 1.f)
      : probability_offset_(probability_offset), temperature_(temperature) {}

  void set_time_components(bool time_components) {
    time_components_ = time_components;
  }

  // Use prefix to load the various weights and biases associated with the model
  void LoadRaw(const std::string& path, const std::string& prefix,
               bool zipped) {
    // compiler gets confused by putting this inside CHECK, thinks it is
    // multiple arguments to CHECK itself.
    auto LoadLayer =
        csrblocksparse::LoadSparseLayer<ProjWeightType, ProjRhsType,
                                        DiskWeightType>;
    CHECK(LoadLayer(prefix + "proj_", zipped, &proj_layer_, path).ok());

    auto LoadMixLayer =
        csrblocksparse::LoadLogitLayer<MixWeightType, ProjMatMulOutType,
                                       DiskWeightType>;
    CHECK(LoadMixLayer(prefix + "mix_", zipped, path, &mix_layer_).ok());

    auto LoadMeanLayer =
        csrblocksparse::LoadLogitLayer<MeanWeightType, ProjMatMulOutType,
                                       DiskWeightType>;
    CHECK(LoadMeanLayer(prefix + "means_", zipped, path, &mean_layer_).ok());

    auto LoadScaleLayer =
        csrblocksparse::LoadLogitLayer<ScaleWeightType, ProjMatMulOutType,
                                       DiskWeightType>;
    CHECK(LoadScaleLayer(prefix + "scales_", zipped, path, &scale_layer_).ok());
  }

  ~ProjectAndSample() {}

  int PrepareForThreads(int num_threads) {
    if (num_threads == num_threads_) return num_threads_;
    num_threads_ = num_threads;
    InitLoadedLayers(num_threads);
    if (num_threads > 1) {
      barrier_ = absl::make_unique<csrblocksparse::SpinBarrier>(num_threads_);
    } else {
      barrier_ = nullptr;
    }
    CHECK_EQ(num_threads, proj_layer_.PrepareForThreads(num_threads));
    CHECK_EQ(1, mix_layer_.PrepareForThreads(1));
    CHECK_EQ(1, mean_layer_.PrepareForThreads(1));
    CHECK_EQ(1, scale_layer_.PrepareForThreads(1));
    return this->num_threads_;
  }

  // Runs the proj layer on the proj_h input, and whichever sampling is
  // required. Returns the value of the sample, or places samples in
  // output_samples for MoL with depth > 1.
  void GetSamples(const csrblocksparse::MutableVectorView<ProjRhsType>& proj_h,
                  int tid, std::minstd_rand* thread_local_gen,
                  csrblocksparse::CacheAlignedVector<ScratchType>* sample_tmp,
                  int num_samples, int* output_samples) {
    absl::Time t_start;
    if (time_components_) t_start = absl::Now();
    auto output = proj_out_.slice(0);
    proj_layer_.MatVec(proj_h, /*relu=*/true, tid, num_proj_replicas_,
                       proj_layer_.rows(), &output);
    if (barrier_ != nullptr) barrier_->barrier();
    if (time_components_ && tid == 0) {
      absl::Time t_now = absl::Now();
      proj_duration_ += t_now - t_start;
      t_start = t_now;
    }
    MolSamples(tid, thread_local_gen, num_samples, output_samples);
  }

  // The next multiple of 8 of the output size of the mix layer. This
  // will be set as the size of the scratch space mixes_, because calling
  // mixes_.Sample() requires that the size is a multiple of 8.
  int expanded_mixes_size() const {
    return (mixes_size() / 8 + (mixes_size() % 8 == 0 ? 0 : 1)) * 8;
  }

  std::size_t ModelSize() const {
    return proj_layer_.bytes() + mix_layer_.bytes() + mean_layer_.bytes() +
           scale_layer_.bytes();
  }

  std::string ReportTiming() const {
    std::string times =
        absl::StrCat(absl::ToDoubleSeconds(proj_duration_), "\t",
                     absl::ToDoubleSeconds(mixture_of_logistics_duration_),
                     "\t", absl::ToDoubleSeconds(samp_duration_), "\n");
    LOG(INFO) << "Times=proj, mixture_of_logistics, samp=" << times;
    return times;
  }

 private:
  void InitLoadedLayers(int num_threads) {
    const int size = proj_size();
    int output_bins = expanded_mixes_size();
#ifdef __AVX2__
    num_proj_replicas_ = num_threads;
#else
    num_proj_replicas_ = 1;
#endif

    // working space for activations
    proj_out_ =
        std::move(csrblocksparse::FatCacheAlignedVector<ProjMatMulOutType>(
            size, num_proj_replicas_));
    mixes_ = std::move(
        csrblocksparse::CacheAlignedVector<MixMatMulOutType>(output_bins));
    // If the number of output_bins has been rounded up, the
    // extra bins will go unset by SPMM, so we initialize the entire result
    // vector with a value that will not disturb the softmax calculation.
    mixes_.FillWith(
        static_cast<MixMatMulOutType>(std::numeric_limits<float>::lowest()));
    CHECK_EQ(size, mix_layer_.cols());
    CHECK_EQ(size, mean_layer_.cols());
    CHECK_EQ(size, scale_layer_.cols());
    means_ = std::move(
        csrblocksparse::CacheAlignedVector<MeanMatMulOutType>(output_bins));
    scales_ = std::move(
        csrblocksparse::CacheAlignedVector<ScaleMatMulOutType>(output_bins));
    mol_sample_tmp_ = csrblocksparse::CacheAlignedVector<float>(output_bins);
  }

  void MolSamples(int tid, std::minstd_rand* thread_local_gen, int num_samples,
                  int* output_samples) {
    DCHECK_NE(output_samples, nullptr);
    absl::Time t_start;
    if (time_components_) t_start = absl::Now();
    if (tid == 0) {
      // If there are two threads, we run the mix layer and its sampling in one,
      // and the mean + scale layers in the other. If there are more than two
      // threads, the others are not used, as more than 2 threads isn't really
      // helpful.
      mix_layer_.MatVec(proj_out_.slice(std::min(tid, num_proj_replicas_ - 1)),
                        /*relu=*/false, 0, /*replicas*/ 1, /*stride*/ 0,
                        &mixes_);
      int mixtures_per_sample = mixes_.size() / num_samples;
      for (int i = 0; i < num_samples; i++) {
        output_samples[i] = mixes_.ScalarSample(
            temperature_, thread_local_gen, &mol_sample_tmp_, tid,
            i * mixtures_per_sample, (i + 1) * mixtures_per_sample);
      }
    }
    if (tid == num_threads_ - 1) {
      mean_layer_.MatVec(proj_out_.slice(std::min(tid, num_proj_replicas_ - 1)),
                         /*relu=*/false, 0, /*replicas*/ 1, /*stride*/ 0,
                         &means_);
      scale_layer_.MatVec(
          proj_out_.slice(std::min(tid, num_proj_replicas_ - 1)),
          /*relu=*/false, 0, /*replicas*/ 1, /*stride*/ 0, &scales_);
    }
    if (barrier_ != nullptr) barrier_->barrier();
    if (tid > 0) return;
    if (time_components_) {
      absl::Time t_now = absl::Now();
      mixture_of_logistics_duration_ += t_now - t_start;
      t_start = t_now;
    }
    for (int s = 0; s < num_samples; s++) {
      int index = output_samples[s];
      float mean = static_cast<float>(means_[index]);
      float scale = static_cast<float>(scales_[index]);
      // Softplus the scale.
      scale = logf(expf(scale) + 1.0f);
      std::uniform_real_distribution<float> dist;

      // Truncated logistic distribution.
      const float kProbabilityScale = 1.0f - 2.0f * probability_offset_;
      float prob =
          dist(*thread_local_gen) * kProbabilityScale + probability_offset_;
      float f_result = mean + scale * log((1.0f - prob) / prob);
      int result = std::min(
          static_cast<int>(std::numeric_limits<int16_t>::max()),
          std::max(static_cast<int>(std::numeric_limits<int16_t>::min()),
                   static_cast<int>(f_result * 256)));
      output_samples[s] = result;
    }

    if (time_components_) {
      absl::Time t_now = absl::Now();
      samp_duration_ += t_now - t_start;
      t_start = t_now;
    }
  }

  int proj_size() const { return proj_layer_.rows(); }
  int mixes_size() const {
    int output_bins = mix_layer_.rows();
#ifdef __AVX2__
    output_bins = ((output_bins + kSIMDWidth - 1) / kSIMDWidth) * kSIMDWidth;
#endif
    return output_bins;
  }

  float probability_offset_;
  float temperature_;

  int num_threads_ = 0;
  int num_proj_replicas_ = 0;
  std::unique_ptr<csrblocksparse::SpinBarrier> barrier_;
  // Parameters of the model.
  csrblocksparse::SparseLinearLayer<ProjWeightType, ProjRhsType> proj_layer_;
  csrblocksparse::SparseLinearLayer<MixWeightType, ProjMatMulOutType>
      mix_layer_;
  csrblocksparse::SparseLinearLayer<MeanWeightType, ProjMatMulOutType>
      mean_layer_;
  csrblocksparse::SparseLinearLayer<ScaleWeightType, ProjMatMulOutType>
      scale_layer_;
  // Scratch space for computation
  csrblocksparse::FatCacheAlignedVector<ProjMatMulOutType> proj_out_;
  csrblocksparse::CacheAlignedVector<MixMatMulOutType> mixes_;
  csrblocksparse::CacheAlignedVector<MeanMatMulOutType> means_;
  csrblocksparse::CacheAlignedVector<ScaleMatMulOutType> scales_;
  csrblocksparse::CacheAlignedVector<float> mol_sample_tmp_;

  bool time_components_ = false;
  absl::Duration proj_duration_;
  absl::Duration mixture_of_logistics_duration_;
  absl::Duration samp_duration_;
  // Maximum possible width of a SIMD register in floats.
  static constexpr int kSIMDWidth = 16;
};

}  // namespace codec
}  // namespace chromemedia

#endif  // LYRA_CODEC_PROJECT_AND_SAMPLE_H_
