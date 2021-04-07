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

#include "project_and_sample.h"

#include <cmath>
#include <random>
#include <string>
#include <tuple>
#include <vector>

// placeholder for get runfiles header.
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/strings/str_format.h"
#include "include/ghc/filesystem.hpp"
#include "exported_layers_test.h"
#include "lyra_types.h"
#include "sparse_inference_matrixvector.h"

namespace chromemedia {
namespace codec {

namespace {

static constexpr int kNumSplitBands = 4;
static constexpr int kTestNumGruHiddens = 4;
static constexpr int kExpandedMixesSize = 8;
static constexpr char kPrefix[] = "lyra_";

// For creating typed-tests. We want to test the template class,
// ProjectAndSampleTest, instantiated with different types:
//   1. float: C++'s generic floating point.
//   2. csrblocksparse::fixed16_type: a type that will be used in our Lyra
//      implementation. See the build rule for
//      audio/chromemedia/lyra/codec:wavegru_model_impl.
// Different types would require different tolerance, hence the template class
// Tolerance below.
template <typename WeightTypeKind>
struct Tolerance {
  // Unspecialized Tolerance class does not define |kTolerance|, so an attempt
  // to test a ComputeType that is not one of
  // {float, csrblocksparse::fixed16_type} will result in a compile error.
};

template <>
struct Tolerance<float> {
  static constexpr float kTolerance = 1e-7f;
};

template <>
struct Tolerance<csrblocksparse::fixed16_type> {
  // Fixed-point arithmetic is less accurate than floating-point; hence a higher
  // tolerance.
  static constexpr float kTolerance = 1.5e-2f;
};

// A matcher to compare a tuple (used in testing::Pointwise) and verify that
// the relative error between the two elements is within a tolerance.
MATCHER_P(IsRelativelyClose, relative_tolerance,
          absl::StrFormat("%s approximately equal (relative error <= %g)",
                          negation ? "are not" : "are", relative_tolerance)) {
  const float actual = static_cast<float>(::testing::get<0>(arg));
  const float expected = static_cast<float>(::testing::get<1>(arg));
  return (std::abs(actual - expected) / std::abs(expected)) <=
         relative_tolerance;
}

template <typename WeightTypeKind>
class ProjectAndSampleTest : public ::testing::Test {
 public:
  ProjectAndSampleTest()
      : project_and_sample_layer_(),
        gru_hiddens_(kTestNumGruHiddens, static_cast<ProjRhsType>(0.5f)),
        gru_hiddens_view_(gru_hiddens_.data(), kTestNumGruHiddens, 1),
        testdata_dir_(ghc::filesystem::current_path() /
                      "testdata"),
        scratch_space_(kExpandedMixesSize) {}

 protected:
  using ProjectAndSampleType =
      ProjectAndSample<ProjectAndSampleTypes<WeightTypeKind, 8, 8, 8, 8, 8>>;
  using ScratchType = typename ProjectAndSampleType::ScratchType;
  using ProjRhsType = typename ProjectAndSampleType::ProjRhsType;

  static constexpr float kTolerance = Tolerance<WeightTypeKind>::kTolerance;

  ProjectAndSampleType project_and_sample_layer_;

  // Input to the project-and-sample layer. In Wavegru it is the hidden states
  // of the GRU layer.
  std::vector<ProjRhsType> gru_hiddens_;
  const csrblocksparse::MutableVectorView<ProjRhsType> gru_hiddens_view_;
  const ghc::filesystem::path testdata_dir_;

  // Scratch space for GetSample().
  csrblocksparse::CacheAlignedVector<ScratchType> scratch_space_;
};

using WeightTypeKinds = ::testing::Types<float, csrblocksparse::fixed16_type>;
TYPED_TEST_SUITE(ProjectAndSampleTest, WeightTypeKinds);

TYPED_TEST(ProjectAndSampleTest, LoadAndPrepareSucceed) {
  this->project_and_sample_layer_.LoadRaw(this->testdata_dir_.string(), kPrefix,
                                          /*zipped=*/true);

  EXPECT_EQ(this->project_and_sample_layer_.expanded_mixes_size(),
            kExpandedMixesSize);

  EXPECT_GT(this->project_and_sample_layer_.ModelSize(), 0);
  for (const int num_threads : {1, 2, 4}) {
    EXPECT_EQ(this->project_and_sample_layer_.PrepareForThreads(num_threads),
              num_threads);
  }
}

TYPED_TEST(ProjectAndSampleTest, GetSamplesReturnGoldenValues) {
  this->project_and_sample_layer_.LoadRaw(this->testdata_dir_.string(), kPrefix,
                                          /*zipped=*/true);

  // The result should match the golden values regardless of number of threads.
  const std::vector<int> expected_samples = {-104, -1387, 238, -220};
  for (const int num_threads : {1, 2, 4}) {
    this->project_and_sample_layer_.PrepareForThreads(num_threads);

    // Make sampling deterministic.
    const std::minstd_rand::result_type kSeed = 42;
    std::vector<std::minstd_rand> gen(num_threads, std::minstd_rand(kSeed));
    std::vector<int> actual_samples(kNumSplitBands);
    auto f = [&](csrblocksparse::SpinBarrier* barrier, int tid) {
      this->project_and_sample_layer_.GetSamples(
          this->gru_hiddens_view_, /*tid=*/tid, &gen[tid],
          &this->scratch_space_, kNumSplitBands, actual_samples.data());
      barrier->barrier();
    };
    LaunchOnThreadsWithBarrier(num_threads, f);
    EXPECT_THAT(actual_samples,
                testing::Pointwise(IsRelativelyClose(TestFixture::kTolerance),
                                   expected_samples));
  }
}

TYPED_TEST(ProjectAndSampleTest, ReportTiming) {
  this->project_and_sample_layer_.LoadRaw(this->testdata_dir_.string(), kPrefix,
                                          /*zipped=*/true);
  this->project_and_sample_layer_.set_time_components(true);
  this->project_and_sample_layer_.PrepareForThreads(1);
  std::minstd_rand gen;
  std::vector<int> actual_samples(kNumSplitBands);
  this->project_and_sample_layer_.GetSamples(
      this->gru_hiddens_view_, /*tid=*/0, &gen, &this->scratch_space_,
      kNumSplitBands, actual_samples.data());

  // Verify that the timing is non-empty. We do not care about the content.
  const std::string timing = this->project_and_sample_layer_.ReportTiming();
  EXPECT_FALSE(timing.empty());
}

// Test that exported layers with fixed-point and float weights produce
// matching results.
using csrblocksparse::fixed16_type;
using FixedProjectAndSampleType =
    ProjectAndSample<ProjectAndSampleTypes<fixed16_type>>;
static constexpr int kNumGruHiddens = 1024;
static constexpr int kProjSize = 512;

LayerParams ProjectionLayerParams(int num_input_channels, int num_filters,
                                  bool relu, const std::string& model_path,
                                  const std::string& prefix) {
  return LayerParams{
      .num_input_channels = num_input_channels,
      .num_filters = num_filters,
      .length = 1,
      .kernel_size = 1,
      .dilation = 1,
      .stride = 1,
      .relu = relu,
      .skip_connection = false,
      .type = LayerType::kConv1D,
      .num_threads = 1,
      .per_column_barrier = false,
      .from = LayerParams::FromDisk{.path = model_path, .zipped = true},
      .prefix = prefix};
}

struct ProjLayerTypes {
  using FloatLayerType = LayerWrapper<float, float, float, float>;
  using FixedLayerType =
      LayerWrapper<FixedProjectAndSampleType::ProjWeightType,
                   FixedProjectAndSampleType::ProjRhsType,
                   FixedProjectAndSampleType::ProjMatMulOutType,
                   FixedProjectAndSampleType::DiskWeightType>;
  static LayerParams Params(const std::string& model_path) {
    return ProjectionLayerParams(kNumGruHiddens, kProjSize, true, model_path,
                                 "lyra_16khz_proj_");
  }
};

struct ScaleLayerTypes {
  using FloatLayerType = LayerWrapper<float, float, float, float>;
  using FixedLayerType =
      LayerWrapper<FixedProjectAndSampleType::ScaleWeightType,
                   FixedProjectAndSampleType::ProjMatMulOutType,
                   FixedProjectAndSampleType::ScaleMatMulOutType,
                   FixedProjectAndSampleType::DiskWeightType>;
  static LayerParams Params(const std::string& model_path) {
    return ProjectionLayerParams(kProjSize, kNumSplitBands * kExpandedMixesSize,
                                 false, model_path, "lyra_16khz_scales_");
  }
};

struct MeanLayerTypes {
  using FloatLayerType = LayerWrapper<float, float, float, float>;
  using FixedLayerType =
      LayerWrapper<FixedProjectAndSampleType::MeanWeightType,
                   FixedProjectAndSampleType::ProjMatMulOutType,
                   FixedProjectAndSampleType::MeanMatMulOutType,
                   FixedProjectAndSampleType::DiskWeightType>;
  static LayerParams Params(const std::string& model_path) {
    return ProjectionLayerParams(kProjSize, kNumSplitBands * kExpandedMixesSize,
                                 false, model_path, "lyra_16khz_means_");
  }
};

struct MixLayerTypes {
  using FloatLayerType = LayerWrapper<float, float, float, float>;
  using FixedLayerType =
      LayerWrapper<FixedProjectAndSampleType::MixWeightType,
                   FixedProjectAndSampleType::ProjMatMulOutType,
                   FixedProjectAndSampleType::MixMatMulOutType,
                   FixedProjectAndSampleType::DiskWeightType>;
  static LayerParams Params(const std::string& model_path) {
    return ProjectionLayerParams(kProjSize, kNumSplitBands * kExpandedMixesSize,
                                 false, model_path, "lyra_16khz_mix_");
  }
};

using LayerTypesList = testing::Types<ProjLayerTypes, ScaleLayerTypes,
                                      MeanLayerTypes, MixLayerTypes>;
INSTANTIATE_TYPED_TEST_SUITE_P(ProjectAndSample, ExportedLayersTest,
                               LayerTypesList);

}  // namespace
}  // namespace codec
}  // namespace chromemedia
