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

#include "causal_convolutional_conditioning.h"

#include <algorithm>
#include <iterator>
#include <string>
#include <vector>

// Placeholder for get runfiles header.
#include "absl/types/span.h"
#include "exported_layers_test.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "include/ghc/filesystem.hpp"
#include "lyra_config.h"
#include "lyra_types.h"
#include "sparse_matmul/sparse_matmul.h"

namespace chromemedia {
namespace codec {

// Use a test peer to access the private transpose_conv_2_buffer_ and make the
// test independent of the projection layer of each architecture.
template <typename WeightTypeKind>
class CausalConvolutionalConditioningPeer {
 public:
  using ConditioningType = CausalConvolutionalConditioning<ConditioningTypes<
      WeightTypeKind, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6>>;

  static LayerParams Conv1DParams(int feature_depth, int num_cond_hiddens,
                                  int num_threads,
                                  const std::string& model_path,
                                  const std::string& prefix) {
    return ConditioningType::Conv1DParams(feature_depth, num_cond_hiddens,
                                          num_threads, model_path, prefix);
  }

  static LayerParams DilatedParams(int num_cond_hiddens, int level,
                                   int num_threads,
                                   const std::string& model_path,
                                   const std::string& prefix) {
    return ConditioningType::DilatedParams(num_cond_hiddens, level, num_threads,
                                           model_path, prefix);
  }

  static LayerParams TransposeParams(int num_cond_hiddens, int level,
                                     int num_threads,
                                     const std::string& model_path,
                                     const std::string& prefix) {
    return ConditioningType::TransposeParams(num_cond_hiddens, level,
                                             num_threads, model_path, prefix);
  }

  static LayerParams ConvCondParams(int num_cond_hiddens, int num_hiddens,
                                    int num_threads,
                                    const std::string& model_path,
                                    const std::string& prefix) {
    return ConditioningType::ConvCondParams(num_cond_hiddens, num_hiddens,
                                            num_threads, model_path, prefix);
  }

  static LayerParams ConvToGatesParams(int num_hiddens, int num_threads,
                                       const std::string& model_path,
                                       const std::string& prefix) {
    return ConditioningType::ConvToGatesParams(num_hiddens, num_threads,
                                               model_path, prefix);
  }

  CausalConvolutionalConditioningPeer(int feature_depth, int num_cond_hiddens,
                                      int num_hiddens, int num_samples_per_hop,
                                      int num_frames_per_packet,
                                      int num_threads, const std::string& path,
                                      const std::string& prefix)
      : conditioning_stack_(feature_depth, num_cond_hiddens, num_hiddens,
                            num_samples_per_hop, num_frames_per_packet,
                            num_threads, 0.0f, path, prefix) {}

  void Precompute(const csrblocksparse::FatCacheAlignedVector<float>& input,
                  int num_threads) {
    conditioning_stack_.Precompute(input, num_threads);
  }

  std::vector<float> Transpose2() {
    const auto output_from_transpose_2 =
        conditioning_stack_.conv_cond_layer_->InputViewToUpdate();
    return std::vector<float>(
        output_from_transpose_2.data(),
        output_from_transpose_2.data() +
            output_from_transpose_2.rows() * output_from_transpose_2.cols());
  }

 private:
  ConditioningType conditioning_stack_;
};

namespace {

static const int kCondUpsamplingRatio = 8;
static const int kNumSamplesPerHop = GetNumSamplesPerHop(16000);
static const int kNumSamplesPerCondOutput =
    kNumSamplesPerHop / kCondUpsamplingRatio;

// For creating typed-tests. We want to test the template class,
// CausalConvolutionalConditioning, instantiated with different types:
//   1. float: C++'s generic floating point.
//   2. csrblocksparse::fixed16_type: a type that is used in our Lyra
//      implementation. See the build rule for :wavegru_model_impl.
// Different types would require different tolerance, hence the template class
// Tolerance below.
template <typename ComputeType>
struct Tolerance {
  // Unspecialized Tolerance class does not define |kTolerance|, so an attempt
  // to test a ComputeType that is not one of
  // {float, csrblocksparse::fixed16_type} will result in a compile error.
};

template <>
struct Tolerance<float> {
  static constexpr float kTolerance = 1e-6f;
};

template <>
struct Tolerance<csrblocksparse::fixed16_type> {
  // Fixed-point arithmetic is less accurate than floating-point; hence a higher
  // tolerance.
  // We have a fixed number of bits to allocate to the
  // integer/mantissa fixed point representation. When we use fixed
  // representations clipping leads to unacceptable quality results, so we
  // allocate more bits to the integer vs mantissa component to prevent
  // clipping. This leads to much less precise results as compared to using full
  // floats in calculations.
  static constexpr float kTolerance = 2e-2f;
};

template <typename ComputeType>
class CausalConvolutionalConditioningTest : public ::testing::Test {
 public:
  CausalConvolutionalConditioningTest()
      : testdata_dir_(ghc::filesystem::current_path() / "testdata") {}

 protected:
  const float kTolerance = Tolerance<ComputeType>::kTolerance;
  const ghc::filesystem::path testdata_dir_;
};

using ComputeTypes = ::testing::Types<float, csrblocksparse::fixed16_type>;
TYPED_TEST_SUITE(CausalConvolutionalConditioningTest, ComputeTypes);

// This tests that the conditioning stack matches the results
// ConvConditioningStack produce when initialized with the same hyperparameters.
// The weight matrices are stored as .raw.gz in the testdata/ directory.
TYPED_TEST(CausalConvolutionalConditioningTest,
           ConditioningStackMatchTensorflow) {
  const int kNumFeatures = 3;
  const int kNumCondHiddens = 8;
  const int kNumHiddens = 4;
  const int kNumThreads = 1;
  const int kNumInvalidPaddedFrames = 8;
  const int kNumTotalFrames = 10;
  std::vector<float> features;
  ASSERT_TRUE(csrblocksparse::ReadArrayFromFile("codec.gz", &features,
                                                this->testdata_dir_.c_str())
                  .ok());
  // Obtained from the TensorFlow architecture loaded with the weights used in
  // testdata/ directory.
  // These values were generated 1 in every 80 print outputs is selected and
  // each row rearranged accordingly:
  // python: r1 r2 u1 u2 e1 e2
  // C++:    r1 u1 e1 r2 u2 e2
  // The weights that are loaded in the cpp impl are the transpose of the
  // TensorFlow values.
  // We do not expect the first |kNumInvalidPaddedFrames| to match this
  // implementation because of the additional initial padding used in
  // TensorFlow.
  std::vector<float> expected_cond_out;
  ASSERT_TRUE(csrblocksparse::ReadArrayFromFile("transpose_2.gz",
                                                &expected_cond_out,
                                                this->testdata_dir_.c_str())
                  .ok());

  CausalConvolutionalConditioningPeer<TypeParam> conditioning_stack(
      kNumFeatures, kNumCondHiddens, kNumHiddens, kNumSamplesPerHop,
      kNumFramesPerPacket, kNumThreads, this->testdata_dir_.string(), "lyra");
  csrblocksparse::FatCacheAlignedVector<float> input(kNumFeatures, 1);

  // Run through |kNumInvalidPaddedFrames| without checking the results.
  for (int i = 0; i < kNumInvalidPaddedFrames; ++i) {
    std::copy(features.begin() + i * input.size(),
              features.begin() + (i + 1) * input.size(), input.data());
    conditioning_stack.Precompute(input, kNumThreads);
  }

  for (int i = kNumInvalidPaddedFrames; i < kNumTotalFrames; ++i) {
    std::copy(features.begin() + i * input.size(),
              features.begin() + (i + 1) * input.size(), input.data());
    conditioning_stack.Precompute(input, kNumThreads);

    const auto transpose_2_result = conditioning_stack.Transpose2();
    const auto expected_transpose_2_result =
        expected_cond_out.begin() +
        (i - 1) * kCondUpsamplingRatio * kNumCondHiddens;

    // For testing with csrblocksparse's fixed-point types, there is no
    // overloaded arithmetic operators (e.g. operator+, operator<). So we cannot
    // use testing::Pointwise(). Convert each element to float before
    // comparison.
    for (int k = 0; k < transpose_2_result.size(); ++k) {
      // Since we are inside a derived class template, C++ requires us to visit
      // the members of CausalConvolutionalConditioningTest via 'this'. See
      // https://isocpp.org/wiki/faq/templates#nondependent-name-lookup-members
      EXPECT_NEAR(transpose_2_result[k], expected_transpose_2_result[k],
                  this->kTolerance);
    }
  }
}

TYPED_TEST(CausalConvolutionalConditioningTest,
           MultipleThreadYieldsSameResult) {
  using ConditioningType = CausalConvolutionalConditioning<ConditioningTypes<
      TypeParam, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6>>;

  const int kNumCondHiddens = 8;
  const int kNumHiddens = 4;
  const std::vector<std::vector<float>> kFeatures = {{0.0f, 0.0f, 0.0f},
                                                     {0.0f, 0.0f, 0.0f},
                                                     {1.0f, 1.0f, 1.0f},
                                                     {0.0f, 0.0f, 0.0f}};

  ConditioningType no_multithreading(
      kFeatures.at(0).size(), kNumCondHiddens, kNumHiddens, kNumSamplesPerHop,
      kNumFramesPerPacket, 1, 0.0f, this->testdata_dir_.string(), "lyra");
  ConditioningType two_threads(
      kFeatures.at(0).size(), kNumCondHiddens, kNumHiddens, kNumSamplesPerHop,
      kNumFramesPerPacket, 2, 0.0f, this->testdata_dir_.string(), "lyra");
  csrblocksparse::FatCacheAlignedVector<float> input(kFeatures.at(0).size(), 1);
  for (int i = 0; i < kFeatures.size(); ++i) {
    std::copy(kFeatures.at(i).begin(), kFeatures.at(i).end(), input.data());
    no_multithreading.Precompute(input, 1);
    two_threads.Precompute(input, 2);

    for (int j = 0; j < kCondUpsamplingRatio; ++j) {
      auto no_multithreading_output =
          no_multithreading.AtStep(j * kNumSamplesPerCondOutput);
      auto two_threads_output =
          two_threads.AtStep(j * kNumSamplesPerCondOutput);
      for (int k = 0; k < no_multithreading_output.size(); ++k) {
        EXPECT_FLOAT_EQ(static_cast<float>(no_multithreading_output[k]),
                        static_cast<float>(two_threads_output[k]));
      }
    }
  }
}

TYPED_TEST(CausalConvolutionalConditioningTest,
           MultipleFramesPerPacketYieldsSameResult) {
  using ConditioningType = CausalConvolutionalConditioning<ConditioningTypes<
      TypeParam, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6>>;

  const int kNumCondHiddens = 8;
  const int kNumHiddens = 4;
  const int kNumThreads = 1;
  const std::vector<std::vector<float>> kFeatures = {{0.0f, 0.0f, 0.0f},
                                                     {0.0f, 0.0f, 0.0f},
                                                     {1.0f, 1.0f, 1.0f},
                                                     {0.0f, 0.0f, 0.0f}};
  ConditioningType one_frame_conditioning(
      kFeatures.at(0).size(), kNumCondHiddens, kNumHiddens, kNumSamplesPerHop,
      /*num_frames_per_packet=*/1, kNumThreads, 0.0f,
      this->testdata_dir_.string(), "lyra");
  ConditioningType two_frames_conditioning(
      kFeatures.at(0).size(), kNumCondHiddens, kNumHiddens, kNumSamplesPerHop,
      /*num_frames_per_packet=*/2, kNumThreads, 0.0f,
      this->testdata_dir_.string(), "lyra");
  csrblocksparse::FatCacheAlignedVector<float> input(kFeatures.at(0).size(), 1);

  std::vector<float> one_frame_output_to_compare;
  std::vector<float> two_frames_output_to_compare;
  int one_frame_step = 0;
  int two_frames_step = 0;
  for (int i = 0; i < kFeatures.size(); ++i) {
    std::copy(kFeatures.at(i).begin(), kFeatures.at(i).end(), input.data());
    one_frame_conditioning.Precompute(input, kNumThreads);
    two_frames_conditioning.Precompute(input, kNumThreads);
    auto to_float = [](auto x) { return static_cast<float>(x); };

    // For one frame per packet, collect the output every frame.
    for (int j = 0; j < one_frame_conditioning.num_samples();
         j += kNumSamplesPerCondOutput) {
      const auto& one_frame_output =
          one_frame_conditioning.AtStep(one_frame_step);
      std::transform(one_frame_output.begin(), one_frame_output.end(),
                     std::back_inserter(one_frame_output_to_compare), to_float);
      one_frame_step += kNumSamplesPerCondOutput;
    }

    // For two frames per packet, collect the output every 2 frames.
    if (i % 2 == 1) {
      for (int j = 0; j < two_frames_conditioning.num_samples();
           j += kNumSamplesPerCondOutput) {
        const auto& two_frames_output =
            two_frames_conditioning.AtStep(two_frames_step);
        std::transform(two_frames_output.begin(), two_frames_output.end(),
                       std::back_inserter(two_frames_output_to_compare),
                       to_float);
        two_frames_step += kNumSamplesPerCondOutput;
      }
    }
  }

  EXPECT_THAT(
      one_frame_output_to_compare,
      testing::Pointwise(testing::FloatEq(), two_frames_output_to_compare));
}

// Test that exported layers with fixed-point and float weights produce
// matching results.
using csrblocksparse::fixed16_type;
using FloatConditioningType =
    CausalConvolutionalConditioning<ConditioningTypes<float>>;
using FixedConditioningType =
    CausalConvolutionalConditioning<ConditioningTypes<fixed16_type>>;
static constexpr int kNumFeatures = 160;
static constexpr int kNumCondHiddens = 512;
static constexpr int kNumGruHiddens = 1024;

struct Conv1DLayerTypes {
  using FloatLayerType = FloatConditioningType::Conv1DLayerType;
  using FixedLayerType = FixedConditioningType::Conv1DLayerType;
  static LayerParams Params(const std::string& model_path) {
    return CausalConvolutionalConditioningPeer<float>::Conv1DParams(
        kNumFeatures, kNumCondHiddens, 1, model_path, "lyra_16khz");
  }
};

struct CondStack0LayerTypes {
  using FloatLayerType = FloatConditioningType::CondStack0LayerType;
  using FixedLayerType = FixedConditioningType::CondStack0LayerType;
  static LayerParams Params(const std::string& model_path) {
    return CausalConvolutionalConditioningPeer<float>::DilatedParams(
        kNumCondHiddens, 0, 1, model_path, "lyra_16khz");
  }
};

struct CondStack1LayerTypes {
  using FloatLayerType = FloatConditioningType::CondStack1LayerType;
  using FixedLayerType = FixedConditioningType::CondStack1LayerType;
  static LayerParams Params(const std::string& model_path) {
    return CausalConvolutionalConditioningPeer<float>::DilatedParams(
        kNumCondHiddens, 1, 1, model_path, "lyra_16khz");
  }
};

struct CondStack2LayerTypes {
  using FloatLayerType = FloatConditioningType::CondStack2LayerType;
  using FixedLayerType = FixedConditioningType::CondStack2LayerType;
  static LayerParams Params(const std::string& model_path) {
    return CausalConvolutionalConditioningPeer<float>::DilatedParams(
        kNumCondHiddens, 2, 1, model_path, "lyra_16khz");
  }
};

struct Transpose0LayerTypes {
  using FloatLayerType = FloatConditioningType::Transpose0LayerType;
  using FixedLayerType = FixedConditioningType::Transpose0LayerType;
  static LayerParams Params(const std::string& model_path) {
    return CausalConvolutionalConditioningPeer<float>::TransposeParams(
        kNumCondHiddens, 0, 1, model_path, "lyra_16khz");
  }
};

struct Transpose1LayerTypes {
  using FloatLayerType = FloatConditioningType::Transpose1LayerType;
  using FixedLayerType = FixedConditioningType::Transpose1LayerType;
  static LayerParams Params(const std::string& model_path) {
    return CausalConvolutionalConditioningPeer<float>::TransposeParams(
        kNumCondHiddens, 1, 1, model_path, "lyra_16khz");
  }
};

struct Transpose2LayerTypes {
  using FloatLayerType = FloatConditioningType::Transpose2LayerType;
  using FixedLayerType = FixedConditioningType::Transpose2LayerType;
  static LayerParams Params(const std::string& model_path) {
    return CausalConvolutionalConditioningPeer<float>::TransposeParams(
        kNumCondHiddens, 2, 1, model_path, "lyra_16khz");
  }
};

struct ConvCondLayerTypes {
  using FloatLayerType = FloatConditioningType::ConvCondLayerType;
  using FixedLayerType = FixedConditioningType::ConvCondLayerType;
  static LayerParams Params(const std::string& model_path) {
    return CausalConvolutionalConditioningPeer<float>::ConvCondParams(
        kNumCondHiddens, kNumGruHiddens, 1, model_path, "lyra_16khz");
  }
};

struct ConvToGatesLayerTypes {
  using FloatLayerType = FloatConditioningType::ConvToGatesLayerType;
  using FixedLayerType = FixedConditioningType::ConvToGatesLayerType;
  static LayerParams Params(const std::string& model_path) {
    return CausalConvolutionalConditioningPeer<float>::ConvToGatesParams(
        kNumGruHiddens, 1, model_path, "lyra_16khz");
  }
};

using LayerTypesList =
    testing::Types<Conv1DLayerTypes, CondStack0LayerTypes, CondStack1LayerTypes,
                   CondStack2LayerTypes, Transpose0LayerTypes,
                   Transpose1LayerTypes, Transpose2LayerTypes,
                   ConvCondLayerTypes, ConvToGatesLayerTypes>;
INSTANTIATE_TYPED_TEST_SUITE_P(CausalConvolutionalConditioning,
                               ExportedLayersTest, LayerTypesList);

}  // namespace
}  // namespace codec
}  // namespace chromemedia
