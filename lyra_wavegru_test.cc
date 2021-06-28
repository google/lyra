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

#include "lyra_wavegru.h"

#include <memory>
#include <string>
#include <tuple>

#if !defined(USE_FIXED16) && !defined(USE_BFLOAT16)
#include "exported_layers_test.h"
#endif  // !defined(USE_FIXED16) && !defined(USE_BFLOAT16)

// Placeholder for get runfiles header.
#include "absl/strings/str_format.h"
#include "gtest/gtest.h"
#include "include/ghc/filesystem.hpp"
#include "lyra_config.h"
#include "sparse_matmul/sparse_matmul.h"  // IWYU pragma: keep

namespace chromemedia {
namespace codec {
namespace {

static const int kNumThreads[] = {1, 2, 4};
static const char kPrefixTemplate[] = "lyra_%dkhz";

#ifdef USE_FIXED16
using ComputeType = csrblocksparse::fixed16_type;
#elif USE_BFLOAT16
using ComputeType = csrblocksparse::bfloat16;
#else
using ComputeType = float;
#endif  // USE_FIXED16

class LyraWavegruTest
    : public testing::TestWithParam<testing::tuple<int, int>> {
 protected:
  LyraWavegruTest()
      : sample_rate_hz_(GetInternalSampleRate(std::get<1>(GetParam()))),
        lyra_wavegru_(LyraWavegru<ComputeType>::Create(
            std::get<0>(GetParam()),
            ghc::filesystem::current_path() / "wavegru",
            absl::StrFormat(kPrefixTemplate, sample_rate_hz_ / 1000))) {}

  const int sample_rate_hz_;
  std::unique_ptr<LyraWavegru<ComputeType>> lyra_wavegru_;
};

TEST_P(LyraWavegruTest, ModelExistsProdFeatures) {
  EXPECT_NE(lyra_wavegru_, nullptr);
}

INSTANTIATE_TEST_SUITE_P(
    ThreadsAndSampleRates, LyraWavegruTest,
    testing::Combine(testing::ValuesIn(kNumThreads),
                     testing::ValuesIn(kSupportedSampleRates)));

// Test that exported layers with fixed-point and float weights produce
// matching results.
// Run only in the first of the three related test targets: {lyra_wavegru_test,
// lyra_wavegru_test_fixed16, lyra_wavegru_test_bfloat16}.
#if !defined(USE_FIXED16) && !defined(USE_BFLOAT16)

using csrblocksparse::fixed16_type;
static constexpr int kNumGruHiddens = 1024;
static constexpr int kNumSplitBands = 4;

struct ArLayerTypes {
  using FloatLayerType = LyraWavegru<float>::ArLayerType;
  using FixedLayerType = LyraWavegru<fixed16_type>::ArLayerType;
  static LayerParams Params(const std::string& model_path) {
    return LayerParams{
        .num_input_channels = kNumSplitBands,
        .num_filters = 3 * kNumGruHiddens,
        .length = 1,
        .kernel_size = 1,
        .dilation = 1,
        .stride = 1,
        .relu = false,
        .skip_connection = false,
        .type = LayerType::kConv1D,
        .num_threads = 1,
        .per_column_barrier = false,
        .from = LayerParams::FromDisk{.path = model_path, .zipped = true},
        .prefix = "lyra_16khz_ar_to_gates_"};
  }
};

struct GruLayerTypes {
  using FloatLayerType = LyraWavegru<float>::GruLayerType;
  using FixedLayerType = LyraWavegru<fixed16_type>::GruLayerType;
  static LayerParams Params(const std::string& model_path) {
    return LayerParams{
        .num_input_channels = kNumGruHiddens,
        .num_filters = 3 * kNumGruHiddens,
        .length = 1,
        .kernel_size = 1,
        .dilation = 1,
        .stride = 1,
        .relu = false,
        .skip_connection = false,
        .type = LayerType::kConv1D,
        .num_threads = 1,
        .per_column_barrier = false,
        .from = LayerParams::FromDisk{.path = model_path, .zipped = true},
        .prefix = "lyra_16khz_gru_layer_"};
  }
};

using LayerTypesList = testing::Types<ArLayerTypes, GruLayerTypes>;
INSTANTIATE_TYPED_TEST_SUITE_P(Wavegru, ExportedLayersTest, LayerTypesList);

#endif  // !defined(USE_FIXED16) && !defined(USE_BFLOAT16)

}  // namespace
}  // namespace codec
}  // namespace chromemedia
