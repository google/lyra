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

#include "benchmark_decode_lib.h"

#include <algorithm>
#include <cmath>
#include <fstream>  // IWYU pragma: keep // b/24696850
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <system_error>  // NOLINT(build/c++11)
#include <type_traits>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "architecture_utils.h"
#include "audio/dsp/signal_vector_util.h"
#include "dsp_util.h"
#include "generative_model_interface.h"
#include "glog/logging.h"
#include "include/ghc/filesystem.hpp"
#include "log_mel_spectrogram_extractor_impl.h"
#include "lyra_config.h"
#include "wavegru_model_impl.h"

#ifdef BENCHMARK
#include "absl/base/thread_annotations.h"
#include "absl/time/time.h"
#endif  // BENCHMARK

namespace chromemedia {
namespace codec {

TimingStats GetTimingStats(const std::vector<int64_t>& timings_microsecs) {
  TimingStats timing_stats;
  timing_stats.num_calls = static_cast<int64_t>(timings_microsecs.size());
  timing_stats.mean_microsecs =
      std::accumulate(timings_microsecs.begin(), timings_microsecs.end(), 0l) /
      timing_stats.num_calls;
  timing_stats.max_microsecs =
      *std::max_element(timings_microsecs.begin(), timings_microsecs.end());
  timing_stats.min_microsecs =
      *std::min_element(timings_microsecs.begin(), timings_microsecs.end());
  std::vector<float> variances(timings_microsecs.size());
  std::transform(timings_microsecs.begin() + 1, timings_microsecs.end(),
                 variances.begin(), [&](int64_t timings_microsecs) {
                   return audio_dsp::Square(timings_microsecs -
                                            timing_stats.mean_microsecs) /
                          timing_stats.num_calls;
                 });
  timing_stats.standard_deviation =
      std::sqrt(std::accumulate(variances.begin(), variances.end(), 0.f));

  return timing_stats;
}

void PrintStatsAndWriteCSV(const std::vector<int64_t>& timings,
                           const absl::string_view title) {
  const std::string stats_template =
      "$0 stats for generating $1 frames of audio, max: $2 us, "
      "min: $3 us, mean: $4 us, stdev: $5.";
  auto stats = GetTimingStats(timings);

  const std::string stats_string = absl::Substitute(
      stats_template, title, stats.num_calls, stats.max_microsecs,
      stats.min_microsecs, stats.mean_microsecs, stats.standard_deviation);
  LOG(INFO) << stats_string;

#if !defined __arm__ && !defined __aarch64__
  const ghc::filesystem::path output_dir("/tmp/benchmarks/");
  std::error_code error_code;
  if (!ghc::filesystem::is_directory(output_dir, error_code)) {
    CHECK(ghc::filesystem::create_directories(output_dir, error_code));
  }
  const std::string filename = absl::Substitute("$0.csv", title);
  std::ofstream csv((output_dir / filename).string());
  csv << "Time(us)" << std::endl;
  for (const auto element : timings) {
    csv << element << std::endl;
  }
#endif  // !defined __arm__ && !defined __aarch64__
}

int benchmark_decode(const int num_cond_vectors,
                     const std::string& model_base_path) {
  const std::string model_path =
      chromemedia::codec::GetCompleteArchitecturePath(model_base_path);
  if (num_cond_vectors <= 0) {
    LOG(ERROR) << "The number of conditioning vectors has to be positive.";
    return -1;
  }

  std::unique_ptr<chromemedia::codec::GenerativeModelInterface> model =
      chromemedia::codec::WavegruModelImpl::Create(
          chromemedia::codec::GetNumSamplesPerHop(
              chromemedia::codec::kInternalSampleRateHz),
          chromemedia::codec::kNumFeatures,
          chromemedia::codec::kNumFramesPerPacket,
          LogMelSpectrogramExtractorImpl::GetSilenceValue(), model_path);

  const int num_samples_per_hop = chromemedia::codec::GetNumSamplesPerHop(
      chromemedia::codec::kInternalSampleRateHz);
  const int num_samples_per_frame = chromemedia::codec::GetNumSamplesPerFrame(
      chromemedia::codec::kInternalSampleRateHz);
  const int kNumFeatures = chromemedia::codec::kNumFeatures;
  auto feature_extractor =
      chromemedia::codec::LogMelSpectrogramExtractorImpl::Create(
          chromemedia::codec::kInternalSampleRateHz, kNumFeatures,
          num_samples_per_hop, num_samples_per_frame);
  // Generate a random signal.
  // The characteristics of the signal are not so important, since this is
  // testing benchmarking.  But it should have some variance since silent
  // signals could potentially be handled differently.
  std::uniform_real_distribution<float> distribution(-1.0, 1.0);
  std::default_random_engine generator;
  std::vector<int16_t> random_audio(num_samples_per_hop);

  for (int i = 0; i < num_cond_vectors; ++i) {
    std::generate(random_audio.begin(), random_audio.end(), [&]() {
      return UnitFloatToInt16Scalar(distribution(generator));
    });
    auto features_or =
        feature_extractor->Extract(absl::MakeConstSpan(random_audio));
    if (!features_or.has_value()) {
      LOG(ERROR) << "Could not create random features to give model.";
      return -1;
    }
    model->AddFeatures(features_or.value());
    auto decoded_or = model->GenerateSamples(num_samples_per_hop);
    if (!decoded_or.has_value()) {
      LOG(ERROR) << "Could not generate samples.";
      return -1;
    }
    if (decoded_or->size() != num_samples_per_hop) {
      LOG(ERROR) << "Model generated " << decoded_or->size()
                 << " but should have generated " << num_samples_per_hop;
      return -1;
    }
  }

#ifdef BENCHMARK
  auto cond_stack_timings = model->conditioning_timings_microsecs();
  auto model_timings = model->model_timings_microsecs();

#ifdef USE_FIXED16
  LOG(INFO) << "Using fixed point arithmetic.";
#elif USE_BFLOAT16
  LOG(INFO) << "Using bfloat arithmetic.";
#else
  LOG(INFO) << "Using float arithmetic.";
#endif  // USE_FIXED16

  std::vector<int64_t> combined_timings;
  std::transform(model_timings.begin(), model_timings.end(),
                 cond_stack_timings.begin(),
                 std::back_inserter(combined_timings), std::plus<int64_t>());

  chromemedia::codec::PrintStatsAndWriteCSV(cond_stack_timings,
                                            "conditioning_only");
  chromemedia::codec::PrintStatsAndWriteCSV(model_timings, "model_only");
  chromemedia::codec::PrintStatsAndWriteCSV(combined_timings,
                                            "combined_model_and_conditioning");
#endif  // BENCHMARK
  return 0;
}

}  // namespace codec
}  // namespace chromemedia
