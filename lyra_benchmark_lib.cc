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

#include "lyra_benchmark_lib.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <fstream>  // IWYU pragma: keep // b/24696850
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <system_error>  // NOLINT(build/c++11)
#include <type_traits>
#include <vector>

#ifdef __ANDROID__
#include <android/log.h>
#endif

#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "architecture_utils.h"
#include "audio/dsp/signal_vector_util.h"
#include "dsp_utils.h"
#include "feature_extractor_interface.h"
#include "generative_model_interface.h"
#include "glog/logging.h"  // IWYU pragma: keep
#include "include/ghc/filesystem.hpp"
#include "lyra_components.h"
#include "lyra_config.h"

#ifdef BENCHMARK
#include "absl/base/thread_annotations.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#endif  // BENCHMARK

namespace chromemedia {
namespace codec {

const int kNumQuantizedBits = 120;

// Given an array of ints, computes the max, min, mean, and standard deviation.
// Returns the results as a string.
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

std::optional<std::vector<float>> MaybeRunFeatureExtraction(
    const std::vector<int16_t>& random_audio,
    FeatureExtractorInterface* feature_extractor,
    std::vector<int64_t>* feature_extractor_timings) {
#ifdef BENCHMARK
  const int64_t feature_extractor_start = absl::ToUnixMicros(absl::Now());
#endif  // BENCHMARK
  std::optional<std::vector<float>> features =
      feature_extractor
          ? feature_extractor->Extract(absl::MakeConstSpan(random_audio))
          : std::vector<float>(kNumFeatures, 0);

#ifdef BENCHMARK
  feature_extractor_timings->push_back(absl::ToUnixMicros(absl::Now()) -
                                       feature_extractor_start);
#endif  // BENCHMARK
  return features;
}

std::optional<std::string> MaybeRunQuantizerQuantize(
    const std::vector<float>& features,
    VectorQuantizerInterface* vector_quantizer,
    std::vector<int64_t>* quantizer_quantize_timings) {
#ifdef BENCHMARK
  const int64_t quantizer_quantize_start = absl::ToUnixMicros(absl::Now());
#endif  // BENCHMARK
  std::optional<std::string> quantized_features =
      vector_quantizer ? vector_quantizer->Quantize(features, kNumQuantizedBits)
                       : std::string(kNumQuantizedBits / CHAR_BIT, '\0');
#ifdef BENCHMARK
  quantizer_quantize_timings->push_back(absl::ToUnixMicros(absl::Now()) -
                                        quantizer_quantize_start);
#endif  // BENCHMARK
  return quantized_features;
}

std::optional<std::vector<float>> MaybeRunQuantizerDecode(
    const std::string& quantized_features,
    VectorQuantizerInterface* vector_quantizer,
    std::vector<int64_t>* quantizer_decode_timings) {
#ifdef BENCHMARK
  const int64_t quantizer_decode_start = absl::ToUnixMicros(absl::Now());
#endif  // BENCHMARK
  std::optional<std::vector<float>> lossy_features =
      vector_quantizer
          ? vector_quantizer->DecodeToLossyFeatures(quantized_features)
          : std::vector<float>(kNumFeatures, 0.0f);
#ifdef BENCHMARK
  quantizer_decode_timings->push_back(absl::ToUnixMicros(absl::Now()) -
                                      quantizer_decode_start);
#endif  // BENCHMARK
  return lossy_features;
}

std::optional<std::vector<int16_t>> MaybeRunGenerativeModel(
    const std::vector<float>& lossy_features, const int num_samples_per_hop,
    GenerativeModelInterface* model,
    std::vector<int64_t>* model_decode_timings) {
  std::optional<std::vector<int16_t>> decoded;
#ifdef BENCHMARK
  const int64_t model_decode_start = absl::ToUnixMicros(absl::Now());
#endif  // BENCHMARK
  if (model != nullptr) {
    model->AddFeatures(lossy_features);
    decoded = model->GenerateSamples(num_samples_per_hop);
  } else {
    // The final result is not used anywhere else, so just make `decoded_or`
    // have a value.
    decoded = std::vector<int16_t>(num_samples_per_hop, 0);
  }
#ifdef BENCHMARK
  model_decode_timings->push_back(absl::ToUnixMicros(absl::Now()) -
                                  model_decode_start);
#endif  // BENCHMARK
  return decoded;
}

// Prints stats for the runtime information in |timings|. For desktop, also
// writes to CSV files under /tmp/benchmark/.
void PrintStatsAndWriteCSV(const std::vector<int64_t>& timings,
                           const absl::string_view title) {
  constexpr absl::string_view stats_template =
      "%18s:  max: %5.3f ms  min: %5.3f ms  mean: %5.3f ms  stdev: %5.3f ms";
  auto stats = GetTimingStats(timings);

  // Because benchmarks are performed on a per-hop basis internally, translate
  // the numbers to per-packet (per-frame) ones, which users care more about.
  const std::string stats_string = absl::StrFormat(
      stats_template, title, static_cast<float>(stats.max_microsecs) / 1000.0f,
      static_cast<float>(stats.min_microsecs) / 1000.0f,
      static_cast<float>(stats.mean_microsecs) / 1000.0f,
      static_cast<float>(stats.standard_deviation) / 1000.0f);
#ifdef __ANDROID__
  __android_log_print(ANDROID_LOG_DEBUG, "lyra_benchmark", "%s",
                      stats_string.c_str());
#else
  LOG(INFO) << stats_string;
#endif

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

int lyra_benchmark(const int num_cond_vectors,
                   const std::string& model_base_path,
                   const bool benchmark_feature_extraction,
                   const bool benchmark_quantizer,
                   const bool benchmark_generative_model) {
  if (num_cond_vectors <= 0) {
    LOG(ERROR) << "The number of conditioning vectors has to be positive.";
    return -1;
  }

  const int num_samples_per_hop = GetNumSamplesPerHop(kInternalSampleRateHz);
  const std::string model_path = GetCompleteArchitecturePath(model_base_path);

  std::unique_ptr<FeatureExtractorInterface> feature_extractor =
      benchmark_feature_extraction ? CreateFeatureExtractor(model_path)
                                   : nullptr;

  std::unique_ptr<VectorQuantizerInterface> vector_quantizer =
      benchmark_quantizer ? CreateQuantizer(model_path) : nullptr;

  std::unique_ptr<GenerativeModelInterface> model =
      benchmark_generative_model
          ? CreateGenerativeModel(kNumFeatures, model_path)
          : nullptr;

  std::vector<int64_t> feature_extractor_timings;
  std::vector<int64_t> quantizer_quantize_timings;
  std::vector<int64_t> quantizer_decode_timings;
  std::vector<int64_t> model_decode_timings;

  // Generate a random signal.
  // The characteristics of the signal are not so important, since this is
  // testing benchmarking.  But it should have some variance since silent
  // signals could potentially be handled differently.
  std::uniform_real_distribution<float> distribution(-1.0, 1.0);
  std::default_random_engine generator;
  std::vector<int16_t> random_audio(num_samples_per_hop);

  for (int i = 0; i < num_cond_vectors; ++i) {
    std::generate(random_audio.begin(), random_audio.end(),
                  [&]() { return UnitToInt16Scalar(distribution(generator)); });

    const auto features = MaybeRunFeatureExtraction(
        random_audio, feature_extractor.get(), &feature_extractor_timings);
    if (!features.has_value()) {
      LOG(ERROR) << "Could not create random features to give model.";
      return -1;
    }

    const auto quantized_features = MaybeRunQuantizerQuantize(
        features.value(), vector_quantizer.get(), &quantizer_quantize_timings);
    if (!quantized_features.has_value()) {
      LOG(ERROR) << "Could not quantize features.";
      return -1;
    }

    const auto lossy_features = MaybeRunQuantizerDecode(
        quantized_features.value(), vector_quantizer.get(),
        &quantizer_decode_timings);
    if (!lossy_features.has_value()) {
      LOG(ERROR) << "Could not decode to lossy features.";
      return -1;
    }

    const auto decoded =
        MaybeRunGenerativeModel(lossy_features.value(), num_samples_per_hop,
                                model.get(), &model_decode_timings);
    if (!decoded.has_value()) {
      LOG(ERROR) << "Could not generate samples.";
      return -1;
    }
    if (decoded->size() != num_samples_per_hop) {
      LOG(ERROR) << "Model generated " << decoded->size()
                 << " but should have generated " << num_samples_per_hop;
      return -1;
    }
  }

#ifdef BENCHMARK
  std::vector<int64_t> total_timings;
  for (int i = 0; i < num_cond_vectors; ++i) {
    total_timings.push_back(
        feature_extractor_timings[i] + quantizer_quantize_timings[i] +
        quantizer_decode_timings[i] + model_decode_timings[i]);
  }

  LOG(INFO) << "For generating " << num_cond_vectors << " frames of audio:";
  PrintStatsAndWriteCSV(feature_extractor_timings, "feature_extractor");
  PrintStatsAndWriteCSV(quantizer_quantize_timings, "quantizer_quantize");
  PrintStatsAndWriteCSV(quantizer_decode_timings, "quantizer_decode");
  PrintStatsAndWriteCSV(model_decode_timings, "model_decode");
  PrintStatsAndWriteCSV(total_timings, "total");
#endif  // BENCHMARK
  return 0;
}

}  // namespace codec
}  // namespace chromemedia
