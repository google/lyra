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

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/random/random.h"
#include "absl/types/span.h"
#include "benchmark/benchmark.h"
#include "log_mel_spectrogram_extractor_impl.h"

static constexpr int kTestSampleRateHz = 16000;
static constexpr int kNumMelBins = 10;

void BenchmarkExtractFeatures(benchmark::State& state, const int hop_length,
                              const int window_length) {
  std::unique_ptr<chromemedia::codec::LogMelSpectrogramExtractorImpl>
      feature_extractor_ =
          chromemedia::codec::LogMelSpectrogramExtractorImpl::Create(
              kTestSampleRateHz, hop_length, window_length, kNumMelBins);
  // We create random audio vectors to avoid any caching in the benchmark.
  const int16_t num_rand_vectors = 10000;
  absl::BitGen gen;
  std::vector<int16_t> rand_vec(hop_length);
  std::vector<absl::Span<const int16_t>> audio_vec(num_rand_vectors);
  for (auto& audio : audio_vec) {
    for (auto& sample : rand_vec) {
      sample = absl::Uniform<uint16_t>(gen);
    }
    audio = absl::MakeConstSpan(rand_vec);
  }

  for (auto _ : state) {
    auto features = feature_extractor_->Extract(
        audio_vec[absl::Uniform(gen, 0, num_rand_vectors)]);
  }
}

void BM_ExtractSmallFeatures(benchmark::State& state) {
  BenchmarkExtractFeatures(state, 6, 12);
}

void BM_ExtractMediumFeatures(benchmark::State& state) {
  BenchmarkExtractFeatures(state, 480, 960);
}

void BM_ExtractLargeFeatures(benchmark::State& state) {
  BenchmarkExtractFeatures(state, 2400, 4800);
}

void BM_ExtractMediumFeaturesLongWindows(benchmark::State& state) {
  BenchmarkExtractFeatures(state, 480, 4800);
}

BENCHMARK(BM_ExtractSmallFeatures);
BENCHMARK(BM_ExtractMediumFeatures);
BENCHMARK(BM_ExtractLargeFeatures);
BENCHMARK(BM_ExtractMediumFeaturesLongWindows);
BENCHMARK_MAIN();
