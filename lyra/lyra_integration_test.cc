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

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

// Placeholder for get runfiles header.
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "include/ghc/filesystem.hpp"
#include "lyra/dsp_utils.h"
#include "lyra/log_mel_spectrogram_extractor_impl.h"
#include "lyra/lyra_config.h"
#include "lyra/lyra_decoder.h"
#include "lyra/lyra_encoder.h"
#include "lyra/wav_utils.h"

namespace chromemedia {
namespace codec {
namespace {

static constexpr absl::string_view kWavFiles[] = {
    "sample1_8kHz.wav", "sample1_16kHz.wav", "sample1_32kHz.wav",
    "sample1_48kHz.wav"};

class LyraIntegrationTest
    : public testing::TestWithParam<testing::tuple<absl::string_view, int>> {};

// This tests that decoded audio has similar features as the original.
TEST_P(LyraIntegrationTest, DecodedAudioHasSimilarFeatures) {
  const ghc::filesystem::path wav_dir("lyra/testdata");
  const auto model_path =
      ghc::filesystem::current_path() / std::string("lyra/model_coeffs");

  const auto input_path = ghc::filesystem::current_path() / wav_dir /
                          std::string(std::get<0>(GetParam()));
  absl::StatusOr<ReadWavResult> input_wav_result =
      Read16BitWavFileToVector(input_path);
  CHECK(input_wav_result.ok());

  const int sample_rate_hz = input_wav_result->sample_rate_hz;
  const int num_quantized_bits = std::get<1>(GetParam());
  std::unique_ptr<LyraEncoder> encoder =
      LyraEncoder::Create(sample_rate_hz, input_wav_result->num_channels,
                          GetBitrate(num_quantized_bits),
                          /*enable_dtx=*/false, model_path);
  ASSERT_NE(encoder, nullptr);

  std::unique_ptr<LyraDecoder> decoder = LyraDecoder::Create(
      sample_rate_hz, input_wav_result->num_channels, model_path);
  ASSERT_NE(decoder, nullptr);

  // Keep only 3 seconds to shorten the test duration.
  const int num_samples_per_hop = GetNumSamplesPerHop(sample_rate_hz);
  const int num_samples_per_window = GetNumSamplesPerWindow(sample_rate_hz);
  const int num_hops =
      std::min(3 * sample_rate_hz / num_samples_per_hop,
               static_cast<int>(input_wav_result->samples.size()));
  std::vector<int16_t> input(
      input_wav_result->samples.begin(),
      input_wav_result->samples.begin() + num_hops * num_samples_per_hop);
  std::vector<int16_t> decoded_all;
  decoded_all.reserve(input.size());
  for (int hop_begin = 0; hop_begin < num_hops * num_samples_per_hop;
       hop_begin += num_samples_per_hop) {
    std::optional<std::vector<uint8_t>> encoded = encoder->Encode(
        absl::MakeConstSpan(input.data() + hop_begin, num_samples_per_hop));
    ASSERT_TRUE(encoded.has_value());
    EXPECT_EQ(encoded.value().size(), GetPacketSize(num_quantized_bits));

    ASSERT_TRUE(decoder->SetEncodedPacket(encoded.value()));
    std::optional<std::vector<int16_t>> decoded =
        decoder->DecodeSamples(num_samples_per_hop);
    ASSERT_TRUE(decoded.has_value());
    EXPECT_EQ(decoded.value().size(), num_samples_per_hop);
    decoded_all.insert(decoded_all.end(), decoded.value().begin(),
                       decoded.value().end());
  }

  // We need separate feature extractors for input and decoded features because
  // the objects maintain an internal state.
  std::unique_ptr<LogMelSpectrogramExtractorImpl> input_extractor =
      LogMelSpectrogramExtractorImpl::Create(
          sample_rate_hz, num_samples_per_hop, num_samples_per_window,
          kNumFeatures);
  ASSERT_NE(input_extractor, nullptr);
  std::unique_ptr<LogMelSpectrogramExtractorImpl> decoded_extractor =
      LogMelSpectrogramExtractorImpl::Create(
          sample_rate_hz, num_samples_per_hop, num_samples_per_window,
          kNumFeatures);
  ASSERT_NE(decoded_extractor, nullptr);
  std::vector<std::vector<float>> input_log_mel_spectrogram;
  std::vector<std::vector<float>> decoded_log_mel_spectrogram;
  for (int hop_begin = 0; hop_begin < num_hops * num_samples_per_hop;
       hop_begin += num_samples_per_hop) {
    std::optional<std::vector<float>> input_features = input_extractor->Extract(
        absl::MakeConstSpan(&input.at(hop_begin), num_samples_per_hop));
    ASSERT_TRUE(input_features.has_value());
    EXPECT_THAT(input_features.value(), testing::SizeIs(kNumFeatures));
    input_log_mel_spectrogram.push_back(input_features.value());

    std::optional<std::vector<float>> decoded_features =
        decoded_extractor->Extract(absl::MakeConstSpan(
            &decoded_all.at(hop_begin), num_samples_per_hop));
    ASSERT_TRUE(decoded_features.has_value());
    EXPECT_THAT(decoded_features.value(), testing::SizeIs(kNumFeatures));
    decoded_log_mel_spectrogram.push_back(decoded_features.value());
  }
  ASSERT_EQ(decoded_log_mel_spectrogram.size(),
            input_log_mel_spectrogram.size());

  // Compute spectral distances.
  for (int hop = 0; hop < num_hops; ++hop) {
    absl::Span<const float> input_features =
        absl::MakeConstSpan(input_log_mel_spectrogram.at(hop));
    absl::Span<const float> decoded_features =
        absl::MakeConstSpan(decoded_log_mel_spectrogram.at(hop));
    EXPECT_EQ(input_features.size(), decoded_features.size());
    const auto log_spectral_distance =
        LogSpectralDistance(input_features, decoded_features);
    ASSERT_TRUE(log_spectral_distance.has_value());
    EXPECT_LT(log_spectral_distance.value(), 2.0f) << "hop=" << hop;
  }
}
}  // namespace

INSTANTIATE_TEST_SUITE_P(
    InputPathsAndQuantizedBits, LyraIntegrationTest,
    testing::Combine(testing::ValuesIn(kWavFiles),
                     testing::ValuesIn(GetSupportedQuantizedBits())));

}  // namespace codec
}  // namespace chromemedia
