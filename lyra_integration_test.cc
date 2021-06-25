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
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "glog/logging.h"
// Placeholder for get runfiles header.
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "dsp_util.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "include/ghc/filesystem.hpp"
#include "log_mel_spectrogram_extractor_impl.h"
#include "lyra_config.h"
#include "lyra_decoder.h"
#include "lyra_encoder.h"
#include "wav_util.h"

namespace chromemedia {
namespace codec {
namespace {

static constexpr absl::string_view kWavFiles[] = {
    "48khz_sample_000003.wav", "32khz_sample_000002.wav",
    "16khz_sample_000001.wav", "8khz_sample_000000.wav"};

class LyraIntegrationTest : public testing::TestWithParam<absl::string_view> {};

// This tests that decoded audio has similar features as the original.
TEST_P(LyraIntegrationTest, DecodedAudioHasSimilarFeatures) {
  const ghc::filesystem::path wav_dir("testdata");
  const auto model_path =
      ghc::filesystem::current_path() / std::string("wavegru");

  const auto input_path =
      ghc::filesystem::current_path() / wav_dir / std::string(GetParam());
  absl::StatusOr<ReadWavResult> input_wav_result =
      Read16BitWavFileToVector(input_path);
  CHECK(input_wav_result.ok());

  // Keep only two seconds of the input to shorten the test duration. The two
  // seconds are taken from the middle to avoid the digital silence near the
  // beginning and the end of each WAV file.
  const int test_input_num_samples =
      std::min(2 * input_wav_result->sample_rate_hz,
               static_cast<int>(input_wav_result->samples.size()));
  const int test_input_begin =
      (input_wav_result->samples.size() - test_input_num_samples) / 2;
  std::vector<int16_t> middle_samples(test_input_num_samples);
  std::copy(input_wav_result->samples.begin() + test_input_begin,
            (input_wav_result->samples.begin() + test_input_begin +
             test_input_num_samples),
            middle_samples.begin());

  std::unique_ptr<LyraEncoder> encoder = LyraEncoder::Create(
      input_wav_result->sample_rate_hz, input_wav_result->num_channels,
      kBitrate, /*enable_dtx=*/false, model_path);
  ASSERT_NE(nullptr, encoder);
  EXPECT_EQ(input_wav_result->sample_rate_hz, encoder->sample_rate_hz());
  EXPECT_EQ(input_wav_result->num_channels, encoder->num_channels());
  EXPECT_EQ(kBitrate, encoder->bitrate());

  std::unique_ptr<LyraDecoder> decoder =
      LyraDecoder::Create(input_wav_result->sample_rate_hz,
                          input_wav_result->num_channels, kBitrate, model_path);
  ASSERT_NE(nullptr, decoder);
  EXPECT_EQ(input_wav_result->sample_rate_hz, decoder->sample_rate_hz());
  EXPECT_EQ(input_wav_result->num_channels, decoder->num_channels());
  EXPECT_EQ(kBitrate, decoder->bitrate());

  const int num_samples_per_hop =
      (input_wav_result->sample_rate_hz / encoder->frame_rate());
  const int num_frames = middle_samples.size() / num_samples_per_hop;
  middle_samples.resize(num_frames * num_samples_per_hop);
  std::vector<int16_t> decoded;
  decoded.reserve(middle_samples.size());
  const int num_samples_per_packet = kNumFramesPerPacket * num_samples_per_hop;
  for (int frame = 0; frame < num_frames; frame += kNumFramesPerPacket) {
    absl::optional<std::vector<uint8_t>> encoded_or = encoder->Encode(
        absl::MakeConstSpan(middle_samples.data() + num_samples_per_hop * frame,
                            num_samples_per_packet));
    ASSERT_TRUE(encoded_or.has_value());
    EXPECT_EQ(encoded_or.value().size(), kPacketSize);

    ASSERT_TRUE(decoder->SetEncodedPacket(encoded_or.value()));
    absl::optional<std::vector<int16_t>> decoded_or =
        decoder->DecodeSamples(num_samples_per_packet);
    ASSERT_TRUE(decoded_or.has_value());
    EXPECT_EQ(decoded_or.value().size(), num_samples_per_packet);
    decoded.insert(decoded.end(), decoded_or.value().begin(),
                   decoded_or.value().end());
  }

  // We need separate feature extractors for input and decoded features because
  // the objects maintain an internal state.
  std::unique_ptr<LogMelSpectrogramExtractorImpl> input_extractor =
      LogMelSpectrogramExtractorImpl::Create(input_wav_result->sample_rate_hz,
                                             kNumFeatures, num_samples_per_hop,
                                             2 * num_samples_per_hop);
  ASSERT_NE(nullptr, input_extractor);

  std::unique_ptr<LogMelSpectrogramExtractorImpl> decoded_extractor =
      LogMelSpectrogramExtractorImpl::Create(input_wav_result->sample_rate_hz,
                                             kNumFeatures, num_samples_per_hop,
                                             2 * num_samples_per_hop);
  ASSERT_NE(nullptr, decoded_extractor);

  std::vector<std::vector<float>> input_log_mel_spectrogram;
  std::vector<std::vector<float>> decoded_log_mel_spectrogram;
  for (int frame = 0; frame < num_frames; ++frame) {
    absl::optional<std::vector<float>> input_features_or =
        input_extractor->Extract(
            absl::MakeConstSpan(&middle_samples.at(frame * num_samples_per_hop),
                                num_samples_per_hop));
    ASSERT_TRUE(input_features_or.has_value());
    EXPECT_THAT(input_features_or,
                testing::Optional(testing::SizeIs(kNumFeatures)));
    input_log_mel_spectrogram.push_back(input_features_or.value());

    absl::optional<std::vector<float>> decoded_features_or =
        decoded_extractor->Extract(absl::MakeConstSpan(
            &decoded.at(frame * num_samples_per_hop), num_samples_per_hop));
    ASSERT_TRUE(decoded_features_or.has_value());
    EXPECT_THAT(decoded_features_or,
                testing::Optional(testing::SizeIs(kNumFeatures)));
    decoded_log_mel_spectrogram.push_back(decoded_features_or.value());
  }
  ASSERT_EQ(decoded_log_mel_spectrogram.size(),
            input_log_mel_spectrogram.size());

  // Filling out the buffer because the conditioning stack has one frame
  // look-ahead, which accounts for one frame of delay. The feature extractor's
  // first frame uses padded zeros as well, adding 1 more frame.
  static constexpr int kDelayInFrames = 2;
  for (int frame = 0; frame < num_frames - kDelayInFrames; ++frame) {
    absl::Span<const float> input_features =
        absl::MakeConstSpan(input_log_mel_spectrogram.at(frame));
    absl::Span<const float> decoded_features = absl::MakeConstSpan(
        decoded_log_mel_spectrogram.at(frame + kDelayInFrames));
    EXPECT_EQ(input_features.size(), decoded_features.size());
    const auto log_spectral_distance_or =
        LogSpectralDistance(input_features, decoded_features);
    ASSERT_TRUE(log_spectral_distance_or.has_value());
    EXPECT_LT(log_spectral_distance_or.value(), 2.6f)
        << "frame_index=" << frame;
  }
}
}  // namespace

INSTANTIATE_TEST_SUITE_P(InputPaths, LyraIntegrationTest,
                         testing::ValuesIn(kWavFiles));

}  // namespace codec
}  // namespace chromemedia
