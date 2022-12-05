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

#include "lyra/cli_example/decoder_main_lib.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "absl/flags/marshalling.h"
#include "absl/random/bit_gen_ref.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "glog/logging.h"  // IWYU pragma: keep
#include "include/ghc/filesystem.hpp"
#include "lyra/fixed_packet_loss_model.h"
#include "lyra/gilbert_model.h"
#include "lyra/lyra_config.h"
#include "lyra/lyra_decoder.h"
#include "lyra/wav_utils.h"

namespace chromemedia {
namespace codec {

std::string AbslUnparseFlag(chromemedia::codec::PacketLossPattern pattern) {
  std::ostringstream flag_text;
  for (int i = 0; i < pattern.starts_.size(); ++i) {
    flag_text << pattern.starts_.at(i) << "," << pattern.durations_.at(i)
              << (i < pattern.starts_.size() - 1 ? "," : "");
  }
  return flag_text.str();
}

bool AbslParseFlag(absl::string_view text,
                   chromemedia::codec::PacketLossPattern* pattern,
                   std::string* error) {
  std::vector<std::string> entries;
  if (!absl::ParseFlag(text, &entries, error)) {
    return false;
  }
  if (entries.size() % 2 == 1) {
    *error =
        "Must supply an even number of comma separated values for packet "
        "loss pattern.";
    return false;
  }
  bool is_start = true;
  for (const std::string& s : entries) {
    float value = std::stof(s);
    if (value < 0) {
      *error = "Can not supply values less than zero for packet loss pattern.";
      return false;
    }
    if (is_start) {
      pattern->starts_.push_back(value);
    } else {
      pattern->durations_.push_back(value);
    }
    is_start = !is_start;
  }
  return true;
}

bool DecodeFeatures(const std::vector<uint8_t>& packet_stream, int packet_size,
                    bool randomize_num_samples_requested, absl::BitGenRef gen,
                    LyraDecoder* decoder,
                    PacketLossModelInterface* packet_loss_model,
                    std::vector<int16_t>* decoded_audio) {
  const int num_samples_per_packet =
      GetNumSamplesPerHop(decoder->sample_rate_hz());

  const auto benchmark_start = absl::Now();
  for (int encoded_index = 0; encoded_index < packet_stream.size();
       encoded_index += packet_size) {
    const absl::Span<const uint8_t> encoded_packet =
        absl::MakeConstSpan(packet_stream.data() + encoded_index, packet_size);

    const int frame_index = encoded_index / packet_size;
    const float packet_start_seconds =
        static_cast<float>(frame_index) / decoder->frame_rate();
    std::optional<std::vector<int16_t>> decoded;
    if (packet_loss_model == nullptr || packet_loss_model->IsPacketReceived()) {
      if (!decoder->SetEncodedPacket(encoded_packet)) {
        LOG(ERROR) << "Unable to set encoded packet starting at byte "
                   << encoded_index << " at time " << packet_start_seconds
                   << "s.";
        return false;
      }
    } else {
      VLOG(1) << "Decoding packet starting at " << packet_start_seconds
              << "seconds in PLC mode.";
    }
    int samples_decoded_so_far = 0;
    while (samples_decoded_so_far < num_samples_per_packet) {
      int samples_to_request =
          randomize_num_samples_requested
              ? std::min(absl::Uniform<int>(absl::IntervalOpenClosed, gen, 0,
                                            num_samples_per_packet),
                         num_samples_per_packet - samples_decoded_so_far)
              : num_samples_per_packet;
      VLOG(1) << "Requesting " << samples_to_request
              << " samples for decoding.";
      decoded = decoder->DecodeSamples(samples_to_request);
      if (!decoded.has_value()) {
        LOG(ERROR) << "Unable to decode features starting at byte "
                   << encoded_index;
        return false;
      }
      samples_decoded_so_far += decoded->size();
      decoded_audio->insert(decoded_audio->end(), decoded.value().begin(),
                            decoded.value().end());
    }
  }

  const auto elapsed = absl::Now() - benchmark_start;
  LOG(INFO) << "Elapsed seconds : " << absl::ToInt64Seconds(elapsed);
  LOG(INFO) << "Samples per second : "
            << decoded_audio->size() / absl::ToDoubleSeconds(elapsed);
  return true;
}

bool DecodeFile(const ghc::filesystem::path& encoded_path,
                const ghc::filesystem::path& output_path, int sample_rate_hz,
                int bitrate, bool randomize_num_samples_requested,
                float packet_loss_rate, float average_burst_length,
                const PacketLossPattern& fixed_packet_loss_pattern,
                const ghc::filesystem::path& model_path) {
  auto decoder = LyraDecoder::Create(sample_rate_hz, kNumChannels, model_path);
  if (decoder == nullptr) {
    LOG(ERROR) << "Could not create lyra decoder.";
    return false;
  }
  std::unique_ptr<PacketLossModelInterface> packet_loss_model;
  if (fixed_packet_loss_pattern.starts_.empty()) {
    packet_loss_model =
        GilbertModel::Create(packet_loss_rate, average_burst_length);

  } else {
    packet_loss_model = std::make_unique<FixedPacketLossModel>(
        sample_rate_hz, GetNumSamplesPerHop(sample_rate_hz),
        fixed_packet_loss_pattern.starts_,
        fixed_packet_loss_pattern.durations_);
  }
  if (packet_loss_model == nullptr) {
    LOG(ERROR) << "Could not create packet loss simulator model.";
    return false;
  }
  std::ifstream encoded_stream(encoded_path.string(), std::ios_base::binary);
  if (!encoded_stream.is_open()) {
    LOG(ERROR) << "Open on file " << encoded_path << " failed.";
    return false;
  }

  std::string packet_stream_string{
      std::istreambuf_iterator<char>(encoded_stream),
      std::istreambuf_iterator<char>()};

  const int packet_size = BitrateToPacketSize(bitrate);
  const int stream_size_remainder = packet_stream_string.size() % packet_size;
  if (stream_size_remainder != 0) {
    LOG(WARNING)
        << "Read " << packet_stream_string.size()
        << " bytes from file, which has a remainder when divided by packet "
           "size. Removing the excess bytes from the end and attempting to "
           "decode.";
    packet_stream_string = packet_stream_string.substr(
        0, packet_stream_string.size() - stream_size_remainder);
  }
  if (packet_stream_string.empty()) {
    LOG(ERROR) << "File was empty or incomplete and truncated to empty size.";
    return false;
  }
  std::vector<uint8_t> packet_stream(packet_stream_string.size());
  std::transform(packet_stream_string.begin(), packet_stream_string.end(),
                 packet_stream.begin(),
                 [](char packet) { return static_cast<uint8_t>(packet); });

  std::vector<int16_t> decoded_audio;
  // Use one |gen| across each file. Creating |gen| inside |DecodeFeatures|
  // would use the same pattern for each hop.
  absl::BitGen gen;
  if (!DecodeFeatures(packet_stream, packet_size,
                      randomize_num_samples_requested, gen, decoder.get(),
                      packet_loss_model.get(), &decoded_audio)) {
    LOG(ERROR) << "Unable to decode features for file " << encoded_path;
    return false;
  }

  absl::Status write_status =
      Write16BitWavFileFromVector(output_path.string(), decoder->num_channels(),
                                  decoder->sample_rate_hz(), decoded_audio);
  if (!write_status.ok()) {
    LOG(ERROR) << write_status;
    return false;
  }
  return true;
}

}  // namespace codec
}  // namespace chromemedia
