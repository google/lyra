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

#ifndef LYRA_CODEC_LYRA_CONFIG_H_
#define LYRA_CODEC_LYRA_CONFIG_H_

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iterator>

#include "absl/base/attributes.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "glog/logging.h"
#include "google/protobuf/text_format.h"
#include "include/ghc/filesystem.hpp"
#include "lyra_config.pb.h"

namespace chromemedia {
namespace codec {

// This file is reserved for non-configurable values needed by both the decoder
// and encoder. What those non-configurable values are depends on which project
// is chosen to be compiled.  As a result, a struct holding the configuration
// data is defined to ensure each new target added and each new configuration
// element is explicitly defined.

ABSL_CONST_INIT extern const int kVersionMajor;
ABSL_CONST_INIT extern const int kVersionMinor;
ABSL_CONST_INIT extern const int kVersionMicro;
ABSL_CONST_INIT extern const int kNumFeatures;
ABSL_CONST_INIT extern const int kNumExpectedOutputFeatures;
ABSL_CONST_INIT extern const int kNumChannels;
ABSL_CONST_INIT extern const int kFrameRate;  // Frames sent per second.
ABSL_CONST_INIT extern const int kFrameOverlapFactor;
ABSL_CONST_INIT extern const int kNumFramesPerPacket;
ABSL_CONST_INIT extern const int kPacketSize;
ABSL_CONST_INIT extern const int kBitrate;

inline constexpr int kSupportedSampleRates[] = {8000, 16000, 32000, 48000};
inline constexpr int kInternalSampleRateHz = 16000;
inline constexpr int kNumQuantizationBits = 120;

inline bool IsSampleRateSupported(int sample_rate_hz) {
  return std::find(std::begin(kSupportedSampleRates),
                   std::end(kSupportedSampleRates),
                   sample_rate_hz) != std::end(kSupportedSampleRates);
}

absl::Status AreParamsSupported(int sample_rate_hz, int num_channels,
                                int bitrate,
                                const ghc::filesystem::path& model_path);

// Returns a string of form "|kVersionMajor|.|kVersionMinor|.|kVersionMicro|".
const std::string& GetVersionString();

// Functions to get values depending on sample rate.
inline int GetNumSamplesPerHop(int sample_rate_hz) {
  CHECK_EQ(sample_rate_hz % kFrameRate, 0);
  return sample_rate_hz / kFrameRate;
}

inline int GetNumSamplesPerFrame(int sample_rate_hz) {
  CHECK_EQ(sample_rate_hz % kFrameRate, 0);
  return kFrameOverlapFactor * (sample_rate_hz / kFrameRate);
}

inline int GetInternalSampleRate(int external_sample_rate_hz) {
  return kInternalSampleRateHz;
}

inline int ConvertNumSamplesBetweenSampleRate(int source_num_samples,
                                              int source_sample_rate,
                                              int target_sample_rate) {
  return static_cast<int>(std::ceil(static_cast<float>(source_num_samples) *
                                    static_cast<float>(target_sample_rate) /
                                    static_cast<float>(source_sample_rate)));
}

}  // namespace codec
}  // namespace chromemedia

#endif  // LYRA_CODEC_LYRA_CONFIG_H_
