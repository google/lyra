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

#include "naive_spectrogram_predictor.h"

#include <vector>

#include "log_mel_spectrogram_extractor_impl.h"

namespace chromemedia {
namespace codec {

void NaiveSpectrogramPredictor::FeedFrame(const std::vector<float>& features) {
  last_packet_ = features;
}

std::vector<float> NaiveSpectrogramPredictor::PredictFrame() {
  return last_packet_;
}

NaiveSpectrogramPredictor::NaiveSpectrogramPredictor(int num_features)
    : last_packet_(num_features,
                   LogMelSpectrogramExtractorImpl::GetSilenceValue()) {}

}  // namespace codec
}  // namespace chromemedia
