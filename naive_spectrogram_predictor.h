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

#ifndef LYRA_CODEC_NAIVE_SPECTROGRAM_PREDICTOR_H_
#define LYRA_CODEC_NAIVE_SPECTROGRAM_PREDICTOR_H_

#include <vector>

#include "spectrogram_predictor_interface.h"

namespace chromemedia {
namespace codec {

// A class for doing naive spectrogram frame prediction. This class is
// essentially just a repository for the most recent spectrogram frame seen by
// a packet loss handler. When requested to predict a frame, it will simply
// return this most recent frame.
class NaiveSpectrogramPredictor : public SpectrogramPredictorInterface {
 public:
  // Saves features to last_packet_.
  void FeedFrame(const std::vector<float>& features) override;

  // Returns the most recently seen frame.
  std::vector<float> PredictFrame() override;

  explicit NaiveSpectrogramPredictor(int num_features);

 private:
  std::vector<float> last_packet_;

  friend class NaiveSpectrogramPredictorPeer;
};

}  // namespace codec
}  // namespace chromemedia

#endif  // LYRA_CODEC_NAIVE_SPECTROGRAM_PREDICTOR_H_
