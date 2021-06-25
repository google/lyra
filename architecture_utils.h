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

#ifndef LYRA_CODEC_ARCHITECTURE_UTILS_H_
#define LYRA_CODEC_ARCHITECTURE_UTILS_H_

// Placeholder for get runfiles header.
#include "include/ghc/filesystem.hpp"

namespace chromemedia {
namespace codec {

ghc::filesystem::path GetCompleteArchitecturePath(
    const ghc::filesystem::path& model_path) {
  return model_path;
}

}  // namespace codec
}  // namespace chromemedia

#endif  // LYRA_CODEC_ARCHITECTURE_UTILS_H_
