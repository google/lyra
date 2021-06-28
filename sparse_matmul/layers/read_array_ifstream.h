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

// Low-level array reading function using std::ifstream.

#ifndef LYRA_CODEC_SPARSE_MATMUL_LAYERS_READ_ARRAY_IFSTREAM_H_
#define LYRA_CODEC_SPARSE_MATMUL_LAYERS_READ_ARRAY_IFSTREAM_H_

#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/substitute.h"
#include "include/ghc/filesystem.hpp"

namespace csrblocksparse {
namespace detail {

template <typename T>
absl::Status ReadArrayIfstream(const std::string& file_name,
                               const std::string& path, std::vector<T>* array,
                               int64_t* length) {
  ghc::filesystem::path complete_path(path);
  complete_path /= file_name;
  std::ifstream in_stream(complete_path.u8string(), std::ios::binary);
  if (!in_stream.is_open()) {
    return absl::UnknownError(
        absl::Substitute("Error opening $0", complete_path.string()));
  }

  std::stringstream buffer;
  buffer << in_stream.rdbuf();
  if (buffer.str().empty()) {
    LOG(ERROR) << "File " << complete_path << " was empty.";
    return absl::UnknownError(
        absl::Substitute("File $0 was empty", complete_path.string()));
  }
  std::string contents = buffer.str();
  *length = contents.length();
  int64_t elem = (*length + sizeof(T) - 1) / sizeof(T);
  array->resize(elem);
  std::move(contents.begin(), contents.end(),
            reinterpret_cast<char*>(array->data()));

  return absl::OkStatus();
}

}  // namespace detail
}  // namespace csrblocksparse

#endif  // LYRA_CODEC_SPARSE_MATMUL_LAYERS_READ_ARRAY_IFSTREAM_H_
