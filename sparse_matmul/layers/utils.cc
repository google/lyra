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

// Source for various utility functions related to reading and writing files
// and vectors.  Would be much simpler if Android and Windows supported File.

#include "sparse_matmul/layers/utils.h"

#ifdef _WIN32
#include <Windows.h>

#include <codecvt>
#include <mutex>  // NOLINT
#else
#include <dirent.h>
#endif  // _WIN32

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"

namespace csrblocksparse {

namespace {

// Helper to test if a filename is "." or "..".
template <typename CharType>
bool IsDotOrDotDot(const CharType* filename) {
  if (filename[0] == '.') {
    if (filename[1] == '\0') {
      return true;
    }
    if (filename[1] == '.' && filename[2] == '\0') {
      return true;
    }
  }

  return false;
}

#ifdef _WIN32  // We only define these conversion routines on Win32.
static std::mutex g_converter_mutex;
static std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> g_converter;

std::string Narrow(const std::wstring& wide) {
  std::lock_guard<std::mutex> auto_lock(g_converter_mutex);
  return g_converter.to_bytes(wide);
}

std::wstring Widen(const std::string& narrow) {
  std::lock_guard<std::mutex> auto_lock(g_converter_mutex);
  return g_converter.from_bytes(narrow);
}

inline constexpr char kLongPathPrefix[] = R"(\\?\)";

std::wstring ConvertToWindowsPathFormat(const std::string& path,
                                        int max_path_length = MAX_PATH) {
  if (path.length() + 1 > max_path_length &&
      !absl::StartsWith(path, kLongPathPrefix)) {
    return Widen(absl::StrCat(kLongPathPrefix, path));
  }
  return Widen(path);
}
#endif  // _WIN32

}  // namespace

// Return all files in a given directory.
absl::Status FilesInDirectory(const std::string& path,
                              const std::string& must_contain,
                              std::vector<std::string>* result) {
#ifdef _WIN32
  WIN32_FIND_DATAW child_data;
  HANDLE find_handle = FindFirstFileW(
      ConvertToWindowsPathFormat(absl::StrCat(path, "\\*")).c_str(),
      &child_data);
  if (find_handle == INVALID_HANDLE_VALUE) {
    return absl::UnknownError(
        absl::Substitute("Couldn't open: $0 (error $1)", path, GetLastError()));
  }
  do {
    if (IsDotOrDotDot(child_data.cFileName)) continue;
    const std::string name = Narrow(child_data.cFileName);
    if (name.find(must_contain) == std::string::npos) continue;
    result->push_back(name);
  } while (FindNextFileW(find_handle, &child_data) != 0);
  const auto err = GetLastError();
  FindClose(find_handle);
  if (err != ERROR_NO_MORE_FILES)
    return absl::UnknownError(
        absl::Substitute("Error in FindNextFileW: $0", err));
#else
  DIR* dirp = opendir(path.c_str());
  if (dirp == nullptr) {
    return absl::UnknownError(absl::Substitute("Couldn't open: $0", path));
  }

  dirent* dp;
  errno = 0;
  while ((dp = readdir(dirp)) != nullptr) {
    if (IsDotOrDotDot(dp->d_name)) continue;
    const std::string name(dp->d_name);
    if (name.find(must_contain) == std::string::npos) continue;
    result->push_back(name);
  }
  closedir(dirp);
  if (errno != 0)
    return absl::UnknownError(absl::Substitute("Error in readdir: $0", errno));
#endif  // _WIN32

  return absl::OkStatus();
}

}  // namespace csrblocksparse
