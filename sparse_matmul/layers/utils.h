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

// Various utility functions related to reading and writing files, vectors, etc.
// Would be much simpler if Android supported File.

#ifndef LYRA_CODEC_SPARSE_MATMUL_LAYERS_UTILS_H_
#define LYRA_CODEC_SPARSE_MATMUL_LAYERS_UTILS_H_

#include <errno.h>
#include <stdio.h>
#include <sys/stat.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/substitute.h"
#include "include/ghc/filesystem.hpp"
#include "sparse_matmul/layers/errno_mapping.h"
#include "sparse_matmul/layers/masked_sparse_matrix.h"
#include "sparse_matmul/layers/read_array_ifstream.h"
#include "sparse_matmul/layers/sparse_linear_layer.h"
#include "sparse_matmul/layers/status_macros.h"
#include "sparse_matmul/numerics/fixed_types.h"
#include "sparse_matmul/numerics/float16_types.h"
#include "sparse_matmul/numerics/type_utils.h"
#include "sparse_matmul/vector/cache_aligned_vector.h"
#include "sparse_matmul/zlib_wrapper/zlibwrapper.h"

namespace csrblocksparse {

template <typename T>
void unzip(int64_t st_size, std::vector<T>* array) {
  ZLib z;
  z.SetGzipHeaderMode();
  if (z.HasGzipHeader(reinterpret_cast<char*>(array->data()), st_size)) {
    const std::size_t kMaxBufferSize = 1 << 27;  // 128MB

    Bytef* dest;
    uLongf dest_len = kMaxBufferSize;
    CHECK_EQ(z.UncompressGzipAndAllocate(&dest, &dest_len,
                                         (Bytef*)array->data(), st_size),
             Z_OK);
    CHECK_EQ(dest_len % sizeof(T), 0);
    array->assign(reinterpret_cast<T*>(dest),
                  reinterpret_cast<T*>(dest + dest_len));
    free(dest);
  } else {
    CHECK_EQ(st_size % sizeof(T), 0);
  }
}

// Reads a file that contains an array of a single POD type.  Eventually we
// will replace serializiation with protos, but for now this is the easiest way
// to interface with the rest of the pipeline.
//
// StatusOr might be preferred but does not compile on ARM.
// |DiskType| and |ElemType| template types have no effect in this function
// version and are only used to handle fixed_type disk storage.
template <typename T, typename DiskType = T, typename ElemType = T>
typename std::enable_if<!csrblocksparse::IsFixed16Type<DiskType>::value,
                        absl::Status>::type
ReadArrayFromFile(const std::string& file_name, std::vector<T>* array,
                  const std::string& path = "/data/local/tmp/") {
  int64_t length = 0;
  const absl::Status status =
      detail::ReadArrayIfstream(file_name, path, array, &length);
  if (!status.ok()) {
    return status;
  }
  unzip(length, array);

  return absl::OkStatus();
}

// If the metatype |DiskType| is of fixed16_type, we load int16_ts from disk and
// construct |ElemType| from them. |ElemType|  is necessary because we need to
// know the mantissa/exponent bit split before casting to float. We need a
// separate function template for fixed rather than an if block because the
// compiler will complain bfloat not having an int16_t constructor.
template <typename T, typename DiskType, typename ElemType>
typename std::enable_if<std::is_same<T, float>::value &&
                            csrblocksparse::IsFixed16Type<DiskType>::value,
                        absl::Status>::type
ReadArrayFromFile(const std::string& file_name, std::vector<T>* array,
                  const std::string& path = "/data/local/tmp/") {
  std::vector<int16_t> disk_values;
  SPARSE_MATMUL_RETURN_IF_ERROR(
      ReadArrayFromFile(file_name, &disk_values, path));
  array->resize(disk_values.size());
  std::transform(
      disk_values.begin(), disk_values.end(), array->begin(),
      [](int16_t disk_value) { return static_cast<T>(ElemType(disk_value)); });
  return absl::OkStatus();
}

// Writes a vector to a binary file.  Eventually serialization will be handled
// with protos.
template <typename T>
absl::Status WriteArrayToFile(const std::vector<T>& array,
                              const std::string& file_name,
                              std::string path = "/data/local/tmp/") {
  path = (ghc::filesystem::path(path) / file_name).string();
  FILE* fp = fopen(path.c_str(), "wb");
  if (fp == nullptr)
    return ErrnoToCanonicalStatus(errno,
                                  absl::Substitute("Error opening $0", path));
  size_t write_count = fwrite(array.data(), sizeof(T), array.size(), fp);
  if (write_count != array.size()) {
    return ErrnoToCanonicalStatus(
        errno,
        absl::Substitute(
            "Error writing array, only wrote $0 of $1 elements for file $2",
            write_count, array.size(), path));
  }
  SPARSE_MATMUL_RETURN_IF_ERROR(ErrnoToCanonicalStatus(
      fclose(fp), absl::Substitute("Error closing $0", path)));
  return absl::OkStatus();
}

// Reads an entire layer that consists of weights, bias and mask as a
// SparseLinearLayer.  Eventually this serialization will be handled with
// protos, but the rest of the system currently does naive serialization.
//
// StatusOr might be preferred but does not compile on ARM.
//
// Here |DiskWeightType| is the metatype used to store the weights, usually
// fixed16_type, float, or bfloat.
// For |DiskWeightType| = fixed16_type specialization, this loads a file with a
// "fixed16_weights.raw" suffix which stores int16_ts as its element datatype.
// The disk elements should match fixed16<WeightType::kExponentBits>. This cuts
// down disk storage of weights by
// >= half. For all other types it reads the weights as floats.
template <typename WeightType, typename RhsType,
          typename DiskWeightType = float>
absl::Status LoadGenericLayer(
    const std::string& prefix, bool zipped, const std::string& path,
    float default_bias,
    SparseLinearLayer<WeightType, RhsType>* sparse_linear_layer) {
  std::string fixed_prefix =
      csrblocksparse::IsFixed16Type<DiskWeightType>::value ? "fixed16_" : "";
  std::string extension = zipped ? ".gz" : "";
  std::string weight_name =
      absl::StrCat(prefix, fixed_prefix, "weights.raw", extension);
  std::string mask_name = absl::StrCat(prefix, "mask.raw", extension);
  std::string bias_name = absl::StrCat(prefix, "bias.raw", extension);

  std::vector<float> weight_vector;
  std::vector<float> mask_vector;
  std::vector<float> bias_vector;

  const auto status = ReadArrayFromFile<float, DiskWeightType, WeightType>(
      weight_name, &weight_vector, path);
  SPARSE_MATMUL_RETURN_IF_ERROR(status);
  SPARSE_MATMUL_RETURN_IF_ERROR(
      ReadArrayFromFile(mask_name, &mask_vector, path));
  SPARSE_MATMUL_RETURN_IF_ERROR(
      ReadArrayFromFile(bias_name, &bias_vector, path));

  CHECK(weight_vector.size() == mask_vector.size())
      << "Weight and mask must be"
      << " the same size, weights: " << weight_vector.size()
      << " mask: " << mask_vector.size();
  CHECK(weight_vector.size() % bias_vector.size() == 0)
      << "Weights size must "
         "be a multiple of the bias size. Weights: "
      << weight_vector.size()
      << " "
         "bias: "
      << bias_vector.size()
      << " remainder: " << weight_vector.size() % bias_vector.size();

  int rows = bias_vector.size();
  int cols = weight_vector.size() / rows;

  MaskedSparseMatrix<float> weights_masked(rows, cols, mask_vector.data(),
                                           weight_vector.data());

  weights_masked.template CastWeights<WeightType>();
  using csrmatrix = CsrBlockSparseMatrix<WeightType, RhsType>;

  csrmatrix weights(weights_masked);
  // If the weights were not a multiple of the block size in rows, we need to
  // expand the bias vector to match using the provided default_bias value.
  bias_vector.resize(weights.rows(), default_bias);
  using BiasType = typename TypeOfProduct<WeightType, RhsType>::type;
  CacheAlignedVector<BiasType> bias(bias_vector);

  *sparse_linear_layer = std::move(SparseLinearLayer<WeightType, RhsType>(
      std::move(weights), std::move(bias)));

  return absl::OkStatus();
}
template <typename WeightType, typename RhsType,
          typename DiskWeightType = float>
absl::Status LoadSparseLayer(
    const std::string& prefix, bool zipped,
    SparseLinearLayer<WeightType, RhsType>* sparse_linear_layer,
    const std::string& path = "/data/local/tmp/") {
  return LoadGenericLayer<WeightType, RhsType, DiskWeightType>(
      prefix, zipped, path, 0.0f, sparse_linear_layer);
}
template <typename WeightType, typename RhsType,
          typename DiskWeightType = float>
absl::Status LoadLogitLayer(
    const std::string& prefix, bool zipped, const std::string& path,
    SparseLinearLayer<WeightType, RhsType>* sparse_linear_layer) {
  return LoadGenericLayer<WeightType, RhsType, DiskWeightType>(
      prefix, zipped, path, std::numeric_limits<float>::lowest(),
      sparse_linear_layer);
}

// Reads an entire layer that consists of weights, bias and mask as a
// MaskedLinearLayer.  Eventually this serialization will be handled with
// protos, but the rest of the system currently does naive serialization.
//
// StatusOr might be preferred but does not compile on ARM.
template <typename T>
absl::Status LoadMaskedLayer(const std::string& prefix, bool zipped,
                             MaskedLinearLayer<T>* masked_sparse_matrix,
                             const std::string& path = "/data/local/tmp/") {
  std::string extension = zipped ? ".gz" : "";
  std::string weight_name = absl::StrCat(prefix, "weights.raw", extension);
  std::string mask_name = absl::StrCat(prefix, "mask.raw", extension);
  std::string bias_name = absl::StrCat(prefix, "bias.raw", extension);

  std::vector<float> weight_vector;
  std::vector<float> mask_vector;
  std::vector<float> bias_vector;

  SPARSE_MATMUL_RETURN_IF_ERROR(
      ReadArrayFromFile(weight_name, &weight_vector, path));
  SPARSE_MATMUL_RETURN_IF_ERROR(
      ReadArrayFromFile(mask_name, &mask_vector, path));
  SPARSE_MATMUL_RETURN_IF_ERROR(
      ReadArrayFromFile(bias_name, &bias_vector, path));

  CHECK(weight_vector.size() == mask_vector.size())
      << "Weight and mask must be"
      << " the same size, weights: " << weight_vector.size()
      << " mask: " << mask_vector.size();
  CHECK(weight_vector.size() % bias_vector.size() == 0)
      << "Weights size must "
         "be a multiple of the bias size. Weights: "
      << weight_vector.size()
      << " "
         "bias: "
      << bias_vector.size()
      << " remainder: " << weight_vector.size() % bias_vector.size();

  int rows = bias_vector.size();
  int cols = weight_vector.size() / rows;

  MaskedSparseMatrix<T> weights_masked(rows, cols, mask_vector.data(),
                                       weight_vector.data());
  CacheAlignedVector<T> bias(bias_vector);

  *masked_sparse_matrix =
      MaskedLinearLayer<T>(std::move(weights_masked), std::move(bias));
  return absl::OkStatus();
}

// Load a vector of POD into a CacheAlignedVector.
//
// StatusOr might be preferred but does not compile on ARM.
template <typename T>
absl::Status LoadVector(const std::string& file_name,
                        CacheAlignedVector<T>* cache_aligned_vector,
                        const std::string& path = "/data/local/tmp/") {
  std::vector<float> values;

  SPARSE_MATMUL_RETURN_IF_ERROR(ReadArrayFromFile(file_name, &values, path));

  *cache_aligned_vector = std::move(CacheAlignedVector<T>(values));

  return absl::OkStatus();
}

// Loads a 2D vector from a file.  One of rows or cols can optionally be
// -1 to indicate that dimension should be inferred.
template <typename T>
absl::Status LoadFatVector(const std::string& file_name, int rows, int cols,
                           FatCacheAlignedVector<T>* fat_cache_aligned_vector,
                           const std::string& path = "/data/local/tmp/") {
  // neither can be zero
  CHECK(rows != 0 && cols != 0);
  // only one can be -1
  CHECK(rows != -1 || cols != -1);
  // otherwise must be positive
  CHECK(rows >= -1 && cols >= -1);

  CacheAlignedVector<T> values;

  SPARSE_MATMUL_RETURN_IF_ERROR(LoadVector(file_name, &values, path));

  if (rows > 0)
    CHECK_EQ(values.size() % rows, 0);
  else
    rows = values.size() / cols;

  if (cols > 0)
    CHECK_EQ(values.size() % cols, 0);
  else
    cols = values.size() / rows;

  *fat_cache_aligned_vector = std::move(FatCacheAlignedVector<T>(values, rows));

  return absl::OkStatus();
}

// Return all files in a given directory
// If only File worked on Android and Windows...
absl::Status FilesInDirectory(const std::string& path,
                              const std::string& must_contain,
                              std::vector<std::string>* result);

}  // namespace csrblocksparse

#endif  // LYRA_CODEC_SPARSE_MATMUL_LAYERS_UTILS_H_
