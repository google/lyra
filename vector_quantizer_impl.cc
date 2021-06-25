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

#include "vector_quantizer_impl.h"

#include <bitset>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "Eigen/LU"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/types/optional.h"
#include "audio/dsp/signal_vector_util.h"
#include "glog/logging.h"
#include "include/ghc/filesystem.hpp"
#include "sparse_matmul/sparse_matmul.h"

namespace chromemedia {
namespace codec {
namespace {
std::vector<std::vector<std::vector<float>>> CodeVectorsToCodebooks(
    const std::vector<float>& flattened_code_vectors,
    const std::vector<int16_t>& codebook_dimensions) {
  // |codebook_dimensions| is a flattened array of arrays, where the inner
  // arrays stored two values per codebook, the number of code vectors and the
  // dimensionality of the vectors. Therefore, dividing the length of
  // |codebook_dimensions| by two gives the number of codebooks.
  const int num_codebooks = static_cast<int>(
      std::ceil(static_cast<float>(codebook_dimensions.size()) / 2.f));
  std::vector<std::vector<std::vector<float>>> codebooks(num_codebooks);

  // |flattened_index| points to the start of an N-dimensional vector stored
  // in |flattened_code_vectors|.
  int flattened_index = 0;
  for (int i = 0; i < num_codebooks; ++i) {
    const int dimensionality = codebook_dimensions[2 * i + 1];

    auto& current_codebook = codebooks[i];
    current_codebook.resize(codebook_dimensions[2 * i]);
    for (auto& current_code_vector : current_codebook) {
      // Build a vector from |flattened_code_vectors| starting at index
      // |flattened_index| ending at |flattened_index| + |dimensionality|.
      current_code_vector = std::vector<float>(
          flattened_code_vectors.begin() + flattened_index,
          flattened_code_vectors.begin() + flattened_index + dimensionality);
      flattened_index += dimensionality;
    }
  }

  return codebooks;
}
}  // namespace

std::unique_ptr<VectorQuantizerImpl> VectorQuantizerImpl::Create(
    int num_features, int num_bits, const ghc::filesystem::path& model_path) {
  const std::string kPrefix = "lyra_16khz_quant_";

  // Open gzipped vqs as arrays.
  std::vector<float> mean_vector_array;
  auto status = csrblocksparse::ReadArrayFromFile(
      kPrefix + "mean_vectors.gz", &mean_vector_array, model_path.string());
  if (!status.ok()) {
    LOG(ERROR) << "Couldn't read " << model_path / (kPrefix + "mean_vectors.gz")
               << ": " << status.message();
    return nullptr;
  }

  if (num_bits > kMaxNumQuantizedBits) {
    LOG(ERROR) << "Specified number of bits " << num_bits
               << "exceeds the compile-time maximum " << kMaxNumQuantizedBits;
    return nullptr;
  }

  std::vector<float> flat_transformation_matrix_array;
  status = csrblocksparse::ReadArrayFromFile(kPrefix + "transmat.gz",
                                             &flat_transformation_matrix_array,
                                             model_path.string());
  if (!status.ok()) {
    LOG(ERROR) << "Couldn't read " << model_path / (kPrefix + "transmat.gz")
               << ": " << status.message();
    return nullptr;
  }

  std::vector<float> flattened_code_vectors;
  status = csrblocksparse::ReadArrayFromFile(kPrefix + "code_vectors.gz",
                                             &flattened_code_vectors,
                                             model_path.string());
  if (!status.ok()) {
    LOG(ERROR) << "Couldn't read " << model_path / (kPrefix + "code_vectors.gz")
               << ": " << status.message();
    return nullptr;
  }

  std::vector<int16_t> codebook_dimensions;
  status = csrblocksparse::ReadArrayFromFile(kPrefix + "codebook_dimensions.gz",
                                             &codebook_dimensions,
                                             model_path.string());
  if (!status.ok()) {
    LOG(ERROR) << "Couldn't read "
               << model_path / (kPrefix + "codebook_dimensions.gz") << ": "
               << status.message();
    return nullptr;
  }

  const Eigen::Map<Eigen::RowVectorXf> mean_vector(mean_vector_array.data(),
                                                   num_features);
  const Eigen::Map<
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      transformation_matrix(flat_transformation_matrix_array.data(),
                            num_features, num_features);
  return Create(num_features, num_bits, mean_vector, transformation_matrix,
                flattened_code_vectors, codebook_dimensions);
}

std::unique_ptr<VectorQuantizerImpl> VectorQuantizerImpl::Create(
    int num_features, int num_bits, const Eigen::RowVectorXf& mean_vector,
    const Eigen::MatrixXf& transformation_matrix,
    const std::vector<float>& flattened_code_vectors,
    const std::vector<int16_t>& codebook_dimensions) {
  if (mean_vector.size() != num_features) {
    LOG(ERROR) << "Expected mean vector to be length " << num_features
               << " but was of size " << mean_vector.size();
    return nullptr;
  }
  if (transformation_matrix.cols() == 0 ||
      transformation_matrix.cols() != transformation_matrix.rows()) {
    LOG(ERROR) << "Expected transformation matrix to be square with size "
               << num_features << "x" << num_features << " but was "
               << transformation_matrix.rows() << "x"
               << transformation_matrix.cols();
    return nullptr;
  }

  if (mean_vector.size() != transformation_matrix.cols()) {
    LOG(ERROR) << "Rows of mean vector " << mean_vector.size()
               << " do not match " << transformation_matrix.cols()
               << " columns of transformation matrix.";
    return nullptr;
  }

  if (!Eigen::FullPivLU<Eigen::MatrixXf>(transformation_matrix)
           .isInvertible()) {
    LOG(ERROR) << "Transformation Matrix is not invertible.";
    return nullptr;
  }

  int total_dimensionality = 0;
  for (int i = 1; i < codebook_dimensions.size(); i += 2) {
    total_dimensionality += codebook_dimensions[i];
  }
  if (total_dimensionality != num_features) {
    LOG(ERROR) << "Codebook must have the same dimensionality as the "
               << "feature space (" << total_dimensionality << " vs "
               << num_features << ").";
    return nullptr;
  }

  const std::vector<std::vector<std::vector<float>>> codebooks =
      CodeVectorsToCodebooks(flattened_code_vectors, codebook_dimensions);
  for (const auto& codebook : codebooks) {
    if (codebook.empty()) {
      LOG(ERROR) << "Codebook did not have any code vectors in it.";
      return nullptr;
    }
  }

  return absl::WrapUnique(new VectorQuantizerImpl(
      num_features, num_bits, mean_vector, transformation_matrix, codebooks));
}

VectorQuantizerImpl::VectorQuantizerImpl(
    int num_features, int num_bits, const Eigen::RowVectorXf& mean_vector,
    const Eigen::MatrixXf& transformation_matrix,
    const std::vector<std::vector<std::vector<float>>>& codebooks)
    : num_bits_(num_bits),
      num_features_(num_features),
      mean_vector_(mean_vector),
      transformation_matrix_(transformation_matrix),
      inverse_transformation_matrix_(transformation_matrix.inverse()),
      codebooks_(codebooks) {}

absl::optional<std::string> VectorQuantizerImpl::Quantize(
    const std::vector<float>& features) const {
  if (features.size() != num_features_) {
    LOG(ERROR) << "There were " << features.size()
               << " features to be quantized but expected " << num_features_;
    return absl::nullopt;
  }

  Eigen::RowVectorXf projected_features(num_features_);
  for (int i = 0; i < features.size(); ++i) {
    projected_features(i) = features.at(i);
  }
  // Project into klt space.
  projected_features =
      (projected_features - mean_vector_) * transformation_matrix_;

  std::bitset<kMaxNumQuantizedBits> quantized_bits = 0;
  uint32_t bit_shift_amount = 0;
  uint32_t start_dimension = 0;
  // Iterate over all the codebooks.
  for (const auto& codebook : codebooks_) {
    // The number of bits needed to represent all code vectors in this code
    // book.
    int current_num_bits = std::ceil(std::log2(codebook.size()));
    if (current_num_bits == 0) {
      break;
    }
    bit_shift_amount += current_num_bits;
    // There will be at least 1 code vector within each codebook.
    uint32_t dimensionality = codebook.at(0).size();

    // Take the sub dimensions of projected_features from current codebook
    // dimensionality.
    Eigen::RowVectorXf sub_projected_feature =
        projected_features.segment(start_dimension, dimensionality);
    start_dimension += dimensionality;
    int chosen_index = FindNearest(sub_projected_feature, codebook);

    // Fill quantized_bits starting from the MSB to the LSB with the bits from
    // subsequent chosen_indexes.
    // Eg for the two bit patterns appended in order and NUM_BITS bits = 16
    //
    // 0b10001, current_num_bits = 5 ->
    //   bit_shift_amount = 5
    //   chosen_index_shift = 16 - 5 = 11
    // 0b0010, current_num_bits = 4 ->
    //   bit_shift_amount = 5 + 4 = 9
    //   chosen_index_shift = 16 - 9 = 7
    //
    //  quantized_bits will become:
    //  |1 0 0 0 1 0 0 1 0 _ _ _ _ _ _ _|
    //   |       |       |             |
    //  bit15  bit11    bit7          bit0

    int chosen_index_shift = quantized_bits.size() - bit_shift_amount;
    quantized_bits |= std::bitset<quantized_bits.size()>(chosen_index)
                      << chosen_index_shift;
  }

  return quantized_bits.to_string().substr(0, num_bits_);
}

std::vector<float> VectorQuantizerImpl::DecodeToLossyFeatures(
    const std::string& quantized_features) const {
  const std::bitset<kMaxNumQuantizedBits> quantized_bits(quantized_features);
  Eigen::RowVectorXf features(num_features_);
  int dimension = 0;
  int bit_shift_amount = num_bits_;
  for (auto& codebook : codebooks_) {
    // Shift right by the total bit width minus the accumulated bit width so
    // far.
    int current_num_bits = static_cast<int>(
        std::ceil(std::log2(static_cast<float>(codebook.size()))));
    CHECK(bit_shift_amount >= current_num_bits);
    bit_shift_amount -= current_num_bits;
    // Mask off bits we have already seen.
    const std::bitset<kMaxNumQuantizedBits> kPreviouslySeenMask(
        (1 << current_num_bits) - 1);
    int code_vector_index =
        ((quantized_bits >> bit_shift_amount) & kPreviouslySeenMask).to_ulong();

    for (auto component : codebook.at(code_vector_index)) {
      features(dimension) = component;
      ++dimension;
    }
  }
  // Project back into the log mel spectrogram domain.
  features = features * inverse_transformation_matrix_ + mean_vector_;

  return std::vector<float>(features.data(), features.data() + features.size());
}

int VectorQuantizerImpl::FindNearest(
    const Eigen::RowVectorXf& sub_projected_features,
    const std::vector<std::vector<float>>& codebook) const {
  float min_dist = std::numeric_limits<float>::max();
  int chosen_index = 0;
  int current_index = 0;

  // Iterate over all the code vectors in this subspace.
  for (const auto& code_vector : codebook) {
    float dist = 0;
    // Iterate over each dimension of the current code vector.
    for (int dimension = 0; dimension < sub_projected_features.size();
         ++dimension) {
      // Find the closest l2 distance of each code vector.
      dist += audio_dsp::Square(sub_projected_features(dimension) -
                                code_vector.at(dimension));
      // No need to keep calculating distance if we are already over the
      // previous min.
      if (dist >= min_dist) {
        break;
      }
    }
    if (dist < min_dist) {
      min_dist = dist;
      chosen_index = current_index;
    }
    current_index++;
  }

  return chosen_index;
}

}  // namespace codec
}  // namespace chromemedia
