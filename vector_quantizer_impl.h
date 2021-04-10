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

#ifndef LYRA_CODEC_VECTOR_QUANTIZER_IMPL_H_
#define LYRA_CODEC_VECTOR_QUANTIZER_IMPL_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "Eigen/Core"
#include "absl/types/optional.h"
#include "include/ghc/filesystem.hpp"
#include "vector_quantizer_interface.h"

namespace chromemedia {
namespace codec {

class VectorQuantizerImpl : public VectorQuantizerInterface {
 public:
  // Returns nullptr if the dimensions of mean_vector and transformation matrix
  // do not match NumFeatures or if codebooks contains unexpected number of
  // code_vectors or if the transformation matrix is not invertible.
  static std::unique_ptr<VectorQuantizerImpl> Create(
      int num_features, int num_bits, const ghc::filesystem::path& model_path);

  // Returns nullptr if the dimensions of mean_vector and transformation_matrix
  // do not match num_features or if codebooks contains unexpected number of
  // code_vectors or if the transformation matrix is not invertible.
  static std::unique_ptr<VectorQuantizerImpl> Create(
      int num_features, int num_bits, const Eigen::RowVectorXf& mean_vector,
      const Eigen::MatrixXf& transformation_matrix,
      const std::vector<float>& code_vectors,
      const std::vector<int16_t>& codebook_dimensions);

  // Quantizes the features using vector quantization. Results will have
  // |NumQuantizedBits| bits.
  absl::optional<std::string> Quantize(
      const std::vector<float>& features) const override;

  // Unpacks the string of bits and looks up the KLT features. Then multiplies
  // by the inverse transformation matrix and adds the mean.
  std::vector<float> DecodeToLossyFeatures(
      const std::string& quantized_features) const override;

 private:
  static constexpr int kMaxNumQuantizedBits = 200;

  VectorQuantizerImpl(
      int num_features, int num_bits, const Eigen::RowVectorXf& mean_vector,
      const Eigen::MatrixXf& transformation_matrix,
      const std::vector<std::vector<std::vector<float>>>& codebooks);

  VectorQuantizerImpl() = delete;
  // Find the closest (l2) code vector to sub_projected_features.
  int FindNearest(const Eigen::RowVectorXf& sub_projected_features,
                  const std::vector<std::vector<float>>& codebook) const;

  const int num_bits_;
  const int num_features_;
  // The mean of each feature over the dataset.
  const Eigen::RowVectorXf mean_vector_;
  // The transformation matrix used for projecting the mean subtracted features
  // vector into the klt feature space.
  const Eigen::MatrixXf transformation_matrix_;
  // Store the inverse for DecodeToLossyFeatures.
  const Eigen::MatrixXf inverse_transformation_matrix_;
  // The first dimension contains the subspaces of the klt feature space,
  // of length |subvector_indexes_ + 1|. The second dimension contains the
  // code vector points in that subspace. The third dimension contains
  // the components of the code vector, of length |subvector_indexes[i]|.
  const std::vector<std::vector<std::vector<float>>> codebooks_;

  friend class VectorQuantizerImplPeer;
};

}  // namespace codec
}  // namespace chromemedia

#endif  // LYRA_CODEC_VECTOR_QUANTIZER_IMPL_H_
