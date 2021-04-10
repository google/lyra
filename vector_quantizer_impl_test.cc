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
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

// placeholder for get runfiles header.
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/memory/memory.h"
#include "absl/strings/string_view.h"
#include "Eigen/Core"
#include "include/ghc/filesystem.hpp"
#include "lyra_config.h"

namespace chromemedia {
namespace codec {
namespace {

static constexpr int kTestNumFeatures = 4;
static constexpr int kTestNumBits = 120;

// A peer class for testing so that mean_vector / transformation_matrix /
// codebooks may be injected.
class VectorQuantizerImplPeer {
 public:
  static std::unique_ptr<VectorQuantizerImplPeer> Create(
      const std::vector<float>& mean_vector,
      const std::vector<std::vector<float>>& transformation_matrix,
      const std::vector<float> flattened_code_vectors,
      const std::vector<int16_t> codebook_dimensions) {
    // It is simplest to store the mean vector and transformation matrix as
    // const vectors and convert them to eigen types here.
    Eigen::VectorXf eigen_mean_vector(mean_vector.size());
    for (int i = 0; i < mean_vector.size(); ++i) {
      eigen_mean_vector(i) = mean_vector.at(i);
    }
    Eigen::MatrixXf eigen_transformation_matrix(
        transformation_matrix.size(), transformation_matrix.at(0).size());
    for (int i = 0; i < transformation_matrix.size(); ++i) {
      for (int j = 0; j < transformation_matrix.at(0).size(); ++j) {
        eigen_transformation_matrix(i, j) = transformation_matrix.at(i).at(j);
      }
    }

    auto quantizer = VectorQuantizerImpl::Create(
        kTestNumFeatures, kTestNumBits, eigen_mean_vector,
        eigen_transformation_matrix, flattened_code_vectors,
        codebook_dimensions);
    if (quantizer == nullptr) {
      return nullptr;
    }
    return absl::WrapUnique(new VectorQuantizerImplPeer(std::move(quantizer)));
  }

  absl::optional<std::string> Quantize(
      const std::vector<float>& features) const {
    return quantizer_->Quantize(features);
  }

  std::vector<float> DecodeToLossyFeatures(
      const std::string& quantized_features) const {
    return quantizer_->DecodeToLossyFeatures(quantized_features);
  }

 private:
  explicit VectorQuantizerImplPeer(
      std::unique_ptr<VectorQuantizerImpl> quantizer)
      : quantizer_(std::move(quantizer)) {}

  std::unique_ptr<VectorQuantizerImpl> quantizer_;
};

class VectorQuantizerImplTest : public testing::Test {
 public:
  VectorQuantizerImplTest()
      : model_path_(ghc::filesystem::current_path() /
                    "wavegru"),
        quantizer_(VectorQuantizerImplPeer::Create(
            mean_vector_, transformation_matrix_, flattened_code_vectors_,
            codebook_dimensions_)) {}

 protected:
  static constexpr int kNumQuantizedBits = 120;
  const std::vector<float> mean_vector_ = {0.1, 0.5, 0.3, -0.2};
  const std::vector<std::vector<float>> transformation_matrix_ = {
      {-0.2, 0.6, -0.8, -0.5},
      {-0.7, -0.4, -0.8, 0.3},
      {0.4, 0.1, -0.5, 0.7},
      {0.2, -0.2, 0.9, -0.8},
  };
  const std::vector<int16_t> codebook_dimensions_ = {
      // Codebook 1: 2 vectors of dimension 2.
      2, 2,
      // Codebook 2: 4 vectors of dimension 2.
      4, 2};
  const std::vector<float> flattened_code_vectors_ = {
      // Codebook 1: 2 vectors.
      0.5, 0.5,    // vector 1
      -0.5, -0.5,  // vector 2
      // Codebook 2: 4 vectors.
      0.25, -0.25,   // vector 1
      -0.25, 0.25,   // vector 2
      -0.25, -0.25,  // vector 3
      0.25, 0.25     // vector 4
  };
  const ghc::filesystem::path model_path_;

  std::unique_ptr<VectorQuantizerImplPeer> quantizer_;
};

TEST_F(VectorQuantizerImplTest, CodebookWithWrongTotalDimensionalityFails) {
  const std::vector<int16_t> invalid_codebook_dimensions = {
      // 1 vector of dimension 2.
      1, 2,
      // 4 vectors of dimension 1, making the total dimensionality defined
      // by the codebook to be 3, which is not the same as the number of
      // features (4).
      4, 1};

  EXPECT_EQ(nullptr, VectorQuantizerImplPeer::Create(
                         mean_vector_, transformation_matrix_,
                         flattened_code_vectors_, invalid_codebook_dimensions));
}

TEST_F(VectorQuantizerImplTest, CodebookWithNoElementsCreateFails) {
  const std::vector<float> invalid_flattened_code_vectors = {
      // Valid codebook.
      0.5, 0.5,
      // No code vectors in this codebook causes Create to return nullptr;
  };
  const std::vector<int16_t> invalid_codebook_dimensions = {
      // 1 vector of dimension 2.
      1, 2,
      // 0 vector of dimension 2.
      0, 2};

  EXPECT_EQ(nullptr,
            VectorQuantizerImplPeer::Create(
                mean_vector_, transformation_matrix_,
                invalid_flattened_code_vectors, invalid_codebook_dimensions));
}

TEST_F(VectorQuantizerImplTest, MeanVectorTooSmallCreateFails) {
  const std::vector<float> invalid_mean_vector(
      mean_vector_.begin(), mean_vector_.begin() + kTestNumFeatures - 1);

  EXPECT_EQ(nullptr, VectorQuantizerImplPeer::Create(
                         invalid_mean_vector, transformation_matrix_,
                         flattened_code_vectors_, codebook_dimensions_));
}

TEST_F(VectorQuantizerImplTest, TransformationMatrixTooSmallCreateFails) {
  const std::vector<std::vector<float>> invalid_transformation_matrix({
      transformation_matrix_.at(0),
      transformation_matrix_.at(1),
      transformation_matrix_.at(2),
  });

  EXPECT_EQ(nullptr, VectorQuantizerImplPeer::Create(
                         mean_vector_, invalid_transformation_matrix,
                         flattened_code_vectors_, codebook_dimensions_));
}

TEST_F(VectorQuantizerImplTest, QuantizeTooFewFeatures) {
  const std::vector<float> features(kTestNumFeatures - 1);
  auto quantized_or = quantizer_->Quantize(features);
  EXPECT_FALSE(quantized_or.has_value());
}

TEST_F(VectorQuantizerImplTest, QuantizeTooManyFeatures) {
  const std::vector<float> features(kTestNumFeatures + 1);
  auto quantized_or = quantizer_->Quantize(features);
  EXPECT_FALSE(quantized_or.has_value());
}

// Test that Quantize on features performs equivalently to the mean removal of
// features followed by matrix multiplication by the transformation matrix.
TEST_F(VectorQuantizerImplTest, QuantizeValidNumFeatures) {
  const std::vector<float> features(
      {0.9083545, -0.63350268, 0.9596105, -0.67812588});
  // After mean removal and projection this vector is {0.8, 1.1, -0.5, 0.1}.
  // The VqConfigs split this into
  // {0.8, 1.1} with 1 bit -> closest to {0.5, 0.5} -> bit pattern 0b0.
  // {-0.5, 0.1} with 2 bits -> closest to {-0.25, 0.25} -> bit pattern 0b01.
  // So the concatenated expected result is 0b001 in the MSB.
  std::bitset<kNumQuantizedBits> expected(0);
  const uint32_t kExpectedBitShift = kNumQuantizedBits - 3;
  expected |= std::bitset<kNumQuantizedBits>(0b001) << kExpectedBitShift;

  auto quantized_or = quantizer_->Quantize(features);

  EXPECT_TRUE(quantized_or.has_value());
  EXPECT_THAT(quantized_or.value(), expected.to_string());
}

TEST_F(VectorQuantizerImplTest, DecodeToLossyFeaturesValidNumFeatures) {
  // Bit pattern 0b110 corresponds to the quantized features {-0.5, -0.5, -0.25,
  // -0.25} in the klt domain. After multiplying by the inverse of the
  // transformation matrix and adding the mean the result is {0.18114592,
  // 1.51749929,  0.47358171,  0.59523003}.
  std::bitset<kNumQuantizedBits> quantized(0b110);
  quantized <<= kNumQuantizedBits - 3;
  const std::vector<float> expected_features(
      {0.18114592, 1.51749929, 0.47358171, 0.59523003});

  std::vector<float> features =
      quantizer_->DecodeToLossyFeatures(quantized.to_string());

  EXPECT_THAT(features,
              testing::Pointwise(testing::FloatEq(), expected_features));
}

TEST_F(VectorQuantizerImplTest, DefaultCreateSucceedsWithProdNumFeatures) {
  auto quantizer = VectorQuantizerImpl::Create(
      kNumFramesPerPacket * kNumFeatures, 120, model_path_);

  EXPECT_NE(quantizer, nullptr);
}

TEST_F(VectorQuantizerImplTest, TooManyBits) {
  // kMaxNumQuantizedBits is 200; try 201.
  auto quantizer = VectorQuantizerImpl::Create(
      kNumFramesPerPacket * kNumFeatures, 201, model_path_);
  EXPECT_EQ(quantizer, nullptr);
}

}  // namespace
}  // namespace codec
}  // namespace chromemedia
