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

#include "sparse_matmul/layers/utils.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <type_traits>
#include <vector>

#include "absl/flags/flag.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "include/ghc/filesystem.hpp"
#include "sparse_matmul/layers/csr_blocksparse_matrix.h"
#include "sparse_matmul/layers/errno_mapping.h"
#include "sparse_matmul/layers/sparse_linear_layer.h"
#include "sparse_matmul/numerics/fast_transcendentals.h"
#include "sparse_matmul/numerics/fixed_types.h"
#include "sparse_matmul/numerics/float16_types.h"
#include "sparse_matmul/numerics/test_utils.h"
#include "sparse_matmul/numerics/type_utils.h"
#include "sparse_matmul/vector/cache_aligned_vector.h"

namespace csrblocksparse {
namespace {

static constexpr char kTempOutputDir[] =
    "third_party/lyra_codec/sparse_matmul/layers/testdata/";
static constexpr int kTestExponentBits = 5;

template <typename ComputeType>
class CsrBlockSparseMatrixUtilsTest : public testing::Test {
 protected:
  CsrBlockSparseMatrixUtilsTest()
      : output_dir_((ghc::filesystem::path(testing::TempDir()) / kTempOutputDir)
                        .string()) {
    if (std::is_floating_point<ComputeType>::value) {
      tolerance_ = 1e-5;
    } else if (csrblocksparse::IsCustomFloatType<ComputeType>::value) {
      // Casting float --> bfloat truncates the least significant 16 bits from
      // the mantissa, thus the larger the exponent bits the larger the rounding
      // error.
      // The exponent for max_val is 2^4, meaning the max rounding error
      // for the weight input is ~ 0.124. The tolerance is 2x this because
      // although the intermediate multiplications are accumulated in float,
      // the output is cast to bfloat.
      // Placeholder for internal diagram.
      float max_val =
          std::pow<float>(2, kTestExponentBits) -
          std::pow<float>(2, -fixed16<kTestExponentBits>::kMantissaBits);
      tolerance_ = 2 * (max_val - static_cast<float>(ComputeType(max_val)));
    } else {
      tolerance_ = std::pow<float>(2, -MantissaBitsOf<ComputeType>::value);
    }
  }

  void SetUp() override {
    std::error_code error_code;
    ghc::filesystem::create_directories(output_dir_, error_code);
    ASSERT_FALSE(error_code);
  }

  void TearDown() override {
    std::error_code error_code;
    ghc::filesystem::remove_all(output_dir_, error_code);
    ASSERT_FALSE(error_code);
  }

  const std::string output_dir_;
  float tolerance_;
};

void GenerateRandomWeightBiasMaskVectors(
    int weight_vector_size, int bias_vector_size,
    std::vector<float>* weight_vector, std::vector<float>* bias_vector,
    std::vector<float>* mask_vector, std::vector<float>* masked_weight_vector) {
  weight_vector->resize(weight_vector_size);
  bias_vector->resize(bias_vector_size);
  mask_vector->resize(weight_vector_size);
  masked_weight_vector->resize(weight_vector_size);
  // Fill Weight and Bias with random values between +/-[2^|kTestExponentBits| -
  // 1] - 0.5 to prevent clipping in the fixed16 case when the weight and bias
  // are added with all 1s in the exponent and mantissa.
  const float max_abs_random_value =
      std::pow<float>(2, kTestExponentBits - 1) - 0.5;
  std::uniform_real_distribution<float> distribution(-max_abs_random_value,
                                                     max_abs_random_value);
  std::default_random_engine generator(1337);
  std::generate(weight_vector->begin(), weight_vector->end(),
                [&]() { return distribution(generator); });
  std::generate(bias_vector->begin(), bias_vector->end(),
                [&]() { return distribution(generator); });
  std::bernoulli_distribution mask_distribution(0.5);
  std::generate(mask_vector->begin(), mask_vector->end(),
                [&]() { return mask_distribution(generator) ? 1 : 0; });
  // Construct the combined weight and mask vector.
  std::transform(mask_vector->begin(), mask_vector->end(),
                 weight_vector->begin(), masked_weight_vector->begin(),
                 [&](float mask_value, float weight_value) {
                   return mask_value * weight_value;
                 });
}

using ComputeTypes =
    testing::Types<float, csrblocksparse::fixed16<kTestExponentBits>,
                   csrblocksparse::bfloat16>;
TYPED_TEST_SUITE(CsrBlockSparseMatrixUtilsTest, ComputeTypes);

TYPED_TEST(CsrBlockSparseMatrixUtilsTest, LoadLayer) {
  const int kWeightVectorSize = 16;
  const int kBiasVectorSize = 4;
  std::vector<float> ref_weight_vector;
  std::vector<float> ref_bias_vector;
  std::vector<float> ref_mask_vector;
  std::vector<float> ref_masked_weight_vector;

  GenerateRandomWeightBiasMaskVectors(
      kWeightVectorSize, kBiasVectorSize, &ref_weight_vector, &ref_bias_vector,
      &ref_mask_vector, &ref_masked_weight_vector);

  // This fixed16_weights.raw vector should only be read by LoadGenericLayer
  // when |TypeParam| is a fixed16_type.
  std::vector<int16_t> fixed_weight_vector(ref_weight_vector.size());
  std::transform(ref_weight_vector.begin(), ref_weight_vector.end(),
                 fixed_weight_vector.begin(), [](float weight) {
                   return fixed16<kTestExponentBits>(weight).raw_val();
                 });
  ASSERT_TRUE(WriteArrayToFile(fixed_weight_vector, "fixed16_weights.raw",
                               this->output_dir_)
                  .ok());
  ASSERT_TRUE(
      WriteArrayToFile(ref_weight_vector, "weights.raw", this->output_dir_)
          .ok());
  ASSERT_TRUE(
      WriteArrayToFile(ref_bias_vector, "bias.raw", this->output_dir_).ok());
  ASSERT_TRUE(
      WriteArrayToFile(ref_mask_vector, "mask.raw", this->output_dir_).ok());

  // Read in the weights, mask, and bias to a layer.
  SparseLinearLayer<TypeParam, TypeParam> actual_layer;
  using DiskWeightType =
      typename std::conditional<csrblocksparse::IsFixed16Type<TypeParam>::value,
                                csrblocksparse::fixed16_type, TypeParam>::type;
  auto status = LoadGenericLayer<TypeParam, TypeParam, DiskWeightType>(
      /*prefix=*/"", /*zipped=*/false, this->output_dir_,
      /*default_bias=*/0.f, &actual_layer);
  ASSERT_TRUE(status.ok());
  // Multiply the read in layer with an identity matrix so we just get
  // the weights added with bias.
  std::vector<TypeParam> identity(kBiasVectorSize * kBiasVectorSize,
                                  TypeParam(0.f));
  for (int i = 0; i < identity.size(); i += kBiasVectorSize + 1) {
    identity.at(i) = TypeParam(1.f);
  }
  FatCacheAlignedVector<TypeParam> masked_weights_plus_bias(kBiasVectorSize,
                                                            kBiasVectorSize);
  actual_layer.SpMM_bias(
      VectorView<TypeParam>(identity.data(), /*rows=*/kBiasVectorSize,
                            /*cols=*/kBiasVectorSize),
      &masked_weights_plus_bias);
  // |masked_weights_plus_bias| - bias = masked weights.
  for (int col = 0; col < masked_weights_plus_bias.cols(); col++) {
    MutableVectorView<TypeParam> col_data = masked_weights_plus_bias.slice(col);
    for (int row = 0; row < masked_weights_plus_bias.rows(); row++) {
      int flat_index = row * masked_weights_plus_bias.cols() + col;
      EXPECT_NEAR(static_cast<float>(col_data[row]) - ref_bias_vector.at(row),
                  ref_masked_weight_vector.at(flat_index), this->tolerance_);
    }
  }
}
}  // namespace
}  // namespace csrblocksparse
