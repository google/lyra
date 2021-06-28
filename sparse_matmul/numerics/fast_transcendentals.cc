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

#include "sparse_matmul/numerics/fast_transcendentals.h"

namespace csrblocksparse {

// Maximum desired precision of the output.
static constexpr int kMaxMantissaBits = 14;

// Returns (and builds if not done yet) a static data table that implements
// tanh on fixed32 input, returning another fixed32 with the given number of
// mantissa bits (which is assumed to be less than the input mantissa bits).
// NOTE that this function is intended to be used only with fixed16 outputs that
// are sign-extended to 32 bits for convenience, and will return a nullptr
// if asked for more than |kMaxMantissaBits| of precision in the output table.
const int32_t* TanhTable(int num_mantissa_bits_out) {
  if (num_mantissa_bits_out > kMaxMantissaBits) return nullptr;
  // Static data dynamically created and never destructed.
  static const int32_t* tanh_luts[kMaxMantissaBits];
  if (tanh_luts[num_mantissa_bits_out - 1] == nullptr) {
    // Total bits is number each side of the binary point.
    int tanh_lut_bits = num_mantissa_bits_out + kNumTanhExpBits;
    // Offset is the number of negative numbers represented.
    int tanh_offset = 1 << tanh_lut_bits;
    // Size is double the offset plus one more for zero.
    int tanh_size = tanh_offset * 2 + 1;
    // Conversion between int and float.
    float float_factor = static_cast<float>(1 << num_mantissa_bits_out);
    int* tanh_lut = new int[tanh_size];
    // Initialize the table.
    for (int i = 0; i < tanh_size; ++i) {
      float x = (i - tanh_offset) / float_factor;
      tanh_lut[i] = static_cast<int>(std::round(tanhf(x) * float_factor));
    }
    tanh_luts[num_mantissa_bits_out - 1] = tanh_lut;
  }
  return tanh_luts[num_mantissa_bits_out - 1];
}

// As TanhTable, but for Sigmoid.
const int32_t* SigmoidTable(int num_mantissa_bits_out) {
  if (num_mantissa_bits_out > kMaxMantissaBits) return nullptr;
  // Static data dynamically created and never destructed.
  static const int32_t* sigmoid_luts[kMaxMantissaBits];
  if (sigmoid_luts[num_mantissa_bits_out - 1] == nullptr) {
    // Total bits is number each side of the binary point minus one for the fact
    // that the gradient never exceeds 1/4. (Could probably use -2.)
    int sigmoid_lut_bits =
        num_mantissa_bits_out + kNumSigmoidExpBits - kNumExtraSigmoidShiftBits;
    // Offset is the number of negative numbers represented.
    int sigmoid_offset = 1 << sigmoid_lut_bits;
    // Size is double the offset plus one more for zero.
    int sigmoid_size = sigmoid_offset * 2 + 1;
    // Conversion between int and float.
    float float_factor = static_cast<float>(1 << num_mantissa_bits_out);
    int* sigmoid_lut = new int[sigmoid_size];
    // Initialize the table.
    for (int i = 0; i < sigmoid_size; ++i) {
      constexpr int kSigmoidFactor = 1 << kNumExtraSigmoidShiftBits;
      float x = ((i - sigmoid_offset) * kSigmoidFactor) / float_factor;
      float sigmoid = 1.0f / (1.0f + expf(-x));
      sigmoid_lut[i] = static_cast<int>(std::round(sigmoid * float_factor));
    }
    sigmoid_luts[num_mantissa_bits_out - 1] = sigmoid_lut;
  }
  return sigmoid_luts[num_mantissa_bits_out - 1];
}

}  // namespace csrblocksparse
