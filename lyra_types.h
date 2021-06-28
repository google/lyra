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

#ifndef LYRA_CODEC_LYRA_TYPES_H_
#define LYRA_CODEC_LYRA_TYPES_H_

#include <type_traits>

#include "layer_wrapper.h"
#include "sparse_matmul/sparse_matmul.h"

namespace chromemedia {
namespace codec {

// Type inference for the LyraWavegru class, including bit allocations for
// fixed-point types.
template <typename WeightTypeKind,
          int kArToGatesWeightExponentBits = 13,  // ar_to_gates log
          int kArToGatesRhsExponentBits = 0,      // conv_ar_input i
          int kGruWeightExponentBits = 4,         // gru_layer w
          int kGruStateExponentBits = 1,          // always in [-1, 1]
          int kGruRhsExponentBits = 13,           // gru_layer i
          class Enable = void>
struct WavegruTypes;

template <typename WeightTypeKind, int kArToGatesWeightExponentBits,
          int kArToGatesRhsExponentBits, int kGruWeightExponentBits,
          int kGruStateExponentBits, int kGruRhsExponentBits>
struct WavegruTypes<
    WeightTypeKind, kArToGatesWeightExponentBits, kArToGatesRhsExponentBits,
    kGruWeightExponentBits, kGruStateExponentBits, kGruRhsExponentBits,
    typename std::enable_if<
        std::is_same<WeightTypeKind, csrblocksparse::bfloat16>::value ||
        std::is_same<WeightTypeKind, float>::value>::type> {
  using DiskWeightType = float;
  using ArWeightType = float;
  using ArRhsType = float;
  using ArOutputType = float;
  using GruWeightType = float;
  using GruStateType = float;
  using GruRhsType = float;
  using ScratchType = float;
};

template <typename WeightTypeKind, int kArToGatesWeightExponentBits,
          int kArToGatesRhsExponentBits, int kGruWeightExponentBits,
          int kGruStateExponentBits, int kGruRhsExponentBits>
struct WavegruTypes<
    WeightTypeKind, kArToGatesWeightExponentBits, kArToGatesRhsExponentBits,
    kGruWeightExponentBits, kGruStateExponentBits, kGruRhsExponentBits,
    typename std::enable_if<std::is_same<
        WeightTypeKind, csrblocksparse::fixed16_type>::value>::type> {
  using DiskWeightType = csrblocksparse::fixed16_type;
  using ArWeightType = csrblocksparse::fixed16<kArToGatesWeightExponentBits>;
  using ArRhsType = csrblocksparse::fixed16<kArToGatesRhsExponentBits>;
  using ArOutputType =
      typename csrblocksparse::TypeOfProduct<ArWeightType, ArRhsType>::type;
  using GruWeightType = csrblocksparse::fixed16<kGruWeightExponentBits>;
  using GruStateType = csrblocksparse::fixed16<kGruStateExponentBits>;
  using GruRhsType = csrblocksparse::fixed32<kGruRhsExponentBits>;
#if defined __ARM_NEON || defined __aarch64__
  using ScratchType = int;
#else
  using ScratchType = float;
#endif  // defined __ARM_NEON || defined __aarch64__
};

template <typename WeightTypeKind,
          int kConv1DWeightExponentBits = 2,       // conv1d w
          int kConv1DRhsExponentBits = 1,          // conv1d i
          int kCondStack0WeightExponentBits = 1,   // conditioning_stack_0 w
          int kCondStack0RhsExponentBits = 3,      // conditioning_stack_0 i
          int kCondStack1WeightExponentBits = 1,   // conditioning_stack_1 w
          int kCondStack1RhsExponentBits = 3,      // conditioning_stack_1 i
          int kCondStack2WeightExponentBits = 1,   // conditioning_stack_2 w
          int kCondStack2RhsExponentBits = 3,      // conditioning_stack_2 i
          int kTranspose0WeightExponentBits = 1,   // transpose_0 w
          int kTranspose0RhsExponentBits = 4,      // transpose_0 i
          int kTranspose1WeightExponentBits = 1,   // transpose_1 w
          int kTranspose1RhsExponentBits = 3,      // transpose_1 i
          int kTranspose2WeightExponentBits = 2,   // transpose_2 w
          int kTranspose2RhsExponentBits = 3,      // transpose_2 i
          int kConvCondWeightExponentBits = 1,     // conv_cond w
          int kConvCondRhsExponentBits = 2,        // conv_cond i
          int kConvToGatesWeightExponentBits = 4,  // conv_to_gates w
          int kConvToGatesRhsExponentBits = 5,     // conv_to_gates i
          class Enable = void>
struct ConditioningTypes;

template <typename WeightTypeKind, int kConv1DWeightExponentBits,
          int kConv1DRhsExponentBits, int kCondStack0WeightExponentBits,
          int kCondStack0RhsExponentBits, int kCondStack1WeightExponentBits,
          int kCondStack1RhsExponentBits, int kCondStack2WeightExponentBits,
          int kCondStack2RhsExponentBits, int kTranspose0WeightExponentBits,
          int kTranspose0RhsExponentBits, int kTranspose1WeightExponentBits,
          int kTranspose1RhsExponentBits, int kTranspose2WeightExponentBits,
          int kTranspose2RhsExponentBits, int kConvCondWeightExponentBits,
          int kConvCondRhsExponentBits, int kConvToGatesWeightExponentBits,
          int kConvToGatesRhsExponentBits>
struct ConditioningTypes<
    WeightTypeKind, kConv1DWeightExponentBits, kConv1DRhsExponentBits,
    kCondStack0WeightExponentBits, kCondStack0RhsExponentBits,
    kCondStack1WeightExponentBits, kCondStack1RhsExponentBits,
    kCondStack2WeightExponentBits, kCondStack2RhsExponentBits,
    kTranspose0WeightExponentBits, kTranspose0RhsExponentBits,
    kTranspose1WeightExponentBits, kTranspose1RhsExponentBits,
    kTranspose2WeightExponentBits, kTranspose2RhsExponentBits,
    kConvCondWeightExponentBits, kConvCondRhsExponentBits,
    kConvToGatesWeightExponentBits, kConvToGatesRhsExponentBits,
    typename std::enable_if<
        std::is_same<WeightTypeKind, csrblocksparse::bfloat16>::value ||
        std::is_same<WeightTypeKind, float>::value>::type> {
  using DiskWeightType = float;

  using Conv1DWeightType = WeightTypeKind;
  using Conv1DRhsType = float;

  using CondStack0WeightType = WeightTypeKind;
  using CondStack0RhsType = float;

  using CondStack1WeightType = WeightTypeKind;
  using CondStack1RhsType = float;

  using CondStack2WeightType = WeightTypeKind;
  using CondStack2RhsType = float;

  using Transpose0WeightType = WeightTypeKind;
  using Transpose0RhsType = float;

  using Transpose1WeightType = WeightTypeKind;
  using Transpose1RhsType = float;

  using Transpose2WeightType = WeightTypeKind;
  using Transpose2RhsType = float;

  using ConvCondWeightType = WeightTypeKind;
  using ConvCondRhsType = float;
  using ConvCondOutputType = float;

  using ConvToGatesWeightType = WeightTypeKind;
  using ConvToGatesRhsType = float;

  using ConvToGatesOutType = float;
  using OutputType = typename WavegruTypes<WeightTypeKind>::GruRhsType;
};

template <typename WeightTypeKind, int kConv1DWeightExponentBits,
          int kConv1DRhsExponentBits, int kCondStack0WeightExponentBits,
          int kCondStack0RhsExponentBits, int kCondStack1WeightExponentBits,
          int kCondStack1RhsExponentBits, int kCondStack2WeightExponentBits,
          int kCondStack2RhsExponentBits, int kTranspose0WeightExponentBits,
          int kTranspose0RhsExponentBits, int kTranspose1WeightExponentBits,
          int kTranspose1RhsExponentBits, int kTranspose2WeightExponentBits,
          int kTranspose2RhsExponentBits, int kConvCondWeightExponentBits,
          int kConvCondRhsExponentBits, int kConvToGatesWeightExponentBits,
          int kConvToGatesRhsExponentBits>
struct ConditioningTypes<
    WeightTypeKind, kConv1DWeightExponentBits, kConv1DRhsExponentBits,
    kCondStack0WeightExponentBits, kCondStack0RhsExponentBits,
    kCondStack1WeightExponentBits, kCondStack1RhsExponentBits,
    kCondStack2WeightExponentBits, kCondStack2RhsExponentBits,
    kTranspose0WeightExponentBits, kTranspose0RhsExponentBits,
    kTranspose1WeightExponentBits, kTranspose1RhsExponentBits,
    kTranspose2WeightExponentBits, kTranspose2RhsExponentBits,
    kConvCondWeightExponentBits, kConvCondRhsExponentBits,
    kConvToGatesWeightExponentBits, kConvToGatesRhsExponentBits,
    typename std::enable_if<std::is_same<
        WeightTypeKind, csrblocksparse::fixed16_type>::value>::type> {
  using DiskWeightType = csrblocksparse::fixed16_type;

  using Conv1DWeightType = csrblocksparse::fixed16<kConv1DWeightExponentBits>;
  using Conv1DRhsType = csrblocksparse::fixed16<kConv1DRhsExponentBits>;

  using CondStack0WeightType =
      csrblocksparse::fixed16<kCondStack0WeightExponentBits>;
  using CondStack0RhsType = csrblocksparse::fixed16<kCondStack0RhsExponentBits>;

  using CondStack1WeightType =
      csrblocksparse::fixed16<kCondStack1WeightExponentBits>;
  using CondStack1RhsType = csrblocksparse::fixed16<kCondStack1RhsExponentBits>;

  using CondStack2WeightType =
      csrblocksparse::fixed16<kCondStack2WeightExponentBits>;
  using CondStack2RhsType = csrblocksparse::fixed16<kCondStack2RhsExponentBits>;

  using Transpose0WeightType =
      csrblocksparse::fixed16<kTranspose0WeightExponentBits>;
  using Transpose0RhsType = csrblocksparse::fixed16<kTranspose0RhsExponentBits>;

  using Transpose1WeightType =
      csrblocksparse::fixed16<kTranspose1WeightExponentBits>;
  using Transpose1RhsType = csrblocksparse::fixed16<kTranspose1RhsExponentBits>;

  using Transpose2WeightType =
      csrblocksparse::fixed16<kTranspose2WeightExponentBits>;
  using Transpose2RhsType = csrblocksparse::fixed16<kTranspose2RhsExponentBits>;

  using ConvCondWeightType =
      csrblocksparse::fixed16<kConvCondWeightExponentBits>;
  using ConvCondRhsType = csrblocksparse::fixed16<kConvCondRhsExponentBits>;
  using ConvCondOutputType =
      typename csrblocksparse::TypeOfProduct<ConvCondWeightType,
                                             ConvCondRhsType>::type;

  using ConvToGatesWeightType =
      csrblocksparse::fixed16<kConvToGatesWeightExponentBits>;
  using ConvToGatesRhsType =
      csrblocksparse::fixed16<kConvToGatesRhsExponentBits>;

  using ConvToGatesOutType =
      typename csrblocksparse::TypeOfProduct<ConvToGatesWeightType,
                                             ConvToGatesRhsType>::type;
  using OutputType = typename WavegruTypes<WeightTypeKind>::GruRhsType;
};

template <typename WeightTypeKind,
          int kProjWeightExponentBits = 2,     // proj w
          int kScaleWeightExponentBits = 3,    // scales w
          int kMeanWeightExponentBits = 3,     // means w
          int kMixWeightExponentBits = 4,      // mix w
          int kProjMatMulOutExponentBits = 4,  // scales i
          class Enable = void>
struct ProjectAndSampleTypes;

template <typename WeightTypeKind, int kProjWeightExponentBits,
          int kScaleWeightExponentBits, int kMeanWeightExponentBits,
          int kMixWeightExponentBits, int kProjMatMulOutExponentBits>
struct ProjectAndSampleTypes<
    WeightTypeKind, kProjWeightExponentBits, kScaleWeightExponentBits,
    kMeanWeightExponentBits, kMixWeightExponentBits, kProjMatMulOutExponentBits,
    typename std::enable_if<
        std::is_same<WeightTypeKind, csrblocksparse::bfloat16>::value ||
        std::is_same<WeightTypeKind, float>::value>::type> {
  using DiskWeightType = float;
  using ScratchType = float;
  using ProjWeightType = WeightTypeKind;
  // The input to the projection layer is the GRU state.
  using ProjRhsType = typename WavegruTypes<WeightTypeKind>::GruStateType;
  using ProjMatMulOutType = WeightTypeKind;
  using ScaleWeightType = WeightTypeKind;
  using MeanWeightType = WeightTypeKind;
  using MixWeightType = WeightTypeKind;
};

template <typename WeightTypeKind, int kProjWeightExponentBits,
          int kScaleWeightExponentBits, int kMeanWeightExponentBits,
          int kMixWeightExponentBits, int kProjMatMulOutExponentBits>
struct ProjectAndSampleTypes<
    WeightTypeKind, kProjWeightExponentBits, kScaleWeightExponentBits,
    kMeanWeightExponentBits, kMixWeightExponentBits, kProjMatMulOutExponentBits,
    typename std::enable_if<std::is_same<
        WeightTypeKind, csrblocksparse::fixed16_type>::value>::type> {
  using DiskWeightType = csrblocksparse::fixed16_type;
#if defined(__aarch64__)
  using ScratchType = int;
#else
  using ScratchType = float;
#endif
  using ProjWeightType = csrblocksparse::fixed16<kProjWeightExponentBits>;
  // The input to the projection layer is the GRU state.
  using ProjRhsType = typename WavegruTypes<WeightTypeKind>::GruStateType;
  using ProjMatMulOutType = csrblocksparse::fixed16<kProjMatMulOutExponentBits>;
  using ScaleWeightType = csrblocksparse::fixed16<kScaleWeightExponentBits>;
  using MeanWeightType = csrblocksparse::fixed16<kMeanWeightExponentBits>;
  using MixWeightType = csrblocksparse::fixed16<kMixWeightExponentBits>;
};

}  // namespace codec
}  // namespace chromemedia

#endif  // LYRA_CODEC_LYRA_TYPES_H_
