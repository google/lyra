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

#ifndef LYRA_CODEC_SPARSE_INFERENCE_MATRIXVECTOR_H_
#define LYRA_CODEC_SPARSE_INFERENCE_MATRIXVECTOR_H_

#include "glog/logging.h"
#include "absl/status/status.h"

// [internal] Start of sparse_inference_matrixvector declarations.

#if defined __aarch64__
#include <arm_neon.h>
#endif
#include <cstdint>
#if defined(__x86_64__) || defined(__i386__) || defined(_WIN32)
#include <cpuid.h>
#endif
#include <random>
#include <string>
#include <thread>  // NOLINT
#include <vector>

namespace csrblocksparse {

enum class ARInputsMode {
  k0ARInputs,
  k2ARInputs,
  k3ARInputs,
};

class SpinBarrier {
 public:
  explicit SpinBarrier(int num_threads)
      : num_threads_(num_threads), threads_at_barrier_(0), barrier_step_(0) {}

  void barrier();

 private:
  const int num_threads_;
  std::atomic<int32_t> threads_at_barrier_;
  std::atomic<uint32_t> barrier_step_;
};

class ProducerConsumer {
 public:
  ProducerConsumer(int num_producers, int num_consumers);

  inline void produce();

  inline void consume();
  int num_producers() const;
  int num_consumers() const;

 private:
  const int num_producers_;
  const int num_consumers_;
  std::atomic<int32_t> producers_ready_;
  std::atomic<int32_t> consumers_passed_;
};

using Thread = std::thread;

class fixed16_type {};
class fixed32_type {};

template <int ExponentBits>
class fixed16 : fixed16_type {
 public:
  static constexpr int kExponentBits = ExponentBits;
  static constexpr int kMantissaBits = 16 - ExponentBits - 1;

  fixed16() = default;
  explicit fixed16(float x);
  explicit fixed16(int16_t x);

  explicit operator float() const;

  int raw_val() const;

 private:
  inline float fixed16_to_float(int16_t x) const;

  inline int16_t float_to_fixed16(float x) const;

  int16_t val_;
};

template <int ExponentBits>
class fixed32 : fixed32_type {
 public:
  static constexpr int kExponentBits = ExponentBits;
  static constexpr int kMantissaBits = 32 - ExponentBits - 1;

  fixed32() = default;
  explicit fixed32(float x);
  explicit fixed32(int32_t x);

  explicit operator float() const;

  int raw_val() const;

 private:
  inline float fixed32_to_float(int32_t x) const;

  inline int32_t float_to_fixed32(float x) const;

  int32_t val_;
};

class bfloat16 {
 public:
  bfloat16() = default;
  explicit bfloat16(float x);
  explicit bfloat16(uint16_t x);
  static constexpr int kMantissaBits = 8;

  explicit operator float() const;

 private:
  inline uint16_t float_to_bfloat16(float x) const;

  inline float bfloat16_to_float(uint32_t as_int) const;

  uint16_t val_;
};

template <typename T>
struct IsCustomFloatType
    : std::integral_constant<bool, std::is_same<T, bfloat16>::value> {};

template <typename T>
struct IsAnyFloatType
    : std::integral_constant<bool, std::is_floating_point<T>::value ||
                                       IsCustomFloatType<T>::value> {};

template <typename T>
struct IsFixed16Type
    : std::integral_constant<bool, std::is_base_of<fixed16_type, T>::value> {};

template <typename T>
struct IsFixed32Type
    : std::integral_constant<bool, std::is_base_of<fixed32_type, T>::value> {};

template <typename T>
struct IsFixedType : std::integral_constant<bool, IsFixed16Type<T>::value ||
                                                      IsFixed32Type<T>::value> {
};

template <typename LhsType, typename RhsType, class Enable = void>
struct TypeOfProduct {};

template <typename LhsType, typename RhsType>
struct TypeOfProduct<
    LhsType, RhsType,
    typename std::enable_if<IsAnyFloatType<LhsType>::value &&
                            IsAnyFloatType<RhsType>::value>::type> {
  using type = float;
};

template <typename LhsType, typename RhsType>
struct TypeOfProduct<
    LhsType, RhsType,
    typename std::enable_if<IsFixed16Type<LhsType>::value &&
                            IsFixed16Type<RhsType>::value>::type> {
  static_assert(LhsType::kMantissaBits + RhsType::kMantissaBits < 31,
                "Sum of mantissa bits must not exceed 31.");
  using type = fixed32<31 - LhsType::kMantissaBits - RhsType::kMantissaBits>;
};

template <typename T, class Enable = void>
struct MantissaBitsOf {
  static constexpr int value = 1;
};

namespace detail {

#if defined __AVX__

#if defined __AVX2__

template <typename Type>
struct IsAddableFixedTypes
    : std::integral_constant<bool, IsFixed32Type<Type>::value ||
                                       IsFixed16Type<Type>::value> {};
template <typename Type>
struct ShouldEnableGenericAdd
    : std::integral_constant<bool, !IsAddableFixedTypes<Type>::value> {};

#else  // No AVX2.

template <typename Type>
struct ShouldEnableGenericAdd : std::true_type {};

#endif  // __AVX2__

template <typename Type>
typename std::enable_if<IsFixed32Type<Type>::value>::type SumVectors(
    int start, int end, const Type* add1, const Type* add2, Type* result);

template <typename Type>
typename std::enable_if<IsFixed16Type<Type>::value>::type SumVectors(
    int start, int end, const Type* add1, const Type* add2, Type* result);

#elif defined __aarch64__

template <typename Type>
struct IsAddableFixedTypes
    : std::integral_constant<bool, IsFixed32Type<Type>::value ||
                                       IsFixed16Type<Type>::value> {};
template <typename Type>
struct ShouldEnableGenericAdd
    : std::integral_constant<bool, !IsAddableFixedTypes<Type>::value> {};

template <typename Type>
typename std::enable_if<IsFixed32Type<Type>::value>::type SumVectors(
    int start, int end, const Type* add1, const Type* add2, Type* result);

template <typename Type>
typename std::enable_if<IsFixed16Type<Type>::value>::type SumVectors(
    int start, int end, const Type* add1, const Type* add2, Type* result);

#else  // defined __aarch64__

template <typename Type>
struct ShouldEnableGenericAdd : std::true_type {};

#endif  // defined __AVX__

template <typename Type>
typename std::enable_if<ShouldEnableGenericAdd<Type>::value>::type SumVectors(
    int start, int end, const Type* add1, const Type* add2, Type* result);

}  // namespace detail

template <typename T>
class MutableVectorView;
template <typename T>
class VectorView;

template <typename DataType>
class CacheAlignedVector {
 public:
  using value_type = DataType;

  explicit CacheAlignedVector(std::size_t size);

  explicit CacheAlignedVector(const std::vector<DataType>& input);

  template <typename InputType>
  explicit CacheAlignedVector(const std::vector<InputType>& input);

  CacheAlignedVector(const DataType* input, int size);

  template <typename InputType>
  explicit CacheAlignedVector(const InputType* input, int size);

  CacheAlignedVector();

  ~CacheAlignedVector();

  CacheAlignedVector(CacheAlignedVector const& other);
  CacheAlignedVector(CacheAlignedVector const& other, int start, int end);

  void operator=(CacheAlignedVector const& other);

  CacheAlignedVector(CacheAlignedVector<DataType>&& other);

  CacheAlignedVector<DataType>& operator=(CacheAlignedVector<DataType>&& other);

  VectorView<DataType> AsView() const;

  MutableVectorView<DataType> AsMutableView();

  void PrepareForThreads(const std::vector<int>& split_points,
                         int block_height);

  void FillRandom(float min = -10.f, float max = 10.f);

  void FillZero();

  void FillOnes();

  void FillWith(const DataType& value);

  template <class Q = DataType>
  typename std::enable_if<std::is_same<Q, float>::value, int>::type Sample(
      float temperature = 1.f);

#if defined __aarch64__
  template <class Q = DataType>
  typename std::enable_if<std::is_same<Q, float>::value, int>::type Sample(
      float temperature, std::minstd_rand* gen,
      CacheAlignedVector<float>* scratch) const;

  template <class Q = DataType>
  static inline int32x4_t vmul_temp_fixed(int32x4_t x, int32x2_t inv_temp);

  template <class Q = DataType>
  static inline int float_to_fixed(float x);

  template <class Q = DataType>
  static inline float fixed_to_float(int x);

  template <class Q = DataType>
  typename std::enable_if<IsFixed32Type<Q>::value, int>::type Sample(
      float temperature, std::minstd_rand* gen,
      CacheAlignedVector<int>* scratch) const;
#endif  // defined __aarch64__

  template <class Q = DataType>
#if defined __aarch64__
  typename std::enable_if<
      !std::is_same<Q, float>::value && !IsFixed32Type<Q>::value, int>::type
#else
  int
#endif
  Sample(float temperature, std::minstd_rand* gen,
         CacheAlignedVector<float>* scratch, int tid = 0,
         SpinBarrier* barrier = nullptr) const;

  int ScalarSample(float temperature, std::minstd_rand* gen,
                   CacheAlignedVector<float>* scratch, int tid = 0,
                   const int mindex = 0, const int maxdex = -1,
                   SpinBarrier* barrier = nullptr) const;

#if defined __AVX2__
  inline int ThreadMax(int t_start, int t_end) const;

  template <int kMantissaBits>
  inline float ApplyExpAndSum(int max_value, float* scratch_ptr);

  inline void FindSamplePoint(const float* scratch_ptr, float* random_target,
                              int* start, int* end);
#endif  // __AVX2__ code

  template <class Q = DataType>
  typename std::enable_if<IsFixed32Type<Q>::value, int>::type ThreadMax(
      int tid) const;

  template <class Q = DataType>
  typename std::enable_if<!IsFixed32Type<Q>::value, int>::type ReducingSample(
      std::minstd_rand* gen, CacheAlignedVector<float>* scratch, int tid = 0,
      float temperature = 1.0f, SpinBarrier* barrier = nullptr);

  template <class Q = DataType>
  typename std::enable_if<IsFixed32Type<Q>::value, int>::type ReducingSample(
      std::minstd_rand* gen, CacheAlignedVector<float>* scratch, int tid = 0,
      float temperature = 1.0f, SpinBarrier* barrier = nullptr);

  template <class Q = DataType>
  typename std::enable_if<std::is_same<Q, float>::value, void>::type Exp();

  template <class Q = DataType>
  typename std::enable_if<std::is_same<Q, float>::value, void>::type Sigmoid();

  template <class Q>
  typename std::enable_if<
      IsFixed32Type<DataType>::value && IsFixed32Type<Q>::value, void>::type
  Sigmoid(const int32_t* sigmoid_table, CacheAlignedVector<Q>* result);

  template <class Q = DataType>
  typename std::enable_if<std::is_same<Q, float>::value, void>::type Tanh();

  template <class Q>
  typename std::enable_if<
      IsFixed32Type<DataType>::value && IsFixed32Type<Q>::value, void>::type
  Tanh(const int32_t* tanh_table, CacheAlignedVector<Q>* result);

  template <class Q = DataType>
  typename std::enable_if<IsFixed32Type<Q>::value, const int32_t*>::type
  cast_data() const;
  template <class Q = DataType>
  typename std::enable_if<IsFixed16Type<Q>::value, const int16_t*>::type
  cast_data() const;
  template <class Q = DataType>
  typename std::enable_if<!(IsFixed32Type<Q>::value || IsFixed16Type<Q>::value),
                          const Q*>::type
  cast_data() const;
  const DataType* begin() const;
  const DataType* end() const;
  const DataType* data() const;
  DataType* data();

  const DataType& operator[](int pos) const;
  DataType& operator[](int pos);

  std::size_t size() const;
  bool empty() const;
  std::size_t bytes() const;

  int rows() const;
  int cols() const;

  int col_stride() const;

  void Print() const;

  float maximum() const;

 private:
  void resize(std::size_t size);

  std::size_t size_;
  DataType* data_;
  std::vector<int> maxes_;
  std::vector<int> thread_starts_;
#if defined __AVX__ || defined __AVX2__
  static constexpr int kCacheLineSize = 64;
  static constexpr int kSIMDWidth = 8;
#else
  static constexpr int kCacheLineSize = 128;
  static constexpr int kSIMDWidth = 4;
#endif  // __AVX__
  std::unique_ptr<std::minstd_rand> gen_;
};

template <typename T>
class FatCacheAlignedVector {
 public:
  FatCacheAlignedVector();
  FatCacheAlignedVector(int rows, int cols);
  FatCacheAlignedVector(const CacheAlignedVector<T>& vector, int rows);
  template <typename U>
  explicit FatCacheAlignedVector(const FatCacheAlignedVector<U>& vector);
  FatCacheAlignedVector(CacheAlignedVector<T>&& vector, int rows);

  VectorView<T> slice(const int col) const;
  MutableVectorView<T> slice(const int col);

  const T* data() const;
  T* data();
  template <class Q = T>
  typename std::enable_if<IsFixed16Type<Q>::value, const int16_t*>::type
  cast_data() const;
  template <class Q = T>
  typename std::enable_if<IsFixed32Type<Q>::value, const int32_t*>::type
  cast_data() const;
  template <class Q = T>
  typename std::enable_if<!(IsFixed16Type<Q>::value || IsFixed32Type<Q>::value),
                          const Q*>::type
  cast_data() const;

  int rows() const;
  int cols() const;
  int size() const;
  bool empty() const;
  std::size_t bytes() const;

  void reshape(int rows, int cols);

  float maximum() const;

  int col_stride() const;

  void FillOnes();
  void FillZero();
  void FillRandom(float min = -10.f, float max = 10.f);

  const T& operator[](int pos) const;
  T& operator[](int pos);

 private:
  CacheAlignedVector<T> vector_;
  int rows_;
  int cols_;
};

template <typename T>
class MutableVectorView {
 public:
  using value_type = T;

  explicit MutableVectorView(T* data = nullptr, int rows = 0, int cols = 0,
                             int col_stride = 0);

  explicit MutableVectorView(CacheAlignedVector<T>* vector);

  explicit MutableVectorView(CacheAlignedVector<T>* vector, int pos = 0,
                             int rows = 0);

  explicit MutableVectorView(FatCacheAlignedVector<T>* vector);

  MutableVectorView(FatCacheAlignedVector<T>* vector, int pos, int rows);

  T* data();
  const T* data() const;

  template <class Q = T>
  typename std::enable_if<IsFixed32Type<Q>::value, const int32_t*>::type
  cast_data() const;
  template <class Q = T>
  typename std::enable_if<IsFixed16Type<Q>::value, const int16_t*>::type
  cast_data() const;
  template <class Q = T>
  typename std::enable_if<!(IsFixed32Type<Q>::value || IsFixed16Type<Q>::value),
                          const Q*>::type
  cast_data() const;

  int cols() const;

  int rows() const;

  bool empty() const;

  int col_stride() const;

  std::size_t bytes() const;

  void reshape(int rows, int cols);

  const T& operator[](int pos) const;
  T& operator[](int pos);

 protected:
  T* data_;
  int rows_;
  int cols_;
  int col_stride_;
};

template <typename T>
class VectorView : public MutableVectorView<const T> {
 public:
  using value_type = T;

  explicit VectorView(const MutableVectorView<T>& other);

  explicit VectorView(const T* data = nullptr, int rows = 0, int cols = 0,
                      int col_stride = 0);

  explicit VectorView(const CacheAlignedVector<T>& vector);

  explicit VectorView(const CacheAlignedVector<T>& vector, int pos = 0,
                      int rows = 0);

  explicit VectorView(const FatCacheAlignedVector<T>& vector);

  VectorView(const FatCacheAlignedVector<T>& vector, int pos, int rows);

  VectorView<T>& operator=(const MutableVectorView<T>& other);
};

template <typename T>
class MaskedSparseMatrix;

class ThreadBounds {
 public:
  ThreadBounds();

  void PrepareForThreads(int block_width, int block_height, int num_threads,
                         int reduced_rows_per_cache_row, int reduced_rows,
                         const int* nnz_per_row);

  template <typename WeightType>
  const WeightType* OffsetWeights(const WeightType* weights, int tid) const;
  template <typename RhsIndType>
  const RhsIndType* OffsetRhsIndices(const RhsIndType* rhs_indices,
                                     int tid) const;
  template <typename BiasType>
  const BiasType* OffsetBias(const BiasType* bias, int tid) const;
  template <typename OutType>
  OutType* OffsetOutput(OutType* output, int tid) const;
  int StartRow(int tid) const;
  const std::vector<int>& row_starts() const;

 private:
  void ComputeThreadSplitPoints(int num_threads, int reduced_rows_per_cache_row,
                                int reduced_rows, const int* nnz_per_row);

  int block_width_;
  int block_height_;
  std::vector<int> row_starts_;
  std::vector<int> weight_starts_;
  std::vector<int> rhs_indices_starts_;
  std::vector<int> bias_starts_;
};

class MatmulBase {
 public:
  MatmulBase() {
#if defined(__x86_64__) || defined(__i386__) || defined(_WIN32)
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx) != 0) {
      using_avx_ = (ecx & bit_AVX) != 0;
      if (using_avx_) {
        __get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx);
        using_avx2_ = (ebx & bit_AVX2) != 0;
        using_avx512_ = (ebx & bit_AVX512F) != 0 && (ebx & bit_AVX512DQ) &&
                        (ebx & bit_AVX512BW) != 0;
        VLOG(2) << "avx2 flag=" << using_avx2_ << " 512=" << using_avx512_;
      } else {
        LOG(ERROR) << "AVX not found at all!";
      }
    }
#else
    using_aarch64_ = true;
#endif
  }

 protected:
  bool using_avx512_ = false;
  bool using_avx2_ = false;
  bool using_avx_ = false;
  bool using_aarch64_ = false;
};

constexpr int kGenericSIMDWidth = 4;

template <typename GruStateType, typename InputType, typename SampleType = void>
class GruGates : public MatmulBase {
 public:
  using SampleWeightType = float;
  static constexpr int kSIMDWidth = kGenericSIMDWidth;

  template <ARInputsMode kInputsMode = ARInputsMode::k2ARInputs,
            bool kSplitGates = false>
  void GruWithARInput(int start, int end, int state_size,
                      const InputType* gru_recurrent_ptr,
                      const InputType* input_ptr, GruStateType* gru_state_ptr,
                      const SampleType* ar_sample0 = nullptr,
                      const SampleType* ar_sample1 = nullptr,
                      const SampleWeightType* ar_01_weights = nullptr,
                      int num_replicas = 1, int replica_stride = 0,
                      const SampleType* ar_sample2 = nullptr,
                      const SampleWeightType* ar_2_weights = nullptr,
                      const InputType* gru_recurrent_other_ptr = nullptr);

  void PlainGru(int start, int end, int state_size,
                const InputType* gru_recurrent_ptr, const InputType* input_ptr,
                GruStateType* gru_state_ptr);
};

#if defined __ARM_NEON || defined __aarch64__
static constexpr int kNeonSIMDWidth = 4;

template <>
class GruGates<float, float, float> : public MatmulBase {
 public:
  static constexpr int kSIMDWidth = kNeonSIMDWidth;

  template <ARInputsMode kInputsMode = ARInputsMode::k2ARInputs,
            bool kSplitGates = false>
  void GruWithARInput(int start, int end, int state_size,
                      const float* gru_recurrent_data, const float* input_data,
                      float* gru_state_data, const float* ar_sample0 = nullptr,
                      const float* ar_sample1 = nullptr,
                      const float* ar_01_weights = nullptr,
                      int num_replicas = 1, int replica_stride = 0,
                      const float* ar_sample2 = nullptr,
                      const float* ar_2_weights = nullptr,
                      const float* gru_recurrent_other_data = nullptr);
};
#endif  // defined __ARM_NEON || defined __aarch64__

template <int kGruStateBits, int kInputBits, int kSampleBits>
class GruGates<fixed16<kGruStateBits>, fixed32<kInputBits>,
               fixed16<kSampleBits>> : public MatmulBase {
 public:
#if defined __ARM_NEON || defined __aarch64__
  static constexpr int kSIMDWidth = kNeonSIMDWidth;
#elif defined __AVX2__
  static constexpr int kSIMDWidth = kAVX2SIMDWidth * 2;
#else  // Generic case.
  static constexpr int kSIMDWidth = kGenericSIMDWidth;
#endif  // __ARM_NEON || defined __aarch64__ / __AVX2__

  using GruStateType = fixed16<kGruStateBits>;
  using InputType = fixed32<kInputBits>;
  using SampleType = fixed16<kSampleBits>;
  using SampleWeightType = float;
  static constexpr int kInputMantissaBits = InputType::kMantissaBits;
  static constexpr int kSampleMantissaBits = SampleType::kMantissaBits;
  static constexpr int kStateMantissaBits = GruStateType::kMantissaBits;
  template <ARInputsMode kInputsMode = ARInputsMode::k2ARInputs,
            bool kSplitGates = false>
  void GruWithARInput(int start, int end, int state_size,
                      const InputType* gru_recurrent_data,
                      const InputType* input_data, GruStateType* gru_state_data,
                      const SampleType* ar_sample0 = nullptr,
                      const SampleType* ar_sample1 = nullptr,
                      const SampleWeightType* ar_01_weights = nullptr,
                      int num_replicas = 1, int replica_stride = 0,
                      const SampleType* ar_sample2 = nullptr,
                      const SampleWeightType* ar_2_weights = nullptr,
                      const InputType* gru_recurrent_other_data = nullptr);
};

template <typename WeightType, typename RhsType>
class Matmul : public MatmulBase {
 public:
  template <typename OutType>
  void MatVec4x4(const WeightType* weights, const RhsType* rhs,
                 const typename TypeOfProduct<WeightType, RhsType>::type* bias,
                 const int32_t* nnz_per_row, const int16_t* rhs_indices,
                 int start_row, int end_row, bool relu, int replicas,
                 int stride, OutType* output);
  template <typename OutType>
  void MatVec8x4(const WeightType* weights, const RhsType* rhs,
                 const typename TypeOfProduct<WeightType, RhsType>::type* bias,
                 const int32_t* nnz_per_row, const int16_t* rhs_indices,
                 int start_row, int end_row, bool relu, int replicas,
                 int stride, OutType* output);
};

template <>
class Matmul<float, float> : public MatmulBase {
 public:
  void MatVec4x4(const float* weights, const float* rhs, const float* bias,
                 const int32_t* nnz_per_row, const int16_t* rhs_indices,
                 int start_row, int end_row, bool relu, int replicas,
                 int stride, float* output);
  void MatVec8x4(const float* weights, const float* rhs, const float* bias,
                 const int32_t* nnz_per_row, const int16_t* rhs_indices,
                 int start_row, int end_row, bool relu, int replicas,
                 int stride, float* output);
};

template <int WeightBits, int RhsBits>
class Matmul<fixed16<WeightBits>, fixed16<RhsBits>> : public MatmulBase {
 public:
  using WeightType = fixed16<WeightBits>;
  using RhsType = fixed16<RhsBits>;

  template <typename OutType>
  void MatVec4x4(const int16_t* weights, const int16_t* rhs,
                 const int32_t* bias, const int32_t* nnz_per_row,
                 const int16_t* rhs_indices, int start_row, int end_row,
                 bool relu, int replicas, int stride, OutType* output);

  template <typename OutType>
  void MatVec8x4(const int16_t* weights, const int16_t* rhs,
                 const int32_t* bias, const int32_t* nnz_per_row,
                 const int16_t* rhs_indices, int start_row, int end_row,
                 bool relu, int replicas, int stride, OutType* output);
};

template <typename WeightType, typename RhsType, typename DeltaType = int16_t>
class CsrBlockSparseMatrix {
 public:
  CsrBlockSparseMatrix();

  CsrBlockSparseMatrix(const uint8_t* const& buffer, const std::size_t& len);

  template <typename InputType>
  CsrBlockSparseMatrix(const MaskedSparseMatrix<InputType>& masked_matrix);

  CsrBlockSparseMatrix(
      const CsrBlockSparseMatrix<WeightType, RhsType, DeltaType>& src_matrix,
      const std::vector<WeightType>& new_weights,
      const std::vector<DeltaType>& new_deltas, const std::vector<int>& new_nnz,
      int cols);

  CsrBlockSparseMatrix SplitByColumn(int start_col, int end_col,
                                     bool keep_rhs_size = false) const;

  CsrBlockSparseMatrix SplitByRow(int start_row, int end_row) const;

  void DoubleBlockHeight();

  std::size_t WriteToFlatBuffer(std::string* csr_flatbuffer);

  void ReadFromFlatBuffer(const uint8_t* const& bytes, const std::size_t& len);

  template <typename RhsClass, typename BiasClass, typename OutClass,
            typename BiasType = typename BiasClass::value_type,
            typename OutType = typename OutClass::value_type>
  void SpMM_bias(const RhsClass& rhs, const BiasClass& bias, OutClass* out,
                 bool relu = false, int tid = 0,
                 SpinBarrier* barrier = nullptr) const;
  template <typename MVRhsType, typename MVBiasType, typename OutType>
  void MatVec(const MVRhsType* rhs, const MVBiasType* bias, bool relu, int tid,
              int replicas, int output_stride, OutType* output);

  int rows() const;
  int cols() const;
  int block_height() const;
  int block_width() const;
  float sparsity() const;
  int num_threads() const;
  const ThreadBounds& thread_bounds() const;
  const CacheAlignedVector<DeltaType>& rhs_indices() const;
  const std::string& name() const;
  void set_name(const std::string& name);
  const std::vector<int>& split_points() const;

  std::size_t bytes() const;

  template <typename RhsClass, typename BiasClass, typename OutClass,
            typename BiasType = typename BiasClass::value_type,
            typename OutType = typename OutClass::value_type>
  typename std::enable_if<!IsFixed32Type<OutType>::value, int>::type
  SpMM_bias_Sample(const RhsClass& rhs, const BiasClass& bias, OutClass* out,
                   float temperature, int tid, SpinBarrier* barrier,
                   std::minstd_rand* gen,
                   CacheAlignedVector<float>* scratch) const;
  template <typename RhsClass, typename BiasClass, typename OutClass,
            typename BiasType = typename BiasClass::value_type,
            typename OutType = typename OutClass::value_type>
  typename std::enable_if<IsFixed32Type<OutType>::value, int>::type
  SpMM_bias_Sample(const RhsClass& rhs, const BiasClass& bias, OutClass* out,
                   float temperature, int tid, SpinBarrier* barrier,
                   std::minstd_rand* gen,
                   CacheAlignedVector<float>* scratch) const;

  void Print() const;

  template <typename OutType = int32_t>
  int PrepareForThreads(int num_threads, int cache_line_size = -1);

  void ComputeRHSIndices();

  void ComputeColDeltas();

  std::vector<int> CumulativeColDeltas() const;

 private:
  constexpr std::size_t FixedParameterSize() const;
  template <typename InputType>
  void DetermineBlockSize(const MaskedSparseMatrix<InputType>& masked_matrix);

  template <typename InputType>
  void MakeColumnsMultiple(const std::vector<int>& row_offsets,
                           std::vector<int>* reduced_mask,
                           std::vector<InputType>* weights);

  template <typename InputType>
  void MaskAndWeightsToCsr(const std::vector<int>& mask,
                           const std::vector<InputType>& weights,
                           std::vector<int>* nnz_per_row,
                           std::vector<int>* col_indices,
                           std::vector<WeightType>* weights_csr);

  template <typename OutType>
  int ReducedRowsPerCacheLine(int override_cache_line_size = -1) const;

  int col_multiple_;
  int rows_;
  int cols_;
  int reduced_rows_;
  int reduced_cols_;
  float sparsity_;
  int block_width_;
  int block_height_;
  int num_threads_;
  std::string name_;

  CacheAlignedVector<WeightType> weights_;
  CacheAlignedVector<DeltaType> col_deltas_;
  CacheAlignedVector<int> nnz_per_row_;
  CacheAlignedVector<DeltaType> rhs_indices_;
  Matmul<WeightType, RhsType> matmul_;
  ThreadBounds thread_bounds_;
  static constexpr int kCacheLineSize = 64;
};

template <typename WeightType, typename RhsType,
          typename BiasType = typename TypeOfProduct<WeightType, RhsType>::type,
          typename DeltaType = int16_t>
class SparseLinearLayer {
 public:
  SparseLinearLayer();

  SparseLinearLayer(CsrBlockSparseMatrix<WeightType, RhsType>&& sparse_matrix,
                    CacheAlignedVector<BiasType>&& bias);
  SparseLinearLayer(
      const SparseLinearLayer<WeightType, RhsType, BiasType, DeltaType>& src);
  SparseLinearLayer& operator=(
      const SparseLinearLayer<WeightType, RhsType, BiasType, DeltaType>& src);

  template <typename RhsClassType, typename OutType>
  void SpMM_bias(const RhsClassType& rhs, OutType* out, bool relu = false,
                 int tid = 0, SpinBarrier* barrier = nullptr) const;
  template <typename RhsClassType, typename OutType>
  int SpMM_bias_Sample(const RhsClassType& rhs, OutType* out, float temperature,
                       int tid, SpinBarrier* barrier, std::minstd_rand* gen,
                       CacheAlignedVector<float>* scratch) const;
  template <typename RhsClassType, typename OutType>
  void MatVec(const RhsClassType& rhs, bool relu, int tid, int replicas,
              int output_stride, OutType* output,
              SpinBarrier* barrier = nullptr);

  int rows() const;
  int cols() const;
  float sparsity() const;
  int block_width() const;
  int block_height() const;
  int num_threads() const;
  const CacheAlignedVector<BiasType>& bias() const;
  const std::vector<int>& split_points() const;
  bool IsSplit() const;

  std::size_t bytes() const;
  void Print() const;

  void DoubleBlockHeight();

  int PrepareForThreads(int num_threads, int cache_line_size = -1);

  void SliceForThreads(const std::vector<int>& split_points);

  void SplitInputs(
      SparseLinearLayer<WeightType, RhsType, BiasType, DeltaType>* part1,
      SparseLinearLayer<WeightType, RhsType, BiasType, DeltaType>* part2);

  void SplitOutputs(
      SparseLinearLayer<WeightType, RhsType, BiasType, DeltaType>* part1,
      SparseLinearLayer<WeightType, RhsType, BiasType, DeltaType>* part2);

 private:
  struct PartLinearLayer {
    PartLinearLayer(const CsrBlockSparseMatrix<WeightType, RhsType>& matrix,
                    const CacheAlignedVector<BiasType>& bias,
                    const CacheAlignedVector<BiasType>& bias_4, int tid,
                    int start_col, int end_col);
    CsrBlockSparseMatrix<WeightType, RhsType> self_matrix;
    CacheAlignedVector<BiasType> full_bias;
    CacheAlignedVector<BiasType> quarter_bias;
    CsrBlockSparseMatrix<WeightType, RhsType> other_matrix;
  };
  CsrBlockSparseMatrix<WeightType, RhsType, DeltaType> sparse_matrix_;
  CacheAlignedVector<BiasType> bias_;
  CacheAlignedVector<BiasType> full_bias_;
  CacheAlignedVector<BiasType> mid_output_;
  std::vector<PartLinearLayer> thread_layers_;
  std::unique_ptr<ProducerConsumer> split_pc_;
  int num_threads_ = 0;
};

template <typename WeightType, typename RhsType>
SparseLinearLayer<WeightType, RhsType> CreateConstantLayer(
    int rows, int cols, float sparsity, float constant = 1.f);

template <typename WeightType, typename RhsType,
          typename DiskWeightType = float>
absl::Status LoadLogitLayer(
    const std::string& prefix, bool zipped, const std::string& path,
    SparseLinearLayer<WeightType, RhsType>* sparse_linear_layer);

template <typename WeightType, typename RhsType,
          typename DiskWeightType = float>
absl::Status LoadSparseLayer(
    const std::string& prefix, bool zipped,
    SparseLinearLayer<WeightType, RhsType>* sparse_linear_layer,
    const std::string& path);

template <typename T, typename DiskType = T, typename ElemType = T>
typename std::enable_if<!csrblocksparse::IsFixed16Type<DiskType>::value,
                        absl::Status>::type
ReadArrayFromFile(const std::string& file_name, std::vector<T>* array,
                  const std::string& path = "/data/local/tmp/");
template <typename T, typename DiskType, typename ElemType>
typename std::enable_if<std::is_same<T, float>::value &&
                            csrblocksparse::IsFixed16Type<DiskType>::value,
                        absl::Status>::type
ReadArrayFromFile(const std::string& file_name, std::vector<T>* array,
                  const std::string& path = "/data/local/tmp/");

}  // namespace csrblocksparse

// [internal] End of sparse_inference_matrixvector declarations.

namespace chromemedia {
namespace codec {

typedef std::function<void(csrblocksparse::SpinBarrier*, int)> Function;
void LaunchOnThreadsWithBarrier(int num_threads, Function&& func);

}  // namespace codec
}  // namespace chromemedia

#endif  // LYRA_CODEC_SPARSE_INFERENCE_MATRIXVECTOR_H_
