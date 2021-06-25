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

#ifndef LYRA_CODEC_SPARSE_MATMUL_VECTOR_CACHE_ALIGNED_VECTOR_H_
#define LYRA_CODEC_SPARSE_MATMUL_VECTOR_CACHE_ALIGNED_VECTOR_H_

#if defined __aarch64__
#include <arm_neon.h>
#endif
#if defined __AVX__ || defined __AVX2__
#include <immintrin.h>
#endif

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <random>
#include <type_traits>

#include "absl/strings/str_format.h"
#include "sparse_matmul/numerics/fast_transcendentals.h"
#include "sparse_matmul/numerics/fixed_types.h"
#include "sparse_matmul/numerics/type_utils.h"
#include "sparse_matmul/os/coop_threads.h"
#include "sparse_matmul/vector/aligned_malloc.h"

namespace csrblocksparse {

template <typename T>
class MutableVectorView;
template <typename T>
class VectorView;

// CacheAlignedVector is a simple vector-like class that makes sure its
// underlying buffer is aligned to a |kCacheLineSize| boundary.  It is meant
// for numeric computation and cannot be used to store objects that are
// not POD as it will neither call their constructors nor destructors.
//
// It is meant to be used with the CSRBlockSparseMatrix class for
// implenting basic neural network layers composed of SpMV.
//
// This class is thread compatible.
template <typename DataType>
class CacheAlignedVector {
  static_assert(std::is_pod<DataType>::value,
                "CacheAlignedVector can only be"
                " used with POD");

 public:
  using value_type = DataType;

  explicit CacheAlignedVector(std::size_t size) : size_(size), data_(nullptr) {
    gen_ = absl::make_unique<std::minstd_rand>(0);
    data_ = reinterpret_cast<DataType*>(
        aligned_malloc(size_ * sizeof(DataType), kCacheLineSize));
  }

  explicit CacheAlignedVector(const std::vector<DataType>& input)
      : size_(input.size()), data_(nullptr) {
    gen_ = absl::make_unique<std::minstd_rand>(0);
    data_ = reinterpret_cast<DataType*>(
        aligned_malloc(size_ * sizeof(DataType), kCacheLineSize));
    memcpy(data_, input.data(), size_ * sizeof(DataType));
  }

  template <typename InputType>
  explicit CacheAlignedVector(const std::vector<InputType>& input)
      : size_(input.size()), data_(nullptr) {
    gen_ = absl::make_unique<std::minstd_rand>(0);
    data_ = reinterpret_cast<DataType*>(
        aligned_malloc(size_ * sizeof(DataType), kCacheLineSize));
    for (int i = 0; i < size_; ++i)
      data_[i] = static_cast<DataType>(input.data()[i]);
  }

  CacheAlignedVector(const DataType* input, int size)
      : size_(size), data_(nullptr) {
    gen_ = absl::make_unique<std::minstd_rand>(0);
    data_ = reinterpret_cast<DataType*>(
        aligned_malloc(size_ * sizeof(DataType), kCacheLineSize));
    memcpy(data_, input, size_ * sizeof(DataType));
  }

  template <typename InputType>
  explicit CacheAlignedVector(const InputType* input, int size)
      : size_(size), data_(nullptr) {
    gen_ = absl::make_unique<std::minstd_rand>(0);
    data_ = reinterpret_cast<DataType*>(
        aligned_malloc(size_ * sizeof(DataType), kCacheLineSize));
    for (int i = 0; i < size_; ++i) data_[i] = static_cast<DataType>(input[i]);
  }

  CacheAlignedVector() : size_(0), data_(nullptr) {}

  ~CacheAlignedVector() {
    aligned_free(data_);
    data_ = nullptr;
    size_ = 0;
  }

  // Copies are _deep_ copies
  CacheAlignedVector(CacheAlignedVector const& other)
      : size_(0), data_(nullptr), gen_(nullptr) {
    if (other.gen_)
      gen_ = absl::make_unique<std::minstd_rand>(std::minstd_rand(*other.gen_));
    this->resize(other.size());
    memcpy(data_, other.data(), size_ * sizeof(DataType));
  }
  // Copies a slice of the input.
  CacheAlignedVector(CacheAlignedVector const& other, int start, int end)
      : size_(0), data_(nullptr), gen_(nullptr) {
    if (other.gen_)
      gen_ = absl::make_unique<std::minstd_rand>(std::minstd_rand(*other.gen_));
    this->resize(end - start);
    memcpy(data_, other.data() + start, size_ * sizeof(DataType));
  }

  void operator=(CacheAlignedVector const& other) {
    if (other.gen_)
      gen_ = absl::make_unique<std::minstd_rand>(std::minstd_rand(*other.gen_));
    else
      gen_.reset(nullptr);
    this->resize(other.size());
    memcpy(data_, other.data(), size_ * sizeof(DataType));
  }

  CacheAlignedVector(CacheAlignedVector<DataType>&& other)
      : size_(0), data_(nullptr), gen_(std::move(other.gen_)) {
    size_ = other.size_;
    data_ = other.data_;
    other.size_ = 0;
    other.data_ = nullptr;
  }

  CacheAlignedVector<DataType>& operator=(
      CacheAlignedVector<DataType>&& other) {
    aligned_free(data_);
    if (other.gen_)
      gen_ = absl::make_unique<std::minstd_rand>(std::move(*other.gen_));
    else
      gen_.reset(nullptr);
    size_ = other.size_;
    data_ = other.data_;
    other.size_ = 0;
    other.data_ = nullptr;
    return *this;
  }

  VectorView<DataType> AsView() const {
    return VectorView<DataType>(this->data(), this->size(), 1);
  }

  MutableVectorView<DataType> AsMutableView() {
    return MutableVectorView<DataType>(this->data(), this->size(), 1);
  }

  // Copies the |split_points| to use in ReducingSample.
  void PrepareForThreads(const std::vector<int>& split_points,
                         int block_height) {
    maxes_.resize(split_points.size() - 1);
    thread_starts_ = split_points;
    for (int t = 0; t < thread_starts_.size(); ++t) {
      thread_starts_[t] *= block_height;
    }
  }

  void FillRandom(float min = -10.f, float max = 10.f) {
    // 10 is smaller than any nonzero bound of the range of any data type.
    std::uniform_real_distribution<float> dist(min, max);
    for (std::size_t i = 0; i < size_; i++) {
      data_[i] = DataType(dist(*gen_));
    }
  }

  void FillZero() {
    for (std::size_t i = 0; i < size_; i++) {
      data_[i] = DataType(0.f);
    }
  }

  void FillOnes() {
    for (std::size_t i = 0; i < size_; i++) {
      data_[i] = DataType(1.f);
    }
  }

  void FillWith(const DataType& value) {
    for (std::size_t i = 0; i < size_; i++) {
      data_[i] = value;
    }
  }

  // Interprets |data_| as logits and samples from the distribution, this
  // version operates IN PLACE and uses an internal random source.
  template <class Q = DataType>
  typename std::enable_if<std::is_same<Q, float>::value, int>::type Sample(
      float temperature = 1.f) {
    return Sample(temperature, gen_.get(), this);
  }

  // Interprets |data_| as logits and samples.  This version requires the random
  // source and temporary memory to be passed in.  It is thread safe assuming
  // no other threads are using the generator and temporary memory.
#if defined __aarch64__
  template <class Q = DataType>
  typename std::enable_if<std::is_same<Q, float>::value, int>::type Sample(
      float temperature, std::minstd_rand* gen,
      CacheAlignedVector<float>* scratch) const {
    DCHECK(scratch->size() >= size_);
    // Round down to nearest multiple of 8.
    int SIMD_iterations = 8 * (size_ / 8);
    float* scratch_ptr = scratch->data();
    std::uniform_real_distribution<float> dist;
    float random_number = dist(*gen);

    float32x4_t sum = vdupq_n_f32(0.f);
    float32x4_t sum1 = vdupq_n_f32(0.f);
    float32x4_t max_value = vdupq_n_f32(std::numeric_limits<float>::lowest());
    float32x4_t max_value1 = vdupq_n_f32(std::numeric_limits<float>::lowest());
    float32x4_t inv_temp = vdupq_n_f32(1.f / temperature);
    // Compute sum of exp(x) for the denominator.
    // Hand unroll by 2, gives speed improvement.
    constexpr int kUnrollFactor = 2;
    constexpr int kElementsPerIter = kUnrollFactor * kSIMDWidth;
    for (std::size_t i = 0; i < SIMD_iterations; i += kElementsPerIter) {
      max_value = vmaxq_f32(vld1q_f32(data_ + i), max_value);
      max_value1 = vmaxq_f32(vld1q_f32(data_ + i + 4), max_value1);
    }

    // Pairwise reduction.
    max_value = vpmaxq_f32(max_value, max_value1);
    // Duplicate (dupq) maximum across vector (maxnmvq).
    float scalar_max_value = vmaxvq_f32(max_value);

    for (int i = SIMD_iterations; i < size_; ++i) {
      scalar_max_value = std::max(data_[i], scalar_max_value);
    }

    max_value = vdupq_n_f32(scalar_max_value);

    for (std::size_t i = 0; i < SIMD_iterations; i += kElementsPerIter) {
      // Load and multiply by temperature.
      float32x4_t x =
          vmulq_f32(vsubq_f32(vld1q_f32(data_ + i), max_value), inv_temp);
      float32x4_t x1 =
          vmulq_f32(vsubq_f32(vld1q_f32(data_ + i + 4), max_value), inv_temp);

      float32x4_t exponent = fast_exp(x);
      float32x4_t exponent1 = fast_exp(x1);

      sum = vaddq_f32(sum, exponent);
      sum1 = vaddq_f32(sum1, exponent1);

      vst1q_f32(scratch_ptr + i, exponent);
      vst1q_f32(scratch_ptr + i + 4, exponent1);
    }

    // Horizontally reduce the two sums.
    sum = vpaddq_f32(sum, sum1);
    sum = vpaddq_f32(sum, sum);
    float denom = vgetq_lane_f32(sum, 0) + vgetq_lane_f32(sum, 1);

    for (int i = SIMD_iterations; i < size_; ++i) {
      float x = (data_[i] - scalar_max_value) / temperature;
      float x_exp = expf(x);
      denom += x_exp;
      scratch_ptr[i] = x_exp;
    }

    // Note: rather than normalize all the probabilities, we can just
    // apply the inverse normalization to the random number.
    random_number *= denom;

    // Now do the scan in serial, return as soon as possible.
    // TODO(b/188821456): This could be made into a parallel SIMD scan
    // followed by a binary search, for a small speedup.
    float cumsum = 0.f;
    for (std::size_t i = 0; i < size_; i++) {
      cumsum += scratch_ptr[i];
      if (cumsum >= random_number) return i;
    }
    return size_ - 1;
  }

  template <class Q = DataType>
  static inline int32x4_t vmul_temp_fixed(int32x4_t x, int32x2_t inv_temp) {
    int32x2_t xh = vget_high_s32(x);
    int32x2_t xl = vget_low_s32(x);
    int32x2_t ph = vqrshrn_n_s64(vmull_s32(xh, inv_temp), Q::kMantissaBits);
    int32x2_t pl = vqrshrn_n_s64(vmull_s32(xl, inv_temp), Q::kMantissaBits);
    return vcombine_s32(pl, ph);
  }

  template <class Q = DataType>
  static inline int float_to_fixed(float x) {
    return static_cast<int>(x * (1 << Q::kMantissaBits));
  }

  template <class Q = DataType>
  static inline float fixed_to_float(int x) {
    const float inv_denom = 1.f / (1 << Q::kMantissaBits);
    return static_cast<float>(x) * inv_denom;
  }

  template <class Q = DataType>
  typename std::enable_if<IsFixed32Type<Q>::value, int>::type Sample(
      float temperature, std::minstd_rand* gen,
      CacheAlignedVector<int>* scratch) const {
    DCHECK(scratch->size() >= size_);
    // Round down to nearest multiple of 8.
    int SIMD_iterations = 8 * (size_ / 8);
    int* scratch_ptr = scratch->data();
    float scalar_inv_temp = 1.f / temperature;

    int32x4_t sum = vdupq_n_s32(0);
    int32x4_t sum1 = vdupq_n_s32(0);
    int32x4_t max_value = vdupq_n_s32(std::numeric_limits<int>::lowest());
    int32x4_t max_value1 = vdupq_n_s32(std::numeric_limits<int>::lowest());
    int32x2_t inv_temp = vdup_n_s32(float_to_fixed(scalar_inv_temp));
    // Compute sum of exp(x) for the denominator.
    // Hand unroll by 2, gives speed improvement.

    const int* data_ptr = reinterpret_cast<int*>(data_);
    constexpr int kUnrollFactor = 2;
    constexpr int kElementsPerIter = kUnrollFactor * kSIMDWidth;
    for (std::size_t i = 0; i < SIMD_iterations; i += kElementsPerIter) {
      max_value = vmaxq_s32(vld1q_s32(data_ptr + i), max_value);
      max_value1 = vmaxq_s32(vld1q_s32(data_ptr + i + kSIMDWidth), max_value1);
    }

    // Pairwise reduction.
    max_value = vpmaxq_s32(max_value, max_value1);
    int scalar_max_value = vmaxvq_s32(max_value);

    for (int i = SIMD_iterations; i < size_; ++i) {
      scalar_max_value = std::max(data_[i].raw_val(), scalar_max_value);
    }
    max_value = vdupq_n_s32(scalar_max_value);
    // We clip all loaded values to a lower bound of the lowest possible arg to
    // exp + the max value that we are going to subtract, to prevent underflow
    // in exp and also to avoid wrap-around with values that are already minint.
    int32x4_t clip_min =
        vdupq_n_s32(scalar_max_value - (80 << MantissaBitsOf<Q>::value));

    for (std::size_t i = 0; i < SIMD_iterations; i += kElementsPerIter) {
      // Load and multiply by temperature.
      int32x4_t loaded = vmaxq_s32(vld1q_s32(data_ptr + i), clip_min);
      int32x4_t x = vmul_temp_fixed(vsubq_s32(loaded, max_value), inv_temp);
      loaded = vmaxq_s32(vld1q_s32(data_ptr + i + kSIMDWidth), clip_min);
      int32x4_t x1 = vmul_temp_fixed(vsubq_s32(loaded, max_value), inv_temp);

      int32x4_t exponent = vcvtq_n_s32_f32(fast_exp_fixed<Q::kExponentBits>(x),
                                           Q::kMantissaBits);
      int32x4_t exponent1 = vcvtq_n_s32_f32(
          fast_exp_fixed<Q::kExponentBits>(x1), Q::kMantissaBits);

      sum = vaddq_s32(sum, exponent);
      sum1 = vaddq_s32(sum1, exponent1);

      vst1q_s32(scratch_ptr + i, exponent);
      vst1q_s32(scratch_ptr + i + kSIMDWidth, exponent1);
    }

    // Horizontally reduce the two sums.
    sum = vpaddq_s32(sum, sum1);
    sum = vpaddq_s32(sum, sum);
    float denom =
        fixed_to_float(vgetq_lane_s32(sum, 0) + vgetq_lane_s32(sum, 1));
    for (int i = SIMD_iterations; i < size_; ++i) {
      float x_exp = fast_exp_fixed<Q::kExponentBits>(
          DataType((data_[i].raw_val() - scalar_max_value) * scalar_inv_temp));

      denom += x_exp;
      scratch_ptr[i] = float_to_fixed(x_exp);
    }

    // Note: rather than normalize all the probabilities, we can just
    // apply the inverse normalization to the random number.
    std::uniform_real_distribution<float> dist;
    int random_number = float_to_fixed(dist(*gen) * denom);

    // Now do the scan in serial, return as soon as possible.
    // TODO(b/188821456): This could be made into a parallel SIMD scan
    // followed by a binary search, for a small speedup.
    int cumsum = 0;
    for (std::size_t i = 0; i < size_; i += kSIMDWidth) {
      int32x4_t next_vals = vld1q_s32(&scratch_ptr[i]);
      cumsum += vaddvq_s32(next_vals);
      if (cumsum >= random_number) {
        int high_sum = vaddv_s32(vget_high_s32(next_vals));
        if (cumsum - high_sum > random_number) {
          // One of the lower ones.
          return (cumsum - high_sum - scratch_ptr[i + 1] > random_number)
                     ? i
                     : i + 1;
        } else {
          // One of the upper ones.
          return (cumsum - scratch_ptr[i + 3] > random_number) ? i + 2 : i + 3;
        }
      }
    }
    return size_ - 1;
  }
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
         SpinBarrier* barrier = nullptr) const {
    return ScalarSample(temperature, gen, scratch, tid, 0, -1, barrier);
  }

  int ScalarSample(float temperature, std::minstd_rand* gen,
                   CacheAlignedVector<float>* scratch, int tid = 0,
                   const int mindex = 0, const int maxdex = -1,
                   SpinBarrier* barrier = nullptr) const {
    // TODO(b/188821456) Don't ignore |tid| and |barrier|. Currently all threads
    // duplicate the same work and ignore |tid| and |barrier|, but they could
    // be used to execute a reducing max over the data before the exp operation.
    DCHECK_EQ(barrier, nullptr);
    DCHECK_EQ(tid, 0);
    DCHECK(scratch->size() >= size_);
    DCHECK(size_ % 8 == 0) << "CacheAlignedVector size must be a multiple of "
                              "8 to allow for maximum SIMD and loop unroll, "
                              "got "
                           << size_ % 8;
    DCHECK(size_ > mindex >= 0);
    DCHECK((maxdex == -1) || (0 <= mindex < maxdex < size_));
    int maxindex = maxdex > 0 ? maxdex : size_;

    float* scratch_ptr = scratch->data();
    std::uniform_real_distribution<float> dist;
    float random_number = dist(*gen);

    float sum = 0.f;
    float max_value = std::numeric_limits<float>::lowest();
    for (int i = mindex; i < maxindex; ++i) {
      max_value = std::max(max_value, static_cast<float>(data_[i]));
    }
    float inv_temperature = 1.f / temperature;
    for (int i = mindex; i < maxindex; ++i) {
      float exponent = fast_exp((static_cast<float>(data_[i]) - max_value) *
                                inv_temperature);
      scratch_ptr[i] = exponent;
      sum += exponent;
    }

    // Note: rather than normalize all the probabilities, we can just
    // apply the inverse normalization to the random number.
    random_number *= sum;

    float cumsum = 0.f;
    for (std::size_t i = mindex; i < maxindex; i++) {
      cumsum += scratch_ptr[i];
      if (cumsum >= random_number) return i;
    }
    return maxindex - 1;
  }

#if defined __AVX2__
  // Some AVX2-only code.
  // Returns the max of |data_| in the range [|t_start|, |t_end|).
  inline int ThreadMax(int t_start, int t_end) const {
    // Note: The AVX2 code requires that the number of threads and the output
    // size be a power of 2. For efficiency purposes, these should be checked
    // when preparing for threads in an architecture class.
    // The output size must be a power of 2 so the binary search for the sample
    // point works correctly.
    // The number of threads must be a power of 2 so that it nicely divides the
    // output size, which has to be a power of 2.
    __m256i maxes =
        _mm256_load_si256(reinterpret_cast<__m256i const*>(data_ + t_start));
    for (int i = t_start + kSIMDWidth; i < t_end; i += kSIMDWidth) {
      __m256i data =
          _mm256_load_si256(reinterpret_cast<__m256i const*>(data_ + i));
      maxes = _mm256_max_epi32(maxes, data);
    }
    // Max within the register.
    // Bring the top lane down to the bottom.
    __m256i other = _mm256_permute4x64_epi64(maxes, 0xe);
    maxes = _mm256_max_epi32(maxes, other);
    // Bring the 2nd 64 bits to the bottom.
    other = _mm256_shuffle_epi32(maxes, 0xe);
    maxes = _mm256_max_epi32(maxes, other);
    // Bring the 2nd 32 bits to the bottom.
    other = _mm256_shuffle_epi32(maxes, 1);
    maxes = _mm256_max_epi32(maxes, other);
    return _mm256_extract_epi32(maxes, 0);
  }

  // Applies exp (approximately) to the difference between |data_| and
  // |max_value|, storing the result in scratch, and returns the sum.
  template <int kMantissaBits>
  inline float ApplyExpAndSum(int max_value, float* scratch_ptr) {
    // Rough approximation for exp(x). See fast_exp_fixed.
    // Constant clipping limit on exp arg. Since its value is never positive,
    // we only need to clip on the negative side.
    constexpr int kClipLimit = -(80 << kMantissaBits);
    __m256i clip_val = _mm256_set1_epi32(kClipLimit);
    // Multiplication factor to convert x from log base e to log base 2, shifted
    // by an amount that lines up the binary point with the float32
    // representation, after the multiplication
    static const int kLogFactor = (1 << (23 - kMantissaBits)) / logf(2.f);
    __m256i log_factor = _mm256_set1_epi32(kLogFactor);
    // Fix the exponent bias and add the additive fudge factor for the mantissa
    // to finish the approximate conversion.
    constexpr int kAddConstant = (127 << 23) - 366000;
    __m256i constant = _mm256_set1_epi32(kAddConstant);
    // Broadcast the max_value.
    __m256i max_val = _mm256_set1_epi32(max_value);
    // Add the max to the |clip_val|, so it can be used before the subtraction.
    clip_val = _mm256_add_epi32(clip_val, max_val);
    // The sum of the exps.
    __m256 sum1 = _mm256_setzero_ps();
    for (int i = 0; i < size_; i += kSIMDWidth) {
      // |data_| - |max_value|.
      __m256i data =
          _mm256_load_si256(reinterpret_cast<__m256i const*>(data_ + i));
      // Clip to negative limit before the subtraction of |max_val| to avoid
      // wrap-around with min-int values.
      data = _mm256_max_epi32(data, clip_val);
      __m256i difference = _mm256_sub_epi32(data, max_val);
      // Exponent trick exp.
      // Multiply by |log_factor|, keeping only the lower 32 bits.
      difference = _mm256_mullo_epi32(difference, log_factor);
      // Add the constant.
      difference = _mm256_add_epi32(difference, constant);
      // Reinterpret the results as float32.
      __m256 float_exp = _mm256_castsi256_ps(difference);
      // Sum the results and save to scratch space.
      _mm256_store_ps(scratch_ptr + i, float_exp);
      sum1 = _mm256_add_ps(sum1, float_exp);
    }
    // Horizontally add the 8 values in sum.
    // Get the top lane down to the bottom.
    __m256 sum2 = _mm256_permute2f128_ps(sum1, sum1, 1);
    sum1 = _mm256_add_ps(sum1, sum2);
    sum1 = _mm256_hadd_ps(sum1, sum1);
    sum1 = _mm256_hadd_ps(sum1, sum1);
    return _mm256_cvtss_f32(sum1);
  }

  // Binary search for the index where the cumulative sum meets random_target.
  inline void FindSamplePoint(const float* scratch_ptr, float* random_target,
                              int* start, int* end) {
    int halfsize = (*end - *start) / 2;
    do {
      // Sum the first half.
      // We sum the section in two independent parts, so we can step down 2
      // levels if we get a hit in this half.
      int quartersize = halfsize / (2 * kSIMDWidth);
      quartersize *= kSIMDWidth;
      halfsize = quartersize * 2;
      // The sums of the quarters.
      __m256 sum1 = _mm256_setzero_ps();
      __m256 sum2 = _mm256_setzero_ps();
      const float* ptr1 = scratch_ptr + *start;
      const float* ptr2 = ptr1 + quartersize;
      for (int i = 0; i < quartersize; i += kSIMDWidth) {
        __m256 data1 = _mm256_load_ps(ptr1 + i);
        __m256 data2 = _mm256_load_ps(ptr2 + i);
        sum1 = _mm256_add_ps(sum1, data1);
        sum2 = _mm256_add_ps(sum2, data2);
      }
      // Horizontally add the two sums, keeping the results separate.
      // Numbering |sum1|=[0-7] and |sum2|=[8-15]...
      sum1 = _mm256_hadd_ps(sum1, sum2);
      // |sum1| now has [0+1, 2+3, 8+9, 10+11, 4+5, 6+7, 12+13, 14+15].
      // Bring the top lane down to the bottom.
      sum2 = _mm256_permute2f128_ps(sum1, sum1, 1);
      sum1 = _mm256_hadd_ps(sum1, sum2);
      // Now |sum1| has [0-3, 8-11, 4-7, 12-15], so swap the middle two
      // elements.
      sum1 = _mm256_shuffle_ps(sum1, sum1, 0xd8);
      sum1 = _mm256_hadd_ps(sum1, sum1);
      // Now |sum1| has [0-7, 8-15, ....].
      float bottom_quarter = _mm256_cvtss_f32(sum1);
      if (bottom_quarter >= *random_target) {
        *end = *start + quartersize;
      } else {
        float bottom_half = _mm256_cvtss_f32(_mm256_hadd_ps(sum1, sum1));
        if (bottom_half >= *random_target) {
          *start += quartersize;
          *end = *start + quartersize;
          *random_target -= bottom_quarter;
        } else {
          *start += halfsize;
          *random_target -= bottom_half;
        }
      }
      halfsize = (*end - *start) / 2;
    } while (halfsize >= kSIMDWidth * 2);
  }
#endif  // __AVX2__ code

  // Fixed32 version.
  template <class Q = DataType>
  typename std::enable_if<IsFixed32Type<Q>::value, int>::type ThreadMax(
      int tid) const {
    int t_start = thread_starts_[tid];
    int t_end = thread_starts_[tid + 1];
#if defined __AVX2__
    return ThreadMax(t_start, t_end);
#else
    // With operator<, could use std::max_element.
    int max_value = data_[t_start].raw_val();
    for (int i = t_start + 1; i < t_end; ++i) {
      max_value = std::max(max_value, data_[i].raw_val());
    }
    return max_value;
#endif
  }

  // As Sample above, except that if |tid| and |barrier| are provided, it will
  // save some time by running a local max in each thread before combining them
  // and doing the rest of the work duplicated across all threads.
  // Fixed32 version.
  template <class Q = DataType>
  typename std::enable_if<!IsFixed32Type<Q>::value, int>::type ReducingSample(
      std::minstd_rand* gen, CacheAlignedVector<float>* scratch, int tid = 0,
      float temperature = 1.0f, SpinBarrier* barrier = nullptr) {
    if (barrier != nullptr) barrier->barrier();
    // Sample only accepts tid of 0, as it would ignore it anyway.
    // All threads duplicate the same work in this path.
    return Sample(temperature, gen, scratch, /*tid=*/0);
  }

  template <class Q = DataType>
  typename std::enable_if<IsFixed32Type<Q>::value, int>::type ReducingSample(
      std::minstd_rand* gen, CacheAlignedVector<float>* scratch, int tid = 0,
      float temperature = 1.0f, SpinBarrier* barrier = nullptr) {
    int max_value;
    if (barrier == nullptr) {
      // There is only one thread.
      max_value = ThreadMax(tid);
    } else {
      // Reduce max using the threads to do some of the work.
      maxes_[tid] = ThreadMax(tid);
      barrier->barrier();
      // The rest of the work is duplicated by all threads.
      max_value = *std::max_element(maxes_.begin(), maxes_.end());
    }
    float* scratch_ptr = scratch->data();
    std::uniform_real_distribution<float> dist;
    float sum = 0.0f;
#if defined __AVX2__
    sum = ApplyExpAndSum<MantissaBitsOf<Q>::value>(max_value, scratch_ptr);
#else
    int clip_limit = max_value - (80 << MantissaBitsOf<Q>::value);
    for (int i = 0; i < size_; ++i) {
      int difference = std::max(data_[i].raw_val(), clip_limit) - max_value;
      float exponent = expf(static_cast<float>(DataType(difference)));
      scratch_ptr[i] = exponent;
      sum += exponent;
    }
#endif  // __AVX2__

    float random_target = dist(*gen) * sum;
    int start = 0;
    int end = size_;

#if defined __AVX2__
    FindSamplePoint(scratch_ptr, &random_target, &start, &end);
    // The scalar code finishes the job from here...
#endif  // __AVX2__
    float cumsum = 0.f;
    for (std::size_t i = start; i < end; i++) {
      cumsum += scratch_ptr[i];
      if (cumsum >= random_target) return i;
    }
    return end - 1;
  }

  template <class Q = DataType>
  typename std::enable_if<std::is_same<Q, float>::value, void>::type Exp() {
#if defined __aarch64__
    DCHECK(size_ % 16 == 0) << "CacheAlignedVector size must be a multiple of "
                               "16 to allow for maximum SIMD and loop unroll "
                               "got "
                            << size_ % 16;
    constexpr int kUnrollFactor = 4;
    constexpr int kElementsPerIter = kUnrollFactor * kSIMDWidth;
    for (std::size_t i = 0; i < size_; i += kElementsPerIter) {
      float32x4_t x = vld1q_f32(data_ + i);
      float32x4_t x1 = vld1q_f32(data_ + i + 4);
      float32x4_t x2 = vld1q_f32(data_ + i + 8);
      float32x4_t x3 = vld1q_f32(data_ + i + 12);

      vst1q_f32(data_ + i, fast_exp(x));
      vst1q_f32(data_ + i + 4, fast_exp(x1));
      vst1q_f32(data_ + i + 8, fast_exp(x2));
      vst1q_f32(data_ + i + 12, fast_exp(x3));
    }
#else
    for (int i = 0; i < size_; ++i) {
      data_[i] = expf(data_[i]);
    }
#endif  // defined __aarch64__
  }

  template <class Q = DataType>
  typename std::enable_if<std::is_same<Q, float>::value, void>::type Sigmoid() {
#if defined __aarch64__
    DCHECK(size_ % 8 == 0) << "CacheAlignedVector size must be a multiple of "
                              "8 to allow for maximum SIMD and loop unroll "
                              "got "
                           << size_ % 8;
    constexpr int kUnrollFactor = 2;
    constexpr int kElementsPerIter = kUnrollFactor * kSIMDWidth;
    for (std::size_t i = 0; i < size_; i += kElementsPerIter) {
      float32x4_t x = vld1q_f32(data_ + i);
      float32x4_t x1 = vld1q_f32(data_ + i + 4);

      vst1q_f32(data_ + i, fast_sigmoid(x));
      vst1q_f32(data_ + i + 4, fast_sigmoid(x1));
    }
#else
    for (int i = 0; i < size_; ++i) {
      data_[i] = 1.f / (1.f + expf(-data_[i]));
    }
#endif  // defined __aarch64__
  }

  template <class Q>
  typename std::enable_if<
      IsFixed32Type<DataType>::value && IsFixed32Type<Q>::value, void>::type
  // For benchmarking only.
  Sigmoid(const int32_t* sigmoid_table, CacheAlignedVector<Q>* result) {
#if defined __AVX2__
    for (int i = 0; i < size_; i += kSIMDWidth) {
      __m256i x_in = _mm256_loadu_si256(reinterpret_cast<__m256i*>(data_ + i));
      __m256i output = fixed32_sigmoid_fixed16<MantissaBitsOf<DataType>::value,
                                               MantissaBitsOf<Q>::value>(
          sigmoid_table, x_in);
      _mm256_store_si256(reinterpret_cast<__m256i*>(result->data() + i),
                         output);
    }
#else
    for (int i = 0; i < size_; ++i) {
      result->data()[i] = 1.f / (1.f + expf(-data_[i]));
    }
#endif  // defined __AVX2__
  }

  template <class Q = DataType>
  typename std::enable_if<std::is_same<Q, float>::value, void>::type Tanh() {
#if defined __aarch64__
    DCHECK(size_ % 8 == 0) << "CacheAlignedVector size must be a multiple of "
                              "8 to allow for maximum SIMD and loop unroll "
                              "got "
                           << size_ % 8;
    constexpr int kUnrollFactor = 2;
    constexpr int kElementsPerIter = kUnrollFactor * kSIMDWidth;
    for (std::size_t i = 0; i < size_; i += kElementsPerIter) {
      float32x4_t x = vld1q_f32(data_ + i);
      float32x4_t x1 = vld1q_f32(data_ + i + 4);

      vst1q_f32(data_ + i, fast_tanh(x));
      vst1q_f32(data_ + i + 4, fast_tanh(x1));
    }
#else
    for (int i = 0; i < size_; ++i) {
      data_[i] = tanhf(data_[i]);
    }
#endif  // defined __aarch64__
  }

  template <class Q>
  typename std::enable_if<
      IsFixed32Type<DataType>::value && IsFixed32Type<Q>::value, void>::type
  // For benchmarking only
  Tanh(const int32_t* tanh_table, CacheAlignedVector<Q>* result) {
#if defined __AVX2__
    for (int i = 0; i < size_; i += kSIMDWidth) {
      __m256i x_in = _mm256_loadu_si256(reinterpret_cast<__m256i*>(data_ + i));
      __m256i output =
          fixed32_tanh_fixed16<MantissaBitsOf<DataType>::value,
                               MantissaBitsOf<Q>::value>(tanh_table, x_in);
      _mm256_store_si256(reinterpret_cast<__m256i*>(result->data() + i),
                         output);
    }
#else
    for (int i = 0; i < size_; ++i) {
      result->data()[i] = tanhf(data_[i]);
    }
#endif  // defined __AVX2__
  }

  // Returns |data_| cast to the correct integer type if fixed point.
  template <class Q = DataType>
  typename std::enable_if<IsFixed32Type<Q>::value, const int32_t*>::type
  cast_data() const {
    return reinterpret_cast<const int32_t*>(data_);
  }
  template <class Q = DataType>
  typename std::enable_if<IsFixed16Type<Q>::value, const int16_t*>::type
  cast_data() const {
    return reinterpret_cast<const int16_t*>(data_);
  }
  template <class Q = DataType>
  typename std::enable_if<!(IsFixed32Type<Q>::value || IsFixed16Type<Q>::value),
                          const Q*>::type
  cast_data() const {
    return data_;
  }
  const DataType* begin() const { return data_; }
  const DataType* end() const { return data_ + size_; }
  const DataType* data() const { return data_; }
  DataType* data() { return data_; }

  const DataType& operator[](int pos) const { return data_[pos]; }
  DataType& operator[](int pos) { return data_[pos]; }

  std::size_t size() const { return size_; }
  bool empty() const { return size_ == 0; }
  std::size_t bytes() const { return size_ * sizeof(DataType); }

  int rows() const { return size_; }
  int cols() const { return 1; }

  // Stride to get to move over by one column (which is the number of rows).
  int col_stride() const { return size_; }

  void Print() const {
    for (int i = 0; i < size(); ++i)
      absl::PrintF("[%d]=%g\n", i, static_cast<float>(data_[i]));
  }

  float maximum() const {
    float max_val = std::numeric_limits<float>::lowest();
    for (int i = 0; i < size_; ++i) {
      max_val = std::max(max_val, std::abs(static_cast<float>(data_[i])));
    }

    return max_val;
  }

 private:
  void resize(std::size_t size) {
    aligned_free(data_);
    size_ = size;
    data_ = reinterpret_cast<DataType*>(
        aligned_malloc(size_ * sizeof(DataType), kCacheLineSize));
  }

  std::size_t size_;
  DataType* data_;
  // Data used by the threaded version for sampling only.
  std::vector<int> maxes_;          // Max value of logits.
  std::vector<int> thread_starts_;  // First index for this thread.
#if defined __AVX__ || defined __AVX2__
  static constexpr int kCacheLineSize = 64;
  static constexpr int kSIMDWidth = 8;
#else
  static constexpr int kCacheLineSize = 128;
  static constexpr int kSIMDWidth = 4;
#endif  // __AVX__
  std::unique_ptr<std::minstd_rand> gen_;
};

// Used for doing Sparse Matrix * Dense Matrix multiplication.  This class is
// not intended to be a general Matrix class, just for the RHS of a SpMM, hence
// the name fat vector rather than Matrix.  The data layout is COLUMN MAJOR.
template <typename T>
class FatCacheAlignedVector {
 public:
  using value_type = T;

  FatCacheAlignedVector() : rows_(0), cols_(0) {}

  // Creates a new vector that is (rows, cols), doesn't init memory.
  FatCacheAlignedVector(int rows, int cols)
      : vector_(rows * cols), rows_(rows), cols_(cols) {}

  // Copies and reshapes vector from (1, size) to (|rows|, size / |rows|).
  FatCacheAlignedVector(const CacheAlignedVector<T>& vector, int rows)
      : vector_(vector), rows_(rows) {
    CHECK_EQ(vector_.size() % rows_, 0);
    cols_ = vector_.size() / rows_;
  }

  template <typename U>
  explicit FatCacheAlignedVector(const FatCacheAlignedVector<U>& vector)
      : vector_(vector.size()), rows_(vector.rows()), cols_(vector.cols()) {
    for (int i = 0; i < vector.size(); ++i) {
      vector_[i] = static_cast<T>(vector[i]);
    }
  }

  // Moves and reshapes vector from (1, size) to (|rows|, size / |rows|)
  FatCacheAlignedVector(CacheAlignedVector<T>&& vector, int rows)
      : vector_(vector), rows_(rows) {
    CHECK_EQ(vector_.size() % rows_, 0);
    cols_ = vector_.size() / rows_;
  }

  VectorView<T> slice(const int col) const {
    return VectorView<T>(this->data() + rows() * col, rows(), 1);
  }
  MutableVectorView<T> slice(const int col) {
    return MutableVectorView<T>(this->data() + rows() * col, rows(), 1);
  }

  const T* data() const { return vector_.data(); }
  T* data() { return vector_.data(); }
  // Returns |data_| cast to the correct integer type if fixed point.
  template <class Q = T>
  typename std::enable_if<IsFixed32Type<Q>::value, const int32_t*>::type
  cast_data() const {
    return vector_.cast_data();
  }
  template <class Q = T>
  typename std::enable_if<IsFixed16Type<Q>::value, const int16_t*>::type
  cast_data() const {
    return vector_.cast_data();
  }
  template <class Q = T>
  typename std::enable_if<!(IsFixed32Type<Q>::value || IsFixed16Type<Q>::value),
                          const Q*>::type
  cast_data() const {
    return vector_.cast_data();
  }

  int rows() const { return rows_; }
  int cols() const { return cols_; }
  int size() const { return rows_ * cols_; }
  bool empty() const { return rows_ == 0 || cols_ == 0; }
  std::size_t bytes() const { return vector_.bytes(); }

  void reshape(int rows, int cols) {
    CHECK_EQ(rows * cols, rows_ * cols_);
    rows_ = rows;
    cols_ = cols;
  }

  float maximum() const { return vector_.maximum(); }

  // Stride to get to move over by one column (which is the number of rows).
  int col_stride() const { return rows_; }

  void FillOnes() { vector_.FillOnes(); }
  void FillZero() { vector_.FillZero(); }
  void FillRandom(float min = -10.f, float max = 10.f) {
    vector_.FillRandom(min, max);
  }

  const T& operator[](int pos) const { return vector_[pos]; }
  T& operator[](int pos) { return vector_[pos]; }

 private:
  CacheAlignedVector<T> vector_;
  int rows_;
  int cols_;
};

// View into a 2D Matrix.  Currently only supports partitions by row.  This is
// expected to be used with underlying data that is COLUMN MAJOR.
template <typename T>
class MutableVectorView {
 public:
  using value_type = T;

  // Construct from a raw pointer, |rows|, |cols| and |col_stride|.
  // |col_stride| will default to |rows| if not specified.
  explicit MutableVectorView(T* data = nullptr, int rows = 0, int cols = 0,
                             int col_stride = 0)
      : data_(data),
        rows_(rows),
        cols_(cols),
        col_stride_(col_stride > 0 ? col_stride : rows) {}

  // Construct from a CacheAlignedVector, must have one column, can optionally
  // specify an offset and row count.
  explicit MutableVectorView(CacheAlignedVector<T>* vector)
      : MutableVectorView(vector->data(), vector->rows(), 1) {}

  explicit MutableVectorView(CacheAlignedVector<T>* vector, int pos = 0,
                             int rows = 0)
      : MutableVectorView(vector->data() + pos,
                          rows == 0 ? vector->rows() - pos : rows, 1,
                          vector->rows()) {}

  // Construct from a FatCacheAlignedVector, can optionally specify an offset,
  // and row count.  Views that have fewer columns than the original are not
  // supported.
  explicit MutableVectorView(FatCacheAlignedVector<T>* vector)
      : MutableVectorView(vector->data(), vector->rows(), vector->cols()) {}

  MutableVectorView(FatCacheAlignedVector<T>* vector, int pos, int rows)
      : MutableVectorView(vector->data() + pos, rows, vector->cols(),
                          vector->rows()) {}

  T* data() { return data_; }
  const T* data() const { return data_; }

  // Returns |data_| cast to the correct integer type if fixed point.
  template <class Q = T>
  typename std::enable_if<IsFixed32Type<Q>::value, const int32_t*>::type
  cast_data() const {
    return reinterpret_cast<const int32_t*>(data_);
  }
  template <class Q = T>
  typename std::enable_if<IsFixed16Type<Q>::value, const int16_t*>::type
  cast_data() const {
    return reinterpret_cast<const int16_t*>(data_);
  }
  template <class Q = T>
  typename std::enable_if<!(IsFixed32Type<Q>::value || IsFixed16Type<Q>::value),
                          const Q*>::type
  cast_data() const {
    return data_;
  }

  // Number of columns in the underlying (Fat)CacheAlignedVector.
  int cols() const { return cols_; }

  // Number of rows in this view.
  int rows() const { return rows_; }

  // Returns true if there's nothing in the MutableVectorView.
  bool empty() const { return rows_ == 0 || cols_ == 0; }

  // Stride to get to the next column (usually the number of rows in the
  // underlying data structure).
  int col_stride() const { return col_stride_; }

  // Returns the total number of bytes that are "owned" by this view. Uses
  // cols and not col_stride.
  std::size_t bytes() const { return rows_ * cols_ * sizeof(T); }

  void reshape(int rows, int cols) {
    CHECK_EQ(rows * cols, rows_ * cols_);
    rows_ = rows;
    cols_ = cols;
    col_stride_ = rows_;
  }

  const T& operator[](int pos) const { return data_[pos]; }
  T& operator[](int pos) { return data_[pos]; }

 protected:
  T* data_;
  int rows_;
  int cols_;
  int col_stride_;
};

// Specialization of MutableVectorView which is read-only.
template <typename T>
class VectorView : public MutableVectorView<const T> {
 public:
  using value_type = T;

  explicit VectorView(const MutableVectorView<T>& other)
      : MutableVectorView<const T>(other.data(), other.rows(), other.cols(),
                                   other.col_stride()) {}

  // Construct from a raw pointer, |rows|, |cols| and |col_stride|.
  // |col_stride| will default to |rows| if not specified.
  explicit VectorView(const T* data = nullptr, int rows = 0, int cols = 0,
                      int col_stride = 0)
      : MutableVectorView<const T>(data, rows, cols, col_stride) {}

  // Construct from a CacheAlignedVector, must have one column, can optionally
  // specify an offset and row count
  explicit VectorView(const CacheAlignedVector<T>& vector)
      : MutableVectorView<const T>(vector.data(), vector.rows(), 1) {}

  explicit VectorView(const CacheAlignedVector<T>& vector, int pos = 0,
                      int rows = 0)
      : MutableVectorView<const T>(vector.data() + pos,
                                   rows == 0 ? vector.rows() - pos : rows, 1,
                                   vector.rows()) {}

  // Construct from a FatCacheAlignedVector, can optionally specify an offset,
  // and row count.  Views that have fewer columns than the original are not
  // supported.
  explicit VectorView(const FatCacheAlignedVector<T>& vector)
      : MutableVectorView<const T>(vector.data(), vector.rows(),
                                   vector.cols()) {}

  VectorView(const FatCacheAlignedVector<T>& vector, int pos, int rows)
      : MutableVectorView<const T>(vector.data() + pos, rows, vector.cols(),
                                   vector.rows()) {}

  VectorView<T>& operator=(const MutableVectorView<T>& other) {
    this->data_ = other.data();
    this->rows_ = other.rows();
    this->cols_ = other.cols();
    this->col_stride_ = other.col_stride();
    return *this;
  }
};

}  // namespace csrblocksparse
#endif  // LYRA_CODEC_SPARSE_MATMUL_VECTOR_CACHE_ALIGNED_VECTOR_H_
