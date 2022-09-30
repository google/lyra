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

#include "noise_estimator.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "comfort_noise_generator.h"
#include "dsp_utils.h"
#include "gtest/gtest.h"
#include "log_mel_spectrogram_extractor_impl.h"
#include "lyra_config.h"

namespace chromemedia {
namespace codec {

class NoiseEstimatorPeer {
 public:
  explicit NoiseEstimatorPeer(int num_samples_per_hop, int num_hops_per_update,
                              int num_features, float max_smoothing,
                              float bound_decay_factor,
                              std::unique_ptr<LogMelSpectrogramExtractorImpl>
                                  log_mel_spectrogram_extractor)
      : noise_estimator_(num_samples_per_hop, num_hops_per_update, num_features,
                         max_smoothing, bound_decay_factor,
                         std::move(log_mel_spectrogram_extractor)) {}

  void UpdateNoiseEstimate(const std::vector<float>& current_power_db) {
    noise_estimator_.UpdateNoiseEstimate(current_power_db);
  }

  bool ComputeIsNoise(const std::vector<float>& current_power_db) {
    return noise_estimator_.ComputeIsNoise(current_power_db);
  }

 private:
  NoiseEstimator noise_estimator_;
};

namespace {

static constexpr int kTestNumHops = 250;
static constexpr int kTestNumFeatures = 160;
static constexpr int kTestNumSamplesPerWindow = 640;
static constexpr int kTestNumSamplesPerHop = 320;
static constexpr float kMaxPower = 1.f;

class NoiseEstimatorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    noise_estimator_ =
        NoiseEstimator::Create(kInternalSampleRateHz, kTestNumSamplesPerHop,
                               kTestNumSamplesPerWindow, kTestNumFeatures);
    ASSERT_NE(noise_estimator_, nullptr);
  }

  // Adds a small amount of variability to the base noise.
  std::vector<float> RandomNoise(const std::vector<float>& base_noise) {
    const float kMaxAbsEnergy = 1e-1f;
    std::uniform_real_distribution<float> noise_distribution(-kMaxAbsEnergy,
                                                             kMaxAbsEnergy);
    std::vector<float> noise(base_noise);
    for (auto& bin : noise) {
      bin += noise_distribution(generator_);
    }
    return noise;
  }

  std::vector<float> CreateNoiseWithSparsePower(
      const std::vector<float>& noise) {
    // Each frequency bin has a 1 in 10 probability of containing a uniform
    // sparse_energy_noise.
    const int kReciprocalPowerProbability = 10;
    std::vector<float> sparse_energy_noise(noise);
    std::uniform_int_distribution<int> bin_distribution(
        0, kReciprocalPowerProbability);
    for (auto& bin : sparse_energy_noise) {
      if (bin_distribution(generator_) == 0) {
        bin = kMaxPower;
      }
    }
    return sparse_energy_noise;
  }

  // Create a noise vector in which power increases with frequency.
  std::vector<float> BaseNoise() {
    std::vector<float> noise(kTestNumFeatures);
    // Approximate the base noise with a line.
    float rise =
        LogMelSpectrogramExtractorImpl::GetSilenceValue() / kTestNumFeatures;
    for (int i = 0; i < noise.size(); i++) {
      noise.at(i) =
          rise * i + LogMelSpectrogramExtractorImpl::GetSilenceValue();
    }
    return noise;
  }

  void GenerateSamples(std::vector<float> noise_features,
                       ComfortNoiseGenerator& noise_generator,
                       std::vector<int16_t>* output) {
    ASSERT_TRUE(noise_generator.AddFeatures(noise_features));
    auto noise = noise_generator.GenerateSamples(kTestNumSamplesPerHop);
    ASSERT_TRUE(noise.has_value());
    // ASSERT_* macros have a hidden return, this function must return void.
    output->assign(noise->begin(), noise->end());
  }

  std::unique_ptr<NoiseEstimator> noise_estimator_;
  std::default_random_engine generator_;
};

TEST_F(NoiseEstimatorTest, FiveSecondsSparseEnergy) {
  auto noise_generator = ComfortNoiseGenerator::Create(
      kInternalSampleRateHz, kTestNumSamplesPerHop, kTestNumSamplesPerWindow,
      kTestNumFeatures);
  ASSERT_NE(noise_generator, nullptr);
  const std::vector<float> base_noise = BaseNoise();
  std::vector<int16_t> samples;

  for (int i = 0; i < kTestNumHops; ++i) {
    const std::vector<float> sparse_energy_noise =
        CreateNoiseWithSparsePower(base_noise);
    GenerateSamples(sparse_energy_noise, *noise_generator, &samples);
    ASSERT_TRUE(noise_estimator_->ReceiveSamples(samples));
  }

  auto spectral_distance =
      LogSpectralDistance(base_noise, noise_estimator_->noise_estimate());
  ASSERT_TRUE(spectral_distance.has_value());
  EXPECT_LT(spectral_distance.value(), 0.7f);
}

TEST_F(NoiseEstimatorTest, FiveSecondsSilence) {
  auto noise_generator = ComfortNoiseGenerator::Create(
      kInternalSampleRateHz, kTestNumSamplesPerHop, kTestNumSamplesPerWindow,
      kTestNumFeatures);
  ASSERT_NE(noise_generator, nullptr);
  std::vector<float> silence(kTestNumFeatures,
                             LogMelSpectrogramExtractorImpl::GetSilenceValue());
  std::vector<int16_t> samples;

  for (int i = 0; i < kTestNumHops; ++i) {
    GenerateSamples(silence, *noise_generator, &samples);
    ASSERT_TRUE(noise_estimator_->ReceiveSamples(samples));
    // The initial noise estimate of silence should not be updated.
    auto spectral_distance =
        LogSpectralDistance(silence, noise_estimator_->noise_estimate());
    ASSERT_TRUE(spectral_distance.has_value());
    EXPECT_LT(spectral_distance.value(), 0.2f)
        << " Noise estimate dissimilar at frame index " << i;
  }
}

TEST_F(NoiseEstimatorTest, NoiseIdentification) {
  auto feature_extractor = LogMelSpectrogramExtractorImpl::Create(
      kInternalSampleRateHz, kTestNumSamplesPerHop, kTestNumSamplesPerWindow,
      kTestNumFeatures);
  const int kMaxSmoothingHalflifeHops = 20;
  const int kBoundHalfLifeHops = 50;
  NoiseEstimatorPeer noise_estimator_peer = NoiseEstimatorPeer(
      kTestNumSamplesPerHop, /*num_hops_per_update=*/10, kTestNumFeatures,
      std::pow(0.5f, 1.f / kMaxSmoothingHalflifeHops),
      std::pow(0.5f, 1.f / kBoundHalfLifeHops), std::move(feature_extractor));
  std::vector<float> periodic_signal_features(
      kTestNumFeatures, LogMelSpectrogramExtractorImpl::GetSilenceValue());
  //
  for (int i = 0; i < kTestNumFeatures; i += 20) {
    periodic_signal_features.at(i) = kMaxPower;
  }
  const std::vector<float> base_noise = BaseNoise();

  // Warm up on some noise.
  for (int i = 0; i < kTestNumHops; ++i) {
    noise_estimator_peer.UpdateNoiseEstimate(RandomNoise(base_noise));
  }
  EXPECT_TRUE(noise_estimator_peer.ComputeIsNoise(base_noise));
  EXPECT_FALSE(noise_estimator_peer.ComputeIsNoise(periodic_signal_features));
}

}  // namespace
}  // namespace codec
}  // namespace chromemedia
