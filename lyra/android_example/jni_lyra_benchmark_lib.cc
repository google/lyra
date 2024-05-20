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

#include <jni.h>

#include <string>
#include <vector>

#include "absl/random/random.h"
#include "lyra/cli_example/decoder_main_lib.h"
#include "lyra/cli_example/encoder_main_lib.h"
#include "lyra/lyra_benchmark_lib.h"
#include "lyra/lyra_config.h"

extern "C" JNIEXPORT jshortArray
    JNICALL
    /**
     * @brief Encode the audio samples with Lyra encoder
     *
     * @param env - JNI environment (do not change)
     * @param this_obj - the Java object to interact with (do not change)
     * @param samples - the input audio samples (waves)
     * @param sample_length - the length of the audio samples (waves)
     * @param sample_rate_Hz - the sampling rate of the audio samples (waves)
     * @param bitrate - the bit rate to be encoded to
     * @param model_base_path - the path of the trained model
     */
    Java_com_example_android_lyra_MainActivity_encodeSamples(
        JNIEnv* env, jobject this_obj, jshortArray samples, jint sample_length,
        jint sample_rate_Hz, jint bitrate, jstring model_base_path) {
  std::vector<int16_t> samples_vector(sample_length);
  std::vector<uint8_t> features;
  std::vector<int16_t> encoded_samples;
  jshortArray java_encoded_samples = nullptr;
  env->GetShortArrayRegion(samples, jsize{0}, sample_length,
                           &samples_vector[0]);

  const char* cpp_model_base_path = env->GetStringUTFChars(model_base_path, 0);

  absl::BitGen gen;
  if (chromemedia::codec::EncodeWav(
          samples_vector, chromemedia::codec::kNumChannels, sample_rate_Hz,
          bitrate, false, false, cpp_model_base_path, &features)) {
    // write the encoded audio samples to a Java byte array to return
    java_encoded_samples = env->NewShortArray(features.size());
    env->SetShortArrayRegion(java_encoded_samples, 0, features.size(),
                             &encoded_samples[0]);
    return java_encoded_samples;

  } else
    return nullptr;
  /* // Original codes
    if (chromemedia::codec::EncodeWav(
            samples_vector, chromemedia::codec::kNumChannels, sample_rate_Hz,
    bitrate, false, false, cpp_model_base_path, &features) &&
        chromemedia::codec::DecodeFeatures(
            features, chromemedia::codec::BitrateToPacketSize(bitrate),
            false, gen, decoder.get(),
            nullptr, &decoded_audio)) {
      java_decoded_audio = env->NewShortArray(decoded_audio.size());
      env->SetShortArrayRegion(java_decoded_audio, 0, decoded_audio.size(),
                               &decoded_audio[0]);
    }
    env->ReleaseStringUTFChars(model_base_path, cpp_model_base_path);

    return java_decoded_audio;
    */
}

extern "C" JNIEXPORT jshortArray
    JNICALL
    /**
     * @brief Decode the encoded bytes back to audio samples with Lyra decoder
     *
     * @param env - JNI environment (do not change)
     * @param this_obj - the Java object to interact with (do not change)
     * @param samples - the input audio samples (waves)
     * @param sample_length - the length of the audio samples (waves)
     * @param sample_rate_Hz - the sampling rate of the audio samples (waves)
     * @param bitrate - the bit rate to be encoded to
     * @param model_base_path - the path of the trained model
     */
    Java_com_example_android_lyra_MainActivity_decodeSamples(
        JNIEnv* env, jobject this_obj, jshortArray features, jshortArray output,
        jint feature_length, jint sample_rate_Hz, jint bitrate,
        jstring model_base_path) {
  std::vector<int16_t> feature_vector(feature_length);
  std::vector<uint8_t> feature_vector_bytes;
  //std::vector<uint8_t> features;
  std::vector<int16_t> decoded_audio;
  jshortArray java_decoded_audio = nullptr;
  // convert the Java data type to its corresponding C++ data type
  env->GetShortArrayRegion(features, jsize{0}, feature_length, &(feature_vector[0]));

// convert the int16_t vector to the uint8_t - stay for now but can be optimised later
// Note: the reason for adding the following loop is that "GetShortArrayRegion" function 
//       only takes int16_t as the type of its last argument but the "DecodeFeature" 
//       function accepts only uint8_t vector as the input.
feature_vector_bytes.reserve(feature_length);
for (int16_t num : feature_vector) {
    feature_vector_bytes.push_back(static_cast<uint8_t>(num));
}


  const char* cpp_model_base_path = env->GetStringUTFChars(model_base_path, 0);
  std::unique_ptr<chromemedia::codec::LyraDecoder> decoder =
      chromemedia::codec::LyraDecoder::Create(
          16000, chromemedia::codec::kNumChannels, cpp_model_base_path);

  absl::BitGen gen;
  if (chromemedia::codec::DecodeFeatures(
          feature_vector_bytes, chromemedia::codec::BitrateToPacketSize(bitrate),
          /*randomize_num_samples_requested=*/false, gen, decoder.get(),
          nullptr, &decoded_audio)) {
    java_decoded_audio = env->NewShortArray(decoded_audio.size());
    env->SetShortArrayRegion(java_decoded_audio, 0, decoded_audio.size(),
                             &decoded_audio[0]);
  }

  env->ReleaseStringUTFChars(model_base_path, cpp_model_base_path);

  return java_decoded_audio;
}

extern "C" JNIEXPORT int JNICALL
Java_com_example_android_lyra_MainActivity_lyraBenchmark(
    JNIEnv* env, jobject this_obj, jint num_cond_vectors,
    jstring model_base_path) {
  const char* cpp_model_base_path = env->GetStringUTFChars(model_base_path, 0);
  int ret =
      chromemedia::codec::lyra_benchmark(num_cond_vectors, cpp_model_base_path,
                                         /*benchmark_feature_extraction=*/true,
                                         /*benchmark_quantizer=*/true,
                                         /*benchmark_generative_model=*/true);
  env->ReleaseStringUTFChars(model_base_path, cpp_model_base_path);
  return ret;
}
