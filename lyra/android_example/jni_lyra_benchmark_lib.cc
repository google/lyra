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

extern "C" JNIEXPORT jshortArray JNICALL
Java_com_example_android_lyra_MainActivity_encodeAndDecodeSamples(
    JNIEnv* env, jobject this_obj, jshortArray samples, jint sample_length,
    jint bitrate, jstring model_base_path) {
  std::vector<int16_t> samples_vector(sample_length);
  std::vector<uint8_t> features;
  std::vector<int16_t> decoded_audio;
  jshortArray java_decoded_audio = nullptr;
  env->GetShortArrayRegion(samples, jsize{0}, sample_length,
                           &samples_vector[0]);

  const char* cpp_model_base_path = env->GetStringUTFChars(model_base_path, 0);
  std::unique_ptr<chromemedia::codec::LyraDecoder> decoder =
      chromemedia::codec::LyraDecoder::Create(
          16000, chromemedia::codec::kNumChannels, cpp_model_base_path);

  absl::BitGen gen;
  if (chromemedia::codec::EncodeWav(
          samples_vector, chromemedia::codec::kNumChannels, 16000, bitrate,
          false, false, cpp_model_base_path, &features) &&
      chromemedia::codec::DecodeFeatures(
          features, chromemedia::codec::BitrateToPacketSize(bitrate),
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
