# [internal] load cc_fuzz_target.bzl
# [internal] load cc_proto_library.bzl
# [internal] load android_cc_test:def.bzl

package(default_visibility = ["//:__subpackages__"])

# To run all cc_tests in this directory:
# bazel test //lyra:all

# [internal] Command to run dsp_utils_android_test.

# [internal] Command to run lyra_integration_android_test.

exports_files(
    srcs = [
        "lyra_benchmark_lib.cc",
        "lyra_benchmark_lib.h",
        "lyra_components.h",
        "lyra_config.h",
        "lyra_decoder.cc",
        "lyra_decoder.h",
        "lyra_encoder.cc",
        "lyra_encoder.h",
        "model_coeffs/lyragan.tflite",
        "model_coeffs/quantizer.tflite",
        "model_coeffs/soundstream_encoder.tflite",
    ],
)

config_setting(
    name = "android_config",
    values = {"crosstool_top": "//external:android/crosstool"},
)

cc_library(
    name = "architecture_utils",
    hdrs = ["architecture_utils.h"],
    deps = ["@gulrak_filesystem//:filesystem"],
)

cc_library(
    name = "lyra_benchmark_lib",
    srcs = ["lyra_benchmark_lib.cc"],
    hdrs = ["lyra_benchmark_lib.h"],
    deps = [
        ":architecture_utils",
        ":dsp_utils",
        ":feature_extractor_interface",
        ":generative_model_interface",
        ":lyra_components",
        ":lyra_config",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@com_google_audio_dsp//audio/dsp:signal_vector_util",
        "@com_google_glog//:glog",
        "@gulrak_filesystem//:filesystem",
    ],
)

cc_library(
    name = "feature_estimator_interface",
    hdrs = [
        "feature_estimator_interface.h",
    ],
    deps = [
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "zero_feature_estimator",
    hdrs = [
        "zero_feature_estimator.h",
    ],
    deps = [
        ":feature_estimator_interface",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "generative_model_interface",
    hdrs = [
        "generative_model_interface.h",
    ],
    deps = [
        "@com_google_glog//:glog",
    ],
)

cc_library(
    name = "resampler_interface",
    hdrs = [
        "resampler_interface.h",
    ],
    deps = [
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "feature_extractor_interface",
    hdrs = [
        "feature_extractor_interface.h",
    ],
    deps = [
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "lyra_decoder_interface",
    hdrs = [
        "lyra_decoder_interface.h",
    ],
    deps = [
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "lyra_encoder_interface",
    hdrs = [
        "lyra_encoder_interface.h",
    ],
    deps = [
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "vector_quantizer_interface",
    hdrs = [
        "vector_quantizer_interface.h",
    ],
    deps = [
    ],
)

cc_library(
    name = "lyra_gan_model",
    srcs = [
        "lyra_gan_model.cc",
    ],
    hdrs = [
        "lyra_gan_model.h",
    ],
    data = ["model_coeffs/lyragan.tflite"],
    deps = [
        ":dsp_utils",
        ":generative_model_interface",
        ":tflite_model_wrapper",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:span",
        "@com_google_glog//:glog",
        "@gulrak_filesystem//:filesystem",
    ],
)

cc_library(
    name = "lyra_decoder",
    srcs = [
        "lyra_decoder.cc",
    ],
    hdrs = [
        "lyra_decoder.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":buffered_filter_interface",
        ":buffered_resampler",
        ":comfort_noise_generator",
        ":feature_estimator_interface",
        ":generative_model_interface",
        ":lyra_components",
        ":lyra_config",
        ":lyra_decoder_interface",
        ":noise_estimator",
        ":noise_estimator_interface",
        ":vector_quantizer_interface",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@com_google_glog//:glog",
        "@gulrak_filesystem//:filesystem",
    ],
)

cc_library(
    name = "comfort_noise_generator",
    srcs = [
        "comfort_noise_generator.cc",
    ],
    hdrs = [
        "comfort_noise_generator.h",
    ],
    deps = [
        ":dsp_utils",
        ":generative_model_interface",
        ":log_mel_spectrogram_extractor_impl",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/types:span",
        "@com_google_audio_dsp//audio/dsp:number_util",
        "@com_google_audio_dsp//audio/dsp/mfcc",
        "@com_google_audio_dsp//audio/dsp/spectrogram:inverse_spectrogram",
        "@com_google_glog//:glog",
    ],
)

cc_library(
    name = "lyra_encoder",
    srcs = [
        "lyra_encoder.cc",
    ],
    hdrs = [
        "lyra_encoder.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":feature_extractor_interface",
        ":lyra_components",
        ":lyra_config",
        ":lyra_encoder_interface",
        ":noise_estimator",
        ":noise_estimator_interface",
        ":packet",
        ":packet_interface",
        ":resampler",
        ":resampler_interface",
        ":vector_quantizer_interface",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@com_google_glog//:glog",
        "@gulrak_filesystem//:filesystem",
    ],
)

cc_library(
    name = "noise_estimator",
    srcs = [
        "noise_estimator.cc",
    ],
    hdrs = [
        "noise_estimator.h",
    ],
    deps = [
        ":log_mel_spectrogram_extractor_impl",
        ":noise_estimator_interface",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:span",
        "@com_google_audio_dsp//audio/dsp:signal_vector_util",
        "@com_google_glog//:glog",
    ],
)

cc_library(
    name = "noise_estimator_interface",
    hdrs = [
        "noise_estimator_interface.h",
    ],
    deps = [
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "packet_loss_model_interface",
    hdrs = ["packet_loss_model_interface.h"],
    deps = [],
)

cc_library(
    name = "fixed_packet_loss_model",
    srcs = [
        "fixed_packet_loss_model.cc",
    ],
    hdrs = [
        "fixed_packet_loss_model.h",
    ],
    deps = [":packet_loss_model_interface"],
)

cc_library(
    name = "gilbert_model",
    srcs = [
        "gilbert_model.cc",
    ],
    hdrs = [
        "gilbert_model.h",
    ],
    deps = [
        ":packet_loss_model_interface",
        "@com_google_absl//absl/memory",
        "@com_google_glog//:glog",
    ],
)

cc_library(
    name = "lyra_config",
    srcs = ["lyra_config.cc"],
    hdrs = ["lyra_config.h"],
    deps = [
        ":lyra_config_cc_proto",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_glog//:glog",
        "@gulrak_filesystem//:filesystem",
    ],
)

proto_library(
    name = "lyra_config_proto",
    srcs = ["lyra_config.proto"],
)

cc_proto_library(
    name = "lyra_config_cc_proto",
    deps = [":lyra_config_proto"],
)

cc_library(
    name = "lyra_components",
    srcs = [
        "lyra_components.cc",
    ],
    hdrs = [
        "lyra_components.h",
    ],
    deps = [
        ":feature_estimator_interface",
        ":feature_extractor_interface",
        ":generative_model_interface",
        ":lyra_gan_model",
        ":packet",
        ":packet_interface",
        ":residual_vector_quantizer",
        ":soundstream_encoder",
        ":vector_quantizer_interface",
        ":zero_feature_estimator",
        "@gulrak_filesystem//:filesystem",
    ],
)

cc_library(
    name = "log_mel_spectrogram_extractor_impl",
    srcs = [
        "log_mel_spectrogram_extractor_impl.cc",
    ],
    hdrs = [
        "log_mel_spectrogram_extractor_impl.h",
    ],
    deps = [
        ":feature_extractor_interface",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:span",
        "@com_google_audio_dsp//audio/dsp:number_util",
        "@com_google_audio_dsp//audio/dsp/mfcc",
        "@com_google_audio_dsp//audio/dsp/spectrogram",
        "@com_google_glog//:glog",
    ],
)

cc_library(
    name = "soundstream_encoder",
    srcs = [
        "soundstream_encoder.cc",
    ],
    hdrs = [
        "soundstream_encoder.h",
    ],
    data = ["model_coeffs/soundstream_encoder.tflite"],
    deps = [
        ":dsp_utils",
        ":feature_extractor_interface",
        ":tflite_model_wrapper",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:span",
        "@com_google_glog//:glog",
        "@gulrak_filesystem//:filesystem",
    ],
)

cc_library(
    name = "residual_vector_quantizer",
    srcs = [
        "residual_vector_quantizer.cc",
    ],
    hdrs = [
        "residual_vector_quantizer.h",
    ],
    data = [
        "model_coeffs/quantizer.tflite",
    ],
    deps = [
        ":tflite_model_wrapper",
        ":vector_quantizer_interface",
        "@com_google_absl//absl/memory",
        "@com_google_glog//:glog",
        "@gulrak_filesystem//:filesystem",
    ],
)

cc_library(
    name = "packet_interface",
    hdrs = [
        "packet_interface.h",
    ],
    deps = [
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "packet",
    hdrs = ["packet.h"],
    deps = [
        ":packet_interface",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:span",
        "@com_google_glog//:glog",
    ],
)

cc_library(
    name = "buffered_filter_interface",
    hdrs = ["buffered_filter_interface.h"],
)

cc_library(
    name = "buffered_resampler",
    srcs = ["buffered_resampler.cc"],
    hdrs = ["buffered_resampler.h"],
    deps = [
        ":buffered_filter_interface",
        ":resampler",
        ":resampler_interface",
        "@com_google_absl//absl/memory",
        "@com_google_glog//:glog",
    ],
)

cc_test(
    name = "buffered_resampler_test",
    srcs = ["buffered_resampler_test.cc"],
    deps = [
        ":buffered_resampler",
        ":lyra_config",
        ":resampler_interface",
        "//lyra/testing:mock_resampler",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "preprocessor_interface",
    hdrs = [
        "preprocessor_interface.h",
    ],
    deps = [
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "no_op_preprocessor",
    hdrs = [
        "no_op_preprocessor.h",
    ],
    deps = [
        ":preprocessor_interface",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "no_op_preprocessor_test",
    size = "small",
    srcs = ["no_op_preprocessor_test.cc"],
    deps = [
        ":no_op_preprocessor",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_binary(
    name = "lyra_benchmark",
    srcs = [
        "lyra_benchmark.cc",
    ],
    linkopts = select({
        ":android_config": ["-landroid"],
        "//conditions:default": [],
    }),
    deps = [
        ":lyra_benchmark_lib",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/flags:usage",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "lyra_decoder_test",
    size = "large",
    srcs = ["lyra_decoder_test.cc"],
    data = [":tflite_testdata"],
    shard_count = 8,
    deps = [
        ":buffered_filter_interface",
        ":buffered_resampler",
        ":dsp_utils",
        ":feature_estimator_interface",
        ":generative_model_interface",
        ":lyra_components",
        ":lyra_config",
        ":lyra_decoder",
        ":packet_interface",
        ":resampler",
        ":vector_quantizer_interface",
        "//lyra/testing:mock_generative_model",
        "//lyra/testing:mock_noise_estimator",
        "//lyra/testing:mock_vector_quantizer",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@gulrak_filesystem//:filesystem",
    ],
)

cc_test(
    name = "comfort_noise_generator_test",
    size = "small",
    srcs = ["comfort_noise_generator_test.cc"],
    deps = [
        ":comfort_noise_generator",
        ":dsp_utils",
        ":log_mel_spectrogram_extractor_impl",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "lyra_gan_model_test",
    srcs = ["lyra_gan_model_test.cc"],
    deps = [
        ":lyra_config",
        ":lyra_gan_model",
        "@com_google_googletest//:gtest_main",
        "@gulrak_filesystem//:filesystem",
    ],
)

cc_test(
    name = "lyra_integration_test",
    size = "small",
    timeout = "long",
    srcs = ["lyra_integration_test.cc"],
    data = [
        ":tflite_testdata",
        "//lyra/testdata:sample1_16kHz.wav",
        "//lyra/testdata:sample1_32kHz.wav",
        "//lyra/testdata:sample1_48kHz.wav",
        "//lyra/testdata:sample1_8kHz.wav",
    ],
    shard_count = 4,
    deps = [
        ":dsp_utils",
        ":log_mel_spectrogram_extractor_impl",
        ":lyra_config",
        ":lyra_decoder",
        ":lyra_encoder",
        ":wav_utils",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@gulrak_filesystem//:filesystem",
    ],
)

cc_test(
    name = "noise_estimator_test",
    size = "small",
    srcs = ["noise_estimator_test.cc"],
    deps = [
        ":comfort_noise_generator",
        ":dsp_utils",
        ":log_mel_spectrogram_extractor_impl",
        ":lyra_config",
        ":noise_estimator",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "fixed_packet_loss_model_test",
    size = "small",
    srcs = ["fixed_packet_loss_model_test.cc"],
    deps = [
        ":fixed_packet_loss_model",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "gilbert_model_test",
    size = "small",
    srcs = ["gilbert_model_test.cc"],
    deps = [
        ":gilbert_model",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "log_mel_spectrogram_extractor_impl_test",
    size = "small",
    srcs = ["log_mel_spectrogram_extractor_impl_test.cc"],
    deps = [
        ":log_mel_spectrogram_extractor_impl",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_binary(
    name = "log_mel_spectrogram_extractor_impl_benchmark",
    testonly = 1,
    srcs = ["log_mel_spectrogram_extractor_impl_benchmark.cc"],
    deps = [
        ":log_mel_spectrogram_extractor_impl",
        "@com_github_google_benchmark//:benchmark",
        "@com_github_google_benchmark//:benchmark_main",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "soundstream_encoder_test",
    srcs = ["soundstream_encoder_test.cc"],
    deps = [
        ":lyra_config",
        ":soundstream_encoder",
        "@com_google_googletest//:gtest_main",
        "@gulrak_filesystem//:filesystem",
    ],
)

cc_test(
    name = "lyra_encoder_test",
    size = "small",
    srcs = ["lyra_encoder_test.cc"],
    data = [":tflite_testdata"],
    shard_count = 8,
    deps = [
        ":feature_extractor_interface",
        ":lyra_config",
        ":lyra_encoder",
        ":noise_estimator_interface",
        ":packet",
        ":resampler_interface",
        ":vector_quantizer_interface",
        "//lyra/testing:mock_feature_extractor",
        "//lyra/testing:mock_noise_estimator",
        "//lyra/testing:mock_resampler",
        "//lyra/testing:mock_vector_quantizer",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@gulrak_filesystem//:filesystem",
    ],
)

cc_test(
    name = "residual_vector_quantizer_test",
    size = "small",
    srcs = [
        "residual_vector_quantizer_test.cc",
    ],
    deps = [
        ":log_mel_spectrogram_extractor_impl",
        ":lyra_config",
        ":residual_vector_quantizer",
        "@com_google_googletest//:gtest_main",
        "@gulrak_filesystem//:filesystem",
    ],
)

cc_test(
    name = "packet_test",
    size = "small",
    srcs = ["packet_test.cc"],
    deps = [
        ":packet",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "resampler",
    srcs = [
        "resampler.cc",
    ],
    hdrs = ["resampler.h"],
    deps = [
        ":dsp_utils",
        ":resampler_interface",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:span",
        "@com_google_audio_dsp//audio/dsp:resampler_q",
        "@com_google_glog//:glog",
    ],
)

cc_test(
    name = "resampler_test",
    size = "small",
    srcs = ["resampler_test.cc"],
    deps = [
        ":lyra_config",
        ":resampler",
        "@com_google_absl//absl/types:span",
        "@com_google_audio_dsp//audio/dsp:signal_vector_util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "dsp_utils",
    srcs = [
        "dsp_utils.cc",
    ],
    hdrs = ["dsp_utils.h"],
    deps = [
        "@com_google_absl//absl/types:span",
        "@com_google_audio_dsp//audio/dsp:signal_vector_util",
        "@com_google_glog//:glog",
    ],
)

cc_library(
    name = "wav_utils",
    srcs = [
        "wav_utils.cc",
    ],
    hdrs = ["wav_utils.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_audio_dsp//audio/dsp/portable:read_wav_file",
        "@com_google_audio_dsp//audio/dsp/portable:write_wav_file",
    ],
)

cc_library(
    name = "tflite_model_wrapper",
    srcs = [
        "tflite_model_wrapper.cc",
    ],
    hdrs = [
        "tflite_model_wrapper.h",
    ],
    deps = [
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:span",
        "@com_google_glog//:glog",
        "@gulrak_filesystem//:filesystem",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/delegates/xnnpack:xnnpack_delegate",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
    ],
)

cc_test(
    name = "wav_utils_test",
    size = "small",
    srcs = ["wav_utils_test.cc"],
    data = [
        "//lyra/testdata:invalid.wav",
        "//lyra/testdata:sample1_16kHz.wav",
    ],
    deps = [
        ":wav_utils",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
        "@gulrak_filesystem//:filesystem",
    ],
)

cc_test(
    name = "dsp_utils_test",
    size = "small",
    srcs = ["dsp_utils_test.cc"],
    deps = [
        ":dsp_utils",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "tflite_model_wrapper_test",
    srcs = ["tflite_model_wrapper_test.cc"],
    data = ["model_coeffs/lyragan.tflite"],
    deps = [
        ":tflite_model_wrapper",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@gulrak_filesystem//:filesystem",
        "@org_tensorflow//tensorflow/lite:framework",
    ],
)

cc_test(
    name = "lyra_config_test",
    srcs = ["lyra_config_test.cc"],
    data = [":tflite_testdata"],
    deps = [
        ":lyra_config",
        ":lyra_config_cc_proto",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
        "@gulrak_filesystem//:filesystem",
    ],
)

filegroup(
    name = "tflite_testdata",
    data = glob([
        "model_coeffs/*",
    ]),
)

filegroup(
    name = "android_example_assets",
    srcs = glob([
        "model_coeffs/*.tflite",
    ]) + ["model_coeffs/lyra_config.binarypb"],
)
