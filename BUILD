# [internal] load cc_fuzz_target.bzl
# [internal] load cc_proto_library.bzl
# [internal] load android_cc_test:def.bzl

package(default_visibility = [":__subpackages__"])

licenses(["notice"])

# To run all cc_tests in this directory:
# bazel test //:all

# [internal] Command to run dsp_util_android_test.

# [internal] Command to run lyra_integration_android_test.

exports_files(["LICENSE"])

exports_files(
    srcs = [
        "decoder_main.cc",
        "decoder_main_lib.cc",
        "decoder_main_lib.h",
        "encoder_main.cc",
        "encoder_main_lib.cc",
        "encoder_main_lib.h",
        "lyra_components.h",
        "lyra_config.h",
        "lyra_decoder.cc",
        "lyra_decoder.h",
        "lyra_encoder.cc",
        "lyra_encoder.h",
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
    name = "layer_wrapper_interface",
    hdrs = ["layer_wrapper_interface.h"],
    deps = [
        "//sparse_matmul",
    ],
)

cc_library(
    name = "layer_wrapper",
    hdrs = ["layer_wrapper.h"],
    deps = [
        ":dsp_util",
        ":layer_wrapper_interface",
        "//sparse_matmul",
        "@com_google_glog//:glog",
    ],
)

cc_library(
    name = "conv1d_layer_wrapper",
    hdrs = ["conv1d_layer_wrapper.h"],
    deps = [
        ":layer_wrapper",
        "//sparse_matmul",
        "@com_google_absl//absl/memory",
        "@com_google_glog//:glog",
    ],
)

cc_library(
    name = "dilated_convolutional_layer_wrapper",
    hdrs = ["dilated_convolutional_layer_wrapper.h"],
    deps = [
        ":layer_wrapper",
        "//sparse_matmul",
        "@com_google_absl//absl/memory",
        "@com_google_glog//:glog",
    ],
)

cc_library(
    name = "transpose_convolutional_layer_wrapper",
    hdrs = ["transpose_convolutional_layer_wrapper.h"],
    deps = [
        ":layer_wrapper",
        "//sparse_matmul",
        "@com_google_absl//absl/memory",
        "@com_google_glog//:glog",
    ],
)

cc_library(
    name = "layer_wrappers_lib",
    hdrs = ["layer_wrappers_lib.h"],
    deps = [
        ":conv1d_layer_wrapper",
        ":dilated_convolutional_layer_wrapper",
        ":layer_wrapper",
        ":transpose_convolutional_layer_wrapper",
    ],
)

cc_library(
    name = "causal_convolutional_conditioning",
    hdrs = ["causal_convolutional_conditioning.h"],
    deps = [
        ":dsp_util",
        ":layer_wrappers_lib",
        ":lyra_types",
        "//sparse_matmul",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@com_google_glog//:glog",
    ],
)

cc_library(
    name = "benchmark_decode_lib",
    srcs = ["benchmark_decode_lib.cc"],
    hdrs = ["benchmark_decode_lib.h"],
    deps = [
        ":architecture_utils",
        ":dsp_util",
        ":generative_model_interface",
        ":log_mel_spectrogram_extractor_impl",
        ":lyra_config",
        ":wavegru_model_impl",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@com_google_audio_dsp//audio/dsp:signal_vector_util",
        "@com_google_glog//:glog",
        "@gulrak_filesystem//:filesystem",
    ],
)

cc_library(
    name = "generative_model_interface",
    hdrs = [
        "generative_model_interface.h",
    ],
    deps = [
        "@com_google_absl//absl/types:optional",
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
    name = "denoiser_interface",
    hdrs = [
        "denoiser_interface.h",
    ],
    deps = [
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "feature_extractor_interface",
    hdrs = [
        "feature_extractor_interface.h",
    ],
    deps = [
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "lyra_decoder_interface",
    hdrs = [
        "lyra_decoder_interface.h",
    ],
    deps = [
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "lyra_encoder_interface",
    hdrs = [
        "lyra_encoder_interface.h",
    ],
    deps = [
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "vector_quantizer_interface",
    hdrs = [
        "vector_quantizer_interface.h",
    ],
    deps = [
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "filter_banks_interface",
    hdrs = [
        "filter_banks_interface.h",
    ],
)

cc_library(
    name = "wavegru_model_impl",
    srcs = [
        "wavegru_model_impl.cc",
    ],
    hdrs = [
        "wavegru_model_impl.h",
    ],
    copts = [
        "-O3",
    ],
    data = glob(["wavegru/**"]),
    deps = [
        ":buffer_merger",
        ":causal_convolutional_conditioning",
        ":generative_model_interface",
        ":lyra_types",
        ":lyra_wavegru",
        "//sparse_matmul",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:optional",
        "@com_google_glog//:glog",
        "@gulrak_filesystem//:filesystem",
    ],
)

cc_library(
    name = "wavegru_model_impl_fixed16",
    srcs = [
        "wavegru_model_impl.cc",
    ],
    hdrs = [
        "wavegru_model_impl.h",
    ],
    copts = [
        "-O3",
        "-DUSE_FIXED16",
    ],
    data = glob(["wavegru/**"]),
    deps = [
        ":buffer_merger",
        ":causal_convolutional_conditioning",
        ":generative_model_interface",
        ":lyra_types",
        ":lyra_wavegru",
        "//sparse_matmul",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:optional",
        "@com_google_glog//:glog",
        "@gulrak_filesystem//:filesystem",
    ],
)

cc_library(
    name = "naive_spectrogram_predictor",
    srcs = [
        "naive_spectrogram_predictor.cc",
    ],
    hdrs = [
        "naive_spectrogram_predictor.h",
    ],
    deps = [
        ":log_mel_spectrogram_extractor_impl",
        ":spectrogram_predictor_interface",
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
        ":comfort_noise_generator",
        ":generative_model_interface",
        ":lyra_components",
        ":lyra_config",
        ":lyra_decoder_interface",
        ":packet_interface",
        ":packet_loss_handler",
        ":packet_loss_handler_interface",
        ":resampler",
        ":resampler_interface",
        ":vector_quantizer_interface",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@com_google_glog//:glog",
        "@gulrak_filesystem//:filesystem",
    ],
)

cc_library(
    name = "lyra_decoder_fixed16",
    testonly = 1,
    srcs = [
        "lyra_decoder.cc",
    ],
    hdrs = [
        "lyra_decoder.h",
    ],
    copts = ["-DUSE_FIXED16"],
    visibility = ["//visibility:public"],
    deps = [
        ":comfort_noise_generator",
        ":generative_model_interface",
        ":lyra_components_fixed16",
        ":lyra_config",
        ":lyra_decoder_interface",
        ":packet_interface",
        ":packet_loss_handler",
        ":packet_loss_handler_interface",
        ":resampler",
        ":resampler_interface",
        ":vector_quantizer_interface",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@com_google_glog//:glog",
        "@gulrak_filesystem//:filesystem",
    ],
)

cc_library(
    name = "packet_loss_handler",
    srcs = ["packet_loss_handler.cc"],
    hdrs = ["packet_loss_handler.h"],
    deps = [
        ":naive_spectrogram_predictor",
        ":noise_estimator",
        ":noise_estimator_interface",
        ":packet_loss_handler_interface",
        ":spectrogram_predictor_interface",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:optional",
        "@com_google_glog//:glog",
    ],
)

cc_library(
    name = "decoder_main_lib",
    srcs = [
        "decoder_main_lib.cc",
    ],
    hdrs = [
        "decoder_main_lib.h",
    ],
    deps = [
        ":gilbert_model",
        ":lyra_config",
        ":lyra_decoder",
        ":wav_util",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
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
        ":dsp_util",
        ":generative_model_interface",
        ":log_mel_spectrogram_extractor_impl",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@com_google_audio_dsp//audio/dsp:number_util",
        "@com_google_audio_dsp//audio/dsp/mfcc",
        "@com_google_audio_dsp//audio/dsp/spectrogram:inverse_spectrogram",
        "@com_google_glog//:glog",
    ],
)

cc_library(
    name = "lyra_encoder_fixed16",
    srcs = [
        "lyra_encoder.cc",
    ],
    hdrs = [
        "lyra_encoder.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":denoiser_interface",
        ":dsp_util",
        ":feature_extractor_interface",
        ":lyra_components_fixed16",
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
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@com_google_audio_dsp//audio/linear_filters:biquad_filter",
        "@com_google_audio_dsp//audio/linear_filters:biquad_filter_coefficients",
        "@com_google_glog//:glog",
        "@gulrak_filesystem//:filesystem",
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
        ":denoiser_interface",
        ":dsp_util",
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
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@com_google_audio_dsp//audio/linear_filters:biquad_filter",
        "@com_google_audio_dsp//audio/linear_filters:biquad_filter_coefficients",
        "@com_google_glog//:glog",
        "@gulrak_filesystem//:filesystem",
    ],
)

cc_library(
    name = "encoder_main_lib",
    srcs = [
        "encoder_main_lib.cc",
    ],
    hdrs = [
        "encoder_main_lib.h",
    ],
    deps = [
        ":lyra_config",
        ":lyra_encoder",
        ":no_op_preprocessor",
        ":wav_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
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
        "@com_google_absl//absl/types:optional",
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
        "@com_google_absl//absl/types:optional",
    ],
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
        "@com_google_absl//absl/memory",
        "@com_google_glog//:glog",
    ],
)

cc_library(
    name = "packet_loss_handler_interface",
    hdrs = [
        "packet_loss_handler_interface.h",
    ],
    deps = [
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "spectrogram_predictor_interface",
    hdrs = [
        "spectrogram_predictor_interface.h",
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
        "@com_google_protobuf//:protobuf",
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
        ":denoiser_interface",
        ":feature_extractor_interface",
        ":generative_model_interface",
        ":log_mel_spectrogram_extractor_impl",
        ":packet",
        ":packet_interface",
        ":vector_quantizer_impl",
        ":vector_quantizer_interface",
        ":wavegru_model_impl",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status:statusor",
        "@eigen_archive//:eigen",
        "@gulrak_filesystem//:filesystem",
    ],
)

cc_library(
    name = "lyra_components_fixed16",
    srcs = [
        "lyra_components.cc",
    ],
    hdrs = [
        "lyra_components.h",
    ],
    deps = [
        ":denoiser_interface",
        ":feature_extractor_interface",
        ":generative_model_interface",
        ":log_mel_spectrogram_extractor_impl",
        ":packet",
        ":packet_interface",
        ":vector_quantizer_impl",
        ":vector_quantizer_interface",
        ":wavegru_model_impl_fixed16",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status:statusor",
        "@eigen_archive//:eigen",
        "@gulrak_filesystem//:filesystem",
    ],
)

cc_library(
    name = "lyra_types",
    hdrs = ["lyra_types.h"],
    copts = ["-O3"],
    deps = [
        ":layer_wrapper",
        "//sparse_matmul",
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
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@com_google_audio_dsp//audio/dsp:number_util",
        "@com_google_audio_dsp//audio/dsp/mfcc",
        "@com_google_audio_dsp//audio/dsp/spectrogram",
        "@com_google_glog//:glog",
    ],
)

cc_library(
    name = "vector_quantizer_impl",
    srcs = [
        "vector_quantizer_impl.cc",
    ],
    hdrs = [
        "vector_quantizer_impl.h",
    ],
    data = glob(["wavegru/**"]),
    deps = [
        ":vector_quantizer_interface",
        "//sparse_matmul",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:optional",
        "@com_google_audio_dsp//audio/dsp:signal_vector_util",
        "@com_google_glog//:glog",
        "@eigen_archive//:eigen",
        "@gulrak_filesystem//:filesystem",
    ],
)

cc_library(
    name = "packet_interface",
    hdrs = [
        "packet_interface.h",
    ],
    deps = [
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "packet",
    hdrs = ["packet.h"],
    deps = [
        ":packet_interface",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@com_google_glog//:glog",
    ],
)

cc_library(
    name = "lyra_wavegru",
    hdrs = ["lyra_wavegru.h"],
    deps = [
        ":causal_convolutional_conditioning",
        ":dsp_util",
        ":layer_wrappers_lib",
        ":lyra_types",
        ":project_and_sample",
        "//sparse_matmul",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@com_google_glog//:glog",
        "@gulrak_filesystem//:filesystem",
    ],
)

cc_library(
    name = "project_and_sample",
    hdrs = [
        "project_and_sample.h",
    ],
    copts = ["-O3"],
    deps = [
        ":lyra_types",
        "//sparse_matmul",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_glog//:glog",
    ],
)

cc_library(
    name = "filter_banks",
    srcs = ["filter_banks.cc"],
    hdrs = ["filter_banks.h"],
    deps = [
        ":filter_banks_interface",
        ":quadrature_mirror_filter",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:span",
        "@com_google_glog//:glog",
    ],
)

cc_library(
    name = "quadrature_mirror_filter",
    srcs = ["quadrature_mirror_filter.cc"],
    hdrs = ["quadrature_mirror_filter.h"],
    deps = [
        ":dsp_util",
        "@com_google_absl//absl/types:span",
        "@com_google_audio_dsp//audio/linear_filters:biquad_filter",
        "@com_google_audio_dsp//audio/linear_filters:biquad_filter_coefficients",
        "@com_google_glog//:glog",
    ],
)

cc_library(
    name = "buffer_merger",
    srcs = ["buffer_merger.cc"],
    hdrs = ["buffer_merger.h"],
    deps = [
        ":filter_banks",
        ":filter_banks_interface",
        "@com_google_absl//absl/memory",
        "@com_google_glog//:glog",
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
        "@com_google_googletest//:gtest_main",
    ],
)

cc_binary(
    name = "encoder_main",
    srcs = [
        "encoder_main.cc",
    ],
    linkopts = select({
        ":android_config": ["-landroid"],
        "//conditions:default": [],
    }),
    deps = [
        ":architecture_utils",
        ":encoder_main_lib",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/flags:usage",
        "@com_google_absl//absl/strings",
        "@com_google_glog//:glog",
        "@gulrak_filesystem//:filesystem",
    ],
)

cc_binary(
    name = "decoder_main",
    srcs = [
        "decoder_main.cc",
    ],
    linkopts = select({
        ":android_config": ["-landroid"],
        "//conditions:default": [],
    }),
    deps = [
        ":architecture_utils",
        ":decoder_main_lib",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/flags:usage",
        "@com_google_absl//absl/strings",
        "@com_google_glog//:glog",
        "@gulrak_filesystem//:filesystem",
    ],
)

cc_binary(
    name = "benchmark_decode",
    srcs = [
        "benchmark_decode.cc",
    ],
    linkopts = select({
        ":android_config": ["-landroid"],
        "//conditions:default": [],
    }),
    deps = [
        ":benchmark_decode_lib",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/flags:usage",
    ],
)

cc_test(
    name = "lyra_wavegru_test",
    size = "small",
    timeout = "short",
    srcs = ["lyra_wavegru_test.cc"],
    data = glob(["wavegru/**"]),
    deps = [
        ":exported_layers_test",
        ":lyra_config",
        ":lyra_wavegru",
        "//sparse_matmul",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_googletest//:gtest_main",
        "@gulrak_filesystem//:filesystem",
    ],
)

cc_test(
    name = "lyra_wavegru_test_fixed16",
    size = "small",
    timeout = "short",
    srcs = ["lyra_wavegru_test.cc"],
    copts = [
        "-DUSE_FIXED16",
    ],
    data = glob(["wavegru/**"]),
    deps = [
        ":lyra_config",
        ":lyra_wavegru",
        "//sparse_matmul",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_googletest//:gtest_main",
        "@gulrak_filesystem//:filesystem",
    ],
)

cc_test(
    name = "lyra_wavegru_test_bfloat16",
    size = "small",
    timeout = "short",
    srcs = ["lyra_wavegru_test.cc"],
    copts = [
        "-DUSE_BFLOAT16",
    ],
    data = glob(["wavegru/**"]),
    deps = [
        ":lyra_config",
        ":lyra_wavegru",
        "//sparse_matmul",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_googletest//:gtest_main",
        "@gulrak_filesystem//:filesystem",
    ],
)

cc_test(
    name = "project_and_sample_test",
    size = "small",
    timeout = "short",
    srcs = ["project_and_sample_test.cc"],
    data = glob(["wavegru/**"]) + [
        "//testdata:lyra_means_bias.raw.gz",
        "//testdata:lyra_means_fixed16_weights.raw.gz",
        "//testdata:lyra_means_mask.raw.gz",
        "//testdata:lyra_means_weights.raw.gz",
        "//testdata:lyra_mix_bias.raw.gz",
        "//testdata:lyra_mix_fixed16_weights.raw.gz",
        "//testdata:lyra_mix_mask.raw.gz",
        "//testdata:lyra_mix_weights.raw.gz",
        "//testdata:lyra_proj_bias.raw.gz",
        "//testdata:lyra_proj_fixed16_weights.raw.gz",
        "//testdata:lyra_proj_mask.raw.gz",
        "//testdata:lyra_proj_weights.raw.gz",
        "//testdata:lyra_scales_bias.raw.gz",
        "//testdata:lyra_scales_fixed16_weights.raw.gz",
        "//testdata:lyra_scales_mask.raw.gz",
        "//testdata:lyra_scales_weights.raw.gz",
    ],
    deps = [
        ":exported_layers_test",
        ":lyra_types",
        ":project_and_sample",
        "//sparse_matmul",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_googletest//:gtest_main",
        "@gulrak_filesystem//:filesystem",
    ],
)

cc_test(
    name = "lyra_decoder_test",
    size = "large",
    srcs = ["lyra_decoder_test.cc"],
    shard_count = 8,
    deps = [
        ":generative_model_interface",
        ":log_mel_spectrogram_extractor_impl",
        ":lyra_config",
        ":lyra_decoder",
        ":packet",
        ":packet_interface",
        ":packet_loss_handler_interface",
        ":resampler",
        ":resampler_interface",
        ":vector_quantizer_interface",
        "//testing:mock_generative_model",
        "//testing:mock_packet_loss_handler",
        "//testing:mock_resampler",
        "//testing:mock_vector_quantizer",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
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
        "@com_google_absl//absl/types:optional",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "packet_loss_handler_test",
    size = "small",
    srcs = ["packet_loss_handler_test.cc"],
    deps = [
        ":lyra_config",
        ":noise_estimator_interface",
        ":packet_loss_handler",
        ":spectrogram_predictor_interface",
        "//testing:mock_noise_estimator",
        "//testing:mock_spectrogram_predictor",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "naive_spectrogram_predictor_test",
    size = "small",
    srcs = ["naive_spectrogram_predictor_test.cc"],
    deps = [
        ":log_mel_spectrogram_extractor_impl",
        ":lyra_config",
        ":naive_spectrogram_predictor",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "wavegru_model_impl_test",
    size = "small",
    timeout = "short",
    srcs = ["wavegru_model_impl_test.cc"],
    deps = [
        ":lyra_config",
        ":wavegru_model_impl",
        "@com_google_googletest//:gtest_main",
        "@gulrak_filesystem//:filesystem",
    ],
)

cc_library(
    name = "exported_layers_test",
    testonly = 1,
    hdrs = [
        "exported_layers_test.h",
    ],
    deps = [
        ":layer_wrappers_lib",
        ":lyra_types",
        "//sparse_matmul",
        "@com_google_absl//absl/random",
        "@com_google_googletest//:gtest",
        "@gulrak_filesystem//:filesystem",
    ],
)

cc_test(
    name = "lyra_integration_test",
    size = "small",
    timeout = "long",
    srcs = ["lyra_integration_test.cc"],
    data = [
        "//testdata:16khz_sample_000001.wav",
        "//testdata:32khz_sample_000002.wav",
        "//testdata:48khz_sample_000003.wav",
        "//testdata:8khz_sample_000000.wav",
    ],
    shard_count = 4,
    deps = [
        ":dsp_util",
        ":log_mel_spectrogram_extractor_impl",
        ":lyra_config",
        ":lyra_decoder",
        ":lyra_encoder",
        ":wav_util",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_glog//:glog",
        "@com_google_googletest//:gtest_main",
        "@gulrak_filesystem//:filesystem",
    ],
)

cc_test(
    name = "lyra_integration_test_fixed16",
    size = "small",
    timeout = "long",
    srcs = ["lyra_integration_test.cc"],
    copts = ["-DUSE_FIXED16"],
    data = [
        "//testdata:16khz_sample_000001.wav",
        "//testdata:32khz_sample_000002.wav",
        "//testdata:48khz_sample_000003.wav",
        "//testdata:8khz_sample_000000.wav",
    ],
    shard_count = 4,
    deps = [
        ":dsp_util",
        ":log_mel_spectrogram_extractor_impl",
        ":lyra_config",
        ":lyra_decoder_fixed16",
        ":lyra_encoder_fixed16",
        ":wav_util",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_glog//:glog",
        "@com_google_googletest//:gtest_main",
        "@gulrak_filesystem//:filesystem",
    ],
)

cc_test(
    name = "encoder_main_lib_test",
    size = "small",
    srcs = ["encoder_main_lib_test.cc"],
    data = [
        "//testdata:16khz_sample_000001.wav",
        "//testdata:32khz_sample_000002.wav",
        "//testdata:48khz_sample_000003.wav",
        "//testdata:8khz_sample_000000.wav",
    ],
    deps = [
        ":encoder_main_lib",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@gulrak_filesystem//:filesystem",
    ],
)

cc_test(
    name = "decoder_main_lib_test",
    size = "large",
    srcs = ["decoder_main_lib_test.cc"],
    data = [
        "//testdata:incomplete_encoded_frame",
        "//testdata:no_encoded_frames",
        "//testdata:one_encoded_frame_16khz",
        "//testdata:two_encoded_frames_16khz.lyra",
    ],
    shard_count = 4,
    deps = [
        ":decoder_main_lib",
        ":lyra_config",
        ":wav_util",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@gulrak_filesystem//:filesystem",
    ],
)

cc_test(
    name = "noise_estimator_test",
    size = "small",
    srcs = ["noise_estimator_test.cc"],
    deps = [
        ":noise_estimator",
        "@com_google_absl//absl/types:optional",
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
    name = "lyra_encoder_test",
    size = "small",
    srcs = ["lyra_encoder_test.cc"],
    shard_count = 8,
    deps = [
        ":denoiser_interface",
        ":feature_extractor_interface",
        ":lyra_config",
        ":lyra_encoder",
        ":noise_estimator_interface",
        ":packet",
        ":packet_interface",
        ":resampler_interface",
        ":vector_quantizer_interface",
        "//testing:mock_denoiser",
        "//testing:mock_feature_extractor",
        "//testing:mock_noise_estimator",
        "//testing:mock_resampler",
        "//testing:mock_vector_quantizer",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@gulrak_filesystem//:filesystem",
    ],
)

cc_test(
    name = "vector_quantizer_impl_test",
    size = "small",
    srcs = [
        "vector_quantizer_impl_test.cc",
    ],
    deps = [
        ":lyra_config",
        ":vector_quantizer_impl",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@eigen_archive//:eigen",
        "@gulrak_filesystem//:filesystem",
    ],
)

cc_test(
    name = "causal_convolutional_conditioning_test",
    size = "small",
    srcs = ["causal_convolutional_conditioning_test.cc"],
    data = glob(["wavegru/**"]) + [
        "//testdata:codec.gz",
        "//testdata:lyra_conditioning_stack_0_bias.raw.gz",
        "//testdata:lyra_conditioning_stack_0_fixed16_weights.raw.gz",
        "//testdata:lyra_conditioning_stack_0_mask.raw.gz",
        "//testdata:lyra_conditioning_stack_0_weights.raw.gz",
        "//testdata:lyra_conditioning_stack_1_bias.raw.gz",
        "//testdata:lyra_conditioning_stack_1_fixed16_weights.raw.gz",
        "//testdata:lyra_conditioning_stack_1_mask.raw.gz",
        "//testdata:lyra_conditioning_stack_1_weights.raw.gz",
        "//testdata:lyra_conditioning_stack_2_bias.raw.gz",
        "//testdata:lyra_conditioning_stack_2_fixed16_weights.raw.gz",
        "//testdata:lyra_conditioning_stack_2_mask.raw.gz",
        "//testdata:lyra_conditioning_stack_2_weights.raw.gz",
        "//testdata:lyra_conv1d_bias.raw.gz",
        "//testdata:lyra_conv1d_fixed16_weights.raw.gz",
        "//testdata:lyra_conv1d_mask.raw.gz",
        "//testdata:lyra_conv1d_weights.raw.gz",
        "//testdata:lyra_conv_cond_bias.raw.gz",
        "//testdata:lyra_conv_cond_fixed16_weights.raw.gz",
        "//testdata:lyra_conv_cond_mask.raw.gz",
        "//testdata:lyra_conv_cond_weights.raw.gz",
        "//testdata:lyra_conv_to_gates_bias.raw.gz",
        "//testdata:lyra_conv_to_gates_fixed16_weights.raw.gz",
        "//testdata:lyra_conv_to_gates_mask.raw.gz",
        "//testdata:lyra_conv_to_gates_weights.raw.gz",
        "//testdata:lyra_transpose_0_bias.raw.gz",
        "//testdata:lyra_transpose_0_fixed16_weights.raw.gz",
        "//testdata:lyra_transpose_0_mask.raw.gz",
        "//testdata:lyra_transpose_0_weights.raw.gz",
        "//testdata:lyra_transpose_1_bias.raw.gz",
        "//testdata:lyra_transpose_1_fixed16_weights.raw.gz",
        "//testdata:lyra_transpose_1_mask.raw.gz",
        "//testdata:lyra_transpose_1_weights.raw.gz",
        "//testdata:lyra_transpose_2_bias.raw.gz",
        "//testdata:lyra_transpose_2_fixed16_weights.raw.gz",
        "//testdata:lyra_transpose_2_mask.raw.gz",
        "//testdata:lyra_transpose_2_weights.raw.gz",
        "//testdata:transpose_2.gz",
    ],
    deps = [
        ":causal_convolutional_conditioning",
        ":exported_layers_test",
        ":lyra_config",
        ":lyra_types",
        "//sparse_matmul",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@gulrak_filesystem//:filesystem",
    ],
)

cc_library(
    name = "layer_wrapper_test_common",
    testonly = 1,
    hdrs = [
        "layer_wrapper_test_common.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":layer_wrappers_lib",
        "//sparse_matmul",
        "@com_google_absl//absl/memory",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "conv1d_layer_wrapper_test",
    size = "small",
    srcs = ["conv1d_layer_wrapper_test.cc"],
    data = [
        "//testdata:lyra_conv1d_bias.raw.gz",
        "//testdata:lyra_conv1d_fixed16_weights.raw.gz",
        "//testdata:lyra_conv1d_mask.raw.gz",
        "//testdata:lyra_conv1d_weights.raw.gz",
        "//testdata:test_conv1d_bias.raw.gz",
        "//testdata:test_conv1d_fixed16_weights.raw.gz",
        "//testdata:test_conv1d_mask.raw.gz",
        "//testdata:test_conv1d_weights.raw.gz",
    ],
    deps = [
        ":conv1d_layer_wrapper",
        ":layer_wrapper",
        ":layer_wrapper_test_common",
        "//sparse_matmul",
        "@com_google_googletest//:gtest_main",
        "@gulrak_filesystem//:filesystem",
    ],
)

cc_test(
    name = "dilated_convolutional_layer_wrapper_test",
    size = "small",
    srcs = ["dilated_convolutional_layer_wrapper_test.cc"],
    data = [
        "//testdata:lyra_conditioning_stack_2_bias.raw.gz",
        "//testdata:lyra_conditioning_stack_2_fixed16_weights.raw.gz",
        "//testdata:lyra_conditioning_stack_2_mask.raw.gz",
        "//testdata:lyra_conditioning_stack_2_weights.raw.gz",
        "//testdata:test_dilated_bias.raw.gz",
        "//testdata:test_dilated_fixed16_weights.raw.gz",
        "//testdata:test_dilated_mask.raw.gz",
        "//testdata:test_dilated_weights.raw.gz",
    ],
    deps = [
        ":dilated_convolutional_layer_wrapper",
        ":layer_wrapper",
        ":layer_wrapper_test_common",
        "//sparse_matmul",
        "@com_google_googletest//:gtest_main",
        "@gulrak_filesystem//:filesystem",
    ],
)

cc_test(
    name = "transpose_convolutional_layer_wrapper_test",
    size = "small",
    srcs = ["transpose_convolutional_layer_wrapper_test.cc"],
    data = [
        "//testdata:lyra_transpose_2_bias.raw.gz",
        "//testdata:lyra_transpose_2_fixed16_weights.raw.gz",
        "//testdata:lyra_transpose_2_mask.raw.gz",
        "//testdata:lyra_transpose_2_weights.raw.gz",
        "//testdata:test_transpose_bias.raw.gz",
        "//testdata:test_transpose_fixed16_weights.raw.gz",
        "//testdata:test_transpose_mask.raw.gz",
        "//testdata:test_transpose_weights.raw.gz",
    ],
    deps = [
        ":layer_wrapper",
        ":layer_wrapper_test_common",
        ":transpose_convolutional_layer_wrapper",
        "//sparse_matmul",
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
        "@com_google_absl//absl/types:optional",
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
        ":dsp_util",
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
    name = "dsp_util",
    srcs = [
        "dsp_util.cc",
    ],
    hdrs = ["dsp_util.h"],
    deps = [
        "//sparse_matmul",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@com_google_audio_dsp//audio/dsp:signal_vector_util",
        "@com_google_glog//:glog",
    ],
)

cc_library(
    name = "wav_util",
    srcs = [
        "wav_util.cc",
    ],
    hdrs = ["wav_util.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_audio_dsp//audio/dsp/portable:read_wav_file",
        "@com_google_audio_dsp//audio/dsp/portable:write_wav_file",
    ],
)

cc_test(
    name = "wav_util_test",
    size = "small",
    srcs = ["wav_util_test.cc"],
    data = [
        "//testdata:16khz_sample_000001.wav",
        "//testdata:lyra_config.textproto",
    ],
    deps = [
        ":wav_util",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
        "@gulrak_filesystem//:filesystem",
    ],
)

cc_test(
    name = "dsp_util_test",
    size = "small",
    srcs = ["dsp_util_test.cc"],
    deps = [
        ":dsp_util",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "filter_banks_test",
    srcs = ["filter_banks_test.cc"],
    deps = [
        ":filter_banks",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "quadrature_mirror_filter_test",
    srcs = ["quadrature_mirror_filter_test.cc"],
    deps = [
        ":quadrature_mirror_filter",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "buffer_merger_test",
    srcs = ["buffer_merger_test.cc"],
    deps = [
        ":buffer_merger",
        ":filter_banks_interface",
        ":lyra_config",
        "//testing:mock_filter_banks",
        "@com_google_absl//absl/memory",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "lyra_config_test",
    srcs = ["lyra_config_test.cc"],
    deps = [
        ":lyra_config",
        "@com_google_googletest//:gtest_main",
    ],
)

filegroup(
    name = "wavegru_testdata",
    data = glob([
        "wavegru/*.gz",
        "wavegru/*.textproto",
    ]),
)

filegroup(
    name = "android_example_assets",
    srcs = glob([
        "wavegru/*.gz",
        "wavegru/*.textproto",
    ]),
)
