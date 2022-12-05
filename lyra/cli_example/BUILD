package(default_visibility = ["//visibility:public"])

exports_files(
    srcs = [
        "decoder_main.cc",
        "decoder_main_lib.cc",
        "decoder_main_lib.h",
        "encoder_main.cc",
        "encoder_main_lib.cc",
        "encoder_main_lib.h",
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
        "//lyra:lyra_config",
        "//lyra:lyra_encoder",
        "//lyra:no_op_preprocessor",
        "//lyra:wav_utils",
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
    name = "decoder_main_lib",
    srcs = [
        "decoder_main_lib.cc",
    ],
    hdrs = [
        "decoder_main_lib.h",
    ],
    deps = [
        "//lyra:fixed_packet_loss_model",
        "//lyra:gilbert_model",
        "//lyra:lyra_config",
        "//lyra:lyra_decoder",
        "//lyra:packet_loss_model_interface",
        "//lyra:wav_utils",
        "@com_google_absl//absl/flags:marshalling",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/random:bit_gen_ref",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@com_google_glog//:glog",
        "@gulrak_filesystem//:filesystem",
    ],
)

cc_test(
    name = "encoder_main_lib_test",
    size = "small",
    srcs = ["encoder_main_lib_test.cc"],
    data = [
        "//lyra:tflite_testdata",
        "//lyra/testdata:sample1_16kHz.wav",
        "//lyra/testdata:sample1_32kHz.wav",
        "//lyra/testdata:sample1_48kHz.wav",
        "//lyra/testdata:sample1_8kHz.wav",
    ],
    deps = [
        ":encoder_main_lib",
        "@com_google_absl//absl/flags:flag",
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
        "//lyra:tflite_testdata",
        "//lyra/testdata:incomplete_encoded_packet.lyra",
        "//lyra/testdata:no_encoded_packet.lyra",
        "//lyra/testdata:one_encoded_packet_16khz.lyra",
        "//lyra/testdata:two_encoded_packets_16khz.lyra",
    ],
    shard_count = 4,
    deps = [
        ":decoder_main_lib",
        "//lyra:lyra_config",
        "//lyra:wav_utils",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@gulrak_filesystem//:filesystem",
    ],
)

cc_binary(
    name = "encoder_main",
    srcs = [
        "encoder_main.cc",
    ],
    data = ["//lyra:tflite_testdata"],
    linkopts = select({
        "//lyra:android_config": ["-landroid"],
        "//conditions:default": [],
    }),
    deps = [
        ":encoder_main_lib",
        "//lyra:architecture_utils",
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
    data = ["//lyra:tflite_testdata"],
    linkopts = select({
        "//lyra:android_config": ["-landroid"],
        "//conditions:default": [],
    }),
    deps = [
        ":decoder_main_lib",
        "//lyra:architecture_utils",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/flags:usage",
        "@com_google_absl//absl/strings",
        "@com_google_glog//:glog",
        "@gulrak_filesystem//:filesystem",
    ],
)
