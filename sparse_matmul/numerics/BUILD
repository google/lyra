# Base numeric types and transcendental functions.

licenses(["notice"])

cc_library(
    name = "fast_transcendentals",
    srcs = [
        "fast_transcendentals.cc",
    ],
    hdrs = [
        "fast_transcendentals.h",
    ],
    visibility = [
        "//sparse_matmul:__subpackages__",
    ],
    deps = [":types"],
)

cc_library(
    name = "test_utils",
    testonly = 1,
    hdrs = [
        "test_utils.h",
    ],
    visibility = ["//sparse_matmul:__subpackages__"],
    deps = [
        ":types",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "types",
    hdrs = [
        "fixed_types.h",
        "float16_types.h",
        "type_utils.h",
    ],
    visibility = [
        "//sparse_matmul:__subpackages__",
    ],
    deps = [
        "@com_google_glog//:glog",
    ],
)

cc_library(
    name = "fast_transcendentals_cc",
    srcs = ["fast_transcendentals.cc"],
    hdrs = ["fast_transcendentals.h"],
    deps = [":types"],
)

cc_test(
    name = "fasttranscendentals_test",
    size = "small",
    srcs = [
        "fasttranscendentals_test.cc",
    ],
    deps = [
        ":fast_transcendentals",
        ":test_utils",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "fasttranscendentals_test_fast",
    size = "small",
    srcs = [
        "fasttranscendentals_test.cc",
    ],
    copts = ["-DFAST_TRANSCENDENTALS"],
    deps = [
        ":fast_transcendentals",
        ":test_utils",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "fasttranscendentals_test_fast_accurate",
    size = "small",
    srcs = [
        "fasttranscendentals_test.cc",
    ],
    copts = [
        "-DFAST_TRANSCENDENTALS",
        "-DACCURATE_TRANSCENDENTAL_APPROX",
    ],
    deps = [
        ":fast_transcendentals",
        ":test_utils",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "fasttranscendentals_test_fast_accurate_sigmoidastanh",
    size = "small",
    srcs = [
        "fasttranscendentals_test.cc",
    ],
    copts = [
        "-DFAST_TRANSCENDENTALS",
        "-DACCURATE_TRANSCENDENTAL_APPROX",
        "-DSIGMOID_AS_TANH",
    ],
    deps = [
        ":fast_transcendentals",
        ":test_utils",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "fasttranscendentals_test_fast_sigmoidastanh",
    size = "small",
    srcs = [
        "fasttranscendentals_test.cc",
    ],
    copts = [
        "-DFAST_TRANSCENDENTALS",
        "-DSIGMOID_AS_TANH",
    ],
    deps = [
        ":fast_transcendentals",
        ":test_utils",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "fasttranscendentals_test_faster_sigmoid",
    size = "small",
    srcs = [
        "fasttranscendentals_test.cc",
    ],
    copts = [
        "-DFASTER_TRANSCENDENTALS",
    ],
    deps = [
        ":fast_transcendentals",
        ":test_utils",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "fixed_types_test",
    size = "small",
    srcs = [
        "fixed_types_test.cc",
    ],
    deps = [
        ":test_utils",
        ":types",
        "@com_google_googletest//:gtest_main",
    ],
)
