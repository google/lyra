# Sparse/Masked Matrix and Layer.

# [internal] load android_library_selector
# [internal] load android_cc_test:def.bzl

licenses(["notice"])

cc_library(
    name = "layer",
    hdrs = [
        "sparse_linear_layer.h",
    ],
    visibility = [
        "//sparse_matmul:__subpackages__",
    ],
    deps = [
        ":matrix",
        "//sparse_matmul/numerics:types",
        "//sparse_matmul/os:coop_threads",
        "//sparse_matmul/vector:cache_aligned_vector",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_glog//:glog",
    ],
)

cc_library(
    name = "matrix",
    hdrs = [
        "csr_blocksparse_matrix.h",
        "masked_sparse_matrix.h",
    ],
    visibility = [
        "//sparse_matmul:__subpackages__",
    ],
    deps = [
        "//sparse_matmul/compute:kernels",
        "//sparse_matmul/compute:matmul",
        "//sparse_matmul/compute:thread_bounds",
        "//sparse_matmul/numerics:types",
        "//sparse_matmul/os:coop_threads",
        "//sparse_matmul/vector:cache_aligned_vector",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_glog//:glog",
    ],
)

cc_library(
    name = "utils",
    srcs = [
        "utils.cc",
    ],
    hdrs = [
        "read_array_ifstream.h",
        "utils.h",
    ],
    visibility = [
        "//sparse_matmul:__subpackages__",
    ],
    deps = [
        ":layer",
        ":matrix",
        ":status",
        "//sparse_matmul/numerics:types",
        "//sparse_matmul/vector:cache_aligned_vector",
        "//sparse_matmul/zlib_wrapper",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@gulrak_filesystem//:filesystem",
    ],
)

cc_library(
    name = "status",
    srcs = [
        "errno_mapping.cc",
    ],
    hdrs = [
        "errno_mapping.h",
        "status_macros.h",
    ],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
    ],
)

cc_test(
    name = "csrblocksparse_test",
    size = "small",
    srcs = [
        "csrblocksparse_test.cc",
    ],
    data = glob(["testdata/*"]),
    linkopts = select({
        "@bazel_tools//platforms:android": ["-landroid"],
        "//conditions:default": [],
    }),
    shard_count = 10,
    deps = [
        ":status",
        ":utils",
        "//sparse_matmul/compute:matmul",
        "//sparse_matmul/numerics:test_utils",
        "//sparse_matmul/os:coop_threads",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@gulrak_filesystem//:filesystem",
    ],
)

cc_test(
    name = "sparse_linear_layer_test",
    srcs = [
        "sparse_linear_layer_test.cc",
    ],
    deps = [
        ":layer",
        "//sparse_matmul/numerics:test_utils",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "utils_test",
    srcs = ["utils_test.cc"],
    deps = [
        ":layer",
        ":matrix",
        ":status",
        ":utils",
        "//sparse_matmul/numerics:fast_transcendentals",
        "//sparse_matmul/numerics:test_utils",
        "//sparse_matmul/numerics:types",
        "//sparse_matmul/vector:cache_aligned_vector",
        "@com_google_absl//absl/flags:flag",
        "@com_google_googletest//:gtest_main",
        "@gulrak_filesystem//:filesystem",
    ],
)
