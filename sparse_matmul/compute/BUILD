# Low-level computation code, including generic and architecture-specific
# variants.

licenses(["notice"])

cc_library(
    name = "gru_gates",
    srcs = [
        "ar_inputs.h",
        "gru_gates_arm.h",
        "gru_gates_avx_fixed.h",
        "gru_gates_generic.h",
    ],
    hdrs = ["gru_gates.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":matmul",
        "//sparse_matmul/numerics:fast_transcendentals",
        "//sparse_matmul/numerics:types",
        "//sparse_matmul/vector:cache_aligned_vector",
    ],
)

cc_library(
    name = "kernels",
    srcs = [
        "kernels_arm.h",
        "kernels_avx.h",
    ],
    hdrs = [
        "kernels_generic.h",
    ],
    visibility = [
        "//sparse_matmul:__subpackages__",
    ],
    deps = [
        "//sparse_matmul/numerics:fast_transcendentals",
        "//sparse_matmul/numerics:types",
    ],
)

cc_library(
    name = "matmul",
    srcs = [
        "matmul_fixed_avx2.cc",
        "matmul_fixed_avx2.h",
        "matmul_generic.cc",
        "matmul_generic.h",
    ],
    hdrs = [
        "matmul.h",
    ],
    visibility = [
        "//sparse_matmul:__subpackages__",
    ],
    deps = [
        "//sparse_matmul/numerics:types",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "thread_bounds",
    srcs = ["thread_bounds.cc"],
    hdrs = ["thread_bounds.h"],
    visibility = [
        "//sparse_matmul:__subpackages__",
    ],
    deps = [
        "@com_google_glog//:glog",
    ],
)

cc_test(
    name = "gru_gates_test",
    size = "small",
    srcs = [
        "gru_gates_test.cc",
    ],
    deps = [
        ":gru_gates",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)
