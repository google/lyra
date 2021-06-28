# Vector that always aligns its data to the cache line of the host machine.

licenses(["notice"])

cc_library(
    name = "cache_aligned_vector",
    hdrs = [
        "cache_aligned_vector.h",
    ],
    visibility = [
        "//sparse_matmul:__subpackages__",
    ],
    deps = [
        ":aligned_malloc",
        "//sparse_matmul/numerics:fast_transcendentals",
        "//sparse_matmul/numerics:types",
        "//sparse_matmul/os:coop_threads",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_test(
    name = "cachealignedvector_test",
    size = "small",
    srcs = [
        "cachealignedvector_test.cc",
    ],
    copts = [
        "-DFAST_TRANSCENDENTALS",
        "-DSIGMOID_AS_TANH",
    ],
    deps = [
        ":cache_aligned_vector",
        "//sparse_matmul/numerics:test_utils",
        "//sparse_matmul/os:coop_threads",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_binary(
    name = "cachealignedvector_benchmark",
    srcs = [
        "cachealignedvector_benchmark.cc",
    ],
    copts = [
        "-DFAST_TRANSCENDENTALS",
        "-DSIGMOID_AS_TANH",
        "-DACCURATE_TRANSCENDENTAL_APPROX",
    ],
    deps = [
        ":cache_aligned_vector",
        "@com_github_google_benchmark//:benchmark",
        "@com_github_google_benchmark//:benchmark_main",
    ],
)

cc_library(
    name = "aligned_malloc",
    srcs = ["aligned_malloc.cc"],
    hdrs = [
        "aligned_malloc.h",
    ],
)
