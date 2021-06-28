# [internal] load placeholder

licenses(["notice"])

cc_library(
    name = "sparse_matmul",
    hdrs = [
        "sparse_matmul.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//sparse_matmul/compute:gru_gates",
        "//sparse_matmul/layers:layer",
        "//sparse_matmul/layers:matrix",
        "//sparse_matmul/layers:utils",
        "//sparse_matmul/numerics:fast_transcendentals",
        "//sparse_matmul/numerics:types",
        "//sparse_matmul/os:coop_threads",
        "//sparse_matmul/vector:cache_aligned_vector",
    ],  # internal :sparse_matmul deps placeholder
)
