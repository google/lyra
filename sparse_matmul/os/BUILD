# Modules that interact with the operating system, and have no other dependencies.

licenses(["notice"])

cc_library(
    name = "coop_threads",
    srcs = ["coop_threads.cc"],
    hdrs = ["coop_threads.h"],
    visibility = ["//sparse_matmul:__subpackages__"],
    deps = [
        "@com_google_absl//absl/memory",
        "@com_google_glog//:glog",
    ],
)

cc_test(
    name = "coop_threads_test",
    size = "small",
    srcs = [
        "coop_threads_test.cc",
    ],
    deps = [
        ":coop_threads",
        "@com_google_googletest//:gtest_main",
    ],
)
