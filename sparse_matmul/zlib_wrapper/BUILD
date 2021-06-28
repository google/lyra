licenses(["notice"])

cc_library(
    name = "zlib_wrapper",
    srcs = [
        "gzipheader.cc",
        "zlibwrapper.cc",
    ],
    hdrs = [
        "gzipheader.h",
        "zlibwrapper.h",
    ],
    visibility = ["//:__subpackages__"],
    deps = [
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_glog//:glog",
        "@zlib",
    ],
)
