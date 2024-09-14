cc_binary(
    name = "lyra_dll",
    srcs = [
        "dllmain.cc",
    ],
    #data = [":tflite_testdata"],
    linkopts = select({
        "//lyra:android_config": ["-landroid"],
        "//conditions:default": [],
    }),
    deps = [
        "//lyra:lyra_config",
        "//lyra:lyra_encoder",
        "//lyra:lyra_decoder",
        "//lyra:wav_utils",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/flags:usage",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@com_google_glog//:glog",
        "@gulrak_filesystem//:filesystem",
    ],
    linkshared = 1,
    copts = ["/DCOMPILING_DLL"],
    target_compatible_with = [
        "@platforms//cpu:x86_64",
        "@platforms//os:windows",
    ],
)