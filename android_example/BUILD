load("@rules_android//android:rules.bzl", "android_binary", "android_library")
load("@gmaven_rules//:defs.bzl", "gmaven_artifact")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

# Android App rules
cc_library(
    name = "jni_benchmark_decode_lib",
    srcs = ["jni_benchmark_decode_lib.cc"],
    deps = [
        "//:benchmark_decode_lib",
        "//:decoder_main_lib",
        "//:encoder_main_lib",
        "//:lyra_config",
    ],
    alwayslink = True,
)

android_library(
    name = "lyra_android_lib",
    srcs = ["java/com/example/android/lyra/MainActivity.java"],
    custom_package = "com.example.android.lyra",
    manifest = "LibraryManifest.xml",
    resource_files = glob(["res/**/*"]),
    deps = [
        ":jni_benchmark_decode_lib",
        gmaven_artifact("com.android.support.constraint:constraint-layout:aar:1.1.2"),
        gmaven_artifact("com.android.support:appcompat-v7:aar:26.1.0"),
        "@com_android_support_support_annotations_26_1_0",
    ],
)

filegroup(
    name = "assets",
    srcs = ["//:android_example_assets"],
)

android_binary(
    name = "lyra_android_example",
    assets = [
        ":assets",
    ],
    assets_dir = "wavegru",
    custom_package = "com.example.android.lyra",
    manifest = "AndroidManifest.xml",
    deps = [":lyra_android_lib"],
)
