package(default_visibility = ["//:__subpackages__"])

licenses(["notice"])

cc_library(
    name = "mock_generative_model",
    testonly = 1,
    hdrs = [
        "mock_generative_model.h",
    ],
    deps = [
        "//lyra:generative_model_interface",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "mock_feature_extractor",
    testonly = 1,
    hdrs = [
        "mock_feature_extractor.h",
    ],
    deps = [
        "//lyra:feature_extractor_interface",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "mock_vector_quantizer",
    testonly = 1,
    hdrs = [
        "mock_vector_quantizer.h",
    ],
    deps = [
        "//lyra:vector_quantizer_interface",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "mock_noise_estimator",
    testonly = 1,
    hdrs = [
        "mock_noise_estimator.h",
    ],
    deps = [
        "//lyra:noise_estimator_interface",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "mock_lyra_decoder",
    testonly = 1,
    hdrs = [
        "mock_lyra_decoder.h",
    ],
    deps = [
        "//lyra:lyra_decoder_interface",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "mock_lyra_encoder",
    testonly = 1,
    hdrs = [
        "mock_lyra_encoder.h",
    ],
    deps = [
        "//lyra:lyra_encoder_interface",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "mock_resampler",
    testonly = 1,
    hdrs = [
        "mock_resampler.h",
    ],
    deps = [
        "//lyra:resampler",
        "//lyra:resampler_interface",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
    ],
)
