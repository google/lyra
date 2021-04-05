package(default_visibility = ["//:__pkg__"])

licenses(["notice"])

cc_library(
    name = "mock_generative_model",
    testonly = 1,
    hdrs = [
        "mock_generative_model.h",
    ],
    deps = [
        "//:generative_model_interface",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "mock_denoiser",
    testonly = 1,
    hdrs = [
        "mock_denoiser.h",
    ],
    deps = [
        "//:denoiser_interface",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
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
        "//:feature_extractor_interface",
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
        "//:vector_quantizer_interface",
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
        "//:noise_estimator_interface",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "mock_packet_loss_handler",
    testonly = 1,
    hdrs = [
        "mock_packet_loss_handler.h",
    ],
    deps = [
        "//:packet_loss_handler_interface",
        "@com_google_absl//absl/types:optional",
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
        "//:lyra_decoder_interface",
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
        "//:lyra_encoder_interface",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "mock_spectrogram_predictor",
    testonly = 1,
    hdrs = [
        "mock_spectrogram_predictor.h",
    ],
    deps = [
        "//:spectrogram_predictor_interface",
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
        "//:resampler_interface",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "mock_filter_banks",
    testonly = 1,
    hdrs = [
        "mock_filter_banks.h",
    ],
    deps = [
        "//:filter_banks_interface",
        "@com_google_googletest//:gtest",
    ],
)
