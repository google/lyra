package(default_visibility = ["//:__pkg__"])

licenses(["notice"])

exports_files([
    "lyra_config.textproto",
    # Wav files taken from the Common Voice Corpus 1. Prefix describes sample rate of file.
    "48khz_sample_000003.wav",
    "32khz_sample_000002.wav",
    "16khz_sample_000001.wav",
    "8khz_sample_000000.wav",
    # Stereo Wav, each track being white noise.
    "16khz_stereo_white_noise.wav",
    # Wav file whose samples form an increasing sequence in range [0, 16000).
    "16khz_increasing.wav",
    # Wav file whose samples form an increasing sequence in range
    # [-24000, 24000), which is one second for 48 kHz sample rate. Useful as a
    # decoder playback for debugging.
    "48khz_increasing.wav",
    # Wav file containing real speeches. Useful as a encoder playback for
    # debugging.
    "48khz_playback.wav",
    # Not a valid Wav file.
    "invalid.wav",
    # Binary file with 9 bytes of data.
    "incomplete_encoded_frame",
    # Binary file with 16 bytes of data, with sample rate set to 16khz.
    # To change the size of one frame, type the following command on the terminal:
    # 'truncate -s 16 one_encoded_frame'
    "one_encoded_frame_16khz",
    # Binary file with 32 bytes of data. Sample rate of both packets set to 16khz.
    "two_encoded_frames_16khz.lyra",
    # Empty file.
    "no_encoded_frames",
    # Dump file consisting of an encoding stream.
    "encoding_stream_dump.textproto",
    # Dump file consisting of a decoding stream.
    "decoding_stream_dump.textproto",
    # Conditioning stack input.
    "codec.gz",
    # Conditioning stack weights, biases and masks.
    "lyra_conditioning_stack_0_bias.raw.gz",
    "lyra_conditioning_stack_0_fixed16_weights.raw.gz",
    "lyra_conditioning_stack_0_mask.raw.gz",
    "lyra_conditioning_stack_0_weights.raw.gz",
    "lyra_conditioning_stack_1_bias.raw.gz",
    "lyra_conditioning_stack_1_fixed16_weights.raw.gz",
    "lyra_conditioning_stack_1_mask.raw.gz",
    "lyra_conditioning_stack_1_weights.raw.gz",
    "lyra_conditioning_stack_2_bias.raw.gz",
    "lyra_conditioning_stack_2_fixed16_weights.raw.gz",
    "lyra_conditioning_stack_2_mask.raw.gz",
    "lyra_conditioning_stack_2_weights.raw.gz",
    "lyra_conv1d_bias.raw.gz",
    "lyra_conv1d_fixed16_weights.raw.gz",
    "lyra_conv1d_mask.raw.gz",
    "lyra_conv1d_weights.raw.gz",
    "lyra_conv_cond_bias.raw.gz",
    "lyra_conv_cond_fixed16_weights.raw.gz",
    "lyra_conv_cond_mask.raw.gz",
    "lyra_conv_cond_weights.raw.gz",
    "lyra_conv_to_gates_bias.raw.gz",
    "lyra_conv_to_gates_fixed16_weights.raw.gz",
    "lyra_conv_to_gates_mask.raw.gz",
    "lyra_conv_to_gates_weights.raw.gz",
    "lyra_gru_layer_bias.raw.gz",
    "lyra_gru_layer_fixed16_weights.raw.gz",
    "lyra_gru_layer_mask.raw.gz",
    "lyra_gru_layer_weights.raw.gz",
    "lyra_means_bias.raw.gz",
    "lyra_means_fixed16_weights.raw.gz",
    "lyra_means_mask.raw.gz",
    "lyra_means_weights.raw.gz",
    "lyra_mix_bias.raw.gz",
    "lyra_mix_fixed16_weights.raw.gz",
    "lyra_mix_mask.raw.gz",
    "lyra_mix_weights.raw.gz",
    "lyra_proj_bias.raw.gz",
    "lyra_proj_fixed16_weights.raw.gz",
    "lyra_proj_mask.raw.gz",
    "lyra_proj_weights.raw.gz",
    "lyra_scales_bias.raw.gz",
    "lyra_scales_fixed16_weights.raw.gz",
    "lyra_scales_mask.raw.gz",
    "lyra_scales_weights.raw.gz",
    "lyra_transpose_0_bias.raw.gz",
    "lyra_transpose_0_fixed16_weights.raw.gz",
    "lyra_transpose_0_mask.raw.gz",
    "lyra_transpose_0_weights.raw.gz",
    "lyra_transpose_1_bias.raw.gz",
    "lyra_transpose_1_fixed16_weights.raw.gz",
    "lyra_transpose_1_mask.raw.gz",
    "lyra_transpose_1_weights.raw.gz",
    "lyra_transpose_2_bias.raw.gz",
    "lyra_transpose_2_fixed16_weights.raw.gz",
    "lyra_transpose_2_mask.raw.gz",
    "lyra_transpose_2_weights.raw.gz",
    # Golden conditioning stack output.
    "transpose_2.gz",
    # Used in numerical tests for various LayerWrapper subclasses.
    "test_conv1d_bias.raw.gz",
    "test_conv1d_fixed16_weights.raw.gz",
    "test_conv1d_mask.raw.gz",
    "test_conv1d_weights.raw.gz",
    "test_dilated_bias.raw.gz",
    "test_dilated_fixed16_weights.raw.gz",
    "test_dilated_mask.raw.gz",
    "test_dilated_weights.raw.gz",
    "test_transpose_bias.raw.gz",
    "test_transpose_fixed16_weights.raw.gz",
    "test_transpose_mask.raw.gz",
    "test_transpose_weights.raw.gz",
])
