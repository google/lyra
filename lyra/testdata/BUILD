package(default_visibility = ["//:__subpackages__"])

licenses(["notice"])

exports_files([
    # Wav files taken from VCTK and AMI datasets.
    # Suffix describes sample rate of file.
    "sample1_8kHz.wav",
    "sample1_16kHz.wav",
    "sample1_32kHz.wav",
    "sample1_48kHz.wav",
    "sample2_8kHz.wav",
    "sample2_16kHz.wav",
    "sample2_32kHz.wav",
    "sample2_48kHz.wav",
    # Not a valid Wav file.
    "invalid.wav",
    # Binary file with 9 bytes of data.
    "incomplete_encoded_packet.lyra",
    # Binary file with 16 bytes of data, with sample rate set to 16khz.
    # To change the size of one packet, type the following command on the terminal:
    # 'truncate -s 16 one_encoded_packet_16khz.lyra'
    "one_encoded_packet_16khz.lyra",
    # Binary file with 32 bytes of data. Sample rate of both packets set to 16khz.
    "two_encoded_packets_16khz.lyra",
    # Empty file.
    "no_encoded_packet.lyra",
])
