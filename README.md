# Lyra: a generative low bitrate speech codec

## What is Lyra?

[Lyra](https://ai.googleblog.com/2021/08/soundstream-end-to-end-neural-audio.html)
is a high-quality, low-bitrate speech codec that makes voice communication
available even on the slowest networks. To do this it applies traditional codec
techniques while leveraging advances in machine learning (ML) with models
trained on thousands of hours of data to create a novel method for compressing
and transmitting voice signals.

### Overview

The basic architecture of the Lyra codec is quite simple. Features are extracted
from speech every 20ms and are then compressed for transmission at a desired
bitrate between 3.2kbps and 9.2kbps. On the other end, a generative model uses
those features to recreate the speech signal.

Lyra harnesses the power of new natural-sounding generative models to maintain
the low bitrate of parametric codecs while achieving high quality, on par with
state-of-the-art waveform codecs used in most streaming and communication
platforms today.

Computational complexity is reduced by using a cheaper convolutional generative
model called SoundStream, which enables Lyra to not only run on cloud servers,
but also on-device on low-end phones in real time (with a processing latency of
20ms). This whole system is then trained end-to-end on thousands of hours of
speech data with speakers in over 90 languages and optimized to accurately
recreate the input audio.

Lyra is supported on Android, Linux, Mac and Windows.

## Prerequisites

There are a few things you'll need to do to set up your computer to build Lyra.

### Common setup

Lyra is built using Google's build system, Bazel. Install it following these
[instructions](https://docs.bazel.build/versions/master/install.html). Bazel
verson 5.0.0 is required, and some Linux distributions may make an older version
available in their application repositories, so make sure you are using the
required version or newer. The latest version can be downloaded via
[Github](https://github.com/bazelbuild/bazel/releases).

You will also need python3 and numpy installed.

Lyra can be built from Linux using Bazel for an ARM Android target, or a Linux
target, as well as Mac and Windows for native targets.

### Android requirements

Building on android requires downloading a specific version of the android NDK
toolchain. If you develop with Android Studio already, you might not need to do
these steps if ANDROID_HOME and ANDROID_NDK_HOME are defined and pointing at the
right version of the NDK.

1.  Download command line tools from https://developer.android.com/studio
2.  Unzip and cd to the directory
3.  Check the available packages to install in case they don't match the
    following steps.

    ```shell
    bin/sdkmanager  --sdk_root=$HOME/android/sdk --list
    ```

    Some systems will already have the java runtime set up. But if you see an
    error here like `ERROR: JAVA_HOME is not set and no 'java' command could be
    found on your PATH.`, this means you need to install the java runtime with
    `sudo apt install default-jdk` first. You will also need to add `export
    JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64` (type `ls /usr/lib/jvm` to see
    which path was installed) to your $HOME/.bashrc and reload it with `source
    $HOME/.bashrc`.

4.  Install the r21 ndk, android sdk 30, and build tools:

    ```shell
    bin/sdkmanager  --sdk_root=$HOME/android/sdk --install  "platforms;android-30" "build-tools;30.0.3" "ndk;21.4.7075529"
    ```

5.  Add the following to .bashrc (or export the variables)

    ```shell
    export ANDROID_NDK_HOME=$HOME/android/sdk/ndk/21.4.7075529
    export ANDROID_HOME=$HOME/android/sdk
    ```

6.  Reload .bashrc (with `source $HOME/.bashrc`)

## Building

The building and running process differs slightly depending on the selected
platform.

### Building for Linux

You can build the cc_binaries with the default config. `encoder_main` is an
example of a file encoder.

```shell
bazel build -c opt :encoder_main
```

You can run `encoder_main` to encode a test .wav file with some speech in it,
specified by `--input_path`. The `--output_dir` specifies where to write the
encoded (compressed) representation, and the desired bitrate can be specified
using the `--bitrate` flag.

```shell
bazel-bin/encoder_main --input_path=testdata/sample1_16kHz.wav --output_dir=$HOME/temp --bitrate=3200
```

Similarly, you can build decoder_main and use it on the output of encoder_main
to decode the encoded data back into speech.

```shell
bazel build -c opt :decoder_main
bazel-bin/decoder_main --encoded_path=$HOME/temp/sample1_16kHz.lyra --output_dir=$HOME/temp/ --bitrate=3200
```

Note: the default Bazel toolchain is automatically configured and likely uses
gcc/libstdc++ on Linux. This should be satisfactory for most users, but will
differ from the NDK toolchain, which uses clang/libc++. To use a custom clang
toolchain on Linux, see toolchain/README.md and .bazelrc.

### Building for Android

#### Android App

There is an example APK target called `lyra_android_example` that you can build
after you have set up the NDK.

This example is an app with a minimal GUI that has buttons for two options. One
option is to record from the microphone and encode/decode with Lyra so you can
test what Lyra would sound like for your voice. The other option runs a
benchmark that encodes and decodes in the background and prints the timings to
logcat.

```shell
bazel build -c opt android_example:lyra_android_example --config=android_arm64 --copt=-DBENCHMARK
adb install bazel-bin/android_example/lyra_android_example.apk
```

After this you should see an app called "Lyra Example App".

You can open it, and you will see a simple TextView that says the benchmark is
running, and when it finishes.

Press "Record from microphone", say a few words, and then press "Encode and
decode to speaker". You should hear your voice being played back after being
coded with Lyra.

If you press 'Benchmark', you should see something like the following in logcat
on a Pixel 6 Pro when running the benchmark:

```shell
lyra_benchmark:  feature_extractor:  max: 0.685 ms  min: 0.206 ms  mean: 0.219 ms  stdev: 0.000 ms
lyra_benchmark: quantizer_quantize:  max: 0.250 ms  min: 0.076 ms  mean: 0.082 ms  stdev: 0.000 ms
lyra_benchmark:   quantizer_decode:  max: 0.152 ms  min: 0.027 ms  mean: 0.030 ms  stdev: 0.001 ms
lyra_benchmark:       model_decode:  max: 0.560 ms  min: 0.223 ms  mean: 0.237 ms  stdev: 0.000 ms
lyra_benchmark:              total:  max: 1.560 ms  min: 0.541 ms  mean: 0.569 ms  stdev: 0.005 ms
```

This shows that decoding a 50Hz frame (each frame is 20 milliseconds) takes
0.569 milliseconds on average. So decoding is performed at around 35 (20/0.569)
times faster than realtime.

To build your own android app, you can either use the cc_library target outputs
to create a .so that you can use in your own build system. Or you can use it
with an
[`android_binary`](https://docs.bazel.build/versions/master/be/android.html)
rule within bazel to create an .apk file as in this example.

There is a tutorial on building for android with Bazel in the
[bazel docs](https://docs.bazel.build/versions/master/android-ndk.html).

#### Android command-line binaries

There are also the binary targets that you can use to experiment with encoding
and decoding .wav files.

You can build the example cc_binary targets with:

```shell
bazel build -c opt :encoder_main --config=android_arm64
bazel build -c opt :decoder_main --config=android_arm64
```

This builds an executable binary that can be run on android 64-bit arm devices
(not an android app). You can then push it to your android device and run it as
a binary through the shell.

```shell
# Push the binary and the data it needs, including the model and .wav files:
adb push bazel-bin/encoder_main /data/local/tmp/
adb push bazel-bin/decoder_main /data/local/tmp/
adb push model_coeffs/ /data/local/tmp/
adb push testdata/ /data/local/tmp/

adb shell
cd /data/local/tmp
./encoder_main --model_path=/data/local/tmp/model_coeffs --output_dir=/data/local/tmp --input_path=testdata/sample1_16kHz.wav
./decoder_main --model_path=/data/local/tmp/model_coeffs --output_dir=/data/local/tmp --encoded_path=sample1_16kHz.lyra
```

The encoder_main/decoder_main as above should also work.

### Building for Mac

You will need to install the XCode command line tools in addition to the
prerequisites common to all platforms. XCode setup is a required step for using
Bazel on Mac. See this [guide](https://bazel.build/install/os-x) for how to
install XCode command line tools. Lyra has been built successfully using XCode
13.3.

You can follow the instructions in the [Building for Linux](#building-for-linux)
section once this is completed.

### Building for Windows

You will need to install Build Tools for Visual Studio 2019 in addition to the
prerequisites common to all platforms. Visual Studio setup is a required step
for building C++ for Bazel on Windows. See this
[guide](https://bazel.build/install/windows) for how to install MSVC. You may
also need to install python 3 support, which is also described in the guide.

You can follow the instructions in the [Building for Linux](#building-for-linux)
section once this is completed.

## API

For integrating Lyra into any project only two APIs are relevant:
[LyraEncoder](lyra_encoder.h) and [LyraDecoder](lyra_decoder.h).

> DISCLAIMER: At this time Lyra's API and bit-stream are **not** guaranteed to
> be stable and might change in future versions of the code.

On the sending side, `LyraEncoder` can be used to encode an audio stream using
the following interface:

```cpp
class LyraEncoder : public LyraEncoderInterface {
 public:
  static std::unique_ptr<LyraEncoder> Create(
      int sample_rate_hz, int num_channels, int bitrate, bool enable_dtx,
      const ghc::filesystem::path& model_path);

  std::optional<std::vector<uint8_t>> Encode(
      const absl::Span<const int16_t> audio) override;

  bool set_bitrate(int bitrate) override;

  int sample_rate_hz() const override;

  int num_channels() const override;

  int bitrate() const override;

  int frame_rate() const override;
};
```

The static `Create` method instantiates a `LyraEncoder` with the desired sample
rate in Hertz, number of channels and bitrate, as long as those parameters are
supported. Else it returns a nullptr. The `Create` method also needs to know if
DTX should be enabled and where the model weights are stored. It also checks
that these weights exist and are compatible with the current Lyra version.

Given a `LyraEncoder`, any audio stream can be compressed using the `Encode`
method. The provided span of int16-formatted samples is assumed to contain 20ms
of data at the sample rate chosen at `Create` time. As long as this condition is
met the `Encode` method returns the encoded packet as a vector of bytes that is
ready to be stored or transmitted over the network.

The bitrate can be dynamically modified using the `set_bitrate` setter. It
returns true if the desired bitrate is supported and correctly set.

The rest of the `LyraEncoder` methods are just getters for the different
predetermined parameters.

On the receiving end, `LyraDecoder` can be used to decode the encoded packet
using the following interface:

```cpp
class LyraDecoder : public LyraDecoderInterface {
 public:
  static std::unique_ptr<LyraDecoder> Create(
      int sample_rate_hz, int num_channels,
      const ghc::filesystem::path& model_path);

  bool SetEncodedPacket(absl::Span<const uint8_t> encoded) override;

  std::optional<std::vector<int16_t>> DecodeSamples(int num_samples) override;

  int sample_rate_hz() const override;

  int num_channels() const override;

  int frame_rate() const override;

  bool is_comfort_noise() const override;
};
```

Once again, the static `Create` method instantiates a `LyraDecoder` with the
desired sample rate in Hertz and number of channels, as long as those parameters
are supported. Else it returns a `nullptr`. These parameters don't need to be
the same as the ones in `LyraEncoder`. And once again, the `Create` method also
needs to know where the model weights are stored. It also checks that these
weights exist and are compatible with the current Lyra version.

Given a `LyraDecoder`, any packet can be decoded by first feeding it into
`SetEncodedPacket`, which returns true if the provided span of bytes is a valid
Lyra-encoded packet.

Then the int16-formatted samples can be obtained by calling `DecodeSamples`. If
there isn't a packet available, but samples still need to be generated, the
decoder might switch to a comfort noise generation mode, which can be checked
using `is_comfort_noise`.

The rest of the `LyraDecoder` methods are just getters for the different
predetermined parameters.

For an example on how to use `LyraEncoder` and `LyraDecoder` to encode and
decode a stream of audio, please refer to the
[integration test](lyra_integration_test.cc).

## License

Use of this source code is governed by a Apache v2.0 license that can be found
in the LICENSE file.

## Papers

1.  Kleijn, W. B., Lim, F. S., Luebs, A., Skoglund, J., Stimberg, F., Wang, Q.,
    & Walters, T. C. (2018, April).
    [Wavenet based low rate speech coding](https://arxiv.org/pdf/1712.01120). In
    2018 IEEE international conference on acoustics, speech and signal
    processing (ICASSP) (pp. 676-680). IEEE.
2.  Denton, T., Luebs, A., Chinen, M., Lim, F. S., Storus, A., Yeh, H., Kleijn,
    W. B., & Skoglund, J. (2020, November).
    [Handling Background Noise in Neural Speech Generation](https://arxiv.org/pdf/2102.11906).
    In 2020 54th Asilomar Conference on Signals, Systems, and Computers (pp.
    667-671). IEEE.
3.  Kleijn, W. B., Storus, A., Chinen, M., Denton, T., Lim, F. S., Luebs, A.,
    Skoglund, J., & Yeh, H. (2021, June).
    [Generative speech coding with predictive variance regularization](https://arxiv.org/pdf/2102.09660).
    In ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and
    Signal Processing (ICASSP) (pp. 6478-6482). IEEE.
4.  Zeghidour, N., Luebs, A., Omran, A., Skoglund, J., & Tagliasacchi, M.
    (2021).
    [SoundStream: An end-to-end neural audio codec](https://arxiv.org/pdf/2107.03312).
    IEEE/ACM Transactions on Audio, Speech, and Language Processing.
