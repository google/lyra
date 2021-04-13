# Lyra: a generative low bitrate speech codec

## What is Lyra?

[Lyra](https://ai.googleblog.com/2021/02/lyra-new-very-low-bitrate-codec-for.html)
is a high-quality, low-bitrate speech codec that makes voice communication
available even on the slowest networks. To do this it applies traditional codec
techniques while leveraging advances in machine learning (ML) with models
trained on thousands of hours of data to create a novel method for compressing
and transmitting voice signals.

### Overview

The basic architecture of the Lyra codec is quite simple. Features are extracted
from speech every 40ms and are then compressed for transmission at a bitrate of
3kbps. The features themselves are log mel spectrograms, a list of numbers
representing the speech energy in different frequency bands, which have
traditionally been used for their perceptual relevance because they are modeled
after human auditory response. On the other end, a generative model uses those
features to recreate the speech signal.

Lyra harnesses the power of new natural-sounding generative models to maintain
the low bitrate of parametric codecs while achieving high quality, on par with
state-of-the-art waveform codecs used in most streaming and communication
platforms today.

Computational complexity is reduced by using a cheaper recurrent generative
model, a WaveRNN variation, that works at a lower rate, but generates in
parallel multiple signals in different frequency ranges that it later combines
into a single output signal at the desired sample rate. This trick, plus 64-bit
ARM optimizations, enables Lyra to not only run on cloud servers, but also
on-device on mid-range phones, such as Pixel phones, in real time (with a
processing latency of 90ms). This generative model is then trained on thousands
of hours of speech data with speakers in over 70 languages and optimized to
accurately recreate the input audio.

## Prerequisites

There are a few things you'll need to do to set up your computer to build Lyra.

### Common setup

Lyra is built using Google's build system, Bazel. Install it following these
[instructions](https://docs.bazel.build/versions/master/install.html).
Bazel verson 4.0.0 is required, and some Linux distributions may make an older
version available in their application repositories, so make sure you are
using the required version or newer. The latest version can be downloaded via
[Github](https://github.com/bazelbuild/bazel/releases).

Lyra can be built from linux using bazel for an arm android target, or a linux
target.  The android target is optimized for realtime performance.  The linux
target is typically used for development and debugging.

You will also need to install some tools (which may already be on your system).
You can install them with:

```shell
sudo apt update
sudo apt install ninja-build git cmake clang python
```

### Linux requirements

The instructions below are for Ubuntu and have been verified on 20.04.

You will need to install a certain version of clang to ensure ABI compatibility.

```shell
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout 96ef4f307df2

mkdir build_clang
cd build_clang
cmake -G Ninja -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_PROJECTS="clang" -DCMAKE_BUILD_TYPE=release ../llvm
ninja
sudo $(which ninja) install

cd ..
mkdir build_libcxx
cd build_libcxx
cmake -G Ninja -DCMAKE_C_COMPILER=/usr/local/bin/clang -DCMAKE_CXX_COMPILER=/usr/local/bin/clang++ -DLLVM_ENABLE_PROJECTS="libcxx;libcxxabi" -DCMAKE_BUILD_TYPE=release ../llvm
ninja
sudo $(which ninja) install

sudo ldconfig
```

Note: the above will install a particular version of libc++ to /usr/local/lib,
and clang to /usr/local/bin, which the toolchain depends on.

### Android requirements

Building on android requires downloading a specific version of the android NDK
toolchain. If you develop with Android Studio already, you might not need to do
these steps if ANDROID_HOME and ANDROID_NDK_HOME are defined and pointing at the
right version of the NDK.

1. Download the sdk manager from https://developer.android.com/studio
2. Unzip and cd to the directory
3. Check the available packages to install in case they don't match the following steps.

``` shell
bin/sdkmanager  --sdk_root=$HOME/android/sdk --list
```

Some systems will already have the java runtime set up.  But if you see an error
here like `ERROR: JAVA_HOME is not set and no 'java' command could be found
on your PATH.`, this means you need to install the java runtime with `sudo apt
install default-jdk` first. You will also need to add `export
JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64` (type `ls /usr/lib/jvm` to see
which path was installed) to your $HOME/.bashrc and reload it with `source
$HOME/.bashrc`.

4. Install the r21 ndk, android sdk 29, and build tools:

``` shell
bin/sdkmanager  --sdk_root=$HOME/android/sdk --install  "platforms;android-29" "build-tools;29.0.3" "ndk;21.4.7075529"
```

5. Add the following to .bashrc (or export the variables)

``` shell
export ANDROID_NDK_HOME=$HOME/android/sdk/ndk/21.4.7075529
export ANDROID_HOME=$HOME/android/sdk
```

6. Reload .bashrc (with `source $HOME/.bashrc`)

## Building

The building and running process differs slightly depending on the selected
platform.

### Building for Linux

You can build the cc_binaries with the default config.  `encoder_main` is an
example of a file encoder.

```shell
bazel build -c opt :encoder_main
```

You can run `encoder_main` to encode a test .wav file with some speech in it,
specified by `--input_path`.  The `--model_path` flag contains the model data
necessary to encode, and `--output_path` specifies where to write the encoded
(compressed) representation.

```shell
bazel-bin/encoder_main --model_path=wavegru --output_dir=$HOME/temp --input_path=testdata/16khz_sample_000001.wav
```

Similarly, you can build decoder_main and use it on the output of encoder_main
to decode the encoded data back into speech.

```shell
bazel build -c opt :decoder_main
bazel-bin/decoder_main  --model_path=wavegru --output_dir=$HOME/temp/ --encoded_path=$HOME/temp/16khz_sample_000001.lyra
```

### Building for Android

#### Android App
There is an example APK target called `lyra_android_example` that you can build
after you have set up the NDK.

This example is an app with a minimal GUI that has buttons for two options.
One option is to record from the microphone and encode/decode with Lyra so you
can test what Lyra would sound like for your voice. The other option runs a
benchmark that encodes and decodes in the background and prints the timings to
logcat.

```shell
bazel build android_example:lyra_android_example --config=android_arm64 --copt=-DBENCHMARK
adb install bazel-bin/android_example/lyra_android_example.apk
```

After this you should see an app called "Lyra Example App".

You can open it, and you will see a simple TextView that says the benchmark is
running, and when it finishes.

Press "Record from microphone", say a few words (be sure to have your microphone
near your mouth), and then press "Encode and decode to speaker". You should hear
your voice being played back after being coded with Lyra.

If you press 'Benchmark', you should you should see something like the following
in logcat on a Pixel 4 when running the benchmark:

```shell
I  Starting benchmarkDecode()
I  I20210401 11:04:06.898649  6870 lyra_wavegru.h:75] lyra_wavegru running fast multiplication kernels for aarch64.
I  I20210401 11:04:06.900411  6870 layer_wrapper.h:162] |lyra_16khz_ar_to_gates_| layer:  Shape: [3072, 4]. Sparsity: 0
I  I20210401 11:04:07.031975  6870 layer_wrapper.h:162] |lyra_16khz_gru_layer_| layer:  Shape: [3072, 1024]. Sparsity: 0.9375
...
I  I20210401 11:04:26.700160  6870 benchmark_decode_lib.cc:167] Using float arithmetic.
I  I20210401 11:04:26.700352  6870 benchmark_decode_lib.cc:85] conditioning_only stats for generating 2000 frames of audio, max: 506 us, min: 368 us, mean: 391 us, stdev: 10.3923.
I  I20210401 11:04:26.725538  6870 benchmark_decode_lib.cc:85] model_only stats for generating 2000 frames of audio, max: 12690 us, min: 9087 us, mean: 9237 us, stdev: 262.416.
I  I20210401 11:04:26.729460  6870 benchmark_decode_lib.cc:85] combined_model_and_conditioning stats for generating 2000 frames of audio, max: 13173 us, min: 9463 us, mean: 9629 us, stdev: 270.788.
I  Finished benchmarkDecode()
```

This shows that decoding a 25Hz frame (each frame is .04 seconds) takes 9629
microseconds on average (.0096 seconds).  So decoding is performed at around
4.15 (.04/.0096) times faster than realtime.

For even faster decoding, you can use a fixed point representation by building
with `--copt=-DUSE_FIXED16`, although there may be some loss of quality.

To build your own android app, you can either use the cc_library target outputs
to create a .so that you can use in your own build system. Or you can use it
with an [`android_binary`](https://docs.bazel.build/versions/master/be/android.html)
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
# Push the binary and the data it needs, including the model, .wav, and .so files:
adb push bazel-bin/encoder_main /data/local/tmp/
adb push bazel-bin/decoder_main /data/local/tmp/
adb push wavegru/ /data/local/tmp/
adb push testdata/ /data/local/tmp/
adb shell mkdir -p /data/local/tmp/_U_S_S_Csparse_Uinference_Umatrixvector___Ulib_Sandroid_Uarm64
adb push bazel-bin/_solib_arm64-v8a/_U_S_S_Csparse_Uinference_Umatrixvector___Ulib_Sandroid_Uarm64/libsparse_inference.so /data/local/tmp/_U_S_S_Csparse_Uinference_Umatrixvector___Ulib_Sandroid_Uarm64

adb shell
cd /data/local/tmp
./encoder_main --model_path=/data/local/tmp/wavegru --output_dir=/data/local/tmp --input_path=testdata/16khz_sample_000001.wav
./decoder_main --model_path=/data/local/tmp/wavegru --output_dir=/data/local/tmp --encoded_path=16khz_sample_000001.lyra
```

The encoder_main/decoder_main as above should also work.

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

  absl::optional<std::vector<uint8_t>> Encode(
      const absl::Span<const int16_t> audio) override;

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
method. The provided span of int16-formatted samples is assumed to contain 40ms
of data at the sample rate chosen at `Create` time. As long as this condition is
met the `Encode` method returns the encoded packet as a vector of bytes that is
ready to be stored or transmitted over the network.

The rest of the `LyraEncoder` methods are just getters for the different
predetermined parameters.

On the receiving end, `LyraDecoder` can be used to decode the encoded packet
using the following interface:

```cpp
class LyraDecoder : public LyraDecoderInterface {
 public:
  static std::unique_ptr<LyraDecoder> Create(
      int sample_rate_hz, int num_channels, int bitrate,
      const ghc::filesystem::path& model_path);

  bool SetEncodedPacket(absl::Span<const uint8_t> encoded) override;

  absl::optional<std::vector<int16_t>> DecodeSamples(int num_samples) override;

  absl::optional<std::vector<int16_t>> DecodePacketLoss(
      int num_samples) override;

  int sample_rate_hz() const override;

  int num_channels() const override;

  int bitrate() const override;

  int frame_rate() const override;

  bool is_comfort_noise() const override;
};
```

Once again, the static `Create` method instantiates a `LyraDecoder` with the
desired sample rate in Hertz, number of channels and bitrate, as long as those
parameters are supported. Else it returns a `nullptr`. These parameters don't need
to be the same as the ones in `LyraEncoder`. And once again, the `Create` method
also needs to know where the model weights are stored. It also checks that these
weights exist and are compatible with the current Lyra version.

Given a `LyraDecoder`, any packet can be decoded by first feeding it into
`SetEncodedPacket`, which returns true if the provided span of bytes is a valid
Lyra-encoded packet.

Then the int16-formatted samples can be obtained by calling `DecodeSamples`, as
long as the total number of samples obtained this way between any two calls to
`SetEncodedPacket` is less than 40ms of data at the sample rate chose at
`Create` time.

If there isn't a packet available, but samples still need to be generated,
`DecodePacketLoss` can be used, which doesn't have a restriction on the number
of samples.

In those cases, the decoder might switch to a comfort noise generation mode,
which can be checked using `is_confort_noise`.

The rest of the `LyraDecoder` methods are just getters for the different
predetermined parameters.

For an example on how to use `LyraEncoder` and `LyraDecoder` to encode and
decode a stream of audio, please refer to the [integration
test](lyra_integration_test.cc).

## License

Use of this source code is governed by a Apache v2.0 license that can be found
in the LICENSE file.

Please note that there is a closed-source kernel used for math operations that
is linked via a shared object called libsparse_inference.so. We provide the
libsparse_inference.so library to be linked, but are unable to provide source
for it. This is the reason that a specific toolchain/compiler is required.

## Papers

1. Kleijn, W. B., Lim, F. S., Luebs, A., Skoglund, J., Stimberg, F., Wang, Q., &
   Walters, T. C. (2018, April). [Wavenet based low rate speech coding](https://arxiv.org/pdf/1712.01120).
   In 2018 IEEE international conference on acoustics, speech and signal
   processing (ICASSP) (pp. 676-680). IEEE.
2. Denton, T., Luebs, A., Lim, F. S., Storus, A., Yeh, H., Kleijn, W. B., &
   Skoglund, J. (2021). [Handling Background Noise in Neural Speech Generation](https://arxiv.org/pdf/2102.11906).
   arXiv preprint arXiv:2102.11906.
3. Kleijn, W. B., Storus, A., Chinen, M., Denton, T., Lim, F. S., Luebs, A., ...
   & Yeh, H. (2021). [Generative Speech Coding with Predictive Variance
   Regularization](https://arxiv.org/pdf/2102.09660). arXiv preprint
   arXiv:2102.09660.
