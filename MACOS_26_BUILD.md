# Experimental macOS 26 build

This opt-in patch is for Lyra v1.3.2 on Apple Silicon running macOS 26 with
Xcode 26.4. It was validated by building the native `encoder_main` and
`decoder_main` targets.

## Compilation issues addressed

- Bazel 5.3.2's `wrapped_clang` helper is rejected by macOS 26 because it lacks
  an `LC_UUID` load command.
- The moving `glog` dependency expects `@gflags`, while Lyra declares the
  repository under a historical name.
- Lyra's Bazel Skylib 1.0.2 metadata is incompatible with Bazel 6 platform
  constraints.
- The zlib revision pulled through Protobuf redefines `fdopen` before the
  current macOS SDK declares it.
- TensorFlow Lite 2.11 passes `std::abs<float>` in a form rejected by Xcode
  26's C++ library.

## Fix provided

`patches/macos_26_build.patch` updates Bazel to 6.6.0, updates Bazel Skylib,
adds the dependency mapping, applies the macOS-only zlib workaround, and adds
a small TensorFlow Lite compatibility patch.

Apply it only to a clean Lyra v1.3.2 checkout:

```bash
git apply --check patches/macos_26_build.patch
git apply patches/macos_26_build.patch
bazel build -c opt lyra/cli_example:encoder_main
bazel build -c opt lyra/cli_example:decoder_main
```

To discard the opt-in changes:

```bash
git apply -R patches/macos_26_build.patch
```

## Compatibility

This patch is experimental and has only been validated for the native macOS
26 Apple-Silicon build. It does not change any checkout unless explicitly
applied. Linux, Android, and Windows builds have not been validated with the
patched Bazel and dependency versions.
