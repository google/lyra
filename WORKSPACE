########################
# Platform Independent #
########################

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository", "new_git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# GoogleTest/GoogleMock framework.
git_repository(
    name = "com_google_googletest",
    remote = "https://github.com/google/googletest.git",
    tag = "release-1.10.0",
)

# Google benchmark.
http_archive(
    name = "com_github_google_benchmark",
    urls = ["https://github.com/google/benchmark/archive/bf585a2789e30585b4e3ce6baf11ef2750b54677.zip"],  # 2020-11-26T11:14:03Z
    strip_prefix = "benchmark-bf585a2789e30585b4e3ce6baf11ef2750b54677",
    sha256 = "2a778d821997df7d8646c9c59b8edb9a573a6e04c534c01892a40aa524a7b68c",
)

# proto_library, cc_proto_library, and java_proto_library rules implicitly
# depend on @com_google_protobuf for protoc and proto runtimes.
# This statement defines the @com_google_protobuf repo.
git_repository(
    name = "com_google_protobuf",
    remote = "https://github.com/protocolbuffers/protobuf.git",
    tag = "v3.15.4",
)

load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")
protobuf_deps()

# Google Abseil Libs
git_repository(
    name = "com_google_absl",
    remote = "https://github.com/abseil/abseil-cpp.git",
    tag = "20211102.0",
    # Remove after https://github.com/abseil/abseil-cpp/issues/326 is solved.
    patches = [
        "@//patches:com_google_absl_f863b622fe13612433fdf43f76547d5edda0c93001.diff"
    ],
    patch_args = [
        "-p1",
    ]
)

# Filesystem
# The new_* prefix is used because it is not a bazel project and there is
# no BUILD file in that repo.
FILESYSTEM_BUILD = """
cc_library(
  name = "filesystem",
  hdrs = glob(["include/ghc/*"]),
  visibility = ["//visibility:public"],
)
"""

new_git_repository(
    name = "gulrak_filesystem",
    remote = "https://github.com/gulrak/filesystem.git",
    tag = "v1.3.6",
    build_file_content = FILESYSTEM_BUILD
)

# Audio DSP
git_repository(
    name = "com_google_audio_dsp",
    # TODO(b/231448719) use main google repo after merging PR for TF eigen compatibility.
    remote = "https://github.com/mchinen/multichannel-audio-tools.git",
    # There are no tags for this repo, we are synced to bleeding edge.
    commit = "14a45c5a7c965e5ef01fe537bd816ce10a247813",
    repo_mapping = {
        "@com_github_glog_glog" : "@com_google_glog",
        "@eigen3": "@eigen_archive"
    }
)

# Transitive dependencies of Audio DSP.
# Note: eigen is used by Audio DSP, but provided through tensorflow workspace functions.

http_archive(
    name = "fft2d",
    build_file = "fft2d.BUILD",
    sha256 = "ada7e99087c4ed477bfdf11413f2ba8db8a840ba9bbf8ac94f4f3972e2a7cec9",
    urls = [
        "http://www.kurims.kyoto-u.ac.jp/~ooura/fft2d.tgz",
    ],
)

# Google logging
git_repository(
    name = "com_google_glog",
    remote = "https://github.com/google/glog.git",
    branch = "master"
)
# Dependency for glog
git_repository(
    name = "gflags",
    remote = "https://github.com/mchinen/gflags.git",
    branch = "android_linking_fix"
)

# Bazel/build rules

http_archive(
    name = "bazel_skylib",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.0.2/bazel-skylib-1.0.2.tar.gz",
        "https://github.com/bazelbuild/bazel-skylib/releases/download/1.0.2/bazel-skylib-1.0.2.tar.gz",
    ],
    sha256 = "97e70364e9249702246c0e9444bccdc4b847bed1eb03c5a3ece4f83dfe6abc44",
)
load("@bazel_skylib//:workspace.bzl", "bazel_skylib_workspace")
bazel_skylib_workspace()

load("//:android_configure.bzl", "android_configure")
android_configure(name = "local_config_android")

load("@local_config_android//:android_configure.bzl", "android_workspace")
android_workspace()

http_archive(
    name = "rules_android",
    sha256 = "cd06d15dd8bb59926e4d65f9003bfc20f9da4b2519985c27e190cddc8b7a7806",
    strip_prefix = "rules_android-0.1.1",
    urls = ["https://github.com/bazelbuild/rules_android/archive/v0.1.1.zip"],
)

# Google Maven Repository
# See https://github.com/android/android-test/blob/master/WORKSPACE for examples
# of importing android deps.
# The specific versions can be found in:
# https://github.com/android/android-test/blob/master/build_extensions/axt_versions.bzl
# and
# https://developer.android.com/jetpack/androidx/releases/
# This allows us to use "@maven//some_android_package" deps for imports.

RULES_JVM_EXTERNAL_TAG = "4.0"
RULES_JVM_EXTERNAL_SHA = "31701ad93dbfe544d597dbe62c9a1fdd76d81d8a9150c2bf1ecf928ecdf97169"

http_archive(
    name = "rules_jvm_external",
    strip_prefix = "rules_jvm_external-%s" % RULES_JVM_EXTERNAL_TAG,
    sha256 = RULES_JVM_EXTERNAL_SHA,
    url = "https://github.com/bazelbuild/rules_jvm_external/archive/%s.zip" % RULES_JVM_EXTERNAL_TAG,
)

load("@rules_jvm_external//:defs.bzl", "maven_install")

maven_install(
    artifacts = [
        "androidx.annotation:annotation:1.2.0",
        "androidx.appcompat:appcompat:1.3.1",
        "androidx.core:core:1.6.0",
        "androidx.constraintlayout:constraintlayout:2.1.1"
    ],
    repositories = [
        "https://maven.google.com",
        "https://repo1.maven.org/maven2",
    ],
)


# Begin Tensorflow WORKSPACE subset required for TFLite

git_repository(
    name = "org_tensorflow",
    remote = "https://github.com/tensorflow/tensorflow.git",
    # Below is reproducible and equivalent to `tag = "v2.11.0"`
    commit = "d5b57ca93e506df258271ea00fc29cf98383a374",
    shallow_since = "1668561432 -0800"
)

# Check bazel version requirement, which is stricter than TensorFlow's.
load("@bazel_skylib//lib:versions.bzl", "versions")
versions.check("3.7.2")

# TF WORKSPACE Loading functions
# This section uses a subset of the tensorflow WORKSPACE loading by reusing its contents.
# There are four workspace() functions create repos for the dependencies.
# TF's loading is very complicated, and we only need a subset for TFLite.
# If we use the full TF loading sequence, we also run into conflicts and errors on some platforms.

load("@org_tensorflow//tensorflow:workspace3.bzl", "workspace")
workspace()

load("@org_tensorflow//tensorflow:workspace2.bzl", workspace2 = "workspace")
workspace2()

# End Tensorflow WORKSPACE subset required for TFLite
