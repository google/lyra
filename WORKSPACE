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
    branch = "lts_2020_09_23",
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
    remote = "https://github.com/google/multichannel-audio-tools.git",
    # There are no tags for this repo, we are synced to bleeding edge.
    branch = "master",
    repo_mapping = {
        "@com_github_glog_glog" : "@com_google_glog"
    }
)

# Transitive dependencies of Audio DSP.
http_archive(
    name = "eigen_archive",
    build_file = "eigen.BUILD",
    sha256 = "f3d69ac773ecaf3602cb940040390d4e71a501bb145ca9e01ce5464cf6d4eb68",
    strip_prefix = "eigen-eigen-049af2f56331",
    urls = [
        "http://mirror.tensorflow.org/bitbucket.org/eigen/eigen/get/049af2f56331.tar.gz",
        "https://bitbucket.org/eigen/eigen/get/049af2f56331.tar.gz",
    ],
)

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
    name = "com_github_gflags_gflags",
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

android_sdk_repository(
    name = "androidsdk",
    api_level = 29,
    build_tools_version = "29.0.3"
)

android_ndk_repository(
    name = "androidndk",
    api_level = 29
)

http_archive(
    name = "rules_android",
    sha256 = "cd06d15dd8bb59926e4d65f9003bfc20f9da4b2519985c27e190cddc8b7a7806",
    strip_prefix = "rules_android-0.1.1",
    urls = ["https://github.com/bazelbuild/rules_android/archive/v0.1.1.zip"],
)

# Google Maven Repository
GMAVEN_TAG = "20180625-1"

http_archive(
    name = "gmaven_rules",
    strip_prefix = "gmaven_rules-%s" % GMAVEN_TAG,
    url = "https://github.com/bazelbuild/gmaven_rules/archive/%s.tar.gz" % GMAVEN_TAG,
)

load("@gmaven_rules//:gmaven.bzl", "gmaven_rules")

gmaven_rules()
