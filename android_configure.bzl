"""Repository rule for Android SDK and NDK autoconfiguration.
This rule is a no-op unless the required android environment variables are set.
"""

# Based on https://github.com/envoyproxy/envoy-mobile/pull/2039
# Workaround for https://github.com/bazelbuild/bazel/issues/14260

def _android_autoconf_impl(repository_ctx):
    sdk_rule = ""
    if repository_ctx.os.environ.get("ANDROID_HOME"):
        sdk_rule = """
    native.android_sdk_repository(
        name="androidsdk",
        api_level=30,
        build_tools_version="30.0.3",
    )
"""

    ndk_rule = ""
    if repository_ctx.os.environ.get("ANDROID_NDK_HOME"):
        ndk_rule = """
    native.android_ndk_repository(
        name="androidndk",
        api_level=30,
    )
"""

    if ndk_rule == "" and sdk_rule == "":
        sdk_rule = "pass"

    repository_ctx.file("BUILD.bazel", "")
    repository_ctx.file("android_configure.bzl", """
def android_workspace():
    {}
    {}
""".format(sdk_rule, ndk_rule))

android_configure = repository_rule(
    implementation = _android_autoconf_impl,
)
