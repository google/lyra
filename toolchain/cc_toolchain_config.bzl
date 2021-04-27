"""A clang-based c++ toolchain for linux builds.

 See the top-level README.md to set up clang/libc++.
"""

load("@bazel_tools//tools/build_defs/cc:action_names.bzl", "ACTION_NAMES")
load(
    "@bazel_tools//tools/cpp:cc_toolchain_config_lib.bzl",
    "feature",
    "flag_group",
    "flag_set",
    "tool_path",
    "with_feature_set",
)

all_link_actions = [
    ACTION_NAMES.cpp_link_executable,
    ACTION_NAMES.cpp_link_dynamic_library,
    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
]

all_compiler_actions = [
    ACTION_NAMES.cpp_compile,
    ACTION_NAMES.c_compile,
]

def _impl(ctx):
    tool_paths = [
        tool_path(
            name = "gcc",
            path = "/usr/local/bin/clang",
        ),
        tool_path(
            name = "ld",
            path = "/usr/bin/ld",
        ),
        tool_path(
            name = "ar",
            path = "/usr/bin/ar",
        ),
        tool_path(
            name = "cpp",
            path = "/bin/false",
        ),
        tool_path(
            name = "gcov",
            path = "/bin/false",
        ),
        tool_path(
            name = "nm",
            path = "/bin/false",
        ),
        tool_path(
            name = "objdump",
            path = "/bin/false",
        ),
        tool_path(
            name = "strip",
            path = "/bin/false",
        ),
    ]

    features = [
        feature(name = "opt"),
        feature(name = "dbg"),
        feature(
            name = "default_linker_flags",
            enabled = True,
            flag_sets = [
                flag_set(
                    actions = all_link_actions,
                    flag_groups = ([
                        flag_group(
                            flags = [
                                "-L/usr/local/lib",
                                # Add the math library.
                                "-lm",
                                "-lc++",
                            ],
                        ),
                    ]),
                ),
            ],
        ),
        feature(
            name = "default_cpp_compiler_flags",
            enabled = True,
            flag_sets = [
                flag_set(
                    actions = [ACTION_NAMES.cpp_compile],
                    flag_groups = ([
                        flag_group(
                            flags = [
                                "-nostdinc++",
                                "-std=gnu++17",
                                "-I/usr/local/include/c++/v1/",
                            ],
                        ),
                    ]),
                ),
            ],
        ),
        feature(
            name = "default_compiler_flags",
            enabled = True,
            flag_sets = [
                flag_set(
                    actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                    flag_groups = ([
                        flag_group(
                            flags = [
                                "-fPIC",
                            ],
                        ),
                    ]),
                ),
                flag_set(
                    actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                    flag_groups = ([
                        flag_group(
                            flags = [
                                "-O3",
                                "-ffunction-sections",
                                "-fdata-sections",
                                "-falign-functions=32",
                                "-DNDEBUG",
                            ],
                        ),
                    ]),
                    with_features = [with_feature_set(features = ["opt"])],
                ),
                flag_set(
                    actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                    flag_groups = ([
                        flag_group(
                            flags = [
                                "-g",
                            ],
                        ),
                    ]),
                    with_features = [with_feature_set(features = ["dbg"])],
                ),
            ],
        ),
    ]

    return cc_common.create_cc_toolchain_config_info(
        ctx = ctx,
        features = features,
        cxx_builtin_include_directories = [
            "/usr/local/include/c++/v1/",
            "/usr/local/lib/clang/12.0.0/include",
            "/usr/include",
        ],
        toolchain_identifier = "local",
        host_system_name = "local",
        target_system_name = "local",
        target_cpu = "k8",
        target_libc = "unknown",
        compiler = "clang",
        abi_version = "unknown",
        abi_libc_version = "unknown",
        tool_paths = tool_paths,
    )

cc_toolchain_config = rule(
    implementation = _impl,
    attrs = {},
    provides = [CcToolchainConfigInfo],
)
