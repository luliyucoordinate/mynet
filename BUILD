load("@rules_foreign_cc//tools/build_defs:cmake.bzl", "cmake_external")

cmake_external(
    name = "openblas",
    cache_entries = {
        "NOFORTRAN": "on",
        "BUILD_WITHOUT_LAPACK": "no",
    },
    lib_source = "@openblas//:all",
    static_libraries = ["libopenblas.a"],
    visibility = ["//visibility:public"],
)
