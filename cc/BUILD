load("@rules_foreign_cc//tools/build_defs:cmake.bzl", "cmake_external")

cc_test(
    name = "tensor_test",
    srcs = ["tensor_test.cc"],
    deps = [
        ":tensor",
        ":mynet_test_main",
    ],
)

cc_library(
    name = "math_functions",
    srcs = [
        "math_functions.cc",
    ],
    hdrs = ["math_functions.hpp"],
    deps = [
        ":common",
        ":eigen",
    ]
)

cc_library(
    name = "mynet_test_main",
    srcs = [
        "mynet_test_main.cc",
    ],
    hdrs = ["mynet_test_main.hpp"],
    deps = [
        ":common",
    ]
)

cmake_external(
   name = "openblas",
   cache_entries = {
       "NOFORTRAN": "on",
       "BUILD_WITHOUT_LAPACK": "no",
   },
   lib_source = "@openblas//:all",
   static_libraries = ["libopenblas.a"],
)

cmake_external(
   name = "eigen",
   cache_entries = {
       "BLA_VENDOR": "OpenBLAS",
       "BLAS_LIBRARIES": "$EXT_BUILD_DEPS/openblas/lib/libopenblas.a",
   },
   headers_only = True,
   lib_source = "@eigen//:all",
   # Dependency on other cmake_external rule; can also depend on cc_import, cc_library rules
   deps = [":openblas"],
)

cc_library(
    name = "tensor",
    srcs = [
        "tensor.cc",
    ],
    hdrs = ["tensor.hpp"],
    deps = [
        ":common",
        ":syncedmem",
        ":math_functions",
        "//protobuf:mynet_cc_proto",
    ]
)

cc_test(
    name = "filler_test",
    srcs = ["filler_test.cc"],
    deps = [
        ":tensor",
        ":filler",
    ],
)

cc_library(
    name = "filler",
    hdrs = ["filler.hpp"],
    deps = [
        ":common",
        ":eigen",
    ]
)

cc_test(
    name = "syncedmem_test",
    srcs = ["syncedmem_test.cc"],
    deps = [
        ":syncedmem"
    ],
)

cc_library(
    name = "syncedmem",
    srcs = [
        "syncedmem.cc",
        "common.hpp",
    ],
    hdrs = ["syncedmem.hpp"],
    deps = [
        ":common",
    ]
)

cc_library(
    name = "common",
    srcs = ["common.cc"],
    hdrs = ["common.hpp"],
    deps = [
        "@com_google_googletest//:gtest_main",
        "@com_github_gflags_gflags//:gflags",
        "@com_github_google_glog//:glog",
    ]
)