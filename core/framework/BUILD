cc_library(
    name = "tensor",
    srcs = ["tensor.cc"],
    hdrs = ["tensor.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        ":common",
        "//core/schema:mynet_fbs",
    ],
)

cc_test(
    name = "syncedmem_test",
    srcs = [
        "syncedmem_test.cc",
    ],
    deps = [
        ":syncedmem",
    ],
)

cc_library(
    name = "syncedmem",
    srcs = [
        "syncedmem.cc",
    ],
    hdrs = [
        "syncedmem.hpp",
    ],
    deps = [
        ":common",
    ],
)

cc_library(
    name = "common",
    srcs = ["common.cc"],
    hdrs = ["common.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_github_gflags_gflags//:gflags",
        "@com_github_google_glog//:glog",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "math_functions",
    srcs = [
        "math_functions.cc",
    ],
    hdrs = [
        "math_functions.hpp",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":common",
        "//:openblas",
    ],
)
