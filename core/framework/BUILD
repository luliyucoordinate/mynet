cc_test(
    name = "ops_factory_test",
    srcs = ["ops_factory_test.cc"],
    copts = [
        "-std=c++17",
    ],
    deps = [
        ":mynet_test_main",
        ":ops_factory",
        "//core/schema:mynet_fbs",
    ],
)

cc_library(
    name = "ops_factory",
    srcs = [
        "ops_factory.cc",
    ],
    hdrs = ["ops_factory.hpp"],
    copts = [
        "-std=c++17",
    ],
    deps = [
        ":common",
        ":ops",
        "//core/kernels:conv_ops",
    ],
)

cc_library(
    name = "ops",
    srcs = [
        "ops.cc",
    ],
    hdrs = ["ops.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        ":common",
        ":math_functions",
        ":tensor",
        "//core/schema:mynet_fbs",
    ],
)

cc_test(
    name = "tensor_test",
    srcs = ["tensor_test.cc"],
    deps = [
        ":filler",
        ":mynet_test_main",
        ":tensor",
    ],
)

cc_library(
    name = "math_functions",
    srcs = [
        "math_functions.cc",
    ],
    hdrs = ["math_functions.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        ":common",
        "//:openblas",
    ],
)

cc_library(
    name = "mynet_test_main",
    srcs = [
        "mynet_test_main.cc",
    ],
    hdrs = ["mynet_test_main.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        ":common",
    ],
)

cc_library(
    name = "tensor",
    srcs = [
        "tensor.cc",
    ],
    hdrs = ["tensor.hpp"],
    copts = [
        "-std=c++17",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":common",
        ":math_functions",
        ":syncedmem",
        "//core/schema:mynet_fbs",
    ],
)

cc_test(
    name = "filler_test",
    srcs = ["filler_test.cc"],
    deps = [
        ":common",
        ":filler",
        ":mynet_test_main",
        "//core/schema:mynet_fbs",
    ],
)

cc_library(
    name = "filler",
    hdrs = ["filler.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        ":math_functions",
        ":tensor",
    ],
)

cc_test(
    name = "syncedmem_test",
    srcs = ["syncedmem_test.cc"],
    deps = [
        ":syncedmem",
    ],
)

cc_library(
    name = "syncedmem",
    srcs = [
        "common.hpp",
        "syncedmem.cc",
    ],
    hdrs = ["syncedmem.hpp"],
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
