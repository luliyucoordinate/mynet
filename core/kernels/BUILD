cc_test(
    name = "conv_ops_test",
    srcs = ["conv_ops_test.cc"],
    deps = [
        "//core/schema:mynet_fbs",
        "//core/framework:mynet_test_main",
        ":conv_ops",
    ],
    copts = [
        "-std=c++17",
    ],
)

cc_library(
    name = "conv_ops",
    srcs = [
        "conv_ops.cc",
    ],
    hdrs = ["conv_ops.hpp"],
    deps = [
        ":im2col",
        "//core/framework:common",
        "//core/framework:ops",
        "//core/framework:tensor",
        "//core/framework:filler",
        "//core/framework:math_functions",
        "//core/schema:mynet_fbs",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "im2col",
    srcs = [
        "im2col.cc",
    ],
    hdrs = ["im2col.hpp"],
    deps = [
        "@com_github_google_glog//:glog",
    ],
)