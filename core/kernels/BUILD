cc_test(
    name = "conv_ops_test",
    srcs = ["conv_ops_test.cc"],
    copts = [
        "-std=c++17",
    ],
    deps = [
        ":conv_ops",
        "//core/framework:mynet_test_main",
        "//core/schema:mynet_fbs",
    ],
)

cc_library(
    name = "conv_ops",
    srcs = [
        "conv_ops.cc",
    ],
    hdrs = ["conv_ops.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        ":im2col",
        "//core/framework:common",
        "//core/framework:filler",
        "//core/framework:math_functions",
        "//core/framework:op",
        "//core/framework:tensor",
        "//core/schema:mynet_fbs",
    ],
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

cc_test(
    name = "dummy_data_op_test",
    srcs = ["dummy_data_op_test.cc"],
    copts = [
        "-std=c++17",
    ],
    deps = [
        ":dummy_data_op",
        "//core/framework:mynet_test_main",
    ],
)

cc_library(
    name = "dummy_data_op",
    srcs = [
        "dummy_data_op.cc",
    ],
    hdrs = ["dummy_data_op.hpp"],
    copts = [
        "-std=c++17",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//core/framework:common",
        "//core/framework:filler",
        "//core/framework:math_functions",
        "//core/framework:op",
        "//core/framework:op_factory",
        "//core/framework:tensor",
        "//core/schema:mynet_fbs",
    ],
)
