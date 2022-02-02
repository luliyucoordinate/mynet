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

cc_test(
    name = "fully_connected_op_test",
    srcs = ["fully_connected_op_test.cc"],
    copts = [
        "-std=c++17",
    ],
    deps = [
        ":fully_connected_op",
        ":gradient_check_util",
        "//core/framework:mynet_test_main",
        "//core/schema:mynet_fbs",
    ],
)

cc_library(
    name = "fully_connected_op",
    srcs = [
        "fully_connected_op.cc",
    ],
    hdrs = ["fully_connected_op.hpp"],
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

cc_test(
    name = "softmax_op_test",
    srcs = ["softmax_op_test.cc"],
    copts = [
        "-std=c++17",
    ],
    deps = [
        ":gradient_check_util",
        ":softmax_op",
        "//core/framework:mynet_test_main",
        "//core/schema:mynet_fbs",
    ],
)

cc_library(
    name = "softmax_op",
    srcs = [
        "softmax_op.cc",
    ],
    hdrs = ["softmax_op.hpp"],
    copts = [
        "-std=c++17",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//core/framework:common",
        "//core/framework:filler",
        "//core/framework:math_functions",
        "//core/framework:op",
        "//core/framework:tensor",
        "//core/schema:mynet_fbs",
    ],
)

cc_library(
    name = "loss_op",
    srcs = [
        "loss_op.cc",
    ],
    hdrs = ["loss_op.hpp"],
    copts = [
        "-std=c++17",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//core/framework:op",
        "//core/framework:tensor",
    ],
)

cc_test(
    name = "euclidean_loss_op_test",
    srcs = ["euclidean_loss_op_test.cc"],
    copts = [
        "-std=c++17",
    ],
    deps = [
        ":euclidean_loss_op",
        ":gradient_check_util",
        "//core/framework:filler",
        "//core/framework:mynet_test_main",
        "//core/schema:mynet_fbs",
    ],
)

cc_library(
    name = "euclidean_loss_op",
    srcs = [
        "euclidean_loss_op.cc",
    ],
    hdrs = ["euclidean_loss_op.hpp"],
    copts = [
        "-std=c++17",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":loss_op",
        "//core/framework:op_factory",
        "//core/framework:tensor",
    ],
)

cc_test(
    name = "split_op_test",
    srcs = ["split_op_test.cc"],
    copts = [
        "-std=c++17",
    ],
    deps = [
        ":gradient_check_util",
        ":split_op",
        "//core/framework:common",
        "//core/framework:filler",
        "//core/framework:mynet_test_main",
        "//core/framework:tensor",
        "//core/schema:mynet_fbs",
    ],
)

cc_library(
    name = "split_op",
    srcs = [
        "split_op.cc",
    ],
    hdrs = ["split_op.hpp"],
    copts = [
        "-std=c++17",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//core/framework:math_functions",
        "//core/framework:op",
        "//core/framework:op_factory",
        "//core/framework:tensor",
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

cc_library(
    name = "gradient_check_util",
    hdrs = ["gradient_check_util.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//core/framework:op",
        "//core/framework:tensor",
    ],
)
