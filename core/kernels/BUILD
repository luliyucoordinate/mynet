cc_test(
    name = "conv_ops_test",
    srcs = ["conv_ops_test.cc"],
    deps = [
        "//core/schema:mynet_fbs",
        ":conv_ops",
        ":mynet_test_main",
    ],
)

cc_library(
    name = "conv_ops",
    srcs = [
        "conv_ops.cc",
    ],
    hdrs = ["conv_ops.hpp"],
    deps = [
        ":common",
        ":ops",
    ]
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