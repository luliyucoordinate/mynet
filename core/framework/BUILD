load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

cc_library(
    name = "tensor",
    srcs = ["tensor.cc"],
    hdrs = ["tensor.hpp"],
    deps = [
        "//core/schema:mynet_fbs",
    ],
    visibility = ["//visibility:public"],
)