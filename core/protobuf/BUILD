load("@rules_proto//proto:defs.bzl", "proto_library")

cc_proto_library(
    name = "mynet_cc_proto",
    deps = [":mynet_proto"],
    visibility = ["//visibility:public"],
)

proto_library(
    name = "mynet_proto",
    srcs = ["mynet.proto"],
)