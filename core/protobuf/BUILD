load("@rules_proto//proto:defs.bzl", "proto_library")
load("@com_github_google_flatbuffers//:build_defs.bzl", "flatbuffer_cc_library")

cc_proto_library(
    name = "mynet_cc_proto",
    deps = [":mynet_proto"],
    visibility = ["//visibility:public"],
)

proto_library(
    name = "mynet_proto",
    srcs = ["mynet.proto"],
)

flatbuffer_cc_library(
    name = "mynet_fbs",
    srcs = ["mynet.fbs"],
    visibility = ["//visibility:public"],
)