load("@com_github_google_flatbuffers//:build_defs.bzl", "flatbuffer_cc_library")

flatbuffer_cc_library(
    name = "mynet_fbs",
    srcs = [
        "filler.fbs",
        "mynet.fbs",
        "ops.fbs",
        "solver.fbs",
        "tensor.fbs",
    ],
    visibility = ["//visibility:public"],
)

exports_files([
    "mynet.fbs",
    "solver.fbs",
    "tensor.fbs",
    "ops.fbs",
    "filler.fbs",
])
