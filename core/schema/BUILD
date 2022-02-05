load("@com_github_google_flatbuffers//:build_defs.bzl", "flatbuffer_cc_library")

flatbuffer_cc_library(
    name = "mynet_fbs",
    srcs = [
        "filler.fbs",
        "net.fbs",
        "op.fbs",
        "solver.fbs",
        "tensor.fbs",
    ],
    visibility = ["//visibility:public"],
)

exports_files([
    "net.fbs",
    "solver.fbs",
    "tensor.fbs",
    "op.fbs",
    "filler.fbs",
])
