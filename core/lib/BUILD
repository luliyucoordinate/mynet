cc_test(
    name = "io_test",
    srcs = ["io_test.cc"],
    copts = [
        "-std=c++17",
    ],
    data = [
        "//core/test_data:io_read_test.bin",
        "//core/test_data:io_read_test.json",
        "//core/test_data:io_write_test.bin",
        "//core/test_data:io_write_test.json",
    ],
    deps = [
        ":io",
        "//core/framework:mynet_test_main",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "format",
    hdrs = ["format.hpp"],
)

cc_library(
    name = "io",
    srcs = [
        "io.cc",
    ],
    hdrs = ["io.hpp"],
    copts = [
        "-std=c++17",
    ],
    data = [
        "//core/schema:filler.fbs",
        "//core/schema:mynet.fbs",
        "//core/schema:ops.fbs",
        "//core/schema:tensor.fbs",
    ],
    deps = [
        ":format",
        "//core/schema:mynet_fbs",
        "@com_github_google_flatbuffers//:flatbuffers",
        "@com_github_google_glog//:glog",
    ],
)
