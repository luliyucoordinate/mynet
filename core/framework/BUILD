cc_test(
    name = "tensor_test",
    srcs = ["tensor_test.cc"],
    deps = [
        ":tensor",
        ":filler",
        ":mynet_test_main",
    ],
)

cc_library(
    name = "math_functions",
    srcs = [
        "math_functions.cc",
    ],
    hdrs = ["math_functions.hpp"],
    deps = [
        ":common",
        "//:openblas",
    ]
)

cc_library(
    name = "mynet_test_main",
    srcs = [
        "mynet_test_main.cc",
    ],
    hdrs = ["mynet_test_main.hpp"],
    deps = [
        ":common",
    ]
)

cc_library(
    name = "tensor",
    srcs = [
        "tensor.cc",
    ],
    hdrs = ["tensor.hpp"],
    deps = [
        ":common",
        ":syncedmem",
        ":math_functions",
        "//core/protobuf:mynet_cc_proto",
        "//core/schema:mynet_fbs"
    ]
)

cc_test(
    name = "filler_test",
    srcs = ["filler_test.cc"],
    deps = [
        ":filler",
        ":common",
        ":mynet_test_main",
    ],
)

cc_library(
    name = "filler",
    hdrs = ["filler.hpp"],
    deps = [
        ":tensor",
        ":math_functions",
    ]
)

cc_test(
    name = "syncedmem_test",
    srcs = ["syncedmem_test.cc"],
    deps = [
        ":syncedmem"
    ],
)

cc_library(
    name = "syncedmem",
    srcs = [
        "syncedmem.cc",
        "common.hpp",
    ],
    hdrs = ["syncedmem.hpp"],
    deps = [
        ":common",
    ]
)

cc_library(
    name = "common",
    srcs = ["common.cc"],
    hdrs = ["common.hpp"],
    deps = [
        "@com_google_googletest//:gtest_main",
        "@com_github_gflags_gflags//:gflags",
        "@com_github_google_glog//:glog",
    ]
)