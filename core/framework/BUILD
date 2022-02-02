cc_test(
    name = "net_test",
    srcs = ["net_test.cc"],
    copts = [
        "-std=c++17",
    ],
    data = [
        "//core/test_data:tinynet.json",
    ],
    deps = [
        ":net",
        "//core/framework:mynet_test_main",
    ],
)

cc_library(
    name = "net",
    srcs = [
        "net.cc",
    ],
    hdrs = ["net.hpp"],
    copts = [
        "-std=c++17",
    ],
    deps = [
        ":common",
        ":insert_splits",
        ":op",
        ":op_factory",
        ":tensor",
        "//core/kernels:dummy_data_op",
        "//core/kernels:euclidean_loss_op",
        "//core/kernels:fully_connected_op",
        "//core/kernels:split_op",
        "//core/lib:io",
        "//core/schema:mynet_fbs",
    ],
)

cc_library(
    name = "insert_splits",
    srcs = [
        "insert_splits.cc",
    ],
    hdrs = ["insert_splits.hpp"],
    copts = [
        "-std=c++17",
    ],
    deps = [
        ":common",
        ":mynet_test_main",
        ":op",
        "//core/schema:mynet_fbs",
    ],
)

cc_test(
    name = "solver_factory_test",
    srcs = ["solver_factory_test.cc"],
    copts = [
        "-std=c++17",
    ],
    deps = [
        ":mynet_test_main",
        ":solver",
        ":solver_factory",
        "//core/schema:mynet_fbs",
    ],
)

cc_library(
    name = "solver_factory",
    hdrs = ["solver_factory.hpp"],
    copts = [
        "-std=c++17",
    ],
    deps = [
        ":common",
        "//core/schema:mynet_fbs",
    ],
)

cc_library(
    name = "solver",
    hdrs = ["solver.hpp"],
    copts = [
        "-std=c++17",
    ],
    deps = [
        ":common",
        "//core/schema:mynet_fbs",
    ],
)

cc_test(
    name = "op_factory_test",
    srcs = ["op_factory_test.cc"],
    copts = [
        "-std=c++17",
    ],
    deps = [
        ":mynet_test_main",
        ":op_factory",
        "//core/schema:mynet_fbs",
    ],
)

cc_library(
    name = "op_factory",
    srcs = [
        "op_factory.cc",
    ],
    hdrs = ["op_factory.hpp"],
    copts = [
        "-std=c++17",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":common",
        ":op",
        "//core/kernels:conv_ops",
        "//core/kernels:softmax_op",
    ],
)

cc_library(
    name = "op",
    srcs = [
        "op.cc",
    ],
    hdrs = ["op.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        ":common",
        ":math_functions",
        ":tensor",
        "//core/schema:mynet_fbs",
    ],
)

cc_test(
    name = "tensor_test",
    srcs = ["tensor_test.cc"],
    deps = [
        ":filler",
        ":mynet_test_main",
        ":tensor",
    ],
)

cc_library(
    name = "math_functions",
    srcs = [
        "math_functions.cc",
    ],
    hdrs = ["math_functions.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        ":common",
        "//:openblas",
    ],
)

cc_library(
    name = "mynet_test_main",
    srcs = [
        "mynet_test_main.cc",
    ],
    hdrs = ["mynet_test_main.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        ":common",
    ],
)

cc_library(
    name = "tensor",
    srcs = [
        "tensor.cc",
    ],
    hdrs = ["tensor.hpp"],
    copts = [
        "-std=c++17",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":common",
        ":math_functions",
        ":syncedmem",
        "//core/schema:mynet_fbs",
    ],
)

cc_test(
    name = "filler_test",
    srcs = ["filler_test.cc"],
    deps = [
        ":common",
        ":filler",
        ":mynet_test_main",
        "//core/schema:mynet_fbs",
    ],
)

cc_library(
    name = "filler",
    hdrs = ["filler.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        ":math_functions",
        ":tensor",
    ],
)

cc_test(
    name = "syncedmem_test",
    srcs = ["syncedmem_test.cc"],
    deps = [
        ":syncedmem",
    ],
)

cc_library(
    name = "syncedmem",
    srcs = [
        "common.hpp",
        "syncedmem.cc",
    ],
    hdrs = ["syncedmem.hpp"],
    deps = [
        ":common",
    ],
)

cc_library(
    name = "common",
    srcs = ["common.cc"],
    hdrs = [
        "common.hpp",
    ],
    copts = [
        "-std=c++17",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@com_github_gflags_gflags//:gflags",
        "@com_github_google_glog//:glog",
        "@com_google_googletest//:gtest_main",
    ],
)
