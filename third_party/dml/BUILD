load("//tensorflow:tensorflow.bzl", "if_windows", "if_not_windows")

config_setting(
    name = "using_dml",
    define_values = {
        "using_dml": "true",
    },
    visibility = ["//visibility:public"],
)

licenses(["notice"]) # MIT license

cc_library(
    name = "dx_helpers",
    hdrs = ["DirectMLX.h"],
    visibility = ["//visibility:public"],
)