load("//tensorflow:tensorflow.bzl", "if_windows", "if_not_windows")

config_setting(
    name = "using_dml",
    define_values = {
        "using_dml": "true",
    },
    visibility = ["//visibility:public"],
)

licenses(["notice"]) # MIT license

# Adapter for compiling DirectX in Linux.
cc_library(
    name = "winadapter",
    hdrs = if_not_windows([
        "winadapter/d3d12.h",
        "winadapter/d3d12sdklayers.h",
        "winadapter/d3dcommon.h",
        "winadapter/dxcore.h",
        "winadapter/dxcore_interface.h",
        "winadapter/dxgicommon.h",
        "winadapter/dxgiformat.h",
        "winadapter/winadapter.h",
        "winadapter/wrladapter.h",
        "winadapter/stubs/oaidl.h",
        "winadapter/stubs/ocidl.h",
        "winadapter/stubs/rpc.h",
        "winadapter/stubs/rpcndr.h",
        "winadapter/stubs/winapifamily.h",
        "winadapter/stubs/wrl/client.h",
        "winadapter/stubs/wrl/implements.h",
    ]) + if_windows(["winadapter/dxguids.h"]),
    srcs = if_not_windows(["winadapter/dxguids.cc"]),
    linkstatic = 1,
    # includes = ["winadapter", "winadapter/stubs"],
    # strip_include_prefix = "winadapter",
    visibility = ["//visibility:public"]
)

cc_library(
    name = "dx_helpers",
    hdrs = ["DirectMLX.h", "d3dx12.h"],
    visibility = ["//visibility:public"],
)