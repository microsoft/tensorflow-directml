load("//tensorflow:tensorflow.bzl", "if_windows", "if_not_windows")

config_setting(
    name = "using_dml",
    define_values = {
        "using_dml": "true",
    },
    visibility = ["//visibility:public"],
)

licenses(["notice"]) # MIT license

# Adapter for compiling DirectX in Linux. For Windows, this library only
# exposes portability helpers (e.g. uuidof<T>) and IID asserts.
cc_library(
    name = "winadapter",
    hdrs = ["winadapter/dxguids/dxguids.h"] + if_not_windows([
        "winadapter/d3d12.h",
        "winadapter/d3d12sdklayers.h",
        "winadapter/d3dcommon.h",
        "winadapter/dxcore.h",
        "winadapter/dxcore_interface.h",
        "winadapter/dxgicommon.h",
        "winadapter/dxgiformat.h",
        "winadapter/winadapter.h",
        "winadapter/wrladapter.h",
    ]),
    srcs = ["winadapter/dxguids/dxguids.cc"] + if_not_windows([
        "winadapter/stubs/oaidl.h",
        "winadapter/stubs/ocidl.h",
        "winadapter/stubs/rpc.h",
        "winadapter/stubs/rpcndr.h",
        "winadapter/stubs/winapifamily.h",
        "winadapter/stubs/wrl/client.h",
        "winadapter/stubs/wrl/implements.h",
    ]),
    linkstatic = 1,
    includes = ["winadapter/dxguids"] + if_not_windows([
        "winadapter",
        "winadapter/stubs",
    ]),
    visibility = ["//visibility:public"]
)

cc_library(
    name = "dx_helpers",
    hdrs = ["DirectMLX.h", "d3dx12.h"],
    visibility = ["//visibility:public"],
)