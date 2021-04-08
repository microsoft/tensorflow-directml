package(default_visibility = ["//visibility:public"])

config_setting(
    name = "windows",
    values = {"cpu": "x64_windows"},
)

# Core DirectX headers.
cc_library(
    name = "directx_headers",
    hdrs = [
        "include/directx/d3d12.h",
        "include/directx/d3d12compatibility.h",
        "include/directx/d3d12sdklayers.h",
        "include/directx/d3d12shader.h",
        "include/directx/d3d12video.h",
        "include/directx/d3dcommon.h",
        "include/directx/d3dx12.h",
        "include/directx/dxcore.h",
        "include/directx/dxcore_interface.h",
        "include/directx/dxgicommon.h",
        "include/directx/dxgiformat.h",
    ],
    includes = ["include/directx"],
)

# Defines DirectX interface IDs (IIDs), which are necessary for linking.
cc_library(
    name = "directx_guids",
    hdrs = ["include/dxguids/dxguids.h"],
    includes = ["include/dxguids", "include"],
    linkstatic = 1,
)

# Adapter for compiling Win32/DirectX APIs in Linux. For Windows this library only
# exposes portability helpers (e.g. uuidof<T>) and IID asserts.
cc_library(
    name = "directx_winadapter",
    hdrs = select({
        ":windows": [],
        "//conditions:default": [
            "include/wsl/winadapter.h", 
            "include/wsl/wrladapter.h",
        ],
    }),
    includes = ["include/wsl"] + select({
        ":windows": [],
        "//conditions:default": ["include/wsl/stubs"],
    }),
)