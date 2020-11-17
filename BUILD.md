# Building TensorFlow-DirectML

This document contains instructions for producing private builds of tensorflow-directml.

## DirectML Branch

This project is a fork of the official [tensorflow](https://github.com/tensorflow/tensorflow) repository that targets TensorFlow v1.15. This fork will not merge upstream, since TensorFlow does not accept new features for previous releases, so the `master` branch is based off v1.15.0 from the upstream repository. DirectML changes do not appear in the `master` branch.

The `directml` branch is considered the main branch for development in this repository. This branch contains changes related to DirectML as well as security fixes in the upstream repository (e.g. changes in 1.15.1, 1.15.2, 1.15.3., etc.). Since this project is still in early development, the `directml` branch may be unstable and reference a "preview" version of DirectML that is subject to change. We recommend testing pre-built packages of [tensorflow-directml on PyPI](https://pypi.org/project/tensorflow-directml/); these builds are produced from the `directml` branch, but **only** when a stable version of DirectML is referenced. Alternatively, we recommend building against release tags associated with PyPI releases.

## Developer Mode

This repository may periodically reference in-development versions of DirectML for testing new features. For example, experimental APIs are added to `DirectMLPreview.h`, which may have breaking changes; once an API appears in `DirectML.h` it is immutable. A preview build of DirectML requires *developer mode* to be enabled or it will fail to load. This restriction is intended to avoid a long-term dependency on the preview library. **Packages on PyPI will only be released when the repository depends on a stable version of DirectML.**

You can determine if the current state of the repository references an in-development version of DirectML by inspecting `tensorflow/workspace.bzl`. If the version attribute ends with `-dev*`, then developer mode will be required. For example, the following snippet shows a dependency on DirectML 1.5.0-dev1, which requires developer mode.

```
dml_repository(
    name = "dml_redist",
    package = "DirectML",
    version = "1.5.0-dev1",
    source = "https://pkgs.dev.azure.com/ms/DirectML/_packaging/tensorflow-directml/nuget/v3/index.json",
    build_file = "//third_party/dml/redist:BUILD.bazel",
)
```

Developer mode is, as the name indicates, only intended to be used for development! It should not be used for any other purpose. To enable developer mode:

- **Windows**
  - [Toggle "Developer Mode" to "On"](https://docs.microsoft.com/en-us/windows/uwp/get-started/enable-your-device-for-development) in the "For developers" tab of the "Update & Security" section of the Settings app.

- **WSL**
  - Create a plain-text file in your Linux home directory, `~/.directml.conf`, with the contents `devmode = 1`.

## DirectX Development Files

The development headers and libraries used to build DirectX-based applications are included in the Windows SDK; however, this SDK is not available when building for Linux. Additionally, this project may depend on a newer version of DirectML than what is available in Windows SDKs. For these reasons, the DirectX development files are integrated a little differently depending on Windows or Linux builds:

- **Windows**
  -  Direct3D/DXCore Headers: Windows SDK newer than 10.0.19645.0
  -  Direct3D/DXCore Libraries: Windows SDK newer than 10.0.19645.0

- **WSL**
   -  Direct3D/DXCore Headers: `third_party/dml/winadapter`
   -  Direct3D/DXCore Libraries: `/usr/lib/wsl/lib`

For both Windows and WSL, the DirectML headers and libraries are pulled from a redistributable NuGet package. This package is downloaded automatically as a part of the build (see `third_party/dml/redist`). The use of the redistributable DirectML library is governed by a separate license that is found as part of the package (found in `tensorflow_core/python/DirectML_LICENSE.txt` when extracted).

## Build Environment

Make sure to follow the official build setup instructions ([Windows](https://www.tensorflow.org/install/source_windows#:~:text=%20%20%20%20Version%20%20%20,%20Bazel%200.26.1%20%2016%20more%20rows%20), [Linux](https://www.tensorflow.org/install/source)) before proceeding. We've tested the following build environments, and we recommend using a Python environment manager like [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to sandbox your builds.

- **Windows**
  - Visual Studio 2017 (15.x)
  - Windows SDK 10.0.19645.0 or later (requires Windows Insider Build)
  - MSYS2 20190524
  - Bazel 0.24.1

- **WSL**
  - Windows 10 Insider Build with WSL feature enabled (you must have libd3d12.so and libdxcore.so in `C:\Windows\System32\lxss\lib`)
  - Ubuntu 18.04 or 20.04 with g++ package
  - Bazel 0.24.1

**NOTE**: currently, it's only possible to build the Linux version of tensorflow-directml from a WSL environment. The D3D/DXCore libraries are needed for linking and are not yet available through another distribution channel. The DirectML redistributable package requires nuget.exe (a PE executable) to download, which also requires WSL/Windows interop. We hope to remove these restrictions in the near future.

## Build Script

A helper script, build.py, can be used to build this repository with support for DirectML. This script is a thin wrapper around the bazel commands for configuring and build TensorFlow; you may use bazel directly if you prefer, but make sure to include `--config=dml`. Run `build.py --help` for a full list of options, or inspect this file to get a full list of the bazel commands it executes. 

For example, to build the repository and produce a Python wheel use `build.py --package` in a Python 3.5-3.8 environment with `bazel` on the PATH.

