# Building TensorFlow-DirectML

This document contains instructions for producing private builds of tensorflow-directml.

## DirectML Branch

This project is a fork of the official [tensorflow](https://github.com/tensorflow/tensorflow) repository that targets TensorFlow v1.15. This fork will not merge upstream, since TensorFlow does not accept new features for previous releases, so the `master` branch is based off v1.15.0 from the upstream repository. DirectML changes do not appear in the `master` branch.

The `directml` branch is considered the main branch for development in this repository. This branch contains changes related to DirectML as well as security fixes in the upstream repository (e.g. changes in 1.15.1, 1.15.2, 1.15.3., etc.). Since this project is still in early development, the `directml` branch may be unstable and reference a "preview" version of DirectML that is subject to change. We recommend testing pre-built packages of [tensorflow-directml on PyPI](https://pypi.org/project/tensorflow-directml/); these builds are produced from the `directml` branch, but **only** when a stable version of DirectML is referenced. Alternatively, we recommend building against release tags associated with PyPI releases.

## Developer Mode

This repository may periodically reference in-development versions of DirectML for testing new features. For example, experimental APIs are added to `DirectMLPreview.h`, which may have breaking changes; once an API appears in `DirectML.h` it is immutable. A preview build of DirectML requires *developer mode* to be enabled or it will fail to load. This restriction is intended to avoid a long-term dependency on the preview library. **Packages on PyPI will only be released when the repository depends on a stable version of DirectML.**

You can determine if the current state of the repository references an in-development version of DirectML by inspecting `tensorflow/workspace.bzl`. If the package name is `Microsoft.AI.DirectML.Preview`, or the version ends with `-dev*`, then developer mode will be required. For example, the following snippet shows a dependency on DirectML Microsoft.AI.DirectML.Preview.1.5.0-dev20210406, which requires developer mode.

```
dml_repository(
    name = "dml_redist",
    package = "Microsoft.AI.DirectML.Preview",
    version = "1.5.0-dev20210406",
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

The development headers and libraries used to build DirectX-based applications are included in the Windows SDK; however, this SDK is not available when building for Linux, and some of required APIs may not yet exist in public versions of the SDK. For these reasons, the DirectX development files are integrated a little differently in this project.

The header files for Direct3D, DXCore, and DirectML are downloaded automatically. This project does not use any DirectX headers included in the Windows 10 SDK *except* for dxgi1_6.h, which is part of Windows 10 SDK 10.0.17763.0 or newer. Windows builds also require import libraries for D3D12 and DXGI, which are included in Windows 10 SDK 10.0.17763.0 or newer.

Linux builds do not require the Windows 10 SDK at all (no such SDK exists for Linux), but your build machine must have WSL installed with the DirectX shared libraries available under `/usr/lib/wsl/lib`.

The use of the redistributable DirectML library is governed by a separate license that is found as part of the package (found in `tensorflow_core/python/DirectML_LICENSE.txt` when extracted).

## Build Environment

We've tested the following build environments, and we recommend using a Python environment manager like [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to sandbox your builds.

- **Windows**
  - Visual Studio 2017 (15.x)
  - Windows 10 SDK 10.0.17763.0 or newer
  - MSYS2 20190524 or newer
  - Bazel 0.24.1

- **WSL**
  - Windows 10 Insider Build with WSL feature enabled (you must have libd3d12.so and libdxcore.so in `C:\Windows\System32\lxss\lib`)
  - Ubuntu 18.04 or 20.04 with g++ package
  - Bazel 0.24.1

**NOTE**: currently, it's only possible to build the Linux version of tensorflow-directml from a WSL environment. The D3D/DXCore libraries are needed for linking and are not yet available through another distribution channel. The DirectML redistributable package requires nuget.exe (a PE executable) to download, which also requires WSL/Windows interop. We hope to remove these restrictions in the near future.

## Build Script

A helper script, build.py, can be used to build this repository with support for DirectML. This script is a thin wrapper around the bazel commands for configuring and build TensorFlow; you may use bazel directly if you prefer, but make sure to include `--config=dml`. Run `build.py --help` for a full list of options, or inspect this file to get a full list of the bazel commands it executes. 

For example, to build the repository and produce a Python wheel use `build.py --package` in a Python 3.5-3.8 environment with `bazel` on the PATH.

# Detailed Instructions: Windows

These instructions use Miniconda to sandbox your build environment. This isn't strictly necessary, and there are other ways to do this (e.g. virtual machines, containers), but for the purpose of this walk-through you will use a Python virtual environment to build TFDML.

## Install Visual Studio 2017

TensorFlow 1.15 only builds with VS2017. [Download](https://my.visualstudio.com/Downloads?q=visual%20studio%202017&wt.mc_id=o~msft~vscom~older-downloads) and install the community, professional, or enterprise edition.

Make sure you install the Windows 10 SDK as well, which should be version 10.0.17763.0 or newer.

## Install MSYS2

The MSYS2 package contains several POSIX tools, built for Windows, which are required by the Bazel build system (bash shell, wget, bzip2, etc.). Again, you can use the default install directory or relocate it if desired. 

Once MSYS2 is installed, invoke its package manager to install a few additional dependencies (replace `C:\msys64` with the location where you installed MSYS2):

```
PS> C:\msys64\usr\bin\pacman.exe -S git patch unzip
```

## Install Bazel

[Bazel](https://bazel.build/) is the build tool for TensorFlow. Bazel is distributed as an executable and there is no installer step.

1. Download [Bazel 0.24.1](https://github.com/bazelbuild/bazel/releases/download/0.24.1/bazel-0.24.1-windows-x86_64.exe).
2. Copy/rename `bazel-0.24.1-windows-x86_64.exe` to a versioned path, such as `C:\bazel\0.24.1\bazel.exe`. TensorFlow will expect bazel.exe on the `%PATH%`, so renaming the executable while retaining the version within the path is useful.

## Install Miniconda

Miniconda is a bundle that includes [Python](https://www.python.org/), a package and environment manager named [Conda](https://docs.conda.io/projects/conda/en/latest/), and a very small set of Python packages. It is a lighter-weight alternative to Anaconda, which contains hundreds of Python packages that aren't necessary for building. The rest of this document applies equally to Anaconda if you prefer.

Download the latest [Miniconda3 Windows 64-bit](https://docs.conda.io/en/latest/miniconda.html) installer. You can leave all the default settings, but take note of the installation location (you'll need it later). The examples in this doc will reference "c:\miniconda3" as the install location.

## Create Conda Build Environment

Launch a Miniconda prompt (appears as "Anaconda PowerShell Prompt (Miniconda3)" in Windows Start Menu). The examples below will use the PowerShell version, but you can use the CMD version if you prefer. This prompt is a thin wrapper around PowerShell/CMD that adds the Conda installation to the `PATH` environment variable and modifies the display prompt slightly.

After launching the prompt, create a new environment. The examples here use the name `tfdml`, but you can choose any name you like. The idea is to sandbox any packages and dependencies in this environment. This will create a separate copy of Python and its packages in a directory `C:\miniconda3\envs\tfdml`.

**IMPORTANT**: make sure to use Python 3.7, 3.6, or 3.5. TensorFlow 1.15 is not supported on newer versions of Python.

```
(base) PS> conda create --name tfdml python=3.7
```

Next, activate the environment. Activating an environment will set up the `PATH` to use the correct version of Python and its packages.

```
(base) PS> conda activate tfdml
```

Within the activated environment, install the following Python packages required for TensorFlow:

```
(tfdml) PS> pip install six numpy==1.18.5 wheel
(tfdml) PS> pip install keras_applications==1.0.6 --no-deps
(tfdml) PS> pip install keras_preprocessing==1.0.5 --no-deps
```

Finally, we need to augment the `PATH` of the conda environment to include Bazel and MSYS2. 
- Conda will automatically run any scripts under `%CONDA_PREFIX%\etc\conda\activate.d\` when activating the environment.
- Conda will automatically run any scripts under `%CONDA_PREFIX%\etc\conda\deactivate.d\` when deactivating the environment. 
- `%CONDA_PREFIX%` is the path to the Conda environment, such as `C:\miniconda3\envs\tfdev` in these examples. 

**IMPORTANT**: Use the script type appropriate for your shell: `.ps1` for PowerShell, `.bat` for Command Prompt, `.sh` for bash, etc.

First, create the empty activation/deactivation scripts:

```
(tfdev) PS> cd $env:CONDA_PREFIX
(tfdev) PS> New-Item etc\conda\activate.d\path.ps1 -Force
(tfdev) PS> New-Item etc\conda\deactivate.d\path.ps1 -Force
```

Set the contents of the activation script (`%CONDA_PREFIX%\etc\conda\activate.d\path.ps1`). Make sure to use the correct paths where you installed Bazel and MSYS2:

```PowerShell
$env:TFDML_PATH_RESTORE = $env:PATH
$env:PATH = "C:\bazel\0.24.1;$env:PATH"
$env:PATH = "C:\msys64\usr\bin;$env:PATH"
```

Set the contents of the deactivation script (`%CONDA_PREFIX%\etc\conda\activate.d\path.ps1`):

```PowerShell
$env:PATH = $env:TFDML_PATH_RESTORE
```

Restart your conda environment (launch the Miniconda prompt again and activate `tfdev`). You should see both Bazel and MSYS2 tools on the PATH when running the `tfdev` environment:

```
(tfdev) PS> get-command bazel

CommandType     Name          Version    Source
-----------     ----          -------    ------
Application     bazel.exe     0.0.0.0    C:\bazel\0.24.1\bazel.exe


(tfdev) PS> get-command git

CommandType     Name          Version    Source
-----------     ----          -------    ------
Application     git.exe       0.0.0.0    C:\msys64\usr\bin\git.exe
```

## Clone

Clone the repository to a location of your choosing. The examples here will assume you clone to `C:\src\tensorflow-directml` for the sake of brevity, but you may clone wherever you like.

```
PS> cd c:\src
PS> git clone https://github.com/microsoft/tensorflow-directml.git
```

## Build

Remember to activate your build environment whenever you need to build. Change your working directory to the clone location:

```
(base) PS> conda activate tfdml
(tfdml) PS> cd c:\src\tensorflow-directml
```

To produce the Python package run the following:

```
(tfdev) PS> python build.py -p -c release
```

After the package is built you will find a wheel package under `<PATH_TO_CLONE>\..\dml_build\python_package` (e.g. `C:\src\dml_build\python_package` in these examples). You can run `pip install` on the output .whl file to install your locally built copy of TensorFlow-DirectML.

The build script has additional options you can experiment with. To see more details:

```
(tfdev) PS> python build.py --help
```

Note that the `config` parameter accepts debug or release as an argument, but these are largely the same: debug builds are effectively just "release with debug symbols" since the output PDBs for TensorFlow without optimizations are prohibitively large.