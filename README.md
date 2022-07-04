# TensorFlow-DirectML <!-- omit in toc -->

|           | Warnings                                                                                                                                                               |
|-----------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| :warning: | **h5py 3.0.0 and 3.1.0 broke compatibility with TensorFlow. Please make sure that your environment has a different version of h5py before using TensorFlow-DirectML.** |


[TensorFlow](https://www.tensorflow.org/) is an end-to-end open source platform for machine learning. This repository is a fork of [tensorflow](https://github.com/tensorflow/tensorflow) that leverages [DirectML](https://github.com/microsoft/DirectML) to provide cross-vendor hardware acceleration on Windows and the Windows Subsystem for Linux (WSL). TensorFlow with DirectML enables training and inference of complex machine learning models on a wide range of DirectX 12-compatible hardware.

**Latest Release**:  
[![PyPI version](https://badge.fury.io/py/tensorflow-directml.svg)](https://badge.fury.io/py/tensorflow-directml) 

## Questions, Issues, and Feedback

- Frequently asked questions: [FAQ](https://docs.microsoft.com/en-us/windows/ai/directml/gpu-faq)
- Learn about our roadmap: [Wiki](https://github.com/microsoft/tensorflow-directml/wiki)
- Ask a question: [Discussions](https://github.com/microsoft/tensorflow-directml/discussions)
- Report a bug: [Issues](https://github.com/microsoft/tensorflow-directml/issues) 

You can also contact us directly at askdirectml@microsoft.com.

## Getting Started

TensorFlow with DirectML is supported on both the latest versions of Windows and the [Windows Subsystem for Linux](https://docs.microsoft.com/windows/wsl/about). For detailed instructions on getting started, see [GPU accelerated ML training (docs.microsoft.com)](http://aka.ms/gpuinwsldocs).

TensorFlow with DirectML is compatible with TensorFlow 1.15 and is supported for production use. Official Python packages are available on the [tensorflow-directml PyPI project](https://pypi.org/project/tensorflow-directml/), and C library packages are [available for download on GitHub](https://github.com/microsoft/tensorflow-directml/releases).

The DirectML repository includes [a few samples](https://github.com/microsoft/DirectML/tree/master/TensorFlow) that have been tested to work with the latest builds on PyPI. These samples include both inference and training scripts, and you can either train the models from scratch or use the supplied pre-trained weights. However, we encourage testing on any TensorFlow 1.15-compatible models -- if you run into issues, please let us know!

The following resources provide additional background on DirectML and TensorFlow:
- [DirectML GitHub](https://github.com/microsoft/DirectML/)
- [RFC: TensorFlow on DirectML](https://github.com/tensorflow/community/pull/243)
- [TensorFlow homepage](https://www.tensorflow.org/)

## System Requirements

### Windows

* Windows 10 Version 1709, 64-bit (Build 16299 or higher) or Windows 11 Version 21H2, 64-bit (Build 22000 or higher)
* Python x86-64 3.5, 3.6, or 3.7<sup>1</sup>
* One of the following supported GPUs:
  * AMD Radeon R5/R7/R9 2xx series or newer
  * Intel HD Graphics 5xx or newer
  * NVIDIA GeForce GTX 9xx series GPU or newer

<sup>1</sup> Note: Python 3.8 or newer is **not** currently supported. To use the official PyPi packages, the CPython interpreter is required. NumPy 1.19.4 is [requires the KB4598291 to properly work on Windows](https://github.com/numpy/numpy/wiki/FMod-Bug-on-Windows).

### Windows Subsystem for Linux

* Windows 10 Version 21H2, 64-bit (Build 20150 or higher) or Windows 11 Version 21H2, 64-bit (Build 22000 or higher)
* Python x86-64 3.5, 3.6, or 3.7<sup>2</sup>
* One of the following supported GPUs:
  * AMD Radeon R5/R7/R9 2xx series or newer, and [20.20.01.05 driver or newer](https://www.amd.com/en/support)
  * Intel HD Graphics 6xx or newer, and [28.20.100.8322 driver or newer](https://www.intel.com/content/www/us/en/download/19344/intel-graphics-windows-dch-drivers.html)
  * NVIDIA GeForce GTX 9xx series GPU or newer, and [460.20 driver or newer](https://www.nvidia.com/download/index.aspx)

<sup>2</sup> Note: Python 3.8 or newer is **not** currently supported. To use the official PyPi packages, the CPython interpreter is required.

## Contribute

If you would like to contribute to tensorflow-directml, please see our [contribution guidelines](CONTRIBUTING.md) and read the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct). We use [GitHub issues](https://github.com/microsoft/tensorflow/issues) for tracking requests and bugs. **Please do not report security vulnerabilities through public GitHub issues.** See SECURITY.md for more details.

See [BUILD.md](BUILD.md) for instructions on how to produce private builds of tensorflow-directml.

## License

This project is licensed under [Apache License 2.0](LICENSE).

The tensorflow-directml Python wheel binary package includes a redistributable version of the DirectML library, which is downloaded automatically as a part of the build. The use of the redistributable DirectML library is governed by a separate license that is found as part of the package (found in `tensorflow_core/python/DirectML_LICENSE.txt` when extracted).

## Data Collection Notice

The software may collect information about you and your use of the software and send it to Microsoft. Microsoft may use this information to provide services and improve our products and services. You may turn off the telemetry as described in the repository. There are also some features in the software that may enable you and Microsoft to collect data from users of your applications. If you use these features, you must comply with applicable law, including providing appropriate notices to users of your applications together with a copy of Microsoft's privacy statement. Our privacy statement is located at https://go.microsoft.com/fwlink/?LinkID=824704. You can learn more about data collection and use in the help documentation and our privacy statement. Your use of the software operates as your consent to these practices.

### Disabling Telemetry

The official builds of tensorflow-directml (hosted on [PyPI](https://pypi.org/project/tensorflow-directml/)) have data collection enabled. This telemetry is enabled when building with `--config=dml_telemetry` (i.e. the `--telemetry` switch in `build.py`), but it is disabled by default for local builds.

## Trademarks Notice

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow Microsoft's Trademark & Brand Guidelines. Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party's policies.

**TensorFlow, the TensorFlow logo and any related marks are trademarks of Google Inc.**
