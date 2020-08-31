---
name: Bug/Performance Issue
about: Use this template for reporting a bug or a performance issue.

---

# System Information

Knowing your system configuration can help us diagnose issues more easily, so please provide as much information as possible. At a minimum, it is useful to know the following:

- Windows 10 Build/Version (e.g. Version 2004 / Build 19041)
- WSL distribution and its WSL version if testing Linux package (e.g. Ubuntu 20.04 WSL2)
- Python Version (e.g. 3.6.10)
- TensorFlow-DirectML Version (e.g. 1.15.3.dev200626)
- Graphics card driver version (e.g. NVIDIA GTX 1080 - 27.21.14.5193)

You can collect this information more easily by running [print_system_info.py](../../tools/print_system_info.py); make sure to run this in the environment that has Python and tensorflow-directml installed. Below is sample output from running this script within a WSL distro (WSL section will be omitted if running in Windows):

```
Host System
--------------------------------------------------------------------------------
Windows 10 Version  : Windows 10 Enterprise Insider Preview 64-bit (10.0, Build 20145) (20145.rs_onecore_sigma_grfx.200607-1704)
Processor           : Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz (16 CPUs)
Memory              : 32768MB RAM
DirectX Version     : DirectX 12

Windows Subsystem for Linux
--------------------------------------------------------------------------------
WSL Name            : Ubuntu
WSL Distribution    : Ubuntu 20.04 focal
WSL Kernel          : 4.19.121-microsoft-standard

Python Environment
--------------------------------------------------------------------------------
Python Version      : 3.7.7
TensorFlow-DirectML : 1.15.3.dev200626

DirectX Device
--------------------------------------------------------------------------------
Description         : NVIDIA GeForce GTX 1080
Manufacturer        : NVIDIA
Chip Type           : GeForce GTX 1080
Dedicated Memory    : 8079 MB
Driver Version      : 27.21.14.5193
Driver Model        : WDDM 2.7
Driver Date         : 8/2/2020 5:00:00 PM
Feature Levels      : 12_1,12_0,11_1,11_0,10_1,10_0,9_3,9_2,9_1
```

# Repro Details

**Describe the current behavior**

**Describe the expected behavior**

**Code to reproduce the issue**
Provide a reproducible test case that is the bare minimum necessary to generate the problem.

**Other info / logs**
Include any logs or source code that would be helpful to diagnose the problem. If including tracebacks, please include the full traceback. Large logs and files should be attached.
