import sys
import subprocess
import tempfile
import os
import platform
import pkg_resources
import xml.etree.ElementTree as ET

# Determine version of tensorflow-directml
installed_packages = pkg_resources.working_set
tfdml_version = [p.version for p in installed_packages if p.key == "tensorflow-directml"]
if tfdml_version:
    tfdml_version = tfdml_version[0]
else:
    tfdml_version = "Not Installed"

# Collect info from dxdiag.exe in Windows.
# NOTE: NamedTemporaryFile in a 'with' statement leaves the file open, which prevents dxdiag.exe
# from opening it a second time for writing on Windows. We must manually delete it without leaving
# the file open.
dxdiag_path = tempfile.NamedTemporaryFile(suffix=".xml", delete=False).name
try:
    if os.name == "nt":
        subprocess.run(['dxdiag.exe', '/x', dxdiag_path], check=True)
    else:
        dxdiag_path_windows = subprocess.run(
            "wslpath -w {}".format(dxdiag_path), 
            shell=True, 
            check=True,
            capture_output=True, 
            text=True).stdout.rstrip().replace('\\','\\\\')
        subprocess.run('dxdiag.exe /x {}'.format(dxdiag_path_windows), shell=True, check=True)
    with open(dxdiag_path, "r") as dxdiag_log:
        dxdiag = ET.parse(dxdiag_log).getroot()
finally:
    if os.path.exists(dxdiag_path):
        os.remove(dxdiag_path)

print("Host System\n{}".format('-'*80))
print("Windows 10 Version  : {}".format(dxdiag.find("./SystemInformation/OperatingSystem").text))
print("Processor           : {}".format(dxdiag.find("./SystemInformation/Processor").text))
print("Memory              : {}".format(dxdiag.find("./SystemInformation/Memory").text))
print("DirectX Version     : {}".format(dxdiag.find("./SystemInformation/DirectXVersion").text))

if os.name != "nt":
    import distro
    print("\nWindows Subsystem for Linux\n{}".format('-'*80))
    print("WSL Name            : {}".format(os.environ["WSL_DISTRO_NAME"]))
    print("WSL Distribution    : {}".format(" ".join(distro.linux_distribution())))
    print("WSL Kernel          : {}".format(platform.release()))

print("\nPython Environment\n{}".format('-'*80))
print("Python Version      : {}".format(platform.python_version()))
print("TensorFlow-DirectML : {}".format(tfdml_version))

for device in dxdiag.findall("./DisplayDevices/DisplayDevice"):
    print("\nDirectX Device\n{}".format('-'*80))
    print("Description         : {}".format(device.find("./CardName").text))
    print("Manufacturer        : {}".format(device.find("./Manufacturer").text))
    print("Chip Type           : {}".format(device.find("./ChipType").text))
    print("Dedicated Memory    : {}".format(device.find("./DedicatedMemory").text))
    print("Driver Version      : {}".format(device.find("./DriverVersion").text))
    print("Driver Model        : {}".format(device.find("./DriverModel").text))
    print("Driver Date         : {}".format(device.find("./DriverDate").text))
    print("Feature Levels      : {}".format(device.find("./FeatureLevels").text))
