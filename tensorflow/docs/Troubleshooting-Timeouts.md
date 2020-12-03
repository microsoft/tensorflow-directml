# Troubleshooting GPU timeouts in tensorflow-directml

Because tensorflow-directml supports such a wide range of hardware (including consumer gaming and integrated GPUs), it is occasionally possible to encounter a training workload that causes a *GPU timeout* on some hardware/drivers.

GPU timeouts are caused by long-running workloads that exceed the system-defined GPU preemption timeout, which is 2 seconds by default. At the lowest level, the exact circumstance that causes a GPU timeout is when a non-preemptable dispatch runs on the GPU for a time that exceeds the defined limit. A "dispatch" often corresponds to the evaluation of a single TensorFlow operator (like conv2d, or matmul) on the GPU. Because of this, the prevalence of timeouts depends on a wide variety of factors, including the specific hardware and driver configuration used.

If you encounter a GPU timeout in tensorflow-directml, you'll see an error message similar to the following:

```
The DirectML device has encountered an unrecoverable error (DXGI_ERROR_DEVICE_HUNG).
This is most often caused by a timeout occurring on the GPU. Please visit
https://aka.ms/tfdmltimeout for more information and troubleshooting steps.

HRESULT failed with 0x887a0005: readback_heap->Map(0, nullptr, &readback_heap_data)
```

If you see this error, there are several ways to mitigate the problem:

### #1: Ensure you are using the latest version of tensorflow-directml

tensorflow-directml is still in active development. With each release we continually make improvements to stability and performance, which can help to resolve timeouts caused by large training workloads. It's highly recommended to keep your copy of tensorflow-directml updated to the latest whenever possible.

To install the latest [tensorflow-directml package](https://pypi.org/project/tensorflow-directml/) from PyPi, run the following on your command line:

```
pip install tensorflow-directml --upgrade
```

### #2: Ensure your GPU drivers are up-to-date

We recommend using the latest stable drivers from your GPU vendor wherever possible. GPU driver updates can be downloaded from the following locations:

* **For AMD GPUs:** [AMD Drivers and Support](https://www.amd.com/en/support)
* **For Intel GPUs:** [Intel Download Center](https://downloadcenter.intel.com/)
* **For NVIDIA GPUs:** [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx)

### #3: Exclude smaller GPUs from being used with DirectML

By default, all compatible GPUs are made available to be used by DirectML. This means that both integrated and discrete GPUs can be used with DirectML.

On hybrid systems that have *both* integrated and discrete GPUs available (such as those found in many laptops), it may be desirable to limit tensorflow-directml to use only one of the two GPUs. This can be done using the `DML_VISIBLE_DEVICES` environment variable.

For example if your system has both an integrated and a discrete GPU (with device IDs of 0 and 1 respectively) and you want tensorflow-directml to only use the discrete GPU, you can `set DML_VISIBLE_DEVICES=1` to force only the second GPU to be visible.

For more detailed instructions on controlling GPU visibility with tensorflow-directml, see the [FAQ for TensorFlow with DirectML](https://docs.microsoft.com/windows/win32/direct3d12/gpu-faq#i-have-multiple-gpus-how-do-i-select-which-one-is-used-by-directml).

### #4: Try a smaller workload, for example by reducing the batch size

GPU timeouts occur when a non-preemptable workload runs on the GPU for too long. During training, this is most likely to occur when performing expensive operations with very large tensors and/or large batch size. Examples of this include convolutions with large filters and/or large batch sizes, and matrix multiplications with very large tensors.

If timeouts are occurring on your GPU, you can try using smaller tensors or lowering the batch size during training to reduce the compute workload.

### #5: Disable GPU timeouts using system-wide registry key (advanced users only)

If you have tried all of the above and are still experiencing GPU timeouts, it is possible to modify or disable GPU timeout protections (known as Timeout Detection and Recovery, or TDR) in Windows 10.

Note that modifying or disabling timeout protections carries risks to your computer. GPU timeouts protect your system from intentional or accidental denial-of-service of GPU resources, and help to ensure a stable and responsive user experience. If timeouts are disabled and an extremely long-running workload is dispatched to the GPU, you may experience system hangs or other instabilities.

For this reason, it is only recommended to modify timeout settings as a last resort. For more information, see [Timeout detection and recovery (TDR) (docs.microsoft.com)](https://docs.microsoft.com/windows-hardware/drivers/display/timeout-detection-and-recovery).

#### Modifying TDR settings

> **Note:** this is for advanced users only. Incorrectly editing the system registry can result in system instability. Make sure you understand the risks before modifying system registry keys.

To modify the system TDR settings, you will need to edit the following registry key (or create it if it doesn't already exist):

```
KeyPath   : HKEY_LOCAL_MACHINE\System\CurrentControlSet\Control\GraphicsDrivers
KeyValue  : TdrDelay
ValueType : REG_DWORD
ValueData : Number of seconds to delay. The default value is 2 seconds.
```

You should pick a value that is sufficiently large for your training workload to run successfully on your particular GPU. For example, a timeout of 10 seconds is often sufficient for all but the most extreme of workloads.

Changes to this registry key require a system reboot to take effect.

For more information about TDR registry keys, see [Testing and debugging TDR](https://docs.microsoft.com/windows-hardware/drivers/display/tdr-registry-keys).
