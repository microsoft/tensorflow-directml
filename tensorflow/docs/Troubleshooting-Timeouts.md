# Troubleshooting GPU timeouts in tensorflow-directml

Because tensorflow-directml supports such a wide range of hardware (including consumer gaming and integrated GPUs), it is occasionally possible to encounter a training workload that causes a *GPU timeout* on some hardware/drivers.

GPU timeouts are caused by long-running workloads that exceed the system-defined GPU preemption timeout, which is 2 seconds by default. The exact circumstances that cause a GPU timeout depend on a wide variety of factors, including the specific hardware and driver configuration used.

If you encounter a GPU timeout in tensorflow-directml, you'll see an error message similar to the following:

```
The DirectML device has encountered an unrecoverable error (DXGI_ERROR_DEVICE_HUNG).
This is most often caused by a timeout occurring on the GPU. Please visit
https://aka.ms/??? for more information and troubleshooting steps.

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

GPU timeouts occur when a non-preemptable workload runs on the GPU for too long. During training, this is most likely to occur when performing expensive operations with very large tensors and/or large batch size.

If timeouts are occurring on your GPU, you can try using smaller tensors or lowering the batch size during training to reduce the compute workload.

### #5: Disable GPU timeouts using system-wide registry key (advanced users only)


