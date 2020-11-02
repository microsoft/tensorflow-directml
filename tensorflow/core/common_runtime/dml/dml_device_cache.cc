/* Copyright (c) Microsoft Corporation.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/common_runtime/dml/dml_device_cache.h"

#include "tensorflow/core/common_runtime/dml/dml_adapter.h"
#include "tensorflow/core/common_runtime/dml/dml_adapter_impl.h"
#include "tensorflow/core/common_runtime/dml/dml_device.h"

using Microsoft::WRL::ComPtr;

namespace tensorflow {

// For now, only support devices that are guaranteed to work on all datatypes
// that the DML kernels use since we don't have a good way of knowing what the
// users might use the device for. In the future, we might want to use a
// different approach if the data shows that many people with old hardware
// want to use our package.
// TFDML #28244592
static bool SupportsAllDataTypes(const DmlAdapter& adapter) {
  D3D_FEATURE_LEVEL feature_level = adapter.IsComputeOnly()
                                        ? D3D_FEATURE_LEVEL_1_0_CORE
                                        : D3D_FEATURE_LEVEL_11_0;

  ComPtr<ID3D12Device> d3d12_device;
  DML_CHECK_SUCCEEDED(D3D12CreateDevice(adapter.Impl()->Get(), feature_level,
                                        IID_PPV_ARGS(&d3d12_device)));

  ComPtr<IDMLDevice> dml_device =
      CreateDmlDevice(d3d12_device.Get(), DML_CREATE_DEVICE_FLAG_NONE);

  std::array<DML_TENSOR_DATA_TYPE, 8> data_types = {
      DML_TENSOR_DATA_TYPE_FLOAT32, DML_TENSOR_DATA_TYPE_FLOAT16,
      DML_TENSOR_DATA_TYPE_UINT32,  DML_TENSOR_DATA_TYPE_UINT16,
      DML_TENSOR_DATA_TYPE_UINT8,   DML_TENSOR_DATA_TYPE_INT32,
      DML_TENSOR_DATA_TYPE_INT16,   DML_TENSOR_DATA_TYPE_INT8,
  };

  return std::all_of(
      data_types.begin(), data_types.end(),
      [&dml_device](DML_TENSOR_DATA_TYPE data_type) {
        DML_FEATURE_QUERY_TENSOR_DATA_TYPE_SUPPORT query{data_type};
        DML_FEATURE_DATA_TENSOR_DATA_TYPE_SUPPORT support;

        DML_CHECK_SUCCEEDED(dml_device->CheckFeatureSupport(
            DML_FEATURE_TENSOR_DATA_TYPE_SUPPORT, sizeof(query), &query,
            sizeof(support), &support));

        return support.IsSupported;
      });
}

static std::vector<DmlAdapter> FilterAdapters() {
  std::vector<DmlAdapter> adapters = EnumerateAdapters();
  adapters.erase(std::remove_if(adapters.begin(), adapters.end(),
                                [](const DmlAdapter& adapter) {
                                  return !SupportsAllDataTypes(adapter);
                                }),
                 adapters.end());

  return adapters;
}

DmlDeviceCache& DmlDeviceCache::Instance() {
  // Rely on magic statics to initialize this in a thread-safe manner. Note
  // that we never free this instance; it's a per-process singleton that's
  // intentionally leaked to avoid order-of-destruction issues during process
  // exit. This sounds unusual, but is done to explicitly match the behavior
  // of the CUDA device.
  static DmlDeviceCache* instance = new DmlDeviceCache();
  return *instance;
}

uint32_t DmlDeviceCache::GetAdapterCount() const {
  std::unique_lock<std::mutex> lock(mutex_);

  return static_cast<uint32_t>(adapters_.size());
}

// It is a little odd that we require GPUOptions and memory_limit here, as
// those can vary per TF device instance - they're not process-global. We
// handle this by using the options and memory limit that are provided to the
// first device created on this adapter. If subsequent devices are created on
// the same adapter but with different options/memory_limit, they are ignored.
// This is unusual, but matches the behavior of the CUDA device.
const DmlDeviceState* DmlDeviceCache::GetOrCreateDeviceState(
    uint32_t adapter_index, const GPUOptions& gpu_options,
    uint64_t memory_limit_in_bytes) {
  std::unique_lock<std::mutex> lock(mutex_);

  assert(adapter_index < adapters_.size());
  assert(adapters_.size() == device_states_.size());

  if (!device_states_[adapter_index]) {
    const DmlAdapter& adapter = adapters_[adapter_index];

    LOG(INFO) << "DirectML: creating device on adapter " << adapter_index
              << " (" << adapter.Name() << ")";

    device_states_[adapter_index] =
        DmlDeviceState::Create(adapter, gpu_options, memory_limit_in_bytes);
  }

  return device_states_[adapter_index].get();
}

const DmlAdapter& DmlDeviceCache::GetAdapter(uint32_t adapter_index) const {
  return adapters_[adapter_index];
}

DmlDeviceCache::DmlDeviceCache() : adapters_(FilterAdapters()) {
  device_states_.resize(adapters_.size());

  LOG(INFO) << "DirectML device enumeration: found " << adapters_.size()
            << " compatible adapters.";

  if (VLOG_IS_ON(1)) {
    for (size_t i = 0; i < adapters_.size(); ++i) {
      const auto& adapter = adapters_[i];
      auto driver_ver = adapter.DriverVersion().parts;

      VLOG(1) << "DirectML adapter " << i << ": " << adapter.Name();
      VLOG(1) << "    VendorID: 0x" << std::hex << (uint32_t)adapter.VendorID();
      VLOG(1) << "    DeviceID: 0x" << std::hex << adapter.DeviceID();
      VLOG(1) << "    Driver: " << driver_ver.a << "." << driver_ver.b << "."
              << driver_ver.c << "." << driver_ver.d;
      VLOG(1) << "    IsComputeOnly: "
              << (adapter.IsComputeOnly() ? "true" : "false");
    }
  }
}

}  // namespace tensorflow