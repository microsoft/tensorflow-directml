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

#include "dml_util.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/dml/dml_adapter.h"
#include "tensorflow/core/common_runtime/dml/dml_adapter_impl.h"
#include "tensorflow/core/common_runtime/dml/dml_bfc_allocator.h"
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

// Maintains a static cache of device singletons, one per adapter. This class is
// thread-safe.
class DmlDeviceCache {
 public:
  static DmlDeviceCache& Instance() {
    // Rely on magic statics to initialize this in a thread-safe manner. Note
    // that we never free this instance; it's a per-process singleton that's
    // intentionally leaked to avoid order-of-destruction issues during process
    // exit. This sounds unusual, but is done to explicitly match the behavior
    // of the CUDA device.
    static DmlDeviceCache* instance = new DmlDeviceCache();
    return *instance;
  }

  uint32_t GetAdapterCount() const {
    std::unique_lock<std::mutex> lock(mutex_);

    return static_cast<uint32_t>(adapters_.size());
  }

  // It is a little odd that we require GPUOptions and memory_limit here, as
  // those can vary per TF device instance - they're not process-global. We
  // handle this by using the options and memory limit that are provided to the
  // first device created on this adapter. If subsequent devices are created on
  // the same adapter but with different options/memory_limit, they are ignored.
  // This is unusual, but matches the behavior of the CUDA device.
  const DmlDeviceState* GetOrCreateDeviceState(uint32_t adapter_index,
                                               const GPUOptions& gpu_options,
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

  const DmlAdapter& GetAdapter(uint32_t adapter_index) const {
    return adapters_[adapter_index];
  }

 private:
  DmlDeviceCache() : adapters_(FilterAdapters()) {
    device_states_.resize(adapters_.size());

    LOG(INFO) << "DirectML device enumeration: found " << adapters_.size()
              << " compatible adapters.";

    if (VLOG_IS_ON(1)) {
      for (size_t i = 0; i < adapters_.size(); ++i) {
        const auto& adapter = adapters_[i];
        auto driver_ver = adapter.DriverVersion().parts;

        VLOG(1) << "DirectML adapter " << i << ": " << adapter.Name();
        VLOG(1) << "    VendorID: 0x" << std::hex
                << (uint32_t)adapter.VendorID();
        VLOG(1) << "    DeviceID: 0x" << std::hex << adapter.DeviceID();
        VLOG(1) << "    Driver: " << driver_ver.a << "." << driver_ver.b << "."
                << driver_ver.c << "." << driver_ver.d;
        VLOG(1) << "    IsComputeOnly: "
                << (adapter.IsComputeOnly() ? "true" : "false");
      }
    }
  }

  mutable std::mutex mutex_;

  std::vector<DmlAdapter> adapters_;

  // Lazily constructed
  std::vector<std::unique_ptr<DmlDeviceState>> device_states_;
};

static std::unique_ptr<DmlDevice> CreateDevice(
    const SessionOptions& options, const string& name_prefix, int device_index,
    const DmlDeviceState* device_state, int64 memory_limit) {
  string name =
      strings::StrCat(name_prefix, "/device:", DEVICE_DML, ":", device_index);

  const DeviceAttributes attributes = Device::BuildDeviceAttributes(
      name, tensorflow::DeviceType(DEVICE_DML), Bytes(memory_limit),
      DeviceLocality(), device_state->adapter->Name());

  return absl::make_unique<DmlDevice>(device_state, options, attributes);
}

class DmlDeviceFactory : public DeviceFactory {
 public:
  Status ListPhysicalDevices(std::vector<string>* devices) override {
    auto& device_cache = DmlDeviceCache::Instance();

    for (uint32_t i = 0; i < device_cache.GetAdapterCount(); ++i) {
      string name = strings::StrCat("/physical_device:", DEVICE_DML, ":", i);
      devices->push_back(std::move(name));
    }
    return Status::OK();
  }

  Status CreateDevices(const SessionOptions& options, const string& name_prefix,
                       std::vector<std::unique_ptr<Device>>* devices) override {
    auto& device_cache = DmlDeviceCache::Instance();

    // By default, if the device count is not specified, create a device for
    // every available adapter. This matches the behavior of CUDA.
    uint32_t dml_device_count = device_cache.GetAdapterCount();

    auto deviceIterator = options.config.device_count().find(DEVICE_DML);
    if (deviceIterator != options.config.device_count().end()) {
      dml_device_count = deviceIterator->second;
    }

    dml_device_count =
        std::min<uint32_t>(dml_device_count, device_cache.GetAdapterCount());

    if (dml_device_count == 0) {
      // Nothing to do; bail early
      return Status::OK();
    }

    // Filter the full list of adapters by the visible_device_list, if
    // applicable

    std::vector<uint32_t> valid_adapter_indices;

    const bool skip_invalid = false;
    TF_RETURN_IF_ERROR(ParseVisibleDeviceList(
        options.config.gpu_options().visible_device_list(),
        device_cache.GetAdapterCount(), skip_invalid, &valid_adapter_indices));
    dml_device_count =
        std::min<uint32_t>(dml_device_count, valid_adapter_indices.size());

    const auto& gpu_options = options.config.gpu_options();
    const auto& virtual_devices = gpu_options.experimental().virtual_devices();
    double memory_fraction = gpu_options.per_process_gpu_memory_fraction();

    if (!virtual_devices.empty()) {
      if (memory_fraction > 0.0f) {
        return errors::InvalidArgument(
            "It's invalid to set per_process_gpu_memory_fraction when "
            "virtual_devices is set.");
      }

      if (dml_device_count < virtual_devices.size()) {
        return errors::Unknown(
            "Not enough GPUs to create virtual devices. dml_device_count: ",
            dml_device_count, " #virtual_devices: ", virtual_devices.size());
      }

      // We've verified that dml_device_count >= virtual_devices.size().
      dml_device_count = virtual_devices.size();
    }

    int virtual_device_index = 0;

    for (uint32_t i : valid_adapter_indices) {
      const auto& adapter = device_cache.GetAdapter(i);

      if (virtual_devices.empty() ||
          virtual_devices.Get(i).memory_limit_mb_size() == 0) {
        int64 memory_limit = 0;
        if (memory_fraction > 0.0f) {
          // A per_process_gpu_memory_fraction was specified: compute the memory
          // limit as a fraction of TOTAL GPU memory
          uint64_t total_gpu_memory = adapter.GetTotalDedicatedMemory();

          memory_limit = total_gpu_memory * memory_fraction;
        } else {
          // No per_process_gpu_memory_fraction was specified: use a memory
          // limit equal to the AVAILALBLE GPU memory
          uint64_t available_gpu_memory =
              adapter.QueryAvailableDedicatedMemory();

          memory_limit = available_gpu_memory;
        }

        const DmlDeviceState* device_state =
            device_cache.GetOrCreateDeviceState(i, gpu_options, memory_limit);

        auto dml_device =
            CreateDevice(options, name_prefix, virtual_device_index++,
                         device_state, memory_limit);

        devices->push_back(std::move(dml_device));
      } else {
        // A single Virtual Device can be divided in multiple sub-devices
        // depending on the memory allocation required by the user
        const auto& memory_limit_mb = virtual_devices.Get(i).memory_limit_mb();

        for (auto it = memory_limit_mb.begin(); it != memory_limit_mb.end();
             ++it) {
          int64 memory_limit = static_cast<int64>(*it) * (1ll << 20);

          const DmlDeviceState* device_state =
              device_cache.GetOrCreateDeviceState(i, gpu_options, memory_limit);
          auto dml_device =
              CreateDevice(options, name_prefix, virtual_device_index++,
                           device_state, memory_limit);

          devices->push_back(std::move(dml_device));
        }
      }
    }

    return Status::OK();
  }
};

REGISTER_LOCAL_DEVICE_FACTORY(DEVICE_DML, DmlDeviceFactory, 300);
}  // namespace tensorflow
