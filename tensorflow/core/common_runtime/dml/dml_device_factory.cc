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
#include "tensorflow/core/common_runtime/dml/dml_device_cache.h"

using Microsoft::WRL::ComPtr;

namespace tensorflow {

// Constructs a physical_device_desc string from an adapter. DML's
// physical_device_desc is JSON-encoded which allows for easier parsing.
//
// Example CUDA string:
//   'device: 0, name: TITAN V, pci bus id: 0000:65:00.0, compute
//   capability: 7.0'
//
// Example DML string:
//   '{"name": "NVIDIA TITAN V", "vendor_id": 4318, "device_id": 7553,
//   "driver_version": "27.21.14.5206"}'
static std::string GetPhysicalDeviceDesc(const DmlAdapter& adapter) {
  auto driver_ver = adapter.DriverVersion().parts;
  std::string driver_ver_str = strings::StrCat(
      driver_ver.a, ".", driver_ver.b, ".", driver_ver.c, ".", driver_ver.d);

  std::stringstream ss;
  ss << '{';
  ss << "\"name\": \"" << adapter.Name() << "\", ";
  ss << "\"vendor_id\": " << (uint32_t)adapter.VendorID() << ", ";
  ss << "\"device_id\": " << adapter.DeviceID() << ", ";
  ss << "\"driver_version\": \"" << driver_ver_str << "\"";
  ss << '}';

  return ss.str();
}

static std::unique_ptr<DmlDevice> CreateDevice(
    const SessionOptions& options, const string& name_prefix, int device_index,
    const DmlDeviceState* device_state, int64 memory_limit) {
  string name =
      strings::StrCat(name_prefix, "/device:", DEVICE_DML, ":", device_index);

  const DeviceAttributes attributes = Device::BuildDeviceAttributes(
      name, tensorflow::DeviceType(DEVICE_DML), Bytes(memory_limit),
      DeviceLocality(), GetPhysicalDeviceDesc(*device_state->adapter));

  return absl::make_unique<DmlDevice>(device_state, options, attributes);
}

static bool IsUmaAdapter(const DmlAdapter& adapter) {
  D3D_FEATURE_LEVEL feature_level = adapter.IsComputeOnly()
                                        ? D3D_FEATURE_LEVEL_1_0_CORE
                                        : D3D_FEATURE_LEVEL_11_0;

  ComPtr<ID3D12Device> d3d12_device;
  DML_CHECK_SUCCEEDED(D3D12CreateDevice(adapter.Impl()->Get(), feature_level,
                                        IID_PPV_ARGS(&d3d12_device)));

  D3D12_FEATURE_DATA_ARCHITECTURE1 feature_data = {};
  DML_CHECK_SUCCEEDED(d3d12_device->CheckFeatureSupport(
      D3D12_FEATURE_ARCHITECTURE1, &feature_data, sizeof(feature_data)));

  return feature_data.CacheCoherentUMA;
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

          if (IsUmaAdapter(adapter)) {
            // For adapters with unified memory architecture (UMA), add shared
            // memory to the total GPU memory
            total_gpu_memory += adapter.GetTotalSharedMemory();
          }

          memory_limit = total_gpu_memory * memory_fraction;
        } else {
          // No per_process_gpu_memory_fraction was specified: use a memory
          // limit equal to the AVAILALBLE GPU memory
          uint64_t available_gpu_memory = adapter.QueryAvailableLocalMemory();

          memory_limit = available_gpu_memory;
        }

        const DmlDeviceState* device_state =
            device_cache.GetOrCreateDeviceState(i, gpu_options, memory_limit);

        auto dml_device =
            CreateDevice(options, name_prefix, virtual_device_index,
                         device_state, memory_limit);

        TF_RETURN_IF_ERROR(
            device_cache.MapDeviceIdToAdapterIndex(virtual_device_index++, i));

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
              CreateDevice(options, name_prefix, virtual_device_index,
                           device_state, memory_limit);

          TF_RETURN_IF_ERROR(device_cache.MapDeviceIdToAdapterIndex(
              virtual_device_index++, i));

          devices->push_back(std::move(dml_device));
        }
      }
    }

    return Status::OK();
  }
};

REGISTER_LOCAL_DEVICE_FACTORY(DEVICE_DML, DmlDeviceFactory, 300);
}  // namespace tensorflow
