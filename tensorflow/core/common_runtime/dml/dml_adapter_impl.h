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

#pragma once

#include "tensorflow/core/common_runtime/dml/dml_adapter.h"
#include "tensorflow/core/common_runtime/dml/dml_common.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

// Represents a DXCore or DXGI adapter.
class DmlAdapterImpl {
 public:
  /*implicit*/ DmlAdapterImpl(LUID adapterLuid);

#if _WIN32
  /*implicit*/ DmlAdapterImpl(IDXGIAdapter* adapter);
#else
  /*implicit*/ DmlAdapterImpl(IDXCoreAdapter* adapter);
#endif

  IUnknown* Get() const { return adapter_.Get(); }

  DriverVersion DriverVersion() const { return driver_version_; }
  VendorID VendorID() const { return vendor_id_; }
  uint32_t DeviceID() const { return device_id_; }
  const std::string& Name() const { return description_; }
  bool IsComputeOnly() const { return is_compute_only_; }

  uint64_t GetTotalDedicatedMemory() const {
    return dedicated_memory_in_bytes_;
  }

  uint64_t GetTotalSharedMemory() const {
    return shared_memory_in_bytes_;
  }

  uint64_t QueryAvailableLocalMemory() const;

 private:
#if _WIN32
  void Initialize(IDXGIAdapter* adapter);
#else
  void Initialize(IDXCoreAdapter* adapter);
#endif

  Microsoft::WRL::ComPtr<IUnknown> adapter_;

  tensorflow::DriverVersion driver_version_;
  tensorflow::VendorID vendor_id_;
  uint32_t device_id_;
  std::string description_;
  bool is_compute_only_;
  uint64_t dedicated_memory_in_bytes_;
  uint64_t shared_memory_in_bytes_;
};

// Retrieves a list of DML-compatible hardware adapters on the system.
std::vector<DmlAdapterImpl> EnumerateAdapterImpls();

// Parse 'visible_device_list' into a list of adapter indices. See
// tensorflow/core/protobuf/config.proto for more information about the format
// and usage of the 'visible_device_list' string. Setting skip_invalid to true
// causes this function to behave similarly to CUDA_VISIBLE_DEVICES, where
// entries after an invalid index are simply ignored.
Status ParseVisibleDeviceList(const string& visible_device_list,
                              uint32_t num_valid_adapters, bool skip_invalid,
                              /*out*/ std::vector<uint32_t>* adapter_indices);

}  // namespace tensorflow