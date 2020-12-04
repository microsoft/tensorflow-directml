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

#include "tensorflow/core/common_runtime/dml/dml_adapter_impl.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"

#if _WIN32
#include "tensorflow/core/platform/windows/wide_char.h"
#endif

#include <numeric>

using Microsoft::WRL::ComPtr;

namespace tensorflow {

Status ParseVisibleDeviceList(const string& visible_device_list,
                              uint32_t num_valid_adapters, bool skip_invalid,
                              /*out*/ std::vector<uint32_t>* adapter_indices) {
  if (visible_device_list.empty()) {
    // By default, don't filter anything
    adapter_indices->resize(num_valid_adapters);
    std::iota(adapter_indices->begin(), adapter_indices->end(), 0);
    return Status::OK();
  }

  const std::vector<string> indices_str =
      str_util::Split(visible_device_list, ',');
  adapter_indices->clear();
  adapter_indices->reserve(indices_str.size());

  for (const string& index_str : indices_str) {
    uint32_t adapter_index;
    if (!strings::safe_strtou32(index_str, &adapter_index)) {
      if (skip_invalid) {
        break;
      } else {
        return errors::InvalidArgument(
            "Could not parse entry in 'visible_device_list': '", index_str,
            "'. visible_device_list = ", visible_device_list);
      }
    }

    if (adapter_index >= num_valid_adapters) {
      if (skip_invalid) {
        break;
      } else {
        return errors::InvalidArgument(
            "'visible_device_list' listed an invalid GPU id '", adapter_index,
            "' but visible device count is ", num_valid_adapters);
      }
    }

    adapter_indices->push_back(adapter_index);
  }

  // Validate no repeats.
  std::set<uint32_t> adapter_index_set(adapter_indices->begin(),
                                       adapter_indices->end());
  if (adapter_index_set.size() != adapter_index_set.size()) {
    return errors::InvalidArgument(
        "visible_device_list contained a duplicate entry: ",
        visible_device_list);
  }

  return Status::OK();
}

// Filter the list of adapters by the DML_VISIBLE_DEVICES environment
// variable. This is called during adapter enumeration to emulate the behavior
// of CUDA_VISIBLE_DEVICES which hides devices from the process itself. By
// filtering at adapter enumeration time, we can ensure that everybody
// (including the device factory) never sees adapters filtered out by
// DML_VISIBLE_DEVICES.
std::vector<DmlAdapterImpl> FilterAdapterListFromEnvVar(
    absl::Span<const DmlAdapterImpl> adapters) {
  const char* dml_visible_devices = getenv("DML_VISIBLE_DEVICES");
  if (dml_visible_devices == nullptr || strlen(dml_visible_devices) == 0) {
    return std::vector<DmlAdapterImpl>(adapters.begin(), adapters.end());
  }

  std::vector<uint32_t> visible_device_list;

  const bool skip_invalid = true;
  TF_CHECK_OK(ParseVisibleDeviceList(dml_visible_devices, adapters.size(),
                                     skip_invalid, &visible_device_list));

  std::vector<DmlAdapterImpl> filtered_adapters;
  filtered_adapters.reserve(visible_device_list.size());
  for (uint32_t adapter_index : visible_device_list) {
    // This should have been validated by ParseVisibleDeviceList
    assert(adapter_index < adapters.size());

    filtered_adapters.push_back(adapters[adapter_index]);
  }

  return filtered_adapters;
}

DmlAdapterImpl::DmlAdapterImpl(LUID adapter_luid) {
#if _WIN32
  ComPtr<IDXGIFactory4> dxgi_factory;
  DML_CHECK_SUCCEEDED(CreateDXGIFactory(IID_PPV_ARGS(&dxgi_factory)));

  ComPtr<IDXGIAdapter1> adapter;
  DML_CHECK_SUCCEEDED(
      dxgi_factory->EnumAdapterByLuid(adapter_luid, IID_PPV_ARGS(&adapter)));

  Initialize(adapter.Get());
#else
  ComPtr<IDXCoreAdapterFactory> dxcore_factory;
  DML_CHECK_SUCCEEDED(
      DXCoreCreateAdapterFactory(IID_PPV_ARGS(&dxcore_factory)));

  ComPtr<IDXCoreAdapter> adapter;
  DML_CHECK_SUCCEEDED(
      dxcore_factory->GetAdapterByLuid(adapter_luid, IID_PPV_ARGS(&adapter)));

  Initialize(adapter.Get());
#endif
}

#if _WIN32
DmlAdapterImpl::DmlAdapterImpl(IDXGIAdapter* adapter) { Initialize(adapter); }

void DmlAdapterImpl::Initialize(IDXGIAdapter* adapter) {
  DXGI_ADAPTER_DESC desc = {};
  DML_CHECK_SUCCEEDED(adapter->GetDesc(&desc));

  LARGE_INTEGER driver_version;
  DML_CHECK_SUCCEEDED(
      adapter->CheckInterfaceSupport(IID_IDXGIDevice, &driver_version));

  adapter_ = adapter;
  driver_version_ = tensorflow::DriverVersion(driver_version.QuadPart);
  vendor_id_ = static_cast<tensorflow::VendorID>(desc.VendorId);
  device_id_ = desc.DeviceId;
  description_ = WideCharToUtf8(desc.Description);
  dedicated_memory_in_bytes_ = desc.DedicatedVideoMemory;
  shared_memory_in_bytes_ = desc.SharedSystemMemory;
  is_compute_only_ = false;
}

uint64_t DmlAdapterImpl::QueryAvailableLocalMemory() const {
  ComPtr<IDXGIAdapter3> adapter3;
  DML_CHECK_SUCCEEDED(adapter_.As(&adapter3));

  DXGI_QUERY_VIDEO_MEMORY_INFO info = {};
  DML_CHECK_SUCCEEDED(adapter3->QueryVideoMemoryInfo(
      0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info));

  return info.Budget;
}

std::vector<DmlAdapterImpl> EnumerateAdapterImpls() {
  ComPtr<IDXGIFactory6> dxgi_factory;
  DML_CHECK_SUCCEEDED(CreateDXGIFactory(IID_PPV_ARGS(&dxgi_factory)));

  const D3D_FEATURE_LEVEL min_feature_level = D3D_FEATURE_LEVEL_11_0;
  std::vector<DmlAdapterImpl> adapter_infos;

  uint32_t adapter_index = 0;
  ComPtr<IDXGIAdapter1> adapter;
  while (dxgi_factory->EnumAdapterByGpuPreference(
             adapter_index, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
             IID_PPV_ARGS(&adapter)) != DXGI_ERROR_NOT_FOUND) {
    DXGI_ADAPTER_DESC1 desc = {};
    DML_CHECK_SUCCEEDED(adapter->GetDesc1(&desc));

    // Since we enumerate by performance, we can ignore everything that comes
    // after the first software adapter, which includes the IDD adapters. This
    // is necessary for now because IDD adapters don't have the
    // DXGI_ADAPTER_FLAG_SOFTWARE flag, even though they run on software.
    // TFDML #21433167

    // See here for documentation on filtering WARP adapter:
    // https://docs.microsoft.com/en-us/windows/desktop/direct3ddxgi/d3d10-graphics-programming-guide-dxgi#new-info-about-enumerating-adapters-for-windows-8
    const bool is_basic_render_driver_vendor_id =
        desc.VendorId == static_cast<UINT>(VendorID::kMicrosoft);
    const bool is_basic_render_driver_device_id = desc.DeviceId == 0x8c;

    if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE ||
        (is_basic_render_driver_vendor_id &&
         is_basic_render_driver_device_id)) {
      break;
    }

    HRESULT hr = D3D12CreateDevice(adapter.Get(), min_feature_level,
                                   IID_ID3D12Device, nullptr);
    if (SUCCEEDED(hr)) {
      adapter_infos.emplace_back(adapter.Get());
    }

    ++adapter_index;
    adapter = nullptr;
  }

  return FilterAdapterListFromEnvVar(adapter_infos);
}

#else

DmlAdapterImpl::DmlAdapterImpl(IDXCoreAdapter* adapter) { Initialize(adapter); }

void DmlAdapterImpl::Initialize(IDXCoreAdapter* adapter) {
  DXCoreHardwareID hardware_id = {};
  DML_CHECK_SUCCEEDED(
      adapter->GetProperty(DXCoreAdapterProperty::HardwareID, &hardware_id));

  size_t driver_description_size;
  DML_CHECK_SUCCEEDED(adapter->GetPropertySize(
      DXCoreAdapterProperty::DriverDescription, &driver_description_size));

  std::vector<char> driver_description(driver_description_size);
  DML_CHECK_SUCCEEDED(
      adapter->GetProperty(DXCoreAdapterProperty::DriverDescription,
                           driver_description_size, driver_description.data()));

  LARGE_INTEGER driver_version;
  DML_CHECK_SUCCEEDED(adapter->GetProperty(DXCoreAdapterProperty::DriverVersion,
                                           sizeof(driver_version),
                                           &driver_version));

  DML_CHECK_SUCCEEDED(adapter->GetProperty(
      DXCoreAdapterProperty::DedicatedAdapterMemory,
      sizeof(dedicated_memory_in_bytes_), &dedicated_memory_in_bytes_));

  DML_CHECK_SUCCEEDED(adapter->GetProperty(
      DXCoreAdapterProperty::SharedSystemMemory,
      sizeof(shared_memory_in_bytes_), &shared_memory_in_bytes_));

  const bool is_graphics_supported =
      adapter->IsAttributeSupported(DXCORE_ADAPTER_ATTRIBUTE_D3D12_GRAPHICS);

  adapter_ = adapter;
  driver_version_ = tensorflow::DriverVersion(driver_version.QuadPart);
  vendor_id_ = static_cast<tensorflow::VendorID>(hardware_id.vendorID);
  device_id_ = hardware_id.deviceID;
  description_.assign(driver_description.begin(), driver_description.end());
  is_compute_only_ = !is_graphics_supported;
}

uint64_t DmlAdapterImpl::QueryAvailableLocalMemory() const {
  ComPtr<IDXCoreAdapter> dxcore_adapter;
  DML_CHECK_SUCCEEDED(adapter_.As(&dxcore_adapter));

  DXCoreAdapterMemoryBudgetNodeSegmentGroup query = {};
  query.nodeIndex = 0;
  query.segmentGroup = DXCoreSegmentGroup::Local;

  DXCoreAdapterMemoryBudget info = {};
  DML_CHECK_SUCCEEDED(dxcore_adapter->QueryState(
      DXCoreAdapterState::AdapterMemoryBudget, &query, &info));

  return info.budget;
}

std::vector<DmlAdapterImpl> EnumerateAdapterImpls() {
  const GUID dxcore_adapter = DXCORE_ADAPTER_ATTRIBUTE_D3D12_CORE_COMPUTE;

  ComPtr<IDXCoreAdapterFactory> adapter_factory;
  DML_CHECK_SUCCEEDED(
      DXCoreCreateAdapterFactory(IID_PPV_ARGS(&adapter_factory)));

  ComPtr<IDXCoreAdapterList> adapter_list;
  DML_CHECK_SUCCEEDED(adapter_factory->CreateAdapterList(
      1, &dxcore_adapter, IID_PPV_ARGS(&adapter_list)));

  // Sort the adapters so that performant adapters are selected first
  DXCoreAdapterPreference sort_preferences[] = {
      DXCoreAdapterPreference::HighPerformance,
  };

  DML_CHECK_SUCCEEDED(adapter_list->Sort(
      static_cast<uint32_t>(ABSL_ARRAYSIZE(sort_preferences)),
      sort_preferences));

  std::vector<DmlAdapterImpl> adapter_infos;

  for (uint32_t i = 0; i < adapter_list->GetAdapterCount(); i++) {
    ComPtr<IDXCoreAdapter> adapter;
    DML_CHECK_SUCCEEDED(adapter_list->GetAdapter(i, IID_PPV_ARGS(&adapter)));

    bool is_hardware_adapter = false;
    DML_CHECK_SUCCEEDED(adapter->GetProperty(DXCoreAdapterProperty::IsHardware,
                                             &is_hardware_adapter));

    DXCoreHardwareID hardware_id = {};
    DML_CHECK_SUCCEEDED(
        adapter->GetProperty(DXCoreAdapterProperty::HardwareID, &hardware_id));

    // See here for documentation on filtering WARP adapter:
    // https://docs.microsoft.com/en-us/windows/desktop/direct3ddxgi/d3d10-graphics-programming-guide-dxgi#new-info-about-enumerating-adapters-for-windows-8
    const bool is_basic_render_driver_vendor_id =
        hardware_id.vendorID == static_cast<UINT>(VendorID::kMicrosoft);
    const bool is_basic_render_driver_device_id = hardware_id.deviceID == 0x8c;

    // Since we enumerate by performance, we can ignore everything that comes
    // after the first software adapter, which includes the IDD adapters. This
    // is necessary for now because IDD adapters are considered hardware
    // adapters, even though they run on software.
    // TFDML #21433167
    if (!is_hardware_adapter || (is_basic_render_driver_vendor_id &&
                                 is_basic_render_driver_device_id)) {
      break;
    }

    DmlAdapterImpl adapter_impl(adapter.Get());

    D3D_FEATURE_LEVEL feature_level = adapter_impl.IsComputeOnly()
                                          ? D3D_FEATURE_LEVEL_1_0_CORE
                                          : D3D_FEATURE_LEVEL_11_0;

    HRESULT hr = D3D12CreateDevice(adapter.Get(), feature_level,
                                   IID_ID3D12Device, nullptr);
    if (SUCCEEDED(hr)) {
      adapter_infos.push_back(std::move(adapter_impl));
    }
  }

  return FilterAdapterListFromEnvVar(adapter_infos);
}

#endif

}  // namespace tensorflow