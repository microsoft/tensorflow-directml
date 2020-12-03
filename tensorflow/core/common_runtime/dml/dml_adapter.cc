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

#include "tensorflow/core/common_runtime/dml/dml_adapter.h"

#include "tensorflow/core/common_runtime/dml/dml_adapter_impl.h"

namespace tensorflow {

DmlAdapter::DmlAdapter(const DmlAdapterImpl& impl)
    : impl_(std::make_shared<DmlAdapterImpl>(impl)) {}

DmlAdapter::~DmlAdapter() = default;

DriverVersion DmlAdapter::DriverVersion() const {
  return impl_->DriverVersion();
}

VendorID DmlAdapter::VendorID() const { return impl_->VendorID(); }
uint32_t DmlAdapter::DeviceID() const { return impl_->DeviceID(); }
const std::string& DmlAdapter::Name() const { return impl_->Name(); }
bool DmlAdapter::IsComputeOnly() const { return impl_->IsComputeOnly(); }

uint64_t DmlAdapter::GetTotalDedicatedMemory() const {
  return impl_->GetTotalDedicatedMemory();
}

uint64_t DmlAdapter::GetTotalSharedMemory() const {
  return impl_->GetTotalSharedMemory();
}

uint64_t DmlAdapter::QueryAvailableLocalMemory() const {
  return impl_->QueryAvailableLocalMemory();
}

const char* GetVendorName(VendorID id) {
  switch (id) {
    case VendorID::kAmd:
      return "AMD";
    case VendorID::kNvidia:
      return "NVIDIA";
    case VendorID::kMicrosoft:
      return "Microsoft";
    case VendorID::kQualcomm:
      return "Qualcomm";
    case VendorID::kIntel:
      return "Intel";
    default:
      return "Unknown";
  }
}

std::vector<DmlAdapter> EnumerateAdapters() {
  auto impls = EnumerateAdapterImpls();

  std::vector<DmlAdapter> adapters;
  adapters.reserve(impls.size());

  for (auto&& impl : impls) {
    adapters.push_back(DmlAdapter(std::move(impl)));
  }

  return adapters;
}

}  // namespace tensorflow