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

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace tensorflow {

struct DriverVersion {
  union {
    struct {
      uint16_t d;
      uint16_t c;
      uint16_t b;
      uint16_t a;
    } parts;
    uint64_t value;
  };

  DriverVersion() = default;

  explicit DriverVersion(uint64_t value) : value(value) {}

  DriverVersion(uint16_t a, uint16_t b, uint16_t c, uint16_t d) {
    parts.a = a;
    parts.b = b;
    parts.c = c;
    parts.d = d;
  }
};

inline bool operator==(DriverVersion lhs, DriverVersion rhs) {
  return lhs.value == rhs.value;
}
inline bool operator!=(DriverVersion lhs, DriverVersion rhs) {
  return lhs.value != rhs.value;
}
inline bool operator<=(DriverVersion lhs, DriverVersion rhs) {
  return lhs.value <= rhs.value;
}
inline bool operator>=(DriverVersion lhs, DriverVersion rhs) {
  return lhs.value >= rhs.value;
}
inline bool operator<(DriverVersion lhs, DriverVersion rhs) {
  return lhs.value < rhs.value;
}
inline bool operator>(DriverVersion lhs, DriverVersion rhs) {
  return lhs.value > rhs.value;
}

enum class VendorID {
  kAmd = 0x1002,
  kNvidia = 0x10DE,
  kMicrosoft = 0x1414,
  kQualcomm = 0x4D4F4351,
  kIntel = 0x8086,
};

// Use a pimpl to firewall DML/D3D-specific headers from the rest of TF.
class DmlAdapterImpl;

// Represents a DXCore or DXGI adapter.
class DmlAdapter {
 public:
  explicit DmlAdapter(const DmlAdapterImpl& impl);
  ~DmlAdapter();

  const DmlAdapterImpl* Impl() const { return impl_.get(); }

  DriverVersion DriverVersion() const;
  VendorID VendorID() const;
  uint32_t DeviceID() const;
  const std::string& Name() const;
  bool IsComputeOnly() const;
  uint64_t GetTotalDedicatedMemory() const;
  uint64_t GetTotalSharedMemory() const;
  uint64_t QueryAvailableLocalMemory() const;

 private:
  // This object is immutable, so this is shared to allow copies.
  std::shared_ptr<DmlAdapterImpl> impl_;
};

// Retrieves a (statically allocated) string name for the given VendorID, e.g.
// "Microsoft", "Intel", etc.
const char* GetVendorName(VendorID id);

// Retrieves a list of DML-compatible hardware adapters on the system.
std::vector<DmlAdapter> EnumerateAdapters();

}  // namespace tensorflow
