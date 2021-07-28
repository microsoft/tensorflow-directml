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

#include "dml_buffer_region.h"
#include "dml_common.h"

namespace tensorflow {

class DmlAllocator;

// Owns a D3D12 default heap buffer allocated using the DML device's
// allocator. This is essentially a convenience wrapper over a device memory
// allocation as well as the buffer region that spans it. When this object is
// destructed, the device memory is freed to the allocator.
class DmlBuffer {
 public:
  DmlBuffer() = default;
  explicit DmlBuffer(DmlAllocator* allocator, uint64_t size_in_bytes);
  ~DmlBuffer();

  // Move-only
  DmlBuffer(const DmlBuffer&) = delete;
  DmlBuffer& operator=(const DmlBuffer&) = delete;
  DmlBuffer(DmlBuffer&&) = default;
  DmlBuffer& operator=(DmlBuffer&&) = default;

  ID3D12Resource* ResourceInUavState() const;
  ID3D12Resource* ResourceInCopySrcState() const;
  ID3D12Resource* ResourceInCopyDstState() const;
  uint64_t Offset() const;
  uint64_t SizeInBytes() const;
  const D3D12BufferRegion& Region() const { return buffer_region_; }

  DML_BUFFER_BINDING GetBufferBinding() const;

  explicit operator bool() const { return !!buffer_region_; }

 private:
  DmlAllocator* allocator_;  // weak; owned by the DML device factory
  void* ptr_ = nullptr;
  D3D12BufferRegion buffer_region_;
};

}  // namespace tensorflow
