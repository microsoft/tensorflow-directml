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

#include "dml_common.h"

namespace tensorflow {

class D3D12HeapAllocator;

// Represents a region of a D3D12 buffer resource. A buffer region has an
// underlying ID3D12Resource* (of D3D12_RESOURCE_DIMENSION_BUFFER), an offset in
// bytes from the beginning of that buffer, and a size in bytes of the region.
// This object owns the underlying D3D12 resource; once this object is
// destructed the resource is returned back to its parent allocator.
class D3D12BufferRegion {
 public:
  D3D12BufferRegion() = default;
  D3D12BufferRegion(D3D12HeapAllocator* allocator, uint32_t allocation_id,
                  Microsoft::WRL::ComPtr<ID3D12Resource> resource,
                  uint64_t offset, uint64_t size_in_bytes);
  ~D3D12BufferRegion();

  // Move-only
  D3D12BufferRegion(const D3D12BufferRegion&) = delete;
  D3D12BufferRegion& operator=(const D3D12BufferRegion&) = delete;
  D3D12BufferRegion(D3D12BufferRegion&&) = default;
  D3D12BufferRegion& operator=(D3D12BufferRegion&&) = default;

  ID3D12Resource* Resource() const;
  uint64_t Offset() const;
  uint64_t SizeInBytes() const;

  DML_BUFFER_BINDING GetBufferBinding() const;

  explicit operator bool() const { return resource_ != nullptr; }

 private:
  D3D12HeapAllocator* allocator_ = nullptr;
  uint32_t allocation_id_ = 0;
  Microsoft::WRL::ComPtr<ID3D12Resource> resource_;
  uint64_t offset_ = 0;
  uint64_t size_in_bytes_ = 0;
};

}  // namespace tensorflow
