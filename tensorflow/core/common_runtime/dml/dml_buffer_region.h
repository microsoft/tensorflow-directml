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
class D3D12BufferRegion {
 public:
  D3D12BufferRegion() = default;

  // References a region of a resource that remains in a fixed state.
  D3D12BufferRegion(uint64_t offset, uint64_t size_in_bytes,
                    D3D12_RESOURCE_STATES resource_state,
                    ID3D12Resource* resource,
                    ID3D12Resource* resource_copy_src_state = nullptr,
                    ID3D12Resource* resource_copy_dst_state = nullptr);

  // Move-only
  D3D12BufferRegion(const D3D12BufferRegion&) = delete;
  D3D12BufferRegion& operator=(const D3D12BufferRegion&) = delete;
  D3D12BufferRegion(D3D12BufferRegion&&) = default;
  D3D12BufferRegion& operator=(D3D12BufferRegion&&) = default;

  ID3D12Resource* ResourceInFixedState() const;
  ID3D12Resource* ResourceInCopySrcState() const;
  ID3D12Resource* ResourceInCopyDstState() const;
  uint64_t Offset() const;
  uint64_t SizeInBytes() const;
  D3D12_RESOURCE_STATES ResourceState() const { return resource_state_; }

  DML_BUFFER_BINDING GetBufferBinding() const;

  explicit operator bool() const { return resource_ != nullptr; }

  // Creates a subregion at an offset from the start of this region. If no size
  // is provided the region runs to the end of the current region.
  inline D3D12BufferRegion Subregion(uint64_t offset,
                                     uint64_t size_in_bytes = 0) const {
    // start of subregion must be within current region
    CHECK(offset < size_in_bytes_);
    size_in_bytes =
        size_in_bytes == 0 ? size_in_bytes_ - offset : size_in_bytes;
    // end of subregion must be within current region
    CHECK(size_in_bytes <= size_in_bytes_ - offset);

    return D3D12BufferRegion(offset_ + offset, size_in_bytes, resource_state_,
                             resource_, resource_copy_src_state_,
                             resource_copy_dst_state_);
  }

 private:
  ID3D12Resource* resource_ = nullptr;
  ID3D12Resource* resource_copy_src_state_ = nullptr;
  ID3D12Resource* resource_copy_dst_state_ = nullptr;
  uint64_t offset_ = 0;
  uint64_t size_in_bytes_ = 0;
  D3D12_RESOURCE_STATES resource_state_ = D3D12_RESOURCE_STATE_COMMON;
};

}  // namespace tensorflow
