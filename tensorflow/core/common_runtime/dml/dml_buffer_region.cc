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

#include "dml_buffer_region.h"

#include "dml_heap_allocator.h"

namespace tensorflow {

D3D12BufferRegion::D3D12BufferRegion(
    D3D12HeapAllocator* allocator, uint32_t allocation_id,
    Microsoft::WRL::ComPtr<ID3D12Resource> resource, uint64_t offset,
    uint64_t size_in_bytes)
    : allocator_(allocator),
      allocation_id_(allocation_id),
      resource_(std::move(resource)),
      offset_(offset),
      size_in_bytes_(size_in_bytes) {
  assert(resource_ != nullptr);
  assert(size_in_bytes_ != 0);
  assert(resource_->GetDesc().Dimension == D3D12_RESOURCE_DIMENSION_BUFFER);
}

ID3D12Resource* D3D12BufferRegion::Resource() const {
  return resource_ ? resource_.Get() : nullptr;
}

uint64_t D3D12BufferRegion::Offset() const { return resource_ ? offset_ : 0; }

uint64_t D3D12BufferRegion::SizeInBytes() const {
  return resource_ ? size_in_bytes_ : 0;
}

DML_BUFFER_BINDING D3D12BufferRegion::GetBufferBinding() const {
  if (!resource_) {
    return DML_BUFFER_BINDING{};
  }

  return DML_BUFFER_BINDING{resource_.Get(), offset_, size_in_bytes_};
}

D3D12BufferRegion::~D3D12BufferRegion() {
  // resource_ can be null if this object gets moved-from
  if (resource_) {
    // Free the resource back to the allocator
    allocator_->ReleasePlacedResource(allocation_id_, std::move(resource_));
  }
}

}  // namespace tensorflow
