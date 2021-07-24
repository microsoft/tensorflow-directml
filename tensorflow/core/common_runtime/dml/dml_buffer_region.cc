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
    uint64_t offset, uint64_t size_in_bytes,
    Microsoft::WRL::ComPtr<ID3D12Resource> resource_uav_state,
    Microsoft::WRL::ComPtr<ID3D12Resource> resource_copy_src_state,
    Microsoft::WRL::ComPtr<ID3D12Resource> resource_copy_dst_state)
    : offset_(offset),
      size_in_bytes_(size_in_bytes),
      resource_uav_state_(std::move(resource_uav_state)),
      resource_copy_src_state_(std::move(resource_copy_src_state)),
      resource_copy_dst_state_(std::move(resource_copy_dst_state)) {
  // Get a raw pointer to the first non-null resource passed in. At least one
  // resource must be provided.
  first_valid_resource_ = resource_uav_state_.Get();
  if (!first_valid_resource_) {
    first_valid_resource_ = resource_copy_src_state_.Get();
  }
  if (!first_valid_resource_) {
    first_valid_resource_ = resource_copy_dst_state_.Get();
  }
  CHECK(first_valid_resource_ != nullptr);

  // Regions cannot be empty.
  CHECK(size_in_bytes_ != 0);

  // Regions cannot extend beyond the size of the resource.
  uint64_t buffer_size = first_valid_resource_->GetDesc().Width;
  CHECK(offset_ < buffer_size);
  CHECK(size_in_bytes_ <= buffer_size - offset);

  // All three resources, if provided, must be identical aside from state.
  assert(first_valid_resource_->GetDesc().Dimension ==
         D3D12_RESOURCE_DIMENSION_BUFFER);
  assert(!resource_uav_state ||
         (resource_uav_state->GetDesc().Dimension ==
              D3D12_RESOURCE_DIMENSION_BUFFER &&
          resource_uav_state->GetDesc().Width == buffer_size));
  assert(!resource_copy_src_state_ ||
         (resource_copy_src_state_->GetDesc().Dimension ==
              D3D12_RESOURCE_DIMENSION_BUFFER &&
          resource_copy_src_state_->GetDesc().Width == buffer_size));
  assert(!resource_copy_dst_state_ ||
         (resource_copy_dst_state_->GetDesc().Dimension ==
              D3D12_RESOURCE_DIMENSION_BUFFER &&
          resource_copy_dst_state_->GetDesc().Width == buffer_size));
}

ID3D12Resource* D3D12BufferRegion::ResourceInUavState() const {
  return resource_uav_state_.Get();
}

ID3D12Resource* D3D12BufferRegion::ResourceInCopySrcState() const {
  return resource_copy_src_state_.Get();
}

ID3D12Resource* D3D12BufferRegion::ResourceInCopyDstState() const {
  return resource_copy_dst_state_.Get();
}

uint64_t D3D12BufferRegion::Offset() const {
  return first_valid_resource_ ? offset_ : 0;
}

uint64_t D3D12BufferRegion::SizeInBytes() const {
  return first_valid_resource_ ? size_in_bytes_ : 0;
}

DML_BUFFER_BINDING D3D12BufferRegion::GetBufferBinding() const {
  if (!resource_uav_state_) {
    return DML_BUFFER_BINDING{};
  }

  return DML_BUFFER_BINDING{resource_uav_state_.Get(), offset_, size_in_bytes_};
}

}  // namespace tensorflow
