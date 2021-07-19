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
    D3D12_RESOURCE_STATES resource_state,
    Microsoft::WRL::ComPtr<ID3D12Resource> resource,
    Microsoft::WRL::ComPtr<ID3D12Resource> resource_copy_src_state,
    Microsoft::WRL::ComPtr<ID3D12Resource> resource_copy_dst_state)
    : offset_(offset),
      size_in_bytes_(size_in_bytes),
      resource_state_(resource_state),
      resource_(std::move(resource)),
      resource_copy_src_state_(std::move(resource_copy_src_state)),
      resource_copy_dst_state_(std::move(resource_copy_dst_state)) {
  assert(resource_ != nullptr);
  assert(size_in_bytes_ != 0);
  assert(resource_->GetDesc().Dimension == D3D12_RESOURCE_DIMENSION_BUFFER);
  // TODO: if other resources provided they should match size
}

ID3D12Resource* D3D12BufferRegion::ResourceInFixedState() const {
  return resource_ ? resource_.Get() : nullptr;
}

ID3D12Resource* D3D12BufferRegion::ResourceInCopySrcState() const {
  return resource_copy_src_state_ ? resource_copy_src_state_.Get() : nullptr;
}

ID3D12Resource* D3D12BufferRegion::ResourceInCopyDstState() const {
  return resource_copy_dst_state_ ? resource_copy_dst_state_.Get() : nullptr;
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

}  // namespace tensorflow
