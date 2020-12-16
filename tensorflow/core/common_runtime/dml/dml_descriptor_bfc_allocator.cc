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

#include "dml_descriptor_bfc_allocator.h"

namespace tensorflow {

DmlDescriptorAllocator::DmlDescriptorAllocator(
    D3D12DescriptorHeapAllocator* heap_allocator, const string& name)
    : BFCAllocator(new SubAllocatorWrapper(heap_allocator),
                   kMaxTotalDescriptors, kAllowGrowth, name,
                   kEnableGarbageCollection, kMaxAllocationSizeInDescriptors),
      heap_allocator_(heap_allocator) {}

DescriptorAllocation DmlDescriptorAllocator::Alloc(size_t size_in_descriptors) {
  void* p = this->AllocateRaw(0, size_in_descriptors);
  return DescriptorAllocation(this, p, size_in_descriptors);
}

D3D12DescriptorHandles DmlDescriptorAllocator::GetDescriptorHandles(
    const void* ptr) const {
  return heap_allocator_->GetDescriptorHandles(ptr);
}

DescriptorAllocation::DescriptorAllocation(DmlDescriptorAllocator* allocator,
                                           void* p, size_t size_in_descriptors)
    : allocator_(allocator), p_(p), size_in_descriptors_(size_in_descriptors) {}

DescriptorAllocation ::~DescriptorAllocation() {
  if (allocator_ && p_) {
    allocator_->DeallocateRaw(p_);
  }
}

DescriptorAllocation::DescriptorAllocation(DescriptorAllocation&& x) {
  *this = std::move(x);
}

DescriptorAllocation& DescriptorAllocation::operator=(
    DescriptorAllocation&& x) {
  if (this != &x) {
    allocator_ = x.allocator_;
    p_ = x.p_;
    size_in_descriptors_ = x.size_in_descriptors_;

    x.allocator_ = nullptr;
    x.p_ = nullptr;
    x.size_in_descriptors_ = 0;
  }
}

D3D12DescriptorHandles DescriptorAllocation::GetDescriptorHandles() const {
  if (!allocator_ || !p_) {
    return D3D12DescriptorHandles{nullptr, UINT64(-1), SIZE_T(-1)};
  }
  return allocator_->GetDescriptorHandles(p_);
}

}  // namespace tensorflow
