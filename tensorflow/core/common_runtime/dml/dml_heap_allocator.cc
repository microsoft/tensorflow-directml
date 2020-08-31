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

#include "dml_heap_allocator.h"

#include "dml_util.h"
#include "tensorflow/core/lib/strings/numbers.h"

namespace tensorflow {

D3D12HeapAllocator::D3D12HeapAllocator(ID3D12Device* device,
                                       const D3D12_HEAP_PROPERTIES& heap_props,
                                       D3D12_HEAP_FLAGS heap_flags,
                                       D3D12_RESOURCE_FLAGS resource_flags,
                                       D3D12_RESOURCE_STATES initial_state)
    : device_(device),
      heap_properties_(heap_props),
      heap_flags_(heap_flags),
      resource_flags_(resource_flags),
      initial_state_(initial_state) {}

void* D3D12HeapAllocator::Alloc(uint64_t size_in_bytes) {
  if (size_in_bytes == 0) {
    return nullptr;
  }

  // The heap properties and flags are constant, and the D3D12 device is
  // thread-safe so we don't need to hold the lock over the call to CreateHeap
  D3D12_HEAP_DESC heap_desc =
      CD3DX12_HEAP_DESC(size_in_bytes, heap_properties_, 0, heap_flags_);

  Microsoft::WRL::ComPtr<ID3D12Heap> heap;
  HRESULT hr = device_->CreateHeap(&heap_desc, IID_PPV_ARGS(&heap));

  // Return early since we don't have enough memory to allocate the buffer
  if (dml_util::HrIsOutOfMemory(hr)) {
    LOG(WARNING) << "DML allocator out of memory!";
    return nullptr;
  }

  DML_CHECK_SUCCEEDED(hr);

  // We need to access (mutable) state after this point, so we need to lock
  std::unique_lock<std::mutex> lock(mutex_);

  absl::optional<uint32_t> id = TryReserveAllocationID();
  if (!id) {
    LOG(WARNING) << "DML allocator ran out of allocation IDs!";
    return nullptr;
  }

  VLOG(3) << "D3D12HeapAllocator: allocating id=" << *id << ", "
          << strings::HumanReadableNumBytes(size_in_bytes);

  Allocation allocation = {};
  allocation.heap = std::move(heap);

  allocations_by_id_.emplace(*id, std::move(allocation));

  lock.unlock();

  const uint64_t offset = 0;
  return PackPointer(*id, offset);
}

void D3D12HeapAllocator::Free(void* ptr, uint64_t size_in_bytes) {
  CHECK(ptr != nullptr) << "Invalid pointer";

  TaggedPointer tagged_ptr = UnpackPointer(ptr);
  CHECK(tagged_ptr.offset == 0) << "Invalid pointer";

  // We need to access (mutable) state after this point, so we need to lock
  std::unique_lock<std::mutex> lock(mutex_);

  auto it = allocations_by_id_.find(tagged_ptr.allocation_id);

  CHECK(it != allocations_by_id_.end()) << "Invalid pointer";
  DCHECK(it->second.heap->GetDesc().SizeInBytes == size_in_bytes);

  VLOG(3) << "D3D12HeapAllocator: freeing id=" << tagged_ptr.allocation_id
          << ", " << strings::HumanReadableNumBytes(size_in_bytes);

  ReleaseAllocationID(tagged_ptr.allocation_id);

  // Frees the ID3D12Heap
  allocations_by_id_.erase(it);
}

D3D12BufferRegion D3D12HeapAllocator::CreateBufferRegion(
    const void* ptr, uint64_t size_in_bytes) {
  CHECK(ptr != nullptr) << "Invalid pointer";

  TaggedPointer tagged_ptr = UnpackPointer(ptr);

  // We need to access (mutable) state after this point, so we need to lock
  std::unique_lock<std::mutex> lock(mutex_);

  // Find the allocation corresponding to this pointer
  auto it = allocations_by_id_.find(tagged_ptr.allocation_id);
  CHECK(it != allocations_by_id_.end()) << "Invalid pointer";

  Allocation* allocation = &it->second;

  // Retrieve a placed resource that spans the allocation's heap
  Microsoft::WRL::ComPtr<ID3D12Resource> buffer;
  if (!allocation->placed_resource_pool.empty()) {
    // If there are spare resources in the pool, return one of those

    buffer = std::move(allocation->placed_resource_pool.back());
    allocation->placed_resource_pool.pop_back();
  } else {
    // No resources left in the pool; need to create one from scratch

    D3D12_RESOURCE_DESC resource_desc = CD3DX12_RESOURCE_DESC::Buffer(
        allocation->heap->GetDesc().SizeInBytes, resource_flags_);

    DML_CHECK_SUCCEEDED(device_->CreatePlacedResource(
        allocation->heap.Get(), 0, &resource_desc, initial_state_, nullptr,
        IID_PPV_ARGS(&buffer)));
  }

  return D3D12BufferRegion(this, tagged_ptr.allocation_id, std::move(buffer),
                           tagged_ptr.offset, size_in_bytes);
}

void D3D12HeapAllocator::ReleasePlacedResource(
    uint32_t allocation_id, Microsoft::WRL::ComPtr<ID3D12Resource> resource) {
  assert(resource->GetDesc().Dimension == D3D12_RESOURCE_DIMENSION_BUFFER);

  std::unique_lock<std::mutex> lock(mutex_);

  auto it = allocations_by_id_.find(allocation_id);
  CHECK(it != allocations_by_id_.end());

  Allocation* allocation = &it->second;

  // If the sizes don't match, then it means this resource didn't come from this
  // allocator...
  assert(allocation->heap->GetDesc().SizeInBytes == resource->GetDesc().Width);

  // Return the resource to the pool
  allocation->placed_resource_pool.push_back(std::move(resource));
}

absl::optional<uint32_t> D3D12HeapAllocator::TryReserveAllocationID() {
  // The mutex must already be held
  assert(!mutex_.try_lock());

  if (!free_allocation_ids_.empty()) {
    // Return a free ID from the pool
    uint32_t id = free_allocation_ids_.back();
    free_allocation_ids_.pop_back();
    return id;
  }

  static constexpr uint32_t kMaxAllocationID = (1 << kAllocationIDBits) - 1;
  if (current_allocation_id_ == kMaxAllocationID) {
    // We've reached the maximum number of allocations!
    return absl::nullopt;
  }

  ++current_allocation_id_;
  return current_allocation_id_;
}

void D3D12HeapAllocator::ReleaseAllocationID(uint32_t id) {
  // The mutex must already be held
  assert(!mutex_.try_lock());

  // Add it to the pool of free IDs
  free_allocation_ids_.push_back(id);
}

/*static*/ void* D3D12HeapAllocator::PackPointer(uint32_t allocation_id,
                                                 uint64_t offset) {
  DCHECK(allocation_id < (1ull << kAllocationIDBits));
  DCHECK(offset < (1ull << kOffsetBits));

  // Store the allocation ID in the upper bits of the pointer, and the offset in
  // the lower bits
  uint64_t ptr = ((uint64_t)allocation_id << kOffsetBits) | offset;

  return reinterpret_cast<void*>(ptr);
}

/*static*/ D3D12HeapAllocator::TaggedPointer D3D12HeapAllocator::UnpackPointer(
    const void* ptr) {
  uint64_t ptr_val = reinterpret_cast<uint64_t>(ptr);

  static constexpr uint64_t kOffsetMask = (1ull << kOffsetBits) - 1;

  TaggedPointer tagged_ptr;
  tagged_ptr.allocation_id = (ptr_val >> kOffsetBits);
  tagged_ptr.offset = (ptr_val & kOffsetMask);

  return tagged_ptr;
}

}  // namespace tensorflow
