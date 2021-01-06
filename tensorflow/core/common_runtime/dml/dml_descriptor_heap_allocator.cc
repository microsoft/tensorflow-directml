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

#include "dml_descriptor_heap_allocator.h"

#include "dml_util.h"
#include "tensorflow/core/lib/strings/numbers.h"

namespace tensorflow {

D3D12DescriptorHeapAllocator::D3D12DescriptorHeapAllocator(
    ID3D12Device* device, D3D12_DESCRIPTOR_HEAP_TYPE type,
    D3D12_DESCRIPTOR_HEAP_FLAGS flags)
    : device_(device),
      heap_type_(type),
      heap_flags_(flags),
      handle_increment_(device->GetDescriptorHandleIncrementSize(type)) {}

void* D3D12DescriptorHeapAllocator::Alloc(uint64_t size_in_descriptors) {
  if (size_in_descriptors == 0) {
    return nullptr;
  }

  // The heap type and flags are constant, and the D3D12 device is thread-safe
  // so we don't need to hold the lock over the call to CreateDescriptorHeap
  D3D12_DESCRIPTOR_HEAP_DESC heap_desc = {};
  heap_desc.Type = heap_type_;
  heap_desc.NumDescriptors = size_in_descriptors;
  heap_desc.Flags = heap_flags_;

  Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> heap;
  HRESULT hr = device_->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap));

  // Return early since we don't have enough memory to allocate the buffer
  if (dml_util::HrIsOutOfMemory(hr)) {
    LOG(WARNING) << "DML descriptor allocator out of memory!";
    return nullptr;
  }

  DML_CHECK_SUCCEEDED(hr);

  // We need to access (mutable) state after this point, so we need to lock
  std::unique_lock<std::mutex> lock(mutex_);

  absl::optional<uint32_t> id = TryReserveAllocationID();
  if (!id) {
    LOG(WARNING) << "DML descriptor allocator ran out of allocation IDs!";
    return nullptr;
  }

  VLOG(3) << "D3D12DescriptorHeapAllocator: allocating id=" << *id << ", "
          << size_in_descriptors << " descriptors";

  Allocation allocation = {};
  allocation.heap = std::move(heap);

  allocations_by_id_.emplace(*id, std::move(allocation));

  lock.unlock();

  const uint64_t offset = 0;
  return PackPointer(*id, offset);
}

void D3D12DescriptorHeapAllocator::Free(void* ptr,
                                        uint64_t size_in_descriptors) {
  CHECK(ptr != nullptr) << "Invalid pointer";

  TaggedPointer tagged_ptr = UnpackPointer(ptr);
  CHECK(tagged_ptr.offset == 0) << "Invalid pointer";

  // We need to access (mutable) state after this point, so we need to lock
  std::unique_lock<std::mutex> lock(mutex_);

  auto it = allocations_by_id_.find(tagged_ptr.allocation_id);

  CHECK(it != allocations_by_id_.end()) << "Invalid pointer";
  DCHECK(it->second.heap->GetDesc().NumDescriptors == size_in_descriptors);

  VLOG(3) << "D3D12DescriptorHeapAllocator: freeing id="
          << tagged_ptr.allocation_id << ", " << size_in_descriptors
          << " descriptors";

  ReleaseAllocationID(tagged_ptr.allocation_id);

  // Frees the ID3D12DescriptorHeap
  allocations_by_id_.erase(it);
}

D3D12DescriptorHandles D3D12DescriptorHeapAllocator::GetDescriptorHandles(
    const void* ptr) const {
  CHECK(ptr != nullptr) << "Invalid pointer";

  TaggedPointer tagged_ptr = UnpackPointer(ptr);

  // We need to access (mutable) state after this point, so we need to lock
  std::unique_lock<std::mutex> lock(mutex_);

  // Find the allocation corresponding to this pointer
  auto it = allocations_by_id_.find(tagged_ptr.allocation_id);
  CHECK(it != allocations_by_id_.end()) << "Invalid pointer";

  const Allocation* allocation = &it->second;

  D3D12_GPU_DESCRIPTOR_HANDLE gpu_start(
      allocation->heap->GetGPUDescriptorHandleForHeapStart());
  D3D12_CPU_DESCRIPTOR_HANDLE cpu_start(
      allocation->heap->GetCPUDescriptorHandleForHeapStart());

  D3D12DescriptorHandles handles = {};
  handles.heap = allocation->heap.Get();
  handles.gpu = CD3DX12_GPU_DESCRIPTOR_HANDLE(gpu_start, tagged_ptr.offset,
                                              handle_increment_);
  handles.cpu = CD3DX12_CPU_DESCRIPTOR_HANDLE(cpu_start, tagged_ptr.offset,
                                              handle_increment_);

  return handles;
}

absl::optional<uint32_t>
D3D12DescriptorHeapAllocator::TryReserveAllocationID() {
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

void D3D12DescriptorHeapAllocator::ReleaseAllocationID(uint32_t id) {
  // The mutex must already be held
  assert(!mutex_.try_lock());

  // Add it to the pool of free IDs
  free_allocation_ids_.push_back(id);
}

/*static*/ void* D3D12DescriptorHeapAllocator::PackPointer(
    uint32_t allocation_id, uint64_t offset) {
  DCHECK(allocation_id < (1ull << kAllocationIDBits));
  DCHECK(offset < (1ull << kOffsetBits));

  // Store the allocation ID in the upper bits of the pointer, and the offset in
  // the lower bits
  uint64_t ptr = ((uint64_t)allocation_id << kOffsetBits) | offset;

  return reinterpret_cast<void*>(ptr);
}

/*static*/ D3D12DescriptorHeapAllocator::TaggedPointer
D3D12DescriptorHeapAllocator::UnpackPointer(const void* ptr) {
  uint64_t ptr_val = reinterpret_cast<uint64_t>(ptr);

  static constexpr uint64_t kOffsetMask = (1ull << kOffsetBits) - 1;

  TaggedPointer tagged_ptr;
  tagged_ptr.allocation_id = (ptr_val >> kOffsetBits);
  tagged_ptr.offset = (ptr_val & kOffsetMask);

  return tagged_ptr;
}

}  // namespace tensorflow
