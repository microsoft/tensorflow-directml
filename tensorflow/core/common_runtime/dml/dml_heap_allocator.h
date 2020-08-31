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

#include "absl/container/flat_hash_map.h"
#include "dml_buffer_region.h"
#include "dml_common.h"

namespace tensorflow {

class D3D12HeapAllocator {
 public:
  // The largest single allocation supported by this allocator. We use 4GB minus
  // a MB to avoid edge cases in hw/drivers that aren't expecting such large
  // allocations. This value can be overridden using the
  // TF_DIRECTML_MAX_ALLOC_SIZE environment variable.
  static constexpr uint64_t kDefaultMaxAllocationSizeInBytes =
      (1ull << 32) - (1ull << 20);

  D3D12HeapAllocator(ID3D12Device* device,
                     const D3D12_HEAP_PROPERTIES& heap_props,
                     D3D12_HEAP_FLAGS heap_flags,
                     D3D12_RESOURCE_FLAGS resource_flags,
                     D3D12_RESOURCE_STATES initial_state);

  // Creates a D3D12 placed resource buffer over the given memory range. The
  // physical D3D12 resource may be larger than thet requested size, so callers
  // must ensure to use the offset/size returned in the D3D12BufferRegion else
  // risk out of bounds access. Note that in practice the ID3D12Resource is
  // cached, so this call typically has a lower cost than a call to
  // ID3D12Device::CreatePlacedResource.
  D3D12BufferRegion CreateBufferRegion(const void* ptr, uint64_t size_in_bytes);

  void* Alloc(uint64_t size_in_bytes);
  void Free(void* ptr, uint64_t size_in_bytes);

 private:
  std::mutex mutex_;

  Microsoft::WRL::ComPtr<ID3D12Device> device_;
  const D3D12_HEAP_PROPERTIES heap_properties_;
  const D3D12_HEAP_FLAGS heap_flags_;
  const D3D12_RESOURCE_FLAGS resource_flags_;
  const D3D12_RESOURCE_STATES initial_state_;

  // The largest allocation ID we've returned so far (or 0 if we've never done
  // so). Note that our allocation IDs start at 1 (not 0) to ensure that it
  // isn't possible for a valid allocation to have a pointer value of
  // 0x00000000.
  uint32_t current_allocation_id_ = 0;

  // A list of unused allocation IDs. This is for re-use of IDs once they get
  // freed. We only bump the max_allocation_id_ once there are no more free
  // IDs.
  std::vector<uint32_t> free_allocation_ids_;

  struct Allocation {
    Microsoft::WRL::ComPtr<ID3D12Heap> heap;

    // A pool of placed resource buffers created over this allocation's heap.
    // Each placed resource is identical, and covers the entire extent of its
    // parent heap. When a caller wants to access a region of memory in this
    // allocation's heap, one of these resources will be popped out of this
    // vector (or if empty, a new one created) and returned. Once the caller is
    // done with the resource, it is added back to the pool for re-use.
    std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>> placed_resource_pool;
  };

  absl::flat_hash_map<uint32_t, Allocation> allocations_by_id_;

  // Releases a placed resource back to an allocation's pool. Called by
  // D3D12BufferRegion when it gets destructed.
  void ReleasePlacedResource(uint32_t allocation_id,
                             Microsoft::WRL::ComPtr<ID3D12Resource> resource);

  // Retrieves a free allocation ID, or nullopt if no more IDs are available.
  absl::optional<uint32_t> TryReserveAllocationID();

  // Releases an allocation ID back to the pool of IDs.
  void ReleaseAllocationID(uint32_t id);

 private:
  static constexpr uint64_t kAllocationIDBits = 24;
  static constexpr uint64_t kOffsetBits = 40;

  // This allocator encodes the allocation ID into the high bits of the pointers
  // it returns, while the low bits are used as an offset into the allocation.
  // Note that since the layout of bitfields is implementation-defined, you
  // can't just cast a void* into a TaggedPointer: it must be done using masks
  // and shifts.
  struct TaggedPointer {
    uint64_t allocation_id : kAllocationIDBits;
    uint64_t offset : kOffsetBits;
  };

  static_assert(sizeof(TaggedPointer) == sizeof(void*),
                "DML requires a 64-bit architecture");
  static_assert(kAllocationIDBits + kOffsetBits == sizeof(void*) * CHAR_BIT,
                "DML requires a 64-bit architecture");

  static void* PackPointer(uint32_t allocation_id, uint64_t offset);
  static TaggedPointer UnpackPointer(const void* ptr);

  friend class D3D12BufferRegion;
};

}  // namespace tensorflow
