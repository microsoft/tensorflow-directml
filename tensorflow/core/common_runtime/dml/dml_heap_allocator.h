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

// An allocator that makes logically contiguous allocations backed by D3D heaps.
//
// Heaps must fit entirely in either local or non-local memory. Larger heaps
// have a greater chance of getting demoted into non-local memory, which can be
// disastrous for performance. This problem is compounded by the fact that heaps
// may be demoted even if overall local memory usage is within the process'
// budget. Heaps are not necessarily mappable to discontiguous regions of
// physical memory, which means physical memory fragmentation *may* make it
// extremely difficult to accommodate larger heaps.
//
// On D3D hardware that supports tiled resource tier 1+ this class implements
// large allocations through tiling. Each allocation is backed by however many
// small heaps are necessary to cover the requested allocation size. Buffer
// regions retrieved through this allocator are reserved resources that span the
// full collection of heaps assigned to an individual allocation. Tile mappings
// are static.
//
// On hardware that doesn't support tiled resources each allocation is backed by
// a single heap. Buffer regions retrieved through this allocator are placed
// resources that span the full heap assigned to an individual allocation. In
// this case it is better make more but smaller allocations (resulting in
// smaller heaps); this fallback path is only retained as a last resort for
// older hardware.
class D3D12HeapAllocator {
 public:
  // Maximum size of a heap (in tiles) when allocations are tiled. Each tile is
  // 64KB. A default size of 512 tiles (32MB) does a good job of handling local
  // video memory fragmentation without requiring lots of heaps. This value can
  // be overridden using the TF_DIRECTML_MAX_HEAP_TILES environment variable.
  static constexpr uint64_t kDefaultMaxHeapSizeInTiles = 512;

  // The largest single allocation supported by this allocator. We use 4GB minus
  // a MB to avoid edge cases in hw/drivers that aren't expecting such large
  // allocations. This value can be overridden using the
  // TF_DIRECTML_MAX_ALLOC_SIZE environment variable.
  static constexpr uint64_t kDefaultMaxAllocationSizeInBytes =
      (1ull << 32) - (1ull << 20);

  D3D12HeapAllocator(ID3D12Device* device, ID3D12CommandQueue* queue,
                     const D3D12_HEAP_PROPERTIES& heap_props,
                     D3D12_HEAP_FLAGS heap_flags,
                     D3D12_RESOURCE_FLAGS resource_flags,
                     D3D12_RESOURCE_STATES initial_state);

  // Creates a reserved or placed resource buffer over the given memory range.
  // The physical D3D12 resource may be larger than the requested size, so
  // callers must ensure to use the offset/size returned in the
  // D3D12BufferRegion else risk out of bounds access. Note that in practice the
  // ID3D12Resource is cached, so this call typically has a lower cost than a
  // call to ID3D12Device::CreatePlacedResource or CreateReservedResource.
  D3D12BufferRegion CreateBufferRegion(const void* ptr, uint64_t size_in_bytes);

  void* Alloc(uint64_t size_in_bytes);
  void Free(void* ptr, uint64_t size_in_bytes);

  bool TilingEnabled() const { return tiling_enabled_; };

 private:
  std::mutex mutex_;

  Microsoft::WRL::ComPtr<ID3D12Device> device_;
  Microsoft::WRL::ComPtr<ID3D12CommandQueue> queue_;
  const D3D12_HEAP_PROPERTIES heap_properties_;
  const D3D12_HEAP_FLAGS heap_flags_;
  const D3D12_RESOURCE_FLAGS resource_flags_;
  const D3D12_RESOURCE_STATES initial_state_;
  bool tiling_enabled_;
  uint64_t max_heap_size_in_tiles_;

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
    // Heaps backing the memory for the allocation. If tiling is supported an
    // allocation may comprise multiple heaps. If tiling is not supported an
    // allocation will only have a single heap.
    std::vector<Microsoft::WRL::ComPtr<ID3D12Heap>> heaps;

    // Resources created over this allocation's heaps. All three resources are
    // identical aside from being fixed in a single resource state: UAV,
    // COPY_SRC, and COPY_DST respectively. The purpose of duplicate resources
    // is to enable overlapping resources in different states for copying data.
    // Most callers will not (and should not) interact directly with these
    // resources; all three are wrapped by the buffer regions returned from this
    // allocator, and the appropriate resource will be used automatically when
    // performing buffer copies.
    Microsoft::WRL::ComPtr<ID3D12Resource> resource_uav_state;
    Microsoft::WRL::ComPtr<ID3D12Resource> resource_copy_src_state;
    Microsoft::WRL::ComPtr<ID3D12Resource> resource_copy_dst_state;
  };

  absl::flat_hash_map<uint32_t, Allocation> allocations_by_id_;

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

  absl::optional<Allocation> TryCreateTiledAllocation(uint64_t size_in_bytes);
  absl::optional<Allocation> TryCreateUntiledAllocation(uint64_t size_in_bytes);

  friend class D3D12BufferRegion;
};

}  // namespace tensorflow
