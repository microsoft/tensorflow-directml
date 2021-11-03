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
#include "tensorflow/core/util/env_var.h"
namespace tensorflow {

static bool GetTilingEnabled(ID3D12Device* device) {
  bool force_placed_resources = false;
  const char* force_placed_resources_string =
      std::getenv("TF_DIRECTML_FORCE_PLACED_RESOURCES");
  if (force_placed_resources_string != nullptr) {
    if (strcmp("false", force_placed_resources_string) == 0) {
      force_placed_resources = false;
    } else if (strcmp("true", force_placed_resources_string) == 0) {
      force_placed_resources = true;
    } else {
      LOG(ERROR) << "The TF_DIRECTML_FORCE_PLACED_RESOURCES environment "
                    "variable is set "
                    "but could"
                 << " not be parsed: \"" << force_placed_resources_string
                 << "\". Valid"
                 << " values are \"true\" or \"false\". Using default value of "
                    "\"false\".";
    }
  }

  if (!force_placed_resources) {
    D3D12_FEATURE_DATA_D3D12_OPTIONS options = {};
    if (SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS,
                                              &options, sizeof(options)))) {
      return options.TiledResourcesTier >= D3D12_TILED_RESOURCES_TIER_1;
    }
  }

  return false;
}

static uint64_t GetMaxHeapSizeInTiles() {
  int64 override_value = 0;
  Status s =
      ReadInt64FromEnvVar("TF_DIRECTML_MAX_HEAP_TILES", 0, &override_value);
  if (s.ok() && override_value > 0) {
    return static_cast<uint64_t>(override_value);
  }

  return D3D12HeapAllocator::kDefaultMaxHeapSizeInTiles;
}

D3D12HeapAllocator::D3D12HeapAllocator(ID3D12Device* device,
                                       ID3D12CommandQueue* queue,
                                       const D3D12_HEAP_PROPERTIES& heap_props,
                                       D3D12_HEAP_FLAGS heap_flags,
                                       D3D12_RESOURCE_FLAGS resource_flags,
                                       D3D12_RESOURCE_STATES initial_state)
    : device_(device),
      queue_(queue),
      heap_properties_(heap_props),
      heap_flags_(heap_flags),
      resource_flags_(resource_flags),
      initial_state_(initial_state),
      tiling_enabled_(GetTilingEnabled(device)),
      max_heap_size_in_tiles_(GetMaxHeapSizeInTiles()) {
  VLOG(1) << "Tiling enabled = " << tiling_enabled_;
  VLOG(1) << "Max heap size in tiles = " << max_heap_size_in_tiles_;
}

absl::optional<D3D12HeapAllocator::Allocation>
D3D12HeapAllocator::TryCreateTiledAllocation(uint64_t size_in_bytes) {
  Allocation allocation = {};

  // The allocation may be larger than the requested size to ensure a whole
  // number of tiles.
  const uint64_t resource_size_in_tiles =
      1 + (size_in_bytes - 1) / D3D12_TILED_RESOURCE_TILE_SIZE_IN_BYTES;
  const uint64_t resource_size_in_bytes =
      resource_size_in_tiles * D3D12_TILED_RESOURCE_TILE_SIZE_IN_BYTES;
  auto resource_desc =
      CD3DX12_RESOURCE_DESC::Buffer(resource_size_in_bytes, resource_flags_);

  ID3D12Resource** resources[] = {&allocation.resource_uav_state,
                                  &allocation.resource_copy_src_state,
                                  &allocation.resource_copy_dst_state};

  D3D12_RESOURCE_STATES states[] = {initial_state_,
                                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                                    D3D12_RESOURCE_STATE_COPY_DEST};

  for (int i = 0; i < ABSL_ARRAYSIZE(resources); i++) {
    HRESULT create_resource_hr = device_->CreateReservedResource(
        &resource_desc, states[i], nullptr, IID_PPV_ARGS(resources[i]));

    if (dml_util::HrIsOutOfMemory(create_resource_hr)) {
      LOG(WARNING) << "DML allocator out of memory!";
      return absl::nullopt;
    }
    DML_CHECK_SUCCEEDED(create_resource_hr);
  }

  // Reserve enough heaps to store all tiles in the resource.
  const uint64_t heap_count =
      1 + (resource_size_in_tiles - 1) / max_heap_size_in_tiles_;
  allocation.heaps.resize(heap_count);

  // Create heaps and map them to the primary reserved resource.
  D3D12_TILED_RESOURCE_COORDINATE resource_region_start_coordinates = {};
  uint64_t unmapped_resource_tiles = resource_size_in_tiles;
  for (uint64_t i = 0; i < heap_count; i++) {
    // Create heap. The last heap of the allocation may have fewer tiles to
    // avoid wasting space.
    uint64_t heap_size_in_tiles =
        std::min<uint64_t>(unmapped_resource_tiles, max_heap_size_in_tiles_);
    uint64_t heap_size_in_bytes =
        heap_size_in_tiles * D3D12_TILED_RESOURCE_TILE_SIZE_IN_BYTES;
    auto heap_desc =
        CD3DX12_HEAP_DESC(heap_size_in_bytes, heap_properties_, 0, heap_flags_);

    HRESULT create_heap_hr =
        device_->CreateHeap(&heap_desc, IID_PPV_ARGS(&allocation.heaps[i]));
    if (dml_util::HrIsOutOfMemory(create_heap_hr)) {
      LOG(WARNING) << "DML allocator out of memory!";
      return absl::nullopt;
    }
    DML_CHECK_SUCCEEDED(create_heap_hr);

    // Source region in the resource to map.
    D3D12_TILE_REGION_SIZE resource_region_size = {};
    resource_region_size.NumTiles = heap_size_in_tiles;

    // Target range in the current heap to map.
    const D3D12_TILE_RANGE_FLAGS tile_range_flags = D3D12_TILE_RANGE_FLAG_NONE;
    const UINT heap_range_start_offset = 0;
    const UINT heap_range_tile_count = heap_size_in_tiles;

    constexpr UINT numResourceRegions = 1;
    constexpr UINT numHeapRanges = 1;

    // This is a brand new allocation/resource, so the tile mappings are
    // guaranteed to be set (on the GPU timeline) by the time any code can
    // reference the returned resource. We only execute operations on a single
    // hardware queue so there is no need to wait or signal.
    //
    // All resources have identical tile mappings. The repeated call to
    // UpdateTileMappings on all resources instead of using CopyTileMappings is
    // intentional: the latter API is not supported by all versions of PIX.
    for (auto resource : {allocation.resource_uav_state.Get(),
                          allocation.resource_copy_src_state.Get(),
                          allocation.resource_copy_dst_state.Get()}) {
      queue_->UpdateTileMappings(
          resource, numResourceRegions, &resource_region_start_coordinates,
          &resource_region_size, allocation.heaps[i].Get(), numHeapRanges,
          &tile_range_flags, &heap_range_start_offset, &heap_range_tile_count,
          D3D12_TILE_MAPPING_FLAG_NONE);
    }

    resource_region_start_coordinates.X += heap_size_in_tiles;
    unmapped_resource_tiles -= heap_size_in_tiles;
  }

  assert(unmapped_resource_tiles == 0);

  return allocation;
}

absl::optional<D3D12HeapAllocator::Allocation>
D3D12HeapAllocator::TryCreateUntiledAllocation(uint64_t size_in_bytes) {
  Allocation allocation = {};

  // Create the allocation's sole heap. The allocation may be larger than the
  // requested size to ensure a whole number of tiles.
  allocation.heaps.resize(1);
  D3D12_HEAP_DESC heap_desc =
      CD3DX12_HEAP_DESC(size_in_bytes, heap_properties_, 0, heap_flags_);
  HRESULT create_heap_hr =
      device_->CreateHeap(&heap_desc, IID_PPV_ARGS(&allocation.heaps.front()));
  if (dml_util::HrIsOutOfMemory(create_heap_hr)) {
    LOG(WARNING) << "DML allocator out of memory!";
    return absl::nullopt;
  }
  DML_CHECK_SUCCEEDED(create_heap_hr);

  // Create large placed resource that spans the heap.
  D3D12_RESOURCE_DESC resource_desc =
      CD3DX12_RESOURCE_DESC::Buffer(size_in_bytes, resource_flags_);

  ID3D12Resource** resources[] = {&allocation.resource_uav_state,
                                  &allocation.resource_copy_src_state,
                                  &allocation.resource_copy_dst_state};
  D3D12_RESOURCE_STATES states[] = {initial_state_,
                                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                                    D3D12_RESOURCE_STATE_COPY_DEST};

  for (int i = 0; i < ABSL_ARRAYSIZE(resources); i++) {
    HRESULT create_resource_hr = device_->CreatePlacedResource(
        allocation.heaps.front().Get(), 0, &resource_desc, states[i], nullptr,
        IID_PPV_ARGS(resources[i]));
    if (dml_util::HrIsOutOfMemory(create_resource_hr)) {
      LOG(WARNING) << "DML allocator out of memory!";
      return absl::nullopt;
    }
    DML_CHECK_SUCCEEDED(create_resource_hr);
  }

  return allocation;
}

void* D3D12HeapAllocator::Alloc(uint64_t size_in_bytes) {
  if (size_in_bytes == 0) {
    return nullptr;
  }

  // The D3D12 device is thread-safe so we don't need to hold the lock while
  // creating an allocation.
  absl::optional<Allocation> allocation =
      tiling_enabled_ ? TryCreateTiledAllocation(size_in_bytes)
                      : TryCreateUntiledAllocation(size_in_bytes);

  if (!allocation) {
    return nullptr;
  }

  // We need to access (mutable) state after this point, so we need to lock
  std::unique_lock<std::mutex> lock(mutex_);

  absl::optional<uint32_t> id = TryReserveAllocationID();
  if (!id) {
    LOG(WARNING) << "DML allocator ran out of allocation IDs!";
    return nullptr;
  }

  VLOG(3) << "D3D12HeapAllocator: allocating id=" << *id << ", "
          << strings::HumanReadableNumBytes(size_in_bytes);

  allocations_by_id_.emplace(*id, std::move(*allocation));

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

  return D3D12BufferRegion(tagged_ptr.offset, size_in_bytes,
                           allocation->resource_uav_state.Get(),
                           allocation->resource_copy_src_state.Get(),
                           allocation->resource_copy_dst_state.Get());
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
