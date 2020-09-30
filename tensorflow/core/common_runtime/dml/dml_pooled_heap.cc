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

#include "dml_pooled_heap.h"

#include "tensorflow/core/common_runtime/dml/dml_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/numbers.h"

namespace tensorflow {

DmlPooledHeap::DmlPooledHeap(ID3D12Device* device,
                             const D3D12_HEAP_PROPERTIES& heap_props,
                             D3D12_RESOURCE_STATES barrier_state,
                             DmlDeviceRemovedEvent* device_removed_event)
    : device_(device), heap_props_(heap_props), barrier_state_(barrier_state) {
  device_removed_event->AddListener(
      std::bind(&DmlPooledHeap::OnDeviceRemoved, this));
}

static uint64_t Align(uint64_t offset, uint64_t alignment) {
  assert(alignment != 0);
  return (offset + alignment - 1) & ~(alignment - 1);
}

absl::optional<uint64_t> DmlPooledHeap::FindOffsetForAllocation(
    const Chunk& chunk, uint64_t size_in_bytes) {
  assert(size_in_bytes != 0);

  if (chunk.capacity_in_bytes < size_in_bytes) {
    // This chunk isn't even big enough to accommodate this allocation
    return absl::nullopt;
  }

  if (chunk.allocations.empty()) {
    // The entire chunk is empty - allocate from the beginning
    return 0;
  }

  // Chunks are used as ring buffers, which means this allocation should go
  // after the most recent previous allocation

  const auto& last_allocation = chunk.allocations.back();
  uint64_t new_allocation_begin =
      last_allocation.offset_in_chunk + last_allocation.size_in_bytes;
  new_allocation_begin = Align(new_allocation_begin, kAllocationAlignment);

  if (new_allocation_begin + size_in_bytes < new_allocation_begin) {
    // Overflow
    return absl::nullopt;
  }

  const auto& first_allocation = chunk.allocations.front();
  if (first_allocation.offset_in_chunk <= last_allocation.offset_in_chunk) {
    // This is the case where there's potentially free space at the beginning
    // and end of the chunk, but not the middle: e.g.
    //   |------XXXXYYYZZ------|
    //          ^^^^   ^^
    //          first  last

    if (new_allocation_begin + size_in_bytes <= chunk.capacity_in_bytes) {
      // There's enough space between the end of the last allocation and the end
      // of the chunk
      return new_allocation_begin;
    } else {
      // Otherwise there's not enough space at the end of the chunk - try the
      // beginning of the chunk instead
      new_allocation_begin = 0;
      if (new_allocation_begin + size_in_bytes <=
          first_allocation.offset_in_chunk) {
        // There was enough space between the start of the buffer, and the start
        // of the first allocation
        return new_allocation_begin;
      }
    }
  } else {
    // This is the case where there's potentially free space in the middle of
    // the chunk, but not at the edges e.g.
    //   |YYYZZ---------XXXX-|
    //       ^^         ^^^^
    //       last       first

    if (new_allocation_begin + size_in_bytes <=
        first_allocation.offset_in_chunk) {
      // There's enough space between the end of the last allocation, and the
      // start of the first one
      return new_allocation_begin;
    }
  }

  // Not enough space in this chunk to accommodate the requested allocation
  return absl::nullopt;
}

Status DmlPooledHeap::CreateChunk(ID3D12Device* device, uint64_t size_in_bytes,
                                  /*out*/ DmlPooledHeap::Chunk* chunk) {
  assert(chunk != nullptr);

  auto resource_desc = CD3DX12_RESOURCE_DESC::Buffer(size_in_bytes);
  Microsoft::WRL::ComPtr<ID3D12Resource> upload_buffer;
  HRESULT hr = device->CreateCommittedResource(
      &heap_props_, D3D12_HEAP_FLAG_NONE, &resource_desc, barrier_state_,
      nullptr, IID_PPV_ARGS(&upload_buffer));

  // Return early since we don't have enough memory to allocate the buffer
  if (dml_util::HrIsOutOfMemory(hr)) {
    return errors::ResourceExhausted("OOM when allocating a buffer of ",
                                     size_in_bytes, " bytes");
  }

  DML_CHECK_SUCCEEDED(hr);

  *chunk = Chunk{size_in_bytes, std::move(upload_buffer)};

  return Status::OK();
}

static const char* HeapTypeString(D3D12_HEAP_TYPE type) {
  switch (type) {
    case D3D12_HEAP_TYPE_DEFAULT:
      return "D3D12_HEAP_TYPE_DEFAULT";
    case D3D12_HEAP_TYPE_UPLOAD:
      return "D3D12_HEAP_TYPE_UPLOAD";
    case D3D12_HEAP_TYPE_READBACK:
      return "D3D12_HEAP_TYPE_READBACK";
    case D3D12_HEAP_TYPE_CUSTOM:
      return "D3D12_HEAP_TYPE_CUSTOM";
  }

  return "<unknown heap type>";
}

Status DmlPooledHeap::Reserve(uint64_t size_in_bytes,
                              DmlPooledHeap::Chunk** chunk_ptr,
                              /*out*/ uint64_t* offset_in_chunk) {
  if (device_removed_) {
    printf("***************Reserve After Device Removal\n");
    return errors::Unknown(
        "Allocating memory is not allowed when the device has already been "
        "removed.");
  }

  assert(chunk_ptr != nullptr);
  assert(offset_in_chunk != nullptr);

  // Try to find a chunk with enough free space to accommodate the requested
  // allocation size
  for (Chunk& chunk : chunks_) {
    absl::optional<uint64_t> offsetForAllocation =
        FindOffsetForAllocation(chunk, size_in_bytes);
    if (offsetForAllocation) {
      // There's enough space in this chunk - return
      *chunk_ptr = &chunk;
      *offset_in_chunk = *offsetForAllocation;
      return Status::OK();
    }
  }

  // No chunks were able to accommodate the allocation - create a new chunk and
  // return that instead

  // At least double the capacity of the pool
  const uint64_t new_chunk_size =
      std::max({total_capacity_, kMinChunkSize, size_in_bytes});

  DmlPooledHeap::Chunk chunk;
  TF_RETURN_IF_ERROR(CreateChunk(device_.Get(), new_chunk_size, &chunk));

  chunks_.push_back(std::move(chunk));
  total_capacity_ += new_chunk_size;

  // Allocate from the beginning of the new chunk
  *chunk_ptr = &chunks_.back();
  *offset_in_chunk = 0;

  VLOG(3) << "Expanding pooled heap 0x" << this << " ("
          << HeapTypeString(heap_props_.Type) << "), new capacity="
          << strings::HumanReadableNumBytes(total_capacity_);

  return Status::OK();
}

void DmlPooledHeap::ReclaimAllocations() {
  for (Chunk& chunk : chunks_) {
    auto* allocs = &chunk.allocations;

    // Remove all allocations which have had their fences signaled - this
    // indicates that they are no longer being used by the GPU. We can stop as
    // soon as we find an allocation which is still in use, because we only use
    // a single command queue and executions always complete in the order they
    // were submitted.
    while (!allocs->empty() && allocs->front().done_event.IsSignaled()) {
      allocs->pop_front();
    }
  }
}

void DmlPooledHeap::OnDeviceRemoved() {
  chunks_.clear();
  total_capacity_ = 0;
  device_removed_ = true;
}

void DmlPooledHeap::Trim() {
  InvariantChecker checker(this);

  ReclaimAllocations();

  // Release any chunks which have no allocations
  auto it = std::remove_if(chunks_.begin(), chunks_.end(), [](const Chunk& c) {
    return c.allocations.empty();
  });
  chunks_.erase(it, chunks_.end());

  // Re-calculate total capacity
  total_capacity_ = 0;
  for (const auto& chunk : chunks_) {
    total_capacity_ += chunk.capacity_in_bytes;
  }
}

void DmlPooledHeap::AssertInvariants() {
#ifdef _DEBUG

  auto chunk_capacity_comparer = [](const Chunk& lhs, const Chunk& rhs) {
    return lhs.capacity_in_bytes < rhs.capacity_in_bytes;
  };

  // Chunks should be sorted by ascending capacity
  assert(
      std::is_sorted(chunks_.begin(), chunks_.end(), chunk_capacity_comparer));

  // Allocations in a chunk should be sorted by ascending fence value
  for (const auto& chunk : chunks_) {
    auto alloc_fence_value_comparer = [](const Allocation& lhs,
                                         const Allocation& rhs) {
      return lhs.done_event.fence_value < rhs.done_event.fence_value;
    };
    assert(std::is_sorted(chunk.allocations.begin(), chunk.allocations.end(),
                          alloc_fence_value_comparer));
  }

  // Validate chunk properties
  for (const auto& chunk : chunks_) {
    assert(chunk.resource != nullptr);
    assert(chunk.capacity_in_bytes == chunk.resource->GetDesc().Width);
  }

  // Validate allocation properties
  for (const auto& chunk : chunks_) {
    for (const auto& alloc : chunk.allocations) {
      assert(alloc.offset_in_chunk + alloc.size_in_bytes <=
             chunk.capacity_in_bytes);
      assert(alloc.offset_in_chunk % kAllocationAlignment ==
             0);  // Validate alignment
    }
  }

  // Validate no overlapping allocations
  for (const auto& chunk : chunks_) {
    auto alloc_offset_comparer = [](const Allocation& lhs,
                                    const Allocation& rhs) {
      return lhs.offset_in_chunk < rhs.offset_in_chunk;
    };

    std::vector<Allocation> allocations_sorted_by_offset(
        chunk.allocations.begin(), chunk.allocations.end());
    std::sort(allocations_sorted_by_offset.begin(),
              allocations_sorted_by_offset.end(), alloc_offset_comparer);

    for (size_t i = 1; i < allocations_sorted_by_offset.size(); ++i) {
      const auto& alloc = allocations_sorted_by_offset[i - 1];
      const auto& next_alloc = allocations_sorted_by_offset[i];
      assert(alloc.offset_in_chunk + alloc.size_in_bytes <=
             next_alloc.offset_in_chunk);
    }
  }

  // Validate total capacity of pool
  uint64_t calculated_capacity = 0;
  for (const auto& chunk : chunks_) {
    calculated_capacity += chunk.capacity_in_bytes;
  }
  assert(calculated_capacity == total_capacity_);

#endif  // #ifdef _DEBUG
}

}  // namespace tensorflow
