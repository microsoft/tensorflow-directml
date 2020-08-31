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

#include "dml_common.h"
#include "dml_gpu_event.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

// Base class for implementing a non-blocking, ring-buffer style D3D12 heap
// allocator where allocations are automatically freed once usage has completed
// on the GPU.
class DmlPooledHeap {
 public:
  // Releases unused capacity.
  void Trim();

  uint64_t Capacity() const { return total_capacity_; }

 protected:
  static constexpr uint64_t kMinChunkSize = 1024 * 1024;  // 1MB

  // In bytes; as per D3D12 requirement for buffers
  static constexpr uint64_t kAllocationAlignment = 512;

  // A suballoction from a chunk
  struct Allocation {
    uint64_t size_in_bytes;

    // The offset, in bytes, from the beginning of the chunk to the beginning of
    // this allocation
    uint64_t offset_in_chunk;

    // The event that will be signaled to when the GPU is done executing work
    // that uses this allocation
    DmlGpuEvent done_event;
  };

  // Represents a single contiguous heap from which we carve out suballocations.
  // Ranges are suballocated from the heap in a ring-buffer fashion.
  struct Chunk {
    uint64_t capacity_in_bytes;  // The total size of the heap, in bytes
    Microsoft::WRL::ComPtr<ID3D12Resource> resource;

    // Allocations are sorted by ascending fence value - that is, least to most
    // recently allocated
    std::list<Allocation> allocations;
  };

  // Calls AssertInvariants on construction and again on destruction
  class InvariantChecker {
   public:
    InvariantChecker(DmlPooledHeap* parent) : parent_(parent) {
      parent_->AssertInvariants();
    }

    ~InvariantChecker() { parent_->AssertInvariants(); }

   private:
    DmlPooledHeap* parent_;
  };

  DmlPooledHeap(ID3D12Device* device, const D3D12_HEAP_PROPERTIES& heap_props,
                D3D12_RESOURCE_STATES barrier_state);

  // Finds or creates a chunk with enough space to accommodate an allocation of
  // the given size, and returns a pointer to the chunk and allocation offset.
  Status Reserve(uint64_t size_in_bytes,
                 /*out*/ DmlPooledHeap::Chunk** chunk_ptr,
                 /*out*/ uint64_t* offset_in_chunk);

  void ReclaimAllocations();  // Frees all allocations which are no longer being
                              // used by the GPU.

 private:
  // Attempts to find enough unused space in the supplied chunk to accommodate
  // the given allocation size. Returns the offset of that memory if successful,
  // null if there wasn't enough space.
  static absl::optional<uint64_t> FindOffsetForAllocation(
      const Chunk& chunk, uint64_t size_in_bytes);

  Status CreateChunk(ID3D12Device* device, uint64_t size_in_bytes,
                     /*out*/ DmlPooledHeap::Chunk* chunk);
  void AssertInvariants();

  Microsoft::WRL::ComPtr<ID3D12Device> device_;
  D3D12_HEAP_PROPERTIES heap_props_;
  D3D12_RESOURCE_STATES barrier_state_;

  // sorted ascending by capacity (heap size)
  std::vector<Chunk> chunks_;
  uint64_t total_capacity_ = 0;  // Total size of all chunks, in bytes
};

}  // namespace tensorflow
