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

namespace tensorflow {

// A contiguous range of descriptors.
struct DmlDescriptorRange {
  ID3D12DescriptorHeap* heap;
  D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle;
  D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle;
};

// Wraps an ID3D12DescriptorHeap to allocate descriptor ranges.
class DmlDescriptorHeap {
 public:
  // Wraps an existing heap.
  explicit DmlDescriptorHeap(ID3D12DescriptorHeap* heap);

  // Reserves descriptors from the end of the heap. Returns nullopt if there is
  // no space left in the heap.
  absl::optional<DmlDescriptorRange> TryAllocDescriptors(
      uint32_t num_descriptors, DmlGpuEvent completion_event,
      D3D12_DESCRIPTOR_HEAP_FLAGS heap_flags =
          D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE);

  DmlGpuEvent GetLastCompletionEvent() const { return completion_event_; }

  uint32_t GetCapacity() const { return capacity_; }

 private:
  Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> heap_;
  uint32_t capacity_ = 0;
  uint32_t size_ = 0;
  uint32_t handle_increment_size_ = 0;
  CD3DX12_CPU_DESCRIPTOR_HANDLE head_cpu_handle_;
  CD3DX12_GPU_DESCRIPTOR_HANDLE head_gpu_handle_;
  D3D12_DESCRIPTOR_HEAP_FLAGS heap_flags_ = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;

  // Most recent GPU completion event. Allocations are always done at the end,
  // so there is no fragmentation of the heap.
  DmlGpuEvent completion_event_;
};

// Manages a pool of CBV/SRV/UAV descriptors.
class DmlDescriptorPool {
 public:
  DmlDescriptorPool(ID3D12Device* device, uint32_t initial_capacity);

  // Reserves a contiguous range of descriptors from a single descriptor heap.
  // The lifetime of the referenced descriptor heap is managed by the
  // DmlDescriptorPool class. The caller must supply a DmlGpuEvent that informs
  // the pool when the reserved descriptors are no longer required.
  DmlDescriptorRange AllocDescriptors(
      uint32_t num_descriptors, DmlGpuEvent completion_event,
      D3D12_DESCRIPTOR_HEAP_FLAGS heap_flags =
          D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE);

  // Releases all descriptor heaps that contain only descriptors which have
  // completed their work on the GPU.
  void Trim();

  // Returns the total capacity of all heaps.
  uint32_t GetTotalCapacity() const;

 private:
  Microsoft::WRL::ComPtr<ID3D12Device> device_;
  std::vector<DmlDescriptorHeap> heaps_;
  const uint32_t initial_heap_capacity_;

  void CreateHeap(uint32_t num_descriptors,
                  D3D12_DESCRIPTOR_HEAP_FLAGS heap_flags);
};

}  // namespace tensorflow