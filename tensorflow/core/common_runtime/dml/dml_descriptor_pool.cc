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

#include "dml_descriptor_pool.h"

namespace tensorflow {

DmlDescriptorHeap::DmlDescriptorHeap(ID3D12DescriptorHeap* heap)
    : heap_(heap),
      capacity_(heap->GetDesc().NumDescriptors),
      head_cpu_handle_(heap->GetCPUDescriptorHandleForHeapStart()),
      head_gpu_handle_(heap->GetGPUDescriptorHandleForHeapStart()),
      heap_flags_(heap->GetDesc().Flags) {
  Microsoft::WRL::ComPtr<ID3D12Device> device;
  DML_CHECK_SUCCEEDED(heap->GetDevice(IID_PPV_ARGS(&device)));

  handle_increment_size_ =
      device->GetDescriptorHandleIncrementSize(heap->GetDesc().Type);
}

absl::optional<DmlDescriptorRange> DmlDescriptorHeap::TryAllocDescriptors(
    uint32_t num_descriptors, DmlGpuEvent completion_event,
    D3D12_DESCRIPTOR_HEAP_FLAGS heap_flags) {
  // Bail if the desired heap creation flags are incompatible with the existing
  // heap.
  if (heap_flags_ != heap_flags) {
    return absl::nullopt;
  }

  if ((completion_event_.fence != nullptr) &&
      (completion_event_.IsSignaled())) {
    // This class always allocates descriptors from the end of the heap.
    // If the most recent completion event is signaled, then all previous
    // allocations have completed; the entire capacity is available to use.
    size_ = 0;
    head_cpu_handle_ = heap_->GetCPUDescriptorHandleForHeapStart();
    head_gpu_handle_ = heap_->GetGPUDescriptorHandleForHeapStart();
  }

  // The caller will need to create a new heap if there is no space left in this
  // one.
  uint32_t space_remaining = capacity_ - size_;
  if (space_remaining < num_descriptors) {
    return absl::nullopt;
  }

  DmlDescriptorRange range = {heap_.Get(), head_cpu_handle_, head_gpu_handle_};

  size_ += num_descriptors;
  completion_event_ = completion_event;
  head_cpu_handle_.Offset(num_descriptors, handle_increment_size_);
  head_gpu_handle_.Offset(num_descriptors, handle_increment_size_);

  return range;
}

DmlDescriptorPool::DmlDescriptorPool(ID3D12Device* device,
                                     uint32_t initial_capacity)
    : device_(device), initial_heap_capacity_(initial_capacity) {
  CreateHeap(initial_capacity, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE);
}

DmlDescriptorRange DmlDescriptorPool::AllocDescriptors(
    uint32_t num_descriptors, DmlGpuEvent completion_event,
    D3D12_DESCRIPTOR_HEAP_FLAGS heap_flags) {
  // Attempt to allocate from an existing heap.
  for (DmlDescriptorHeap& heap : heaps_) {
    auto descriptor_range =
        heap.TryAllocDescriptors(num_descriptors, completion_event, heap_flags);
    if (descriptor_range.has_value()) {
      return descriptor_range.value();
    }
  }

  // A new descriptor heap must be created.
  uint32_t new_heap_capacity =
      std::max(num_descriptors, initial_heap_capacity_);
  CreateHeap(new_heap_capacity, heap_flags);
  auto descriptor_range = heaps_.back().TryAllocDescriptors(
      num_descriptors, completion_event, heap_flags);
  assert(descriptor_range.has_value());
  return descriptor_range.value();
}

void DmlDescriptorPool::Trim() {
  // Remove any heaps that are not pending execution.
  auto it = std::remove_if(
      heaps_.begin(), heaps_.end(), [](const DmlDescriptorHeap& heap) {
        auto completion_event = heap.GetLastCompletionEvent();
        return !completion_event.fence || completion_event.IsSignaled();
      });

  heaps_.erase(it, heaps_.end());
}

void DmlDescriptorPool::CreateHeap(uint32_t num_descriptors,
                                   D3D12_DESCRIPTOR_HEAP_FLAGS heap_flags) {
  // This pool only manages CBV/SRV/UAV descriptors.
  D3D12_DESCRIPTOR_HEAP_DESC desc = {};
  desc.Flags = heap_flags;
  desc.NumDescriptors = num_descriptors;
  desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;

  Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> heap;
  DML_CHECK_SUCCEEDED(
      device_->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&heap)));

  heaps_.push_back(DmlDescriptorHeap{heap.Get()});
}

uint32_t DmlDescriptorPool::GetTotalCapacity() const {
  uint32_t capacity = 0;

  for (auto& heap : heaps_) {
    capacity += heap.GetCapacity();
  }

  return capacity;
}
}  // namespace tensorflow