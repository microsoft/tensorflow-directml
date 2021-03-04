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

#include "dml_command_allocator_ring.h"
#include "dml_command_queue.h"
#include "dml_common.h"
#include "dml_descriptor_pool.h"
#include "dml_status.h"

namespace tensorflow {
class DmlAllocator;
class DmlCommandQueue;

// Helper that manages and wraps an ID3D12GraphicsCommandList and its backing
// command allocator. This class is NOT thread safe.
class DmlCommandList {
 public:
  // Constructs an DmlExecutionContext that executes on the supplied queue.
  DmlCommandList(ID3D12Device* d3d12_device, IDMLDevice* dml_device,
                 D3D12_COMMAND_LIST_TYPE command_list_type,
                 DmlAllocator* allocator);

  // Queues a CopyBufferRegion (see ID3D12GraphicsCommandList::CopyBufferRegion)
  // for execution. Transition barriers are automatically inserted to transition
  // the source and destination resources to COPY_SOURCE and COPY_DEST if
  // necessary.
  void CopyBufferRegion(ID3D12Resource* dst_buffer, uint64_t dst_offset,
                        D3D12_RESOURCE_STATES dst_state,
                        ID3D12Resource* src_buffer, uint64_t src_offset,
                        D3D12_RESOURCE_STATES src_state, uint64_t byte_count);

  void FillBufferWithPattern(
      ID3D12Resource* dst, uint64_t dst_offset, uint64_t dst_size_in_bytes,
      absl::Span<const uint8_t>
          value /* Data type agnostic value, treated as raw bits */);

  void InitializeOperator(IDMLOperatorInitializer* initializer,
                          IDMLBindingTable* binding_table,
                          ID3D12DescriptorHeap* descriptor_heap);

  void ExecuteOperator(IDMLCompiledOperator* op,
                       IDMLBindingTable* binding_table,
                       ID3D12DescriptorHeap* descriptor_heap);

  void ResourceBarrier(absl::Span<const D3D12_RESOURCE_BARRIER> barriers);

  void UavBarrier();

  void Open(DmlGpuEvent completion_event);
  Status Close();
  ID3D12CommandList* Get() { return d3d_command_list_.Get(); }

 private:
  Microsoft::WRL::ComPtr<ID3D12Device> d3d_device_;
  Microsoft::WRL::ComPtr<IDMLDevice> dml_device_;
  Microsoft::WRL::ComPtr<IDMLCommandRecorder> recorder_;
  Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> d3d_command_list_;

  D3D12_COMMAND_LIST_TYPE command_list_type_;

  // Descriptors are allocated from a pool. The current heap pointer is only
  // used to avoid redundantly setting the same heap; it does not have ownership
  // of the heap object.
  DmlDescriptorPool descriptor_pool_;
  ID3D12DescriptorHeap* current_descriptor_heap_ = nullptr;
  DmlGpuEvent current_completion_event_;

  DmlAllocator* allocator_ = nullptr;
  DmlCommandAllocatorRing<2> command_allocator_ring_;

  void SetDescriptorHeap(ID3D12DescriptorHeap* descriptor_heap);
};

}  // namespace tensorflow