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
  // Constructs a command list.
  DmlCommandList(ID3D12Device* d3d12_device, IDMLDevice* dml_device,
                 D3D12_COMMAND_LIST_TYPE command_list_type);

  // Records a CopyBufferRegion (see
  // ID3D12GraphicsCommandList::CopyBufferRegion) for execution. Transition
  // barriers are automatically inserted to transition the source and
  // destination resources to COPY_SOURCE and COPY_DEST if necessary.
  void CopyBufferRegion(ID3D12Resource* dst_buffer, uint64_t dst_offset,
                        D3D12_RESOURCE_STATES dst_state,
                        ID3D12Resource* src_buffer, uint64_t src_offset,
                        D3D12_RESOURCE_STATES src_state, uint64_t byte_count);

  // Records a ClearUAV with the specified value into the command list.
  void FillBufferWithPattern(
      ID3D12Resource* dst, uint64_t dst_offset, uint64_t dst_size_in_bytes,
      absl::Span<const uint8_t>
          value /* Data type agnostic value, treated as raw bits */);

  // Records DML operator initialization into the command list. It's safe to
  // release the binding table immediately after this is called.
  void InitializeOperator(IDMLOperatorInitializer* initializer,
                          IDMLBindingTable* binding_table,
                          ID3D12DescriptorHeap* descriptor_heap);

  // Records DML operator execution into the command list. It's safe to release
  // the binding table immediately after this is called.
  void ExecuteOperator(IDMLCompiledOperator* op,
                       IDMLBindingTable* binding_table,
                       ID3D12DescriptorHeap* descriptor_heap);

  // Records a resoruce barrier into the command list.
  void ResourceBarrier(absl::Span<const D3D12_RESOURCE_BARRIER> barriers);

  // Records a UAV barrier on all resources into the command list.
  void UavBarrier();

  // Opens the command list for recording, which is required before any of the
  // above methods can be called. The supplied completion event indicates the
  // fence value that will be signaled when the commands recorded to this
  // command list have finished executing on the hardware.
  void Open(DmlGpuEvent completion_event);

  // Closes the command list for recording, which is required before the command
  // list can be executed on a command queue. If any errors occur while
  // recording they will be reported as a status here.
  Status Close();

  // Returns a pointer to the underlying D3D command list.
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

  DmlCommandAllocatorRing<2> command_allocator_ring_;

  void SetDescriptorHeap(ID3D12DescriptorHeap* descriptor_heap);
};

}  // namespace tensorflow