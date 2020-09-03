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
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

class DmlCommandQueue;
class DmlAllocator;

class DmlCommandRecorder {
 public:
  DmlCommandRecorder(ID3D12Device* d3d_device, IDMLDevice* device,
                     std::shared_ptr<DmlCommandQueue> command_queue,
                     DmlAllocator* allocator);

  Status InitializeOperator(IDMLCompiledOperator* op,
                            const DML_BINDING_DESC& persistent_resource_binding,
                            const DML_BINDING_DESC& input_array_binding);

  Status ExecuteOperator(IDMLCompiledOperator* op,
                         const DML_BINDING_DESC& persistent_resource_binding,
                         absl::Span<const DML_BINDING_DESC> input_bindings,
                         absl::Span<const DML_BINDING_DESC> output_bindings);

  Status CopyBufferRegion(ID3D12Resource* dst_buffer, uint64_t dst_offset,
                          ID3D12Resource* src_buffer, uint64_t src_offset,
                          uint64_t byte_count);

  Status FillBufferWithPattern(
      ID3D12Resource* dst, uint64_t dst_offset, uint64_t dst_size_in_bytes,
      absl::Span<const uint8_t>
          value /* Data type agnostic value, treated as raw bits */);

  Status ExecuteCommandList(ID3D12GraphicsCommandList* command_list,
                            _Outptr_ ID3D12Fence** fence,
                            _Out_ uint64_t* completion_value);

  Status ResourceBarrier(absl::Span<const D3D12_RESOURCE_BARRIER> barriers);

  Status CloseAndExecute();

  // If false, there are no pending commands to be submitted which indicates
  // that CloseAndExecute() would be a no-op.
  bool HasUnflushedWork() const;

 private:
  std::shared_ptr<DmlCommandQueue> queue_;
  Microsoft::WRL::ComPtr<ID3D12Device> d3d_device_;
  Microsoft::WRL::ComPtr<IDMLDevice> dml_device_;
  Microsoft::WRL::ComPtr<IDMLOperatorInitializer> initializer_;
  Microsoft::WRL::ComPtr<IDMLCommandRecorder> recorder_;

  // Descriptors are allocated from a pool. The current heap pointer is only
  // used to avoid redundantly setting the same heap; it does not have ownership
  // of the heap object.
  DmlDescriptorPool descriptor_pool_;
  ID3D12DescriptorHeap* current_descriptor_heap_ = nullptr;

  DmlAllocator* allocator_;
  DmlCommandAllocatorRing<2> command_allocator_ring_;

  // The command list currently being recorded into, and whether any command
  // have been recorded yet.
  Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> current_command_list_;
  uint32_t operations_recorded_in_current_command_list_ = 0;

  // Command lists which have been batched up for execution.  The values in
  // pending_command_lists_cacheable_ indicate whether they can be moved into
  // this class's cache after execution, versus if they belong to the caller and
  // were passed to ExecuteCommandList.
  std::vector<Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList>>
      pending_command_lists_;
  std::vector<bool> pending_command_lists_cacheable_;

  // A pool of cached command lists which may be re-used.
  std::deque<Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList>>
      cached_command_lists_;

  void SetDescriptorHeap(ID3D12DescriptorHeap* descriptor_heap);
  void Open();

  // Increments operations_recorded_in_current_command_list_. If the size of the
  // current command list exceeds a certain value (based on heuristic), the
  // command list is flushed.
  Status OnCommandRecorded();
};

}  // namespace tensorflow
