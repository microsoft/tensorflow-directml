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

  void InitializeOperator(IDMLCompiledOperator* op,
                            const DML_BINDING_DESC& persistent_resource_binding,
                            const DML_BINDING_DESC& input_array_binding);

  void ExecuteOperator(IDMLCompiledOperator* op,
                       const DML_BINDING_DESC& persistent_resource_binding,
                       absl::Span<const DML_BINDING_DESC> input_bindings,
                       absl::Span<const DML_BINDING_DESC> output_bindings);

  void CopyBufferRegion(ID3D12Resource* dst_buffer, uint64_t dst_offset,
                        D3D12_RESOURCE_STATES dst_state,
                        ID3D12Resource* src_buffer, uint64_t src_offset,
                        D3D12_RESOURCE_STATES src_state, uint64_t byte_count);

  void FillBufferWithPattern(
      ID3D12Resource* dst, uint64_t dst_offset, uint64_t dst_size_in_bytes,
      absl::Span<const uint8_t>
          value /* Data type agnostic value, treated as raw bits */);

  void ResourceBarrier(absl::Span<const D3D12_RESOURCE_BARRIER> barriers);

  void CloseAndExecute();

  // If false, there are no pending commands to be submitted which indicates
  // that CloseAndExecute() would be a no-op.
  bool HasUnflushedWork() const;

  Status GetStatus() const { return status_; }
  void ResetStatus() { status_ = Status::OK(); }

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

  // A pool of cached command lists which may be re-used.
  std::deque<Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList>>
      cached_command_lists_;

  // Status of the first error encountered when closing the command list.
  // Operations that flush the command list or readback from the GPU should make
  // sure that this status doesn't contain an error before doing so.
  Status status_ = Status::OK();

  void SetDescriptorHeap(ID3D12DescriptorHeap* descriptor_heap);
  void Open();

  // Increments operations_recorded_in_current_command_list_. If the size of the
  // current command list exceeds a certain value (based on heuristic), the
  // command list is flushed.
  void OnCommandRecorded();
};

}  // namespace tensorflow
