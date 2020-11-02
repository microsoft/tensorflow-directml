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

#include "dml_command_queue.h"
#include "dml_command_recorder.h"
#include "dml_common.h"
#include "dml_status.h"

namespace tensorflow {
class DmlAllocator;
class DmlCommandQueue;

// Asynchronously performs GPU work, and automatically manages command list
// recording and submission to queues. Work submitted to the DmlExecutionContext
// is typically recorded onto a command list and may not immediately begin
// execution on the GPU. Call Flush() to force all recorded work to be submitted
// to the command queue for execution on the GPU. This class is NOT thread-safe.
class DmlExecutionContextImpl {
 public:
  // Constructs an DmlExecutionContext that executes on the supplied queue.
  DmlExecutionContextImpl(ID3D12Device* d3d12_device, IDMLDevice* dml_device,
                          ID3D12CommandQueue* queue, DmlAllocator* allocator);

  // Waits for flushed work, discards unflushed work, and discards associated
  // references to prevent circular references. Must be the last call on the
  // object before destruction.
  void Close();

  // Queues a CopyBufferRegion (see ID3D12GraphicsCommandList::CopyBufferRegion)
  // for execution. Transition barriers are automatically inserted to transition
  // the source and destination resources to COPY_SOURCE and COPY_DEST if
  // necessary.
  DmlGpuEvent CopyBufferRegion(ID3D12Resource* dst_buffer, uint64_t dst_offset,
                               D3D12_RESOURCE_STATES dst_state,
                               ID3D12Resource* src_buffer, uint64_t src_offset,
                               D3D12_RESOURCE_STATES src_state,
                               uint64_t byte_count);

  DmlGpuEvent FillBufferWithPattern(
      ID3D12Resource* dst, uint64_t dst_offset, uint64_t dst_size_in_bytes,
      absl::Span<const uint8_t>
          value /* Data type agnostic value, treated as raw bits */);

  DmlGpuEvent InitializeOperator(
      IDMLCompiledOperator* op,
      const DML_BINDING_DESC& persistent_resource_binding,
      const DML_BINDING_DESC& input_array_binding);

  DmlGpuEvent ExecuteOperator(
      IDMLCompiledOperator* op,
      const DML_BINDING_DESC& persistent_resource_binding,
      absl::Span<const DML_BINDING_DESC> input_bindings,
      absl::Span<const DML_BINDING_DESC> output_bindings);

  DmlGpuEvent ResourceBarrier(
      absl::Span<const D3D12_RESOURCE_BARRIER> barriers);

  // Forces all queued work to begin executing on the GPU. This method returns
  // immediately and does not wait for the submitted work to complete execution
  // on the GPU.
  StatusOr<DmlGpuEvent> Flush();

  Status GetCommandRecorderStatus() const;

  // Returns an event which will become signaled when everything submitted to
  // the execution context thus far has completed execution on the GPU,
  // including work that has yet to be flushed to the queue.
  DmlGpuEvent GetCurrentCompletionEvent();

  D3D12_COMMAND_LIST_TYPE GetCommandListTypeForQueue() const;

 private:
  Microsoft::WRL::ComPtr<ID3D12Device> d3d_device_;

  void SetCommandRecorder(DmlCommandRecorder* new_recorder);

  std::shared_ptr<DmlCommandQueue> queue_;

  DmlCommandRecorder* current_recorder_ = nullptr;

  // Up to one of these is active at a time
  DmlCommandRecorder dml_recorder_;  // TODO: merge into this class

  bool closed_ = false;
};

// A thread-safe wrapper over DmlExecutionContextImpl.
class DmlExecutionContext {
 public:
  DmlExecutionContext(ID3D12Device* d3d12_device, IDMLDevice* dml_device,
                      ID3D12CommandQueue* queue, DmlAllocator* allocator);

  void Close() {
    std::unique_lock<std::mutex> lock(mutex_);
    impl_->Close();
  }

  DmlGpuEvent CopyBufferRegion(ID3D12Resource* dst_buffer, uint64_t dst_offset,
                               D3D12_RESOURCE_STATES dst_state,
                               ID3D12Resource* src_buffer, uint64_t src_offset,
                               D3D12_RESOURCE_STATES src_state,
                               uint64_t byte_count) {
    std::unique_lock<std::mutex> lock(mutex_);
    return impl_->CopyBufferRegion(dst_buffer, dst_offset, dst_state,
                                   src_buffer, src_offset, src_state,
                                   byte_count);
  }

  DmlGpuEvent FillBufferWithPattern(ID3D12Resource* dst, uint64_t dst_offset,
                                    uint64_t dst_size_in_bytes,
                                    absl::Span<const uint8_t> value) {
    std::unique_lock<std::mutex> lock(mutex_);
    return impl_->FillBufferWithPattern(dst, dst_offset, dst_size_in_bytes,
                                        value);
  }

  DmlGpuEvent InitializeOperator(
      IDMLCompiledOperator* op,
      const DML_BINDING_DESC& persistent_resource_binding,
      const DML_BINDING_DESC& input_array_binding) {
    std::unique_lock<std::mutex> lock(mutex_);
    return impl_->InitializeOperator(op, persistent_resource_binding,
                                     input_array_binding);
  }

  DmlGpuEvent ExecuteOperator(
      IDMLCompiledOperator* op,
      const DML_BINDING_DESC& persistent_resource_binding,
      absl::Span<const DML_BINDING_DESC> input_bindings,
      absl::Span<const DML_BINDING_DESC> output_bindings) {
    std::unique_lock<std::mutex> lock(mutex_);
    return impl_->ExecuteOperator(op, persistent_resource_binding,
                                  input_bindings, output_bindings);
  }

  DmlGpuEvent ResourceBarrier(
      absl::Span<const D3D12_RESOURCE_BARRIER> barriers) {
    std::unique_lock<std::mutex> lock(mutex_);
    return impl_->ResourceBarrier(barriers);
  }

  StatusOr<DmlGpuEvent> Flush() {
    std::unique_lock<std::mutex> lock(mutex_);
    return impl_->Flush();
  }

  Status GetCommandRecorderStatus() const {
    return impl_->GetCommandRecorderStatus();
  }

  DmlGpuEvent GetCurrentCompletionEvent() {
    std::unique_lock<std::mutex> lock(mutex_);
    return impl_->GetCurrentCompletionEvent();
  }

  D3D12_COMMAND_LIST_TYPE GetCommandListTypeForQueue() const {
    return impl_->GetCommandListTypeForQueue();
  }

 private:
  std::mutex mutex_;
  std::unique_ptr<DmlExecutionContextImpl> impl_;
};

}  // namespace tensorflow
