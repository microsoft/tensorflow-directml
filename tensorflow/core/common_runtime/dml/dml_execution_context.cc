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

#include "dml_execution_context.h"

#include "dml_bfc_allocator.h"

namespace tensorflow {

DmlExecutionContext::DmlExecutionContext(ID3D12Device* d3d12_device,
                                         IDMLDevice* dml_device,
                                         ID3D12CommandQueue* queue,
                                         DmlAllocator* allocator)
    : impl_(absl::make_unique<DmlExecutionContextImpl>(d3d12_device, dml_device,
                                                       queue, allocator)) {}

DmlExecutionContextImpl::DmlExecutionContextImpl(ID3D12Device* d3d12_device,
                                                 IDMLDevice* dml_device,
                                                 ID3D12CommandQueue* queue,
                                                 DmlAllocator* allocator)
    : queue_(std::make_shared<DmlCommandQueue>(queue)),
      dml_recorder_(d3d12_device, dml_device, queue_, allocator) {
  DML_CHECK_SUCCEEDED(
      dml_device->GetParentDevice(IID_PPV_ARGS(d3d_device_.GetAddressOf())));
}

DmlGpuEvent DmlExecutionContextImpl::CopyBufferRegion(
    ID3D12Resource* dst_buffer, uint64_t dst_offset,
    D3D12_RESOURCE_STATES dst_state, ID3D12Resource* src_buffer,
    uint64_t src_offset, D3D12_RESOURCE_STATES src_state, uint64_t byte_count) {
  assert(!closed_);

  SetCommandRecorder(&dml_recorder_);
  dml_recorder_.CopyBufferRegion(dst_buffer, dst_offset, dst_state, src_buffer,
                                 src_offset, src_state, byte_count);
  return GetCurrentCompletionEvent();
}

DmlGpuEvent DmlExecutionContextImpl::FillBufferWithPattern(
    ID3D12Resource* dst, uint64_t dst_offset, uint64_t dst_size_in_bytes,
    absl::Span<const uint8_t>
        value /* Data type agnostic value, treated as raw bits */) {
  assert(!closed_);
  SetCommandRecorder(&dml_recorder_);
  dml_recorder_.FillBufferWithPattern(dst, dst_offset, dst_size_in_bytes,
                                      value);
  return GetCurrentCompletionEvent();
}

DmlGpuEvent DmlExecutionContextImpl::InitializeOperator(
    IDMLCompiledOperator* op,
    const DML_BINDING_DESC& persistent_resource_binding,
    const DML_BINDING_DESC& input_array_binding) {
  assert(!closed_);
  SetCommandRecorder(&dml_recorder_);

  dml_recorder_.InitializeOperator(op, persistent_resource_binding,
                                   input_array_binding);

  return GetCurrentCompletionEvent();
}

DmlGpuEvent DmlExecutionContextImpl::ExecuteOperator(
    IDMLCompiledOperator* op,
    const DML_BINDING_DESC& persistent_resource_binding,
    absl::Span<const DML_BINDING_DESC> input_bindings,
    absl::Span<const DML_BINDING_DESC> output_bindings) {
  assert(!closed_);
  SetCommandRecorder(&dml_recorder_);

  dml_recorder_.ExecuteOperator(op, persistent_resource_binding, input_bindings,
                                output_bindings);

  return GetCurrentCompletionEvent();
}

DmlGpuEvent DmlExecutionContextImpl::ResourceBarrier(
    absl::Span<const D3D12_RESOURCE_BARRIER> barriers) {
  assert(!closed_);
  SetCommandRecorder(&dml_recorder_);

  dml_recorder_.ResourceBarrier(barriers);
  return GetCurrentCompletionEvent();
}

void DmlExecutionContextImpl::SetCommandRecorder(
    DmlCommandRecorder* new_recorder) {
  assert(!closed_);
  current_recorder_ = new_recorder;
}

StatusOr<DmlGpuEvent> DmlExecutionContextImpl::Flush() {
  assert(!closed_);

  if (!current_recorder_) {
    // Nothing to flush
    return GetCurrentCompletionEvent();
  }

  current_recorder_->CloseAndExecute();
  Status recorder_status = current_recorder_->GetStatus();

  if (!recorder_status.ok()) {
    // "Unknown" represents device removals, which are uncoverable failures
    if (!errors::IsUnknown(recorder_status)) {
      current_recorder_->ResetStatus();
      current_recorder_ = nullptr;
    }
    return recorder_status;
  }

  // Just submitted our command list, so we have neither DML or D3D12 work
  // recorded on any of our command lists.
  current_recorder_ = nullptr;

  return DmlExecutionContextImpl::GetCurrentCompletionEvent();
}

Status DmlExecutionContextImpl::GetCommandRecorderStatus() const {
  return current_recorder_ ? current_recorder_->GetStatus() : Status::OK();
}

void DmlExecutionContextImpl::Close() {
  assert(!closed_);

  queue_->Close();
  current_recorder_ = nullptr;
  closed_ = true;
}

DmlGpuEvent DmlExecutionContextImpl::GetCurrentCompletionEvent() {
  assert(!closed_);

  DmlGpuEvent event = queue_->GetCurrentCompletionEvent();

  // If something has been recorded into a command list but not submitted yet,
  // it means that the *next* fence value is the one to signal completion.
  const bool unflushed_work_exists =
      (current_recorder_ != nullptr && current_recorder_->HasUnflushedWork());
  if (unflushed_work_exists) {
    ++event.fence_value;
  }

  return event;
}

D3D12_COMMAND_LIST_TYPE DmlExecutionContextImpl::GetCommandListTypeForQueue()
    const {
  assert(!closed_);
  return queue_->GetType();
}

}  // namespace tensorflow
