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

#include "dml_command_queue.h"

namespace tensorflow {

DmlCommandQueue::DmlCommandQueue(ID3D12CommandQueue* existing_queue)
    : queue_(existing_queue), type_(existing_queue->GetDesc().Type) {
  Microsoft::WRL::ComPtr<ID3D12Device> device;
  DML_CHECK_SUCCEEDED(queue_->GetDevice(IID_PPV_ARGS(&device)));

  DML_CHECK_SUCCEEDED(
      device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_)));
}

void DmlCommandQueue::ExecuteCommandLists(
    absl::Span<ID3D12CommandList*> command_lists) {
  queue_->ExecuteCommandLists(static_cast<uint32_t>(command_lists.size()),
                              command_lists.data());

  ++last_fence_value_;
  DML_CHECK_SUCCEEDED(queue_->Signal(fence_.Get(), last_fence_value_));
}

DmlGpuEvent DmlCommandQueue::GetCurrentCompletionEvent() {
  return DmlGpuEvent{last_fence_value_, fence_};
}

DmlGpuEvent DmlCommandQueue::GetNextCompletionEvent() {
  return DmlGpuEvent{last_fence_value_ + 1, fence_};
}

void DmlCommandQueue::Close() {
  // Wait for flushed work:
  assert(!closing_);
  closing_ = true;
  DmlGpuEvent event = GetCurrentCompletionEvent();
  event.WaitForSignal();
  closing_ = false;
}

}  // namespace tensorflow
