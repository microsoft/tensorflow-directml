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

#include "dml_readback_heap.h"

#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

static D3D12_HEAP_PROPERTIES ReadbackHeapProps() {
  return CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK);
}

DmlReadbackHeap::DmlReadbackHeap(ID3D12Device* device,
                                 DmlExecutionContext* execution_context,
                                 DmlEventQueue* event_queue)
    : DmlPooledHeap(device, ReadbackHeapProps(),
                    D3D12_RESOURCE_STATE_COPY_DEST),
      execution_context_(execution_context),
      event_queue_(event_queue) {
  current_completion_event_.fence_value = 0;
  DML_CHECK_SUCCEEDED(
      device->CreateFence(0, D3D12_FENCE_FLAG_NONE,
                          IID_PPV_ARGS(&current_completion_event_.fence)));
}

StatusOr<DmlGpuEvent> DmlReadbackHeap::ReadbackFromGpu(
    absl::Span<uint8_t> dst, ID3D12Resource* src, uint64_t src_offset,
    D3D12_RESOURCE_STATES src_state) {
  std::unique_lock<std::mutex> lock(mutex_);

  assert(!dst.empty());
  assert(src->GetDesc().Dimension == D3D12_RESOURCE_DIMENSION_BUFFER);

  InvariantChecker checker(this);

  ReclaimAllocations();

  // Allocate space from the upload heap
  Chunk* chunk = nullptr;
  uint64_t offset_in_chunk = 0;
  TF_RETURN_IF_ERROR(Reserve(dst.size(), &chunk, &offset_in_chunk));

  assert(chunk != nullptr);
  assert(offset_in_chunk + dst.size() <= chunk->capacity_in_bytes);

  ID3D12Resource* readback_heap = chunk->resource.Get();

  // Copy from the source resource into the readback heap. `gpu_done_event` is
  // the event that will be signaled when the copy to the readback heap
  // completes on the GPU.
  DmlGpuEvent gpu_done_event = execution_context_->CopyBufferRegion(
      readback_heap, offset_in_chunk, D3D12_RESOURCE_STATE_COPY_DEST, src,
      src_offset, src_state, dst.size());

  // Get the event which will become signaled once the readback into `dst` has
  // fully completed on the CPU.
  ++current_completion_event_.fence_value;
  DmlGpuEvent done_event = current_completion_event_;

  // Note that we don't need to keep a ref on the readback_heap, because the
  // pooled allocator guarantees it'll live until we give the signal
  auto done_callback = [this, dst, readback_heap, offset_in_chunk, done_event] {
    // The device could have been removed before the callback is called
    if (!execution_context_->GetCommandRecorderStatus().ok()) return;

    void* readback_heap_data = nullptr;
    DML_CHECK_SUCCEEDED(readback_heap->Map(0, nullptr, &readback_heap_data));
    readback_heap_data =
        static_cast<byte*>(readback_heap_data) + offset_in_chunk;
    memcpy(dst.data(), readback_heap_data, dst.size());
    readback_heap->Unmap(0, nullptr);

    // We're done - signal the event with its fence value.
    DML_CHECK_SUCCEEDED(done_event.fence->Signal(done_event.fence_value));
  };

  // Add an allocation entry to the chunk
  chunk->allocations.push_back(Allocation{static_cast<uint64_t>(dst.size()),
                                          offset_in_chunk, done_event});

  // Enqueue the done_callback to fire once the copy from src -> readback_heap
  // completes on the GPU. The callback will then perform the copy
  // readback_heap -> dst.
  event_queue_->Enqueue(gpu_done_event, done_callback);
  return done_event;
}

}  // namespace tensorflow
