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

#include "dml_upload_heap.h"

#include "dml_util.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

static D3D12_HEAP_PROPERTIES UploadHeapProps() {
  return CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
}

DmlUploadHeap::DmlUploadHeap(ID3D12Device* device,
                             DmlExecutionContext* execution_context,
                             DmlDeviceRemovedEvent* device_removed_event,
                             ID3D12Device* d3d12_device)
    : DmlPooledHeap(device, UploadHeapProps(),
                    D3D12_RESOURCE_STATE_GENERIC_READ, device_removed_event),
      execution_context_(execution_context),
      device_removed_event_(device_removed_event),
      d3d12_device_(d3d12_device) {}

StatusOr<DmlGpuEvent> DmlUploadHeap::BeginUploadToGpu(
    ID3D12Resource* dst, uint64_t dst_offset, D3D12_RESOURCE_STATES dst_state,
    absl::Span<const uint8_t> src) {
  std::unique_lock<std::mutex> lock(mutex_);

  if (DeviceRemoved()) {
    return errors::Unknown(
        "Uploading data to the GPU attempted after the device has already been "
        "removed.");
  }

  assert(!src.empty());
  assert(dst->GetDesc().Dimension == D3D12_RESOURCE_DIMENSION_BUFFER);

  InvariantChecker checker(this);

  ReclaimAllocations();

  // Allocate space from the upload heap
  Chunk* chunk = nullptr;
  uint64_t offset_in_chunk = 0;
  TF_RETURN_IF_ERROR(Reserve(src.size(), &chunk, &offset_in_chunk));

  assert(chunk != nullptr);
  assert(offset_in_chunk + src.size() <= chunk->capacity_in_bytes);

  // Map the upload heap and copy the source data into it at the specified
  // offset
  void* upload_heap_data = nullptr;
  HRESULT hr = chunk->resource->Map(0, nullptr, &upload_heap_data);

  if (hr == DXGI_ERROR_DEVICE_REMOVED) {
    HRESULT device_removed_reason = d3d12_device_->GetDeviceRemovedReason();
    device_removed_event_->NotifyListeners();
    return DeviceRemovalError(device_removed_reason);
  }

  DML_CHECK_SUCCEEDED(hr);

  memcpy(static_cast<byte*>(upload_heap_data) + offset_in_chunk, src.data(),
         src.size());
  chunk->resource->Unmap(0, nullptr);

  // Copy from the upload heap into the destination resource
  DmlGpuEvent done_event = execution_context_->CopyBufferRegion(
      dst, dst_offset, dst_state, chunk->resource.Get(), offset_in_chunk,
      D3D12_RESOURCE_STATE_GENERIC_READ, src.size());

  // Add an allocation entry to the chunk
  chunk->allocations.push_back(Allocation{static_cast<uint64_t>(src.size()),
                                          offset_in_chunk, done_event});

  return done_event;
}

}  // namespace tensorflow
