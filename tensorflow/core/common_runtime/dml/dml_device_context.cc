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

#include "dml_device_context.h"

#include "dml_bfc_allocator.h"
#include "dml_status.h"
#include "dml_util.h"
#include "tensorflow/core/common_runtime/dma_helper.h"

namespace tensorflow {

void DMLDeviceContext::CopyCPUTensorToDevice(const Tensor* cpu_tensor,
                                             Device* device,
                                             Tensor* device_tensor,
                                             StatusCallback done,
                                             bool sync_dst_compute) const {
  size_t total_bytes = cpu_tensor->TotalBytes();
  if (total_bytes == 0) {
    done(Status::OK());
    return;
  }

  const void* src_data = DMAHelper::base(cpu_tensor);

  D3D12BufferRegion dst = dml_util::CreateBufferForTensor(
      static_cast<DmlDevice*>(device), *device_tensor);
  ID3D12Resource* dst_data = dst.Resource();
  const uint64_t dst_offset = dst.Offset();
  const auto dst_state =
      D3D12_RESOURCE_STATE_UNORDERED_ACCESS;  // GPU resources are always kept
                                              // in UAV state

  auto byte_span = absl::Span<const uint8_t>(
      static_cast<const uint8_t*>(src_data), total_bytes);

  StatusOr<DmlGpuEvent> status_or_event = upload_heap_->BeginUploadToGpu(
      dst_data, dst_offset, dst_state, byte_span);

  // Immediately signal completion even though we haven't actually kicked off
  // the GPU, or waited for it to complete. This is because from the framework's
  // point of view, there's no way for it to observe this state (except when
  // copying a tensor back to CPU, at which point we correctly flush and queue a
  // callback)
  done(status_or_event.status());
}

void DMLDeviceContext::CopyTensorInSameDevice(const Tensor* input_tensor,
                                              Device* device,
                                              Tensor* output_tensor,
                                              StatusCallback done) const {
  auto total_bytes = static_cast<uint64_t>(output_tensor->TotalBytes());
  if (total_bytes == 0) {
    done(Status::OK());
    return;
  }

  D3D12BufferRegion src = dml_util::CreateBufferForTensor(
      static_cast<DmlDevice*>(device), *input_tensor);
  ID3D12Resource* src_data = src.Resource();
  const uint64_t src_offset = src.Offset();
  const auto src_state =
      D3D12_RESOURCE_STATE_UNORDERED_ACCESS;  // GPU resources are always kept
                                              // in UAV state

  D3D12BufferRegion dst = dml_util::CreateBufferForTensor(
      static_cast<DmlDevice*>(device), *output_tensor);
  ID3D12Resource* dst_data = dst.Resource();
  const uint64_t dst_offset = dst.Offset();
  const auto dst_state =
      D3D12_RESOURCE_STATE_UNORDERED_ACCESS;  // GPU resources are always kept
                                              // in UAV state

  (void)execution_context_->CopyBufferRegion(dst_data, dst_offset, dst_state,
                                             src_data, src_offset, src_state,
                                             total_bytes);

  // Immediately signal completion even though we haven't actually kicked off
  // the GPU, or waited for it to complete. This is because from the framework's
  // point of view, there's no way for it to observe this state (except when
  // copying a tensor back to CPU, at which point we correctly flush and queue a
  // callback)
  done(Status::OK());
}

void DMLDeviceContext::CopyDeviceTensorToCPU(const Tensor* device_tensor,
                                             StringPiece edge_name,
                                             Device* device, Tensor* cpu_tensor,
                                             StatusCallback done) {
  size_t total_bytes = cpu_tensor->TotalBytes();
  if (total_bytes == 0) {
    done(Status::OK());
    return;
  }

  D3D12BufferRegion src = dml_util::CreateBufferForTensor(
      static_cast<DmlDevice*>(device), *device_tensor);
  ID3D12Resource* src_data = src.Resource();
  const uint64_t src_offset = src.Offset();
  const auto src_state =
      D3D12_RESOURCE_STATE_UNORDERED_ACCESS;  // GPU resources are always kept
                                              // in UAV state

  void* dst_data = DMAHelper::base(cpu_tensor);

  // Performs a blocking call to synchronize and read back data from the GPU
  // into the destination buffer
  auto byte_span =
      absl::Span<uint8_t>(static_cast<uint8_t*>(dst_data), total_bytes);

  StatusOr<DmlGpuEvent> status_or_event = readback_heap_->ReadbackFromGpu(
      byte_span, src_data, src_offset, src_state);

  if (!status_or_event.ok()) {
    done(status_or_event.status());
    return;
  }

  // We have to kick off the GPU now to prevent a potential deadlock, because
  // we don't know if TF is going to block waiting on this copy to complete.
  status_or_event = execution_context_->Flush();

  if (!status_or_event.ok()) {
    done(status_or_event.status());
    return;
  }

  // Keep a ref on the source tensor to keep it alive until we're done with it
  TensorReference input_ref(*device_tensor);

  // Invoke the "done" callback once the readback completes
  auto callback = [done, input_ref] {
    input_ref.Unref();
    done(Status::OK());
  };
  event_queue_->Enqueue(status_or_event.ConsumeValueOrDie(), callback);
}

}  // namespace tensorflow