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

  D3D12BufferRegion dst = GetBufferForTensor(*device_tensor);

  auto byte_span = absl::Span<const uint8_t>(
      static_cast<const uint8_t*>(src_data), total_bytes);

  StatusOr<DmlGpuEvent> status_or_event =
      upload_heap_->BeginUploadToGpu(dst, byte_span);

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

  D3D12BufferRegion src = GetBufferForTensor(*input_tensor);
  D3D12BufferRegion dst = GetBufferForTensor(*output_tensor);

  (void)execution_context_->CopyBufferRegion(dst,
                                             src.Subregion(0, total_bytes));

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

  D3D12BufferRegion src = GetBufferForTensor(*device_tensor);

  void* dst_data = DMAHelper::base(cpu_tensor);

  // Performs a blocking call to synchronize and read back data from the GPU
  // into the destination buffer
  auto byte_span =
      absl::Span<uint8_t>(static_cast<uint8_t*>(dst_data), total_bytes);

  StatusOr<DmlGpuEvent> status_or_event =
      readback_heap_->ReadbackFromGpu(byte_span, src);

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

DmlGpuEvent DMLDeviceContext::BindAndInitializeOperator(
    IDMLOperatorInitializer* initializer,
    Microsoft::WRL::ComPtr<IDMLBindingTable>&& binding_table,
    ID3D12DescriptorHeap* heap_for_binding_table,
    _In_opt_ const DML_BUFFER_BINDING* temporary_resource_binding,
    _In_opt_ const DML_BUFFER_BINDING* persistent_resource_binding) {
  // Bind the temporary resource
  if (temporary_resource_binding) {
    DML_BINDING_DESC temporary_binding_desc = {DML_BINDING_TYPE_BUFFER,
                                               temporary_resource_binding};
    binding_table->BindTemporaryResource(&temporary_binding_desc);
  }

  // Bind the persistent resource
  if (persistent_resource_binding) {
    DML_BINDING_DESC persistent_binding_desc = {DML_BINDING_TYPE_BUFFER,
                                                persistent_resource_binding};
    binding_table->BindOutputs(1, &persistent_binding_desc);
  }

  return execution_context_->InitializeOperator(
      initializer, std::move(binding_table), heap_for_binding_table);
}

DmlGpuEvent DMLDeviceContext::BindAndExecuteOperator(
    IDMLCompiledOperator* op,
    Microsoft::WRL::ComPtr<IDMLBindingTable>&& binding_table,
    ID3D12DescriptorHeap* heap_for_binding_table,
    _In_opt_ const DML_BUFFER_BINDING* temporary_resource_binding,
    _In_opt_ const DML_BUFFER_BINDING* persistent_resource_binding,
    absl::Span<const absl::optional<DML_BUFFER_BINDING>> input_bindings,
    absl::Span<const absl::optional<DML_BUFFER_BINDING>> output_bindings) {
  // Bind the temporary resource
  DML_BINDING_DESC temporary_binding_desc = {DML_BINDING_TYPE_NONE, nullptr};
  if (temporary_resource_binding) {
    temporary_binding_desc = {DML_BINDING_TYPE_BUFFER,
                              temporary_resource_binding};
  }
  binding_table->BindTemporaryResource(&temporary_binding_desc);

  // Bind the persistent resource
  DML_BINDING_DESC persistent_binding_desc = {DML_BINDING_TYPE_NONE, nullptr};
  if (persistent_resource_binding) {
    persistent_binding_desc = {DML_BINDING_TYPE_BUFFER,
                               persistent_resource_binding};
  }
  binding_table->BindPersistentResource(&persistent_binding_desc);

  // Set up the input bindings
  absl::InlinedVector<DML_BINDING_DESC, 8> input_binding_descs;
  for (const auto& binding : input_bindings) {
    DML_BINDING_DESC desc = {DML_BINDING_TYPE_NONE, nullptr};
    if (binding) {
      desc = {DML_BINDING_TYPE_BUFFER, &binding.value()};
    }

    input_binding_descs.push_back(desc);
  }
  binding_table->BindInputs(static_cast<UINT>(input_binding_descs.size()),
                            input_binding_descs.data());

  // Set up the output bindings
  absl::InlinedVector<DML_BINDING_DESC, 4> output_binding_descs;
  for (const auto& binding : output_bindings) {
    DML_BINDING_DESC desc = {DML_BINDING_TYPE_NONE, nullptr};
    if (binding) {
      desc = {DML_BINDING_TYPE_BUFFER, &binding.value()};
    }

    output_binding_descs.push_back(desc);
  }
  binding_table->BindOutputs(static_cast<UINT>(output_binding_descs.size()),
                             output_binding_descs.data());

  return execution_context_->ExecuteOperator(op, std::move(binding_table),
                                             heap_for_binding_table);
}

DmlGpuEvent DMLDeviceContext::InsertUavBarrier() const {
  return execution_context_->UavBarrier();
}

DmlGpuEvent DMLDeviceContext::GetCurrentCompletionEvent() const {
  return execution_context_->GetCurrentCompletionEvent();
}

void DMLDeviceContext::EnqueueCallbackForGpuEvent(
    DmlGpuEvent gpu_event, std::function<void()> callback) const {
  event_queue_->Enqueue(std::move(gpu_event), std::move(callback));
}

DmlBuffer DMLDeviceContext::AllocateDefaultBuffer(uint64_t num_bytes) const {
  return DmlBuffer(allocator_, num_bytes);
}

D3D12BufferRegion DMLDeviceContext::GetBufferForTensor(
    const Tensor& tensor) const {
  const void* p = tensor.tensor_data().data();

  // DML always requires at least 4 byte alignment in all cases, so both the
  // offset and size must certainly be divisible by 4.
  constexpr uint64_t dml_alignment = 4;

  // The offset and size of the region must be aligned to DirectML's
  // requirement. Each tensor has two sizes:
  //
  // - TotalBytes: num_elements * sizeof_element. This may be too small if the
  // tensor has elements smaller than 4 bytes (e.g. 3x float16 is 6 bytes, but
  // DML needs an 8 byte region).
  //
  // - AllocatedBytes: the size of allocation backing the tensor. This is often
  // larger than TotalBytes since the smallest DML allocation size is 256 bytes.
  //
  // While AllocatedBytes is guaranteed to meet DML's requirement, tensor
  // buffers may be offset within an individual allocation (see Tensor::Slice).
  // Using AllocatedBytes directly can result in a region that extends beyond
  // the bounds of the allocation. Instead we round the total bytes up to an
  // aligned value, which should always fit within the allocated bytes.
  uint64_t size_in_bytes =
      dml_alignment + (tensor.TotalBytes() - 1) / dml_alignment;
  CHECK(size_in_bytes <= tensor.AllocatedBytes());

  auto region = allocator_->CreateBufferRegion(p, size_in_bytes);

  DCHECK(region.Offset() % dml_alignment == 0);
  DCHECK(region.SizeInBytes() % dml_alignment == 0);

  return region;
}

DescriptorAllocation DMLDeviceContext::AllocateDescriptors(
    size_t size_in_descriptors) const {
  return descriptor_allocator_->Alloc(size_in_descriptors);
}

DmlGpuEvent DMLDeviceContext::CopyBufferToBuffer(
    const D3D12BufferRegion& dst, const D3D12BufferRegion& src) const {
  return execution_context_->CopyBufferRegion(dst, src);
}

StatusOr<DmlGpuEvent> DMLDeviceContext::CopyHostToBuffer(
    const D3D12BufferRegion& dst, absl::Span<const uint8_t> src) const {
  return upload_heap_->BeginUploadToGpu(dst, src);
}

// Fills dst region with zeroes.
DmlGpuEvent DMLDeviceContext::ZeroBuffer(const D3D12BufferRegion& dst) const {
  uint8_t pattern[] = {0};
  return FillBufferWithPattern(dst, pattern);
}

// Fills dst region with a repeating byte pattern.
DmlGpuEvent DMLDeviceContext::FillBufferWithPattern(
    const D3D12BufferRegion& dst, absl::Span<const uint8_t> value) const {
  return execution_context_->FillBufferWithPattern(dst, value);
}

}  // namespace tensorflow