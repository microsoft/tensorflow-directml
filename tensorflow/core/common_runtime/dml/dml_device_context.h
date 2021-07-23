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

#include "dml_buffer.h"
#include "dml_buffer_region.h"
#include "dml_descriptor_bfc_allocator.h"
#include "dml_event_queue.h"
#include "dml_execution_context.h"
#include "dml_readback_heap.h"
#include "dml_upload_heap.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/device_base.h"

namespace tensorflow {

class DMLDeviceContext : public DeviceContext {
 private:
  // These are all weak pointers; owned by the DML device factory
  DmlExecutionContext* execution_context_;
  DmlEventQueue* event_queue_;
  DmlUploadHeap* upload_heap_;
  DmlReadbackHeap* readback_heap_;
  DmlAllocator* allocator_;
  DmlDescriptorAllocator* descriptor_allocator_;

 public:
  DMLDeviceContext(DmlExecutionContext* execution_context,
                   DmlEventQueue* event_queue, DmlUploadHeap* upload_heap,
                   DmlReadbackHeap* readback_heap, DmlAllocator* allocator,
                   DmlDescriptorAllocator* descriptor_allocator)
      : execution_context_(execution_context),
        event_queue_(event_queue),
        upload_heap_(upload_heap),
        readback_heap_(readback_heap),
        allocator_(allocator),
        descriptor_allocator_(descriptor_allocator) {}

  // --------------------------------
  // TF::DeviceContext Overrides
  // --------------------------------

  void CopyCPUTensorToDevice(const Tensor* cpu_tensor, Device* device,
                             Tensor* device_tensor, StatusCallback done,
                             bool sync_dst_compute) const override;

  void CopyTensorInSameDevice(const Tensor* input_tensor, Device* device,
                              Tensor* output_tensor,
                              StatusCallback done) const override;

  void CopyDeviceTensorToCPU(const Tensor* device_tensor, StringPiece edge_name,
                             Device* device, Tensor* cpu_tensor,
                             StatusCallback done) override;

  // --------------------------------
  // Allocation & Execution Helpers
  // --------------------------------

  // Initializes a given DML operator on the GPU. Note that this merely
  // queues the initialization; the returned event will enter the signaled
  // state when it completes. Note that we never supply any input bindings,
  // because we never set DML_TENSOR_FLAG_OWNED_BY_DML .
  DmlGpuEvent BindAndInitializeOperator(
      IDMLOperatorInitializer* initializer,
      Microsoft::WRL::ComPtr<IDMLBindingTable>&& binding_table,
      ID3D12DescriptorHeap* heap_for_binding_table,
      _In_opt_ const DML_BUFFER_BINDING* temporary_resource_binding,
      _In_opt_ const DML_BUFFER_BINDING* persistent_resource_binding);

  // Executes a DML operator. Note that this merely queues the execution; the
  // returned event will enter the signaled state when it completes.
  DmlGpuEvent BindAndExecuteOperator(
      IDMLCompiledOperator* op,
      Microsoft::WRL::ComPtr<IDMLBindingTable>&& binding_table,
      ID3D12DescriptorHeap* heap_for_binding_table,
      _In_opt_ const DML_BUFFER_BINDING* temporary_resource_binding,
      _In_opt_ const DML_BUFFER_BINDING* persistent_resource_binding,
      absl::Span<const absl::optional<DML_BUFFER_BINDING>> input_bindings,
      absl::Span<const absl::optional<DML_BUFFER_BINDING>> output_bindings);

  DmlGpuEvent InsertUavBarrier() const;

  DmlGpuEvent GetCurrentCompletionEvent() const;

  // Enqueues a callback to fire when the given GpuEvent enters the signaled
  // state. Note that the callback may be invoked on an arbitrary thread, so it
  // must be thread-safe.
  void EnqueueCallbackForGpuEvent(DmlGpuEvent gpu_event,
                                  std::function<void()> callback) const;

  // Allocates a D3D12 default heap buffer which is at least num_bytes large.
  // When the returned object is destructed, the memory is freed back to the
  // pool.
  DmlBuffer AllocateDefaultBuffer(uint64_t num_bytes) const;

  // Retrives the D3D12 default heap buffer backing the specified tensor.
  D3D12BufferRegion GetBufferForTensor(const Tensor& tensor) const;

  // Allocates a range of D3D12 descriptors at least size_in_descriptors large.
  // When the returned object is destructed, the descriptors are freed back to
  // the pool.
  DescriptorAllocation AllocateDescriptors(size_t size_in_descriptors) const;

  // Copies src to dst (dst needs to be at least as big as src).
  DmlGpuEvent CopyBufferToBuffer(const D3D12BufferRegion& dst,
                                 const D3D12BufferRegion& src) const;

  // Copies src (host memory) to dst (dst needs to be at least as big as src).
  StatusOr<DmlGpuEvent> CopyHostToBuffer(const D3D12BufferRegion& dst,
                                         absl::Span<const uint8_t> src) const;

  // Fills dst region with zeroes.
  DmlGpuEvent ZeroBuffer(const D3D12BufferRegion& dst) const;

  // Fills dst region with a repeating byte pattern.
  DmlGpuEvent FillBufferWithPattern(const D3D12BufferRegion& dst,
                                    absl::Span<const uint8_t> value) const;

  // Fills dst region with a repeat value.
  template <typename T>
  DmlGpuEvent FillBufferWithValue(const D3D12BufferRegion& dst, T value) const {
    static_assert(
        sizeof(value) <= 16,
        "FillBufferWithValue doesn't accept values bigger than 16 bytes.");

    static_assert(
        std::is_trivially_copyable<T>::value,
        "FillBufferWithValue only accepts values that are trivially copyable.");

    auto value_bytes = absl::Span<const uint8_t>(
        reinterpret_cast<const uint8_t*>(&value), sizeof(value));

    return FillBufferWithPattern(dst, value_bytes);
  }
};

}  // namespace tensorflow