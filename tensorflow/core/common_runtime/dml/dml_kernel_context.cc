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

#include "tensorflow/core/common_runtime/dml/dml_kernel_context.h"

#include "tensorflow/core/common_runtime/dml/dml_device.h"
#include "tensorflow/core/common_runtime/dml/dml_util.h"
#include "tensorflow/core/lib/core/errors.h"

using Microsoft::WRL::ComPtr;

namespace tensorflow {

//
// DmlKernelConstruction
//

DmlKernelConstruction::DmlKernelConstruction(
    const DmlDevice* device, OpKernelContext* op_ctx, const NodeDef* def,
    const ShapeHelper* shape_helper,
    absl::Span<const TensorShape> output_shapes,
    std::shared_ptr<const InitializationHelper> init_helper)
    : device_(device),
      op_ctx_(op_ctx),
      def_(def),
      shape_helper_(shape_helper),
      output_shapes_(output_shapes),
      init_helper_(init_helper) {}

IDMLDevice* DmlKernelConstruction::GetDmlDevice() const {
  return device_->GetDmlDevice();
}
ID3D12Device* DmlKernelConstruction::GetD3D12Device() const {
  return device_->GetD3D12Device();
}

OpKernelContext* DmlKernelConstruction::GetOpKernelContext() const {
  return op_ctx_;
}

std::shared_ptr<const InitializationHelper>
DmlKernelConstruction::GetInitializationHelper() const {
  return init_helper_;
}

DmlBuffer DmlKernelConstruction::AllocateDefaultBuffer(
    uint64_t num_bytes) const {
  return DmlBuffer(device_->GetAllocator(), num_bytes);
}

D3D12BufferRegion DmlKernelConstruction::CreateBufferForTensor(
    const Tensor& tensor) const {
  return dml_util::CreateBufferForTensor(device_, tensor);
}

void DmlKernelConstruction::InitializeOperator(
    IDMLCompiledOperator* op,
    _In_opt_ const DML_BUFFER_BINDING* persistent_resource_binding,
    absl::Span<const DML_BUFFER_BINDING> input_bindings) {
  // Set up the persistent resource binding
  DML_BINDING_DESC persistent_binding_desc = {};
  if (persistent_resource_binding) {
    persistent_binding_desc = {DML_BINDING_TYPE_BUFFER,
                               persistent_resource_binding};
  }

  // Set up the input array binding, if necessary. This is used if OWNED_BY_DML
  // is specified.
  DML_BUFFER_ARRAY_BINDING input_array_binding = {};
  DML_BINDING_DESC input_binding_desc = {};
  if (!input_bindings.empty()) {
    input_array_binding.Bindings = input_bindings.data();
    input_array_binding.BindingCount = static_cast<UINT>(input_bindings.size());
    input_binding_desc = {DML_BINDING_TYPE_BUFFER_ARRAY, &input_array_binding};
  }

  device_->GetExecutionContext()->InitializeOperator(
      op, persistent_binding_desc, input_binding_desc);
}

DataType DmlKernelConstruction::GetInputDataType(uint32_t index) const {
  return op_ctx_->input_dtype(index);
}

TensorShape DmlKernelConstruction::GetInputTensorShape(uint32_t index) const {
  return op_ctx_->input_is_ref(index)
             ? op_ctx_->mutable_input(index, false).shape()
             : op_ctx_->input(index).shape();
}

const Tensor& DmlKernelConstruction::GetConstantInputTensor(
    uint32_t index) const {
  CHECK_EQ(op_ctx_->input_memory_type(index), HOST_MEMORY)
      << "Input tensor at index " << index
      << " was not declared as a constant CPU input tensor. To mark a tensor "
         "as being a constant CPU input, it must be set as residing in host "
         "memory during kernel registration.";

  CHECK_NE(BaseType(op_ctx_->input_dtype(index)), DT_RESOURCE)
      << "Input tensor at index " << index
      << " has type DT_RESOURCE or DT_RESOURCE_REF. Resource tensors are never "
         "constant CPU inputs even if they are declared as residing in host "
         "memory.";

  return op_ctx_->input(index);
}

DataType DmlKernelConstruction::GetOutputDataType(uint32_t index) const {
  return op_ctx_->expected_output_dtype(index);
}

const TensorShape& DmlKernelConstruction::GetOutputTensorShape(
    uint32_t index) const {
  return output_shapes_[index];
}

bool DmlKernelConstruction::HasAttr(StringPiece attr_name) const {
  return HasNodeAttr(*def_, attr_name);
}

//
// DmlKernelContext
//

DmlKernelContext::DmlKernelContext(
    const DmlDevice* device, OpKernelContext* op_ctx,
    const InitializationHelper* init_helper,
    absl::Span<const TensorShape> output_shapes,
    absl::Span<const absl::optional<uint32_t>> output_refs_forwarding)
    : device_(device), op_ctx_(op_ctx), init_helper_(init_helper) {
  assert(output_shapes.size() == op_ctx_->num_outputs());

  // Allocate output tensors
  output_tensors_.reserve(output_shapes.size());
  for (int i = 0; i < static_cast<int>(output_shapes.size()); ++i) {
    Tensor* output_tensor = nullptr;

    if (IsRefType(op_ctx_->expected_output_dtype(i))) {
      // Ref types have already been allocated beforehand
      CHECK(i < output_refs_forwarding.size());
      CHECK(output_refs_forwarding[i].has_value());
      op_ctx->forward_ref_input_to_ref_output(*output_refs_forwarding[i], i);
      output_tensor = op_ctx_->mutable_output(i);
    } else {
      OP_REQUIRES_OK(op_ctx_, op_ctx_->allocate_output(i, output_shapes[i],
                                                       &output_tensor));
    }

    output_tensors_.push_back(output_tensor);
  }
}

IDMLDevice* DmlKernelContext::GetDmlDevice() const {
  return device_->GetDmlDevice();
}
ID3D12Device* DmlKernelContext::GetD3D12Device() const {
  return device_->GetD3D12Device();
}

OpKernelContext* DmlKernelContext::GetOpKernelContext() const {
  return op_ctx_;
}

DmlBuffer DmlKernelContext::AllocateDefaultBuffer(uint64_t num_bytes) const {
  return DmlBuffer(device_->GetAllocator(), num_bytes);
}

D3D12BufferRegion DmlKernelContext::CreateBufferForTensor(
    const Tensor& tensor) const {
  return dml_util::CreateBufferForTensor(device_, tensor);
}

DmlGpuEvent DmlKernelContext::ExecuteOperator(
    IDMLCompiledOperator* op,
    _In_opt_ const DML_BUFFER_BINDING* persistent_resource_binding,
    absl::Span<const absl::optional<DML_BUFFER_BINDING>> input_bindings,
    absl::Span<const absl::optional<DML_BUFFER_BINDING>> output_bindings) {
  // Set up the persistent resource binding
  DML_BINDING_DESC persistent_binding_desc = {};
  if (persistent_resource_binding) {
    persistent_binding_desc = {DML_BINDING_TYPE_BUFFER,
                               persistent_resource_binding};
  }

  // Set up the input bindings
  absl::InlinedVector<DML_BINDING_DESC, 8> input_binding_descs;
  for (const auto& binding : input_bindings) {
    DML_BINDING_DESC desc = {DML_BINDING_TYPE_NONE, nullptr};
    if (binding) {
      desc = {DML_BINDING_TYPE_BUFFER, &binding.value()};
    }

    input_binding_descs.push_back(desc);
  }

  // Set up the output bindings
  absl::InlinedVector<DML_BINDING_DESC, 4> output_binding_descs;
  for (const auto& binding : output_bindings) {
    DML_BINDING_DESC desc = {DML_BINDING_TYPE_NONE, nullptr};
    if (binding) {
      desc = {DML_BINDING_TYPE_BUFFER, &binding.value()};
    }

    output_binding_descs.push_back(desc);
  }

  return device_->GetExecutionContext()->ExecuteOperator(
      op, persistent_binding_desc, input_binding_descs, output_binding_descs);
}

DmlGpuEvent DmlKernelContext::GetCurrentCompletionEvent() const {
  return device_->GetExecutionContext()->GetCurrentCompletionEvent();
}

DmlGpuEvent DmlKernelContext::CopyBufferToBuffer(ID3D12Resource* dst,
                                                 uint64_t dst_offset,
                                                 ID3D12Resource* src,
                                                 uint64_t src_offset,
                                                 uint64 size_in_bytes) const {
  return device_->GetExecutionContext()->CopyBufferRegion(
      dst, dst_offset, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, src, src_offset,
      D3D12_RESOURCE_STATE_UNORDERED_ACCESS, size_in_bytes);
}

StatusOr<DmlGpuEvent> DmlKernelContext::CopyHostToBuffer(
    ID3D12Resource* dst, uint64_t dst_offset,
    absl::Span<const uint8_t> src) const {
  return device_->GetUploadHeap()->BeginUploadToGpu(
      dst, dst_offset, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, src);
}

DmlGpuEvent DmlKernelContext::ZeroBuffer(ID3D12Resource* dst, uint64_t offset,
                                         uint64_t size_in_bytes) const {
  uint8_t pattern[] = {0};
  return device_->GetExecutionContext()->FillBufferWithPattern(
      dst, offset, size_in_bytes, pattern);
}

DmlGpuEvent DmlKernelContext::ZeroBuffer(const D3D12BufferRegion& dst) const {
  return ZeroBuffer(dst.Resource(), dst.Offset(), dst.SizeInBytes());
}

DmlGpuEvent DmlKernelContext::InsertUavBarrier() const {
  D3D12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::UAV(nullptr);

  return device_->GetExecutionContext()->ResourceBarrier(
      absl::Span<D3D12_RESOURCE_BARRIER>(&barrier, 1));
}

DmlGpuEvent DmlKernelContext::FillBufferWithPattern(
    ID3D12Resource* dst, uint64_t offset, uint64_t size_in_bytes,
    absl::Span<const uint8_t> pattern) const {
  return device_->GetExecutionContext()->FillBufferWithPattern(
      dst, offset, size_in_bytes, pattern);
}

}  // namespace tensorflow