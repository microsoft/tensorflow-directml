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
    absl::Span<const TensorShape> output_shapes,
    std::shared_ptr<const InitializationHelper> init_helper)
    : device_(device),
      op_ctx_(op_ctx),
      def_(def),
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

DMLDeviceContext* DmlKernelConstruction::GetDmlDeviceContext() const {
  return static_cast<DMLDeviceContext*>(op_ctx_->op_device_context());
}

std::shared_ptr<const InitializationHelper>
DmlKernelConstruction::GetInitializationHelper() const {
  return init_helper_;
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
      absl::InlinedVector<int, 4> candidate_input_indices(
          op_ctx_->num_inputs());
      std::iota(candidate_input_indices.begin(), candidate_input_indices.end(),
                0);

      OP_REQUIRES_OK(op_ctx_, op_ctx_->forward_input_or_allocate_output(
                                  candidate_input_indices, i, output_shapes[i],
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

DMLDeviceContext* DmlKernelContext::GetDmlDeviceContext() const {
  return static_cast<DMLDeviceContext*>(op_ctx_->op_device_context());
}

}  // namespace tensorflow