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

#include "dml_status.h"
#include "tensorflow/core/common_runtime/dml/dml_buffer.h"
#include "tensorflow/core/common_runtime/dml/dml_buffer_region.h"
#include "tensorflow/core/common_runtime/dml/dml_common.h"
#include "tensorflow/core/common_runtime/dml/dml_descriptor_bfc_allocator.h"
#include "tensorflow/core/common_runtime/dml/dml_device_context.h"
#include "tensorflow/core/common_runtime/dml/dml_gpu_event.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

class DmlDevice;
class ShapeHelper;
class InitializationHelper;

// Context supplied to a DML kernel during construction.
class DmlKernelConstruction {
 public:
  DmlKernelConstruction(
      const DmlDevice* device, OpKernelContext* op_ctx, const NodeDef* def,
      absl::Span<const TensorShape> output_shapes,
      std::shared_ptr<const InitializationHelper> init_helper);

  IDMLDevice* GetDmlDevice() const;
  ID3D12Device* GetD3D12Device() const;
  OpKernelContext* GetOpKernelContext() const;
  DMLDeviceContext* GetDmlDeviceContext() const;
  std::shared_ptr<const InitializationHelper> GetInitializationHelper() const;

  // Input tensors
  uint32_t GetInputCount() const { return op_ctx_->num_inputs(); }
  DataType GetInputDataType(uint32_t index) const;
  TensorShape GetInputTensorShape(uint32_t index) const;

  // Retrieves a constant CPU input tensor.
  //
  // Constant CPU input tensors are those which are declared as residing in host
  // memory during kernel registration (by specifying .HostMemory("input_name")
  // during REGISTER_KERNEL_BUILDER). Constant CPU input tensors are available
  // during kernel construction, their contents are guaranteed not to change
  // for the lifetime of the kernel, and their contents are guaranteed to reside
  // in CPU memory. This is useful when an operator defines an input as dynamic
  // but the kernel expects it to be static (like an attribute would be).
  //
  // For example, the ClipByValue operator defines its clip_value_min and
  // clip_value_max inputs as scalar tensors - however DirectML expects it to be
  // provided as part of the operator desc. In this case, the kernel can mark
  // those inputs with .HostMemory which makes them constant CPU inputs, and
  // therefore available during kernel construction.
  //
  // Although the contents of the returned tensor are guaranteed not to change
  // during the lifetime of the kernel, the memory backing the tensor itself is
  // not guaranteed to outlive the DmlKernelConstruction. For this reason,
  // kernels should not store references to the returned tensor which persist
  // beyond the kernel constructor.
  const Tensor& GetConstantInputTensor(uint32_t index) const;

  // Output tensors

  uint32_t GetOutputCount() const { return op_ctx_->num_outputs(); }
  DataType GetOutputDataType(uint32_t index) const;
  const TensorShape& GetOutputTensorShape(uint32_t index) const;

  // See OpKernelConstruction::GetAttr
  template <class T>
  Status GetAttr(StringPiece attr_name, T* value) const {
    return GetNodeAttr(*def_, attr_name, value);
  }

  // See OpKernelConstruction::HasAttr
  bool HasAttr(StringPiece attr_name) const;

 private:
  const DmlDevice* device_;
  OpKernelContext* op_ctx_;
  const NodeDef* def_;
  absl::Span<const TensorShape> output_shapes_;
  std::shared_ptr<const InitializationHelper> init_helper_;
};

// Context supplied to a DML kernel during execution.
class DmlKernelContext {
 public:
  DmlKernelContext(
      const DmlDevice* device, OpKernelContext* op_ctx,
      const InitializationHelper* init_helper,
      absl::Span<const TensorShape> output_shapes,
      absl::Span<const absl::optional<uint32_t>> output_refs_forwarding,
      bool supports_in_place_execution);

  IDMLDevice* GetDmlDevice() const;
  ID3D12Device* GetD3D12Device() const;
  OpKernelContext* GetOpKernelContext() const;
  DMLDeviceContext* GetDmlDeviceContext() const;

  template <typename T>
  const T* GetInitializationHelper() const {
    return static_cast<const T*>(init_helper_);
  }

  Tensor GetInputTensor(int index) const {
    return op_ctx_->input_is_ref(index) ? op_ctx_->mutable_input(index, false)
                                        : op_ctx_->input(index);
  }
  uint32_t GetInputCount() const { return op_ctx_->num_inputs(); }

  Tensor* GetOutputTensor(int index) { return output_tensors_[index]; }
  uint32_t GetOutputCount() const { return op_ctx_->num_outputs(); }

 private:
  const DmlDevice* device_;
  OpKernelContext* op_ctx_;
  const InitializationHelper* init_helper_;

  // These output tensors are owned by the framework, because they're allocated
  // using OpKernelContext::allocate_output()
  absl::InlinedVector<Tensor*, 4> output_tensors_;
};

}  // namespace tensorflow