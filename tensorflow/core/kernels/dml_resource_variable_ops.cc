/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/dml/dml_operator_helper.h"
#include "tensorflow/core/common_runtime/dml/dml_util.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/kernels/dml_kernel_wrapper.h"
#include "tensorflow/core/kernels/dml_ops_common.h"
#include "tensorflow/core/kernels/training_op_helpers.h"

namespace tensorflow {

template <DML_OPERATOR_TYPE op_type, typename DML_OPERATOR_SPECIFIC_DESC>
class DmlUpdateVariableOp : public DmlKernel {
 public:
  using InitHelper = NoOpInitializationHelper;

  explicit DmlUpdateVariableOp(DmlKernelConstruction* ctx,
                               const InitHelper* init_helper) {
    uint32_t tensor_sizes[] = {
        1, 1, 1,
        static_cast<uint32_t>(ctx->GetInputTensorShape(1).num_elements())};

    auto tensor_desc = DmlTensorDesc::Create(ctx->GetInputDataType(1),
                                             tensor_sizes, tensor_sizes);

    // This kernel actually only has one (GPU) input: the right-hand side of the
    // AssignAdd or AssignSub expression. The left-hand side is a resource
    // handle residing in host memory, which we use during Compute to retrieve
    // the actual GPU-backed tensor. This LHS tensor is treated as an inout
    // tensor, which relies on DML's in-place execution.
    DmlTensorInfo rhs_info = {};
    rhs_info.kernel_index = 1;
    rhs_info.desc = tensor_desc;

    DmlKernelTensors tensors = {};
    tensors.inputs = {rhs_info};
    tensors.outputs = {};

    // All inputs/outputs have the same tensor desc
    DML_TENSOR_DESC in_out_desc = tensor_desc.GetDmlDesc();
    DML_OPERATOR_SPECIFIC_DESC op_specific_desc = {
        &in_out_desc,  // ATensor
        &in_out_desc,  // BTensor
        &in_out_desc,  // OutputTensor
    };

    DML_OPERATOR_DESC op_desc = {op_type, &op_specific_desc};
    Initialize(ctx, std::move(tensors), op_desc);
  }

  StatusOr<DmlGpuEvent> Compute(DmlKernelContext* ctx) const override {
    auto* op_ctx = ctx->GetOpKernelContext();

    core::RefCountPtr<Var> variable;
    TF_RETURN_IF_ERROR(
        LookupResource(op_ctx, HandleFromInput(op_ctx, 0), &variable));

    mutex_lock ml(*variable->mu());

    Tensor* var_tensor = variable->tensor();
    const TensorShape& var_shape = variable->tensor()->shape();
    const Tensor& value = ctx->GetInputTensor(1);
    const TensorShape& value_shape = value.shape();

    if (!var_shape.IsSameSize(value_shape)) {
      return errors::InvalidArgument(
          "Cannot update variable with shape ", var_shape.DebugString(),
          " using a Tensor with shape ", value_shape.DebugString(),
          ", shapes must be equal.");
    }

    TF_RETURN_IF_ERROR(PrepareToUpdateVariable(
        op_ctx, var_tensor, variable->copy_on_read_mode.load(),
        &dml_util::CopyTensorInSameDevice));

    D3D12BufferRegion var_resource = ctx->CreateBufferForTensor(*var_tensor);
    D3D12BufferRegion value_resource = ctx->CreateBufferForTensor(value);

    absl::optional<DML_BUFFER_BINDING> input_bindings[] = {
        var_resource.GetBufferBinding(),
        value_resource.GetBufferBinding(),
    };

    // Bind the first input as the output, to take advantage of in-place
    // execution
    absl::optional<DML_BUFFER_BINDING> output_bindings[] = {
        input_bindings[0],
    };

    return ctx->ExecuteOperator(GetCompiledOp(), GetPersistentResourceBinding(),
                                input_bindings, output_bindings);
  }

 private:
  Microsoft::WRL::ComPtr<IDMLCompiledOperator> op_;
};

using DmlAssignAddVariableOp =
    DmlUpdateVariableOp<DML_OPERATOR_ELEMENT_WISE_ADD,
                        DML_ELEMENT_WISE_ADD_OPERATOR_DESC>;
using DmlAssignSubVariableOp =
    DmlUpdateVariableOp<DML_OPERATOR_ELEMENT_WISE_SUBTRACT,
                        DML_ELEMENT_WISE_SUBTRACT_OPERATOR_DESC>;

#define REGISTER_DML_KERNEL(type)                                     \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("AssignAddVariableOp")                                     \
          .Device(DEVICE_DML)                                         \
          .HostMemory("resource")                                     \
          .TypeConstraint<type>("dtype"),                             \
      DmlKernelWrapper<DmlAssignAddVariableOp, NoOutputShapeHelper>); \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("AssignSubVariableOp")                                     \
          .Device(DEVICE_DML)                                         \
          .HostMemory("resource")                                     \
          .TypeConstraint<type>("dtype"),                             \
      DmlKernelWrapper<DmlAssignSubVariableOp, NoOutputShapeHelper>);

// This list of types is the intersection between what DML supports, and what
// CUDA registers for these kernels. Notably (and deliberately) missing from
// this list is int32, even though it's supported by DML.
TF_CALL_DML_FLOAT_TYPES(REGISTER_DML_KERNEL);
TF_CALL_int64(REGISTER_DML_KERNEL);
TF_CALL_bool(REGISTER_DML_KERNEL);

#undef REGISTER_DML_KERNEL

}  // namespace tensorflow
