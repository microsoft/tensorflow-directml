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

template <typename Expression>
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

    DmlTensorInfo lhs_info = {};
    lhs_info.kernel_index = 0;
    lhs_info.desc = tensor_desc;

    DmlTensorInfo rhs_info = {};
    rhs_info.kernel_index = 1;
    rhs_info.desc = tensor_desc;

    DmlKernelTensors tensors = {};
    tensors.inputs = {lhs_info, rhs_info};
    tensors.outputs = {lhs_info};

    auto inputs = GetDmlTensorDescs(tensors.inputs);
    auto scope = dml::Graph(ctx->GetDmlDevice());
    const auto a = dml::InputTensor(scope, 0, inputs[0]);
    const auto b = dml::InputTensor(scope, 1, inputs[1]);
    auto result = Expression()(a, b);

    // TFDML #24881131
    if (Is64BitSignedIntegerType(ctx->GetInputDataType(1))) {
      result = dml::ConvertInt32ToInt64(result);
    }

    Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
        scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

    Initialize(ctx, std::move(tensors), compiled_op.Get());
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

    return DmlKernel::Compute(ctx, input_bindings, output_bindings);
  }
};

#define REGISTER_DML_KERNEL(type)                                        \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("AssignAddVariableOp")                                        \
          .Device(DEVICE_DML)                                            \
          .HostMemory("resource")                                        \
          .TypeConstraint<type>("dtype"),                                \
      DmlKernelWrapper<DmlUpdateVariableOp<std::plus<dml::Expression>>,  \
                       NoOutputShapeHelper>);                            \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("AssignSubVariableOp")                                        \
          .Device(DEVICE_DML)                                            \
          .HostMemory("resource")                                        \
          .TypeConstraint<type>("dtype"),                                \
      DmlKernelWrapper<DmlUpdateVariableOp<std::minus<dml::Expression>>, \
                       NoOutputShapeHelper>);

// This list of types is the intersection between what DML supports, and what
// CUDA registers for these kernels. Notably (and deliberately) missing from
// this list is int32, even though it's supported by DML.
TF_CALL_DML_FLOAT_TYPES(REGISTER_DML_KERNEL);
TF_CALL_int64(REGISTER_DML_KERNEL);

#undef REGISTER_DML_KERNEL

}  // namespace tensorflow
