/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.
Portions Copyright (c) Microsoft Corporation.

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
#include "tensorflow/core/kernels/assign_op.h"
#include "tensorflow/core/kernels/dml_kernel_wrapper.h"
#include "tensorflow/core/kernels/dml_ops_common.h"

namespace tensorflow {

class DmlAssignOp : public AssignOp {
 public:
  explicit DmlAssignOp(OpKernelConstruction* context) : AssignOp(context) {}

  void Copy(OpKernelContext* context, Tensor* lhs, const Tensor& rhs) override {
    DeviceContext* device_context = context->op_device_context();
    Device* device = static_cast<Device*>(context->device());

    device_context->CopyTensorInSameDevice(
        &rhs, device, lhs,
        [context](const Status& s) { OP_REQUIRES_OK(context, s); });
  }
};

template <typename Expression>
class DmlAssignModifyOp : public DmlKernel {
 public:
  using InitHelper = NoOpInitializationHelper;

  explicit DmlAssignModifyOp(DmlKernelConstruction* ctx,
                             const InitHelper* init_helper) {
    CHECK(ctx->GetInputCount() == 2);
    CHECK(ctx->GetOutputCount() == 1);

    // The Assign operators don't support broadcasting, so it's safe to collapse
    // all dimensions into a single one before sending it to DML. This allows us
    // to support tensors with more than 4 or 5 dimensions.
    TensorShape tensor_shape = {ctx->GetOutputTensorShape(0).num_elements()};

    DmlKernelTensors tensors;

    for (uint32_t i = 0; i < ctx->GetInputCount(); ++i) {
      DmlTensorInfo input;
      input.kernel_index = i;
      input.desc = DmlTensorDesc::Create(ctx->GetInputDataType(i), tensor_shape,
                                         tensor_shape);

      tensors.inputs.push_back(std::move(input));
    }

    DmlTensorInfo output;
    output.kernel_index = 0;
    output.desc = DmlTensorDesc::Create(ctx->GetOutputDataType(0), tensor_shape,
                                        tensor_shape);

    tensors.outputs = {output};

    // The input ref and the output ref must refer to the same memory
    tensors.output_refs_forwarding = {0};

    auto inputs = GetDmlTensorDescs(tensors.inputs);
    auto scope = dml::Graph(ctx->GetDmlDevice());
    auto a_tensor = dml::InputTensor(scope, 0, inputs[0]);
    auto b_tensor = dml::InputTensor(scope, 1, inputs[1]);
    auto result = Expression()(a_tensor, b_tensor);

    // TFDML #24881131
    if (Is64BitSignedIntegerType(ctx->GetOutputDataType(0))) {
      result = dml::ConvertInt32ToInt64(result);
    }

    Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
        scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

    Initialize(ctx, std::move(tensors), compiled_op.Get());
  }
};

// Only register 'Assign' on DML for the subset of types also supported by
// 'Variable' (see variable_ops.cc.)
#define REGISTER_DML_KERNEL(type)                                  \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("Assign").Device(DEVICE_DML).TypeConstraint<type>("T"), \
      DmlAssignOp);

TF_CALL_DML_FLOAT_TYPES(REGISTER_DML_KERNEL);
TF_CALL_bool(REGISTER_DML_KERNEL);
TF_CALL_int64(REGISTER_DML_KERNEL);
#undef REGISTER_DML_KERNEL

#define REGISTER_DML_KERNEL(type)                                      \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("AssignAdd").Device(DEVICE_DML).TypeConstraint<type>("T"),  \
      DmlKernelWrapper<DmlAssignModifyOp<std::plus<dml::Expression>>,  \
                       GetOutputShapeAsInputShapeHelper>);             \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("AssignSub").Device(DEVICE_DML).TypeConstraint<type>("T"),  \
      DmlKernelWrapper<DmlAssignModifyOp<std::minus<dml::Expression>>, \
                       GetOutputShapeAsInputShapeHelper>);

TF_CALL_DML_FLOAT_TYPES(REGISTER_DML_KERNEL);
TF_CALL_int64(REGISTER_DML_KERNEL);
#undef REGISTER_DML_KERNEL

}  // namespace tensorflow
