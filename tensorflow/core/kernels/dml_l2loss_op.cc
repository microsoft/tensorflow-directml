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
#include "tensorflow/core/kernels/dml_kernel_wrapper.h"
#include "tensorflow/core/kernels/dml_ops_common.h"

namespace tensorflow {

class DmlL2LossKernel : public DmlKernel {
 public:
  using InitHelper = NoOpInitializationHelper;

  explicit DmlL2LossKernel(DmlKernelConstruction* ctx,
                           const InitHelper* init_helper) {
    CHECK(ctx->GetInputCount() == 1);
    CHECK(ctx->GetOutputCount() == 1);

    // The shape is irrelevant since this op reduces all input elements.
    auto num_elements =
        static_cast<uint32_t>(ctx->GetInputTensorShape(0).num_elements());
    uint32_t dml_input_shape[4] = {1, 1, 1, num_elements};
    uint32_t dml_output_shape[4] = {1, 1, 1, 1};

    DML_TENSOR_DATA_TYPE dml_type =
        GetDmlDataTypeFromTfDataType(ctx->GetInputDataType(0));

    DmlTensorInfo input_info = {};
    input_info.kernel_index = 0;
    input_info.desc = DmlTensorDesc{dml_type, dml_input_shape};

    DmlTensorInfo output_info = {};
    output_info.kernel_index = 0;
    output_info.desc = DmlTensorDesc{dml_type, dml_output_shape};

    DmlKernelTensors tensors = {};
    tensors.inputs = {input_info};
    tensors.outputs = {output_info};

    auto inputs = GetDmlTensorDescs(tensors.inputs);
    auto scope = dml::Graph(ctx->GetDmlDevice());
    auto input = dml::InputTensor(scope, 0, inputs[0]);

    auto loss = dml::Reduce(input, DML_REDUCE_FUNCTION_SUM_SQUARE) * 0.5f;

    Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
        scope.Compile(DML_EXECUTION_FLAG_NONE, {loss});

    Initialize(ctx, std::move(tensors), compiled_op.Get());
  }
};

#define DML_REGISTER_KERNEL(type)                                  \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("L2Loss").Device(DEVICE_DML).TypeConstraint<type>("T"), \
      DmlKernelWrapper<DmlL2LossKernel, ScalarOutputShapeHelper>)
TF_CALL_DML_FLOAT_TYPES(DML_REGISTER_KERNEL);
#undef DML_REGISTER_KERNEL

}  // namespace tensorflow
