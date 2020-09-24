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

class DmlAddNKernel : public DmlKernel {
 public:
  using InitHelper = NoOpInitializationHelper;

  explicit DmlAddNKernel(DmlKernelConstruction* ctx,
                         const InitHelper* init_helper) {
    CHECK(ctx->GetInputCount() >= 1);
    CHECK(ctx->GetOutputCount() == 1);

    // AddN doesn't support broadcasting, so we can simply collapse all
    // dimensions into a single one to support more than 5D
    TensorShape tensor_shape({ctx->GetOutputTensorShape(0).num_elements()});

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

    auto inputs = GetDmlTensorDescs(tensors.inputs);

    if (ctx->GetInputCount() == 1) {
      auto outputs = GetDmlTensorDescs(tensors.outputs);

      DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC identity_desc = {};
      identity_desc.InputTensor = inputs.data();
      identity_desc.OutputTensor = outputs.data();

      DML_OPERATOR_DESC op_desc = {DML_OPERATOR_ELEMENT_WISE_IDENTITY,
                                   &identity_desc};
      Initialize(ctx, std::move(tensors), op_desc);
    } else {
      auto scope = dml::Graph(ctx->GetDmlDevice());
      auto result = dml::InputTensor(scope, 0, inputs[0]);

      for (uint32_t i = 1; i < inputs.size(); ++i) {
        result += dml::InputTensor(scope, i, inputs[i]);
      }

      Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
          scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

      Initialize(ctx, std::move(tensors), compiled_op.Get());
    }
  }

  StatusOr<DmlGpuEvent> Compute(DmlKernelContext* ctx) const override {
    // Currently, 64-bit integers in DML are emulated using 32-bit integers
    // using striding to emulate a larger type. Because we can't guarantee that
    // our output tensor's memory is zero'd, we need to do so manually prior to
    // running running gather.
    Tensor* output = ctx->GetOutputTensor(0);

    if (Is64BitIntegerType(output->dtype())) {
      ctx->ZeroBuffer(ctx->CreateBufferForTensor(*output));
    }

    return DmlKernel::Compute(ctx);
  }
};

#define REGISTER_KERNEL(type)                                    \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("AddN").Device(DEVICE_DML).TypeConstraint<type>("T"), \
      DmlKernelWrapper<DmlAddNKernel, GetOutputShapeAsInputShapeHelper>)

// TODO(b/25387198): A special kernel exists for int32 (see aggregate_ops.cc).
TF_CALL_float(REGISTER_KERNEL);
TF_CALL_half(REGISTER_KERNEL);
TF_CALL_uint32(REGISTER_KERNEL);
TF_CALL_int64(REGISTER_KERNEL);
TF_CALL_uint64(REGISTER_KERNEL);
#undef REGISTER_KERNEL

}  // namespace tensorflow