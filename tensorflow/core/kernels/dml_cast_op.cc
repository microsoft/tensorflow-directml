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

class DmlCastKernel : public DmlKernel {
 public:
  using InitHelper = NoOpInitializationHelper;

  explicit DmlCastKernel(DmlKernelConstruction* ctx,
                         const InitHelper* init_helper) {
    CHECK(ctx->GetInputCount() == 1);
    CHECK(ctx->GetOutputCount() == 1);

    DataType input_dtype = ctx->GetInputDataType(0);
    DataType output_dtype = ctx->GetOutputDataType(0);
    const DML_TENSOR_DATA_TYPE dml_out_dtype =
        GetDmlDataTypeFromTfDataType(output_dtype);

    // TFDML #24881131
    const dml::TensorPolicy out_policy =
        Is64BitUnsignedIntegerType(output_dtype)
            ? GetEmulatedInt64TensorPolicy()
            : dml::TensorPolicy::Default();

    // Tensor shape doesn't matter for Cast, so don't bother with DML's 4D
    // restrictions
    TensorShape tensor_shape({ctx->GetOutputTensorShape(0).num_elements()});

    DmlTensorInfo input;
    input.kernel_index = 0;
    input.desc = DmlTensorDesc::Create(input_dtype, tensor_shape, tensor_shape);

    DmlTensorInfo output;
    output.kernel_index = 0;
    output.desc =
        DmlTensorDesc::Create(output_dtype, tensor_shape, tensor_shape);

    DmlKernelTensors tensors;
    tensors.outputs = {output};
    tensors.inputs = {input};

    auto inputs = GetDmlTensorDescs(tensors.inputs);
    auto scope = dml::Graph(ctx->GetDmlDevice(), out_policy);
    auto input_tensor = dml::InputTensor(scope, 0, inputs[0]);

    // Bool is a special case since it doesn't behave the same as uint8. The
    // uint8 version simply drops the decimals, but bool converts anything that
    // is not 0.0 to True.
    if (output_dtype == DT_BOOL &&
        (input_dtype == DT_HALF || input_dtype == DT_FLOAT)) {
      input_tensor = dml::Ceil(dml::Abs(input_tensor));
    }

    auto result = dml::Cast(input_tensor, dml_out_dtype);

    if (output_dtype == DT_BOOL) {
      result = dml::Clip(result, 0.0, 1.0);
    }

    // TFDML #24881131
    if (Is64BitSignedIntegerType(output_dtype)) {
      result = dml::ConvertInt32ToInt64(result);
    }

    Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
        scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

    Initialize(ctx, std::move(tensors), compiled_op.Get());
  }

  StatusOr<DmlGpuEvent> Compute(DmlKernelContext* ctx) const {
    Tensor* output = ctx->GetOutputTensor(0);

    // TFDML #24881131
    if (Is64BitUnsignedIntegerType(output->dtype())) {
      ctx->GetDmlDeviceContext()->ZeroBuffer(
          ctx->GetDmlDeviceContext()->CreateBufferForTensor(*output));
    }

    return DmlKernel::Compute(ctx);
  }
};

#define DML_REGISTER_KERNEL_OUTPUT(output_type)              \
  REGISTER_KERNEL_BUILDER(                                   \
      Name("Cast")                                           \
          .template TypeConstraint<input_type_alias>("SrcT") \
          .template TypeConstraint<output_type>("DstT")      \
          .Device(DEVICE_DML),                               \
      DmlKernelWrapper<DmlCastKernel, GetOutputShapeAsInputShapeHelper>)

template <typename TInput>
class DmlCastRegistration {
 public:
  DmlCastRegistration() {
    using input_type_alias = TInput;
    TF_CALL_DML_ALL_TYPES(DML_REGISTER_KERNEL_OUTPUT);
  }
};

#define CONCAT_NAME_HELPER(name, unique_id) name##unique_id
#define CONCAT_NAME(name, unique_id) CONCAT_NAME_HELPER(name, unique_id)

#define DML_REGISTER_KERNEL_INPUT(input_type)         \
  static DmlCastRegistration<input_type> CONCAT_NAME( \
      dml_cast_kernel_registration, __COUNTER__);

TF_CALL_DML_ALL_TYPES(DML_REGISTER_KERNEL_INPUT);
#undef DML_REGISTER_KERNEL_INPUT
#undef DML_REGISTER_KERNEL_OUTPUT

}  // namespace tensorflow