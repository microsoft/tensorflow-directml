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

    // Currently, 64-bit integers in DML are emulated using 32-bit integers
    // using striding to emulate a larger type. Because we can't guarantee
    // that our output tensor's memory is zero'd, we need to do so manually
    // prior to performing the cast.
    if (Is64BitIntegerType(ctx->GetOutputDataType(0))) {
      zero_outputs_ = true;
    }

    // Tensor shape doesn't matter for Cast, so don't bother with DML's 4D
    // restrictions
    TensorShape tensor_shape({ctx->GetOutputTensorShape(0).num_elements()});

    DmlTensorInfo input;
    input.kernel_index = 0;
    input.desc = DmlTensorDesc::Create(ctx->GetInputDataType(0), tensor_shape,
                                       tensor_shape);

    DmlTensorInfo output;
    output.kernel_index = 0;
    output.desc = DmlTensorDesc::Create(ctx->GetOutputDataType(0), tensor_shape,
                                        tensor_shape);

    DmlKernelTensors tensors;
    tensors.outputs = {output};
    tensors.inputs = {input};

    auto inputs = GetDmlTensorDescs(tensors.inputs);
    auto outputs = GetDmlTensorDescs(tensors.outputs);

    DML_CAST_OPERATOR_DESC cast_desc = {};
    cast_desc.InputTensor = inputs.data();
    cast_desc.OutputTensor = outputs.data();

    DML_OPERATOR_DESC op_desc = {DML_OPERATOR_CAST, &cast_desc};
    Initialize(ctx, std::move(tensors), op_desc);
  }

  StatusOr<DmlGpuEvent> Compute(DmlKernelContext* ctx) const {
    if (zero_outputs_) {
      Tensor* output = ctx->GetOutputTensor(0);
      ctx->ZeroBuffer(ctx->CreateBufferForTensor(*output));
    }

    return DmlKernel::Compute(ctx);
  }

 private:
  bool zero_outputs_ = false;
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