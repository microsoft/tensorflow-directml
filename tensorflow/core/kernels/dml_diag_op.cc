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

class DiagInitHelper : public InitializationHelper {
 public:
  using Attributes = EmptyAttributes;

  DiagInitHelper(OpKernelContext* ctx, std::shared_ptr<const Attributes> attr) {
    const Tensor& diagonal = ctx->input(0);
    const int num_dims = diagonal.dims();
    OP_REQUIRES(
        ctx, 0 != num_dims,
        errors::InvalidArgument("Input must be at least rank 1, got 0"));
  }
};

class DiagShapeHelper : public ShapeHelper {
 public:
  std::vector<TensorShape> GetOutputShapes(
      OpKernelContext* ctx,
      const InitializationHelper* initialization_helper) const override {
    const Tensor& diagonal = ctx->input(0);
    const int num_dims = diagonal.dims();

    TensorShape output_shape;
    for (int i = 0; i < num_dims; ++i) {
      output_shape.AddDim(diagonal.dim_size(i));
    }
    for (int i = 0; i < num_dims; ++i) {
      output_shape.AddDim(diagonal.dim_size(i));
    }

    return {std::move(output_shape)};
  }
};

class DmlDiagKernel : public DmlKernel {
 public:
  using InitHelper = DiagInitHelper;

  explicit DmlDiagKernel(DmlKernelConstruction* ctx,
                         const InitHelper* init_helper) {
    // Flatten the input into a vector
    TensorShape input_shape(
        {1, 1, 1, ctx->GetInputTensorShape(0).num_elements()});

    DmlTensorInfo input;
    input.kernel_index = 0;
    input.desc = DmlTensorDesc::Create(ctx->GetInputDataType(0), input_shape,
                                       input_shape);

    // TODO #24881131: 64-bit data support should be revisited
    // TFDML #24881131
    auto dtype_tf = ctx->GetOutputDataType(0);
    bool is_64_bit_type = Is64BitIntegerType(dtype_tf);
    auto dtype_dml = is_64_bit_type ? DML_TENSOR_DATA_TYPE_UINT32
                                    : GetDmlDataTypeFromTfDataType(dtype_tf);
    uint64_t end_padding_in_bytes = is_64_bit_type ? sizeof(uint32_t) : 0;
    uint32_t stride_multiplier = is_64_bit_type ? 2 : 1;

    // Flatten the output into a vector and use strides to skip over zeros
    auto num_elements = static_cast<uint32_t>(input_shape.num_elements());
    uint32_t output_sizes[] = {1, 1, 1, num_elements};
    uint32_t output_strides[] = {
        0,
        0,
        0,
        (num_elements + 1) * stride_multiplier,
    };

    DmlTensorInfo output;
    output.kernel_index = 0;
    output.desc = DmlTensorDesc(dtype_dml, output_sizes, output_strides, 0,
                                end_padding_in_bytes);

    DmlKernelTensors tensors;
    tensors.inputs = {input};
    tensors.outputs = {output};

    auto inputs = GetDmlTensorDescs(tensors.inputs);
    auto outputs = GetDmlTensorDescs(tensors.outputs);

    DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC identity_desc = {};
    identity_desc.InputTensor = &inputs[0];
    identity_desc.OutputTensor = &outputs[0];

    DML_OPERATOR_DESC op_desc = {DML_OPERATOR_ELEMENT_WISE_IDENTITY,
                                 &identity_desc};
    Initialize(ctx, std::move(tensors), op_desc);
  }

  StatusOr<DmlGpuEvent> Compute(DmlKernelContext* ctx) const override {
    // Zero the buffer since we use strides to skip over elements
    Tensor* output = ctx->GetOutputTensor(0);
    ctx->ZeroBuffer(ctx->CreateBufferForTensor(*output));

    return DmlKernel::Compute(ctx);
  }
};

#define REGISTER_KERNEL(type)                                    \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("Diag").Device(DEVICE_DML).TypeConstraint<type>("T"), \
      DmlKernelWrapper<DmlDiagKernel, DiagShapeHelper>)

TF_CALL_half(REGISTER_KERNEL);
TF_CALL_float(REGISTER_KERNEL);
TF_CALL_int32(REGISTER_KERNEL);
TF_CALL_int64(REGISTER_KERNEL);

#undef REGISTER_KERNEL

}  // namespace tensorflow