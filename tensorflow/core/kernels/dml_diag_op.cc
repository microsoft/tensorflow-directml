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

#pragma optimize("", off)

#include "tensorflow/core/common_runtime/dml/dml_operator_helper.h"
#include "tensorflow/core/common_runtime/dml/dml_util.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/dml_kernel_wrapper.h"
#include "tensorflow/core/kernels/dml_ops_common.h"

namespace tensorflow {

using Microsoft::WRL::ComPtr;

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
    const auto dtype_tf = ctx->GetOutputDataType(0);
    const bool is_64_bit_type = Is64BitIntegerType(dtype_tf);
    const auto dtype_dml = GetDmlDataTypeFromTfDataType(dtype_tf);
    const uint64_t end_padding_in_bytes = is_64_bit_type ? sizeof(uint32_t) : 0;
    const uint32_t stride_multiplier = is_64_bit_type ? 2 : 1;

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

    // TFDML #24881131
    const uint32_t tensor_policy_multiplier =
        Is64BitUnsignedIntegerType(ctx->GetOutputDataType(0)) ? 2 : 1;

    const auto out_policy = dml::TensorPolicy(
        [tensor_policy_multiplier](DML_TENSOR_DATA_TYPE dataType,
                                   DML_TENSOR_FLAGS flags,
                                   dml::Span<const uint32_t> sizes) {
          uint32_t dimension_count = static_cast<uint32_t>(sizes.size());

          const uint32_t num_elements = std::accumulate(
              sizes.begin(), sizes.end(), 1u, std::multiplies<uint32_t>());

          dml::TensorDimensions strides(dimension_count);
          strides.back() = (num_elements + 1) * tensor_policy_multiplier;

          dml::TensorProperties props = {};
          props.guaranteedBaseOffsetAlignment = 0;
          props.strides = std::move(strides);
          props.totalTensorSizeInBytes = DMLCalcBufferTensorSize(
              dataType, dimension_count, sizes.data(), props.strides->data());
          return props;
        });

    auto scope = dml::Graph(ctx->GetDmlDevice(), out_policy);
    auto input_tensor = dml::InputTensor(scope, 0, inputs[0]);
    auto result = dml::Identity(input_tensor);

    // TFDML #24881131
    if (Is64BitSignedIntegerType(ctx->GetOutputDataType(0))) {
      result = dml::ConvertInt32ToInt64(scope, result, num_elements + 1);
    }

    ComPtr<IDMLCompiledOperator> compiled_op =
        scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

    Initialize(ctx, std::move(tensors), compiled_op.Get());
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