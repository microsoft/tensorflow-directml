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

class CompareAndBitpackInitHelper : public InitializationHelper {
 public:
  using Attributes = EmptyAttributes;

  CompareAndBitpackInitHelper(OpKernelContext* ctx,
                              std::shared_ptr<const Attributes> attr) {
    const Tensor& input_t = ctx->input(0);
    const Tensor& threshold_t = ctx->input(1);
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(threshold_t.shape()),
        errors::InvalidArgument("Compare must be a scalar, but saw shape: ",
                                threshold_t.shape().DebugString()));
    const TensorShape& input_shape = input_t.shape();
    OP_REQUIRES(ctx, TensorShapeUtils::IsVectorOrHigher(input_shape),
                errors::InvalidArgument(
                    "Input should be at least a vector, but saw a scalar."));
    OP_REQUIRES(ctx, input_shape.dim_size(input_shape.dims() - 1) % 8 == 0,
                errors::InvalidArgument(
                    "Inner dimension of input should be "
                    "divisible by ",
                    8, ", but saw shape: ", input_shape.DebugString()));
  }
};

class CompareAndBitpackShapeHelper : public ShapeHelper {
 public:
  std::vector<TensorShape> GetOutputShapes(
      OpKernelContext* ctx,
      const InitializationHelper* initialization_helper) const override {
    TensorShape output_shape = ctx->input(0).shape();
    int rank = output_shape.dims();
    output_shape.set_dim(rank - 1, output_shape.dim_size(rank - 1) / 8);
    return {output_shape};
  }
};

class DmlCompareAndBitpackKernel : public DmlKernel {
 public:
  using InitHelper = CompareAndBitpackInitHelper;

  DmlCompareAndBitpackKernel(DmlKernelConstruction* ctx,
                             const InitHelper* init_helper) {
    TensorShape input_shape({
        ctx->GetInputTensorShape(0).num_elements() / 8,
        8,
    });

    TensorShape output_shape({ctx->GetOutputTensorShape(0).num_elements(), 1});

    DmlTensorInfo input_tensor;
    input_tensor.kernel_index = 0;
    input_tensor.desc = DmlTensorDesc::Create(ctx->GetInputDataType(0),
                                              input_shape, input_shape);

    DmlTensorInfo threshold_tensor;
    threshold_tensor.kernel_index = 1;
    threshold_tensor.desc = DmlTensorDesc::Create(
        ctx->GetInputDataType(1), input_shape, TensorShape({1}));

    DmlTensorInfo output_tensor;
    output_tensor.kernel_index = 0;
    output_tensor.desc = DmlTensorDesc::Create(ctx->GetOutputDataType(0),
                                               output_shape, output_shape);

    DmlKernelTensors tensors;
    tensors.inputs = {input_tensor, threshold_tensor};
    tensors.outputs = {output_tensor};

    auto inputs = GetDmlTensorDescs(tensors.inputs);
    auto scope = dml::Graph(ctx->GetDmlDevice());
    auto input = dml::InputTensor(scope, 0, inputs[0]);
    auto threshold = dml::InputTensor(scope, 1, inputs[1]);

    auto result =
        dml::GreaterThan(input, threshold, DML_TENSOR_DATA_TYPE_UINT32);

    auto bit_shift_start = dml::ScalarUnion(7, DML_TENSOR_DATA_TYPE_INT32);
    auto bit_shift_delta = dml::ScalarUnion(-1, DML_TENSOR_DATA_TYPE_INT32);

    auto bit_shifts =
        dml::FillValueSequence(scope, {1, 1, 1, 8}, DML_TENSOR_DATA_TYPE_UINT32,
                               bit_shift_start, bit_shift_delta);

    bit_shifts = dml::Reinterpret(bit_shifts, input.GetOutputDesc().sizes,
                                  dml::TensorDesc::Dimensions({0, 0, 0, 1}));

    result <<= bit_shifts;
    result = dml::Reduce(result, DML_REDUCE_FUNCTION_SUM, {3});
    result = dml::Cast(result, DML_TENSOR_DATA_TYPE_UINT8);

    Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
        scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

    Initialize(ctx, std::move(tensors), compiled_op.Get());
  }
};

#define REGISTER_DML_KERNEL(type)                                             \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("CompareAndBitpack").Device(DEVICE_DML).TypeConstraint<type>("T"), \
      DmlKernelWrapper<DmlCompareAndBitpackKernel,                            \
                       CompareAndBitpackShapeHelper>)

TF_CALL_float(REGISTER_DML_KERNEL);
TF_CALL_half(REGISTER_DML_KERNEL);
TF_CALL_bool(REGISTER_DML_KERNEL);
#undef REGISTER_DML_KERNEL

}  // namespace tensorflow