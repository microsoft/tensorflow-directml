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

class CrossInitHelper : public InitializationHelper {
 public:
  using Attributes = EmptyAttributes;

  CrossInitHelper(OpKernelContext* ctx,
                  std::shared_ptr<const Attributes> attr) {
    const Tensor& in0 = ctx->input(0);
    const Tensor& in1 = ctx->input(1);
    OP_REQUIRES(ctx, in0.shape() == in1.shape(),
                errors::InvalidArgument("Both inputs must be of same shape: ",
                                        in0.shape().DebugString(), " vs. ",
                                        in1.shape().DebugString()));
    OP_REQUIRES(ctx, in0.dims() >= 1,
                errors::InvalidArgument("Input must be at least 1D",
                                        in0.shape().DebugString()));

    // Cross-products only really make sense for three and
    // seven dimensions, and the latter is very obscure. If there is
    // demand, we could perhaps allow 2D vectors where the last
    // element is taken to be zero, but for now, we simply require
    // that all are 3D.
    auto inner_dim = in0.dim_size(in0.dims() - 1);
    OP_REQUIRES(ctx, inner_dim == 3,
                errors::FailedPrecondition(
                    "Cross-products are only defined for 3-element vectors."));
  }
};

class DmlCrossKernel : public DmlKernel {
 public:
  using InitHelper = CrossInitHelper;

  DmlCrossKernel(DmlKernelConstruction* ctx, const InitHelper* init_helper) {
    TensorShape flat_shape({
        ctx->GetOutputTensorShape(0).num_elements() / 3,
        3,
    });

    DmlTensorInfo first_input;
    first_input.kernel_index = 0;
    first_input.desc =
        DmlTensorDesc::Create(ctx->GetInputDataType(0), flat_shape, flat_shape);

    DmlTensorInfo second_input;
    second_input.kernel_index = 1;
    second_input.desc = first_input.desc;

    DmlTensorInfo output;
    output.kernel_index = 0;
    output.desc = first_input.desc;

    DmlKernelTensors tensors;
    tensors.inputs = {first_input, second_input};
    tensors.outputs = {output};

    auto inputs = GetDmlTensorDescs(tensors.inputs);
    auto scope = dml::Graph(ctx->GetDmlDevice());
    auto input1 = dml::InputTensor(scope, 0, inputs[0]);
    auto input2 = dml::InputTensor(scope, 1, inputs[1]);

    // Generate the indices [-2, -1, 0], which is the same as [1, 2, 0]
    auto indices = dml::Sequence<int32>(scope, -2, 1, {1, 1, 1, 3});

    // Generate the first vector:
    // [a, b, c] -> [b, c, a]
    constexpr uint32_t gather_axis = 3;
    auto vector1 = dml::GatherElements(input1, indices, gather_axis);

    // Generate the second vector:
    // [b, c, a] -> [c, a, b]
    auto vector2 = dml::GatherElements(vector1, indices, gather_axis);

    // Generate the third vector:
    // [x, y, z] -> [y, z, x]
    auto vector3 = dml::GatherElements(input2, indices, 3);

    // Generate the fourth vector:
    // [y, z, x] -> [z, x, y]
    auto vector4 = dml::GatherElements(vector3, indices, 3);

    // Multiply the first and last vectors together:
    // [b, c, a] * [z, x, y] = [bz, cx, ay]
    auto vector14 = vector1 * vector4;

    // Multiply the second and third vectors together:
    // [c, a, b] * [y, z, x] = [cy, az, bx]
    auto vector23 = vector2 * vector3;

    // Finally, subtract the second multiplied vector from the first one:
    // [bz-cy, cx-az, ay-bx]
    auto result = vector14 - vector23;

    Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
        scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

    Initialize(ctx, std::move(tensors), compiled_op.Get());
  }
};

#define REGISTER_DML_KERNEL(type)                                 \
  REGISTER_KERNEL_BUILDER(                                        \
      Name("Cross").Device(DEVICE_DML).TypeConstraint<type>("T"), \
      DmlKernelWrapper<DmlCrossKernel, GetOutputShapeAsInputShapeHelper>)

TF_CALL_float(REGISTER_DML_KERNEL);
TF_CALL_half(REGISTER_DML_KERNEL);
TF_CALL_int32(REGISTER_DML_KERNEL);
#undef REGISTER_DML_KERNEL

}  // namespace tensorflow