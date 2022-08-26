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

template <typename TIndex>
class FillInitializationHelper : public InitializationHelper {
 public:
  using Attributes = EmptyAttributes;

  bool IsNoOpKernel(
      OpKernelContext* ctx,
      absl::Span<const TensorShape> output_shapes) const override {
    // The output shape of this kernel is determined by a dims tensor. Since an
    // empty dims tensor is valid, we only look at the output tensor shapes when
    // determining when to no-op.
    return output_shapes[0].num_elements() == 0;
  }

  FillInitializationHelper(OpKernelContext* ctx,
                           std::shared_ptr<const Attributes> attr) {
    const Tensor& dims_tensor = ctx->input(0);
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(dims_tensor.shape()),
                errors::InvalidArgument("dims must be a vector, got shape ",
                                        dims_tensor.shape().DebugString()));

    const TensorShape& value_shape = ctx->input(1).shape();
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(value_shape),
                errors::InvalidArgument("value must be a scalar, got shape ",
                                        value_shape.DebugString()));

    // We only call TensorShapeUtils::MakeShape for the validation that it
    // provides
    auto output_dims = dims_tensor.flat<TIndex>();
    TensorShape output_shape;
    OP_REQUIRES_OK(
        ctx, TensorShapeUtils::MakeShape(output_dims.data(), output_dims.size(),
                                         &output_shape));
  }
};

template <typename TIndex>
class DmlFillKernel : public DmlKernel {
 public:
  using InitHelper = FillInitializationHelper<TIndex>;

  explicit DmlFillKernel(DmlKernelConstruction* ctx,
                         const InitHelper* init_helper) {
    CHECK(ctx->GetInputCount() == 2);
    CHECK(ctx->GetOutputCount() == 1);

    DmlKernelParams params;

    // Broadcast inputs to match output shape
    params.input_shape = ctx->GetOutputTensorShape(0);

    // The value tensor is at index 1 in TF's kernel
    params.kernel_input_indices = {1};

    DmlKernelTensors tensors = GetTensorInfos(ctx, params);

    auto inputs = GetDmlTensorDescs(tensors.inputs);
    auto scope = dml::Graph(ctx->GetDmlDevice());
    auto input_tensor = dml::InputTensor(scope, 0, inputs[0]);
    auto result = dml::Identity(input_tensor);

    Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
        scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

    Initialize(ctx, std::move(tensors), compiled_op.Get());
  }
};

template <typename TIndex>
using DmlFillWrapper =
    DmlKernelWrapper<DmlFillKernel<TIndex>,
                     GetOutputShapeFromDimsTensorHelper<TIndex, 0>>;

#define REGISTER_DML_KERNEL(TYPE)                                  \
  REGISTER_KERNEL_BUILDER(Name("Fill")                             \
                              .Device(DEVICE_DML)                  \
                              .TypeConstraint<TYPE>("T")           \
                              .TypeConstraint<int32>("index_type") \
                              .HostMemory("dims"),                 \
                          DmlFillWrapper<int32>);                  \
  REGISTER_KERNEL_BUILDER(Name("Fill")                             \
                              .Device(DEVICE_DML)                  \
                              .TypeConstraint<TYPE>("T")           \
                              .TypeConstraint<int64>("index_type") \
                              .HostMemory("dims"),                 \
                          DmlFillWrapper<int64>);

// TODO(b/25387198): A special kernel exists for int32 (see constant_op.cc).
TF_CALL_float(REGISTER_DML_KERNEL);
TF_CALL_half(REGISTER_DML_KERNEL);
TF_CALL_uint8(REGISTER_DML_KERNEL);
TF_CALL_int8(REGISTER_DML_KERNEL);
TF_CALL_uint16(REGISTER_DML_KERNEL);
TF_CALL_int16(REGISTER_DML_KERNEL);
TF_CALL_int64(REGISTER_DML_KERNEL);
TF_CALL_bool(REGISTER_DML_KERNEL);
#undef REGISTER_DML_KERNEL

}  // namespace tensorflow
