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
    auto outputs = GetDmlTensorDescs(tensors.outputs);

    DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC identity_desc = {};
    identity_desc.InputTensor = &inputs[0];
    identity_desc.OutputTensor = &outputs[0];

    DML_OPERATOR_DESC op_desc = {DML_OPERATOR_ELEMENT_WISE_IDENTITY,
                                 &identity_desc};
    Initialize(ctx, std::move(tensors), op_desc);
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
TF_CALL_DML_ALL_TYPES_EXCEPT_INT32(REGISTER_DML_KERNEL)
#undef REGISTER_DML_KERNEL

}  // namespace tensorflow
