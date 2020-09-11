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
#include "tensorflow/core/util/bcast.h"

namespace tensorflow {

class BroadcastToInitHelper : public InitializationHelper {
 public:
  using Attributes = EmptyAttributes;

  BroadcastToInitHelper(OpKernelContext* ctx,
                        std::shared_ptr<const Attributes> attr) {
    const TensorShape& input_shape = ctx->input(0).shape();
    const Tensor& shape_tensor = ctx->input(1);

    TensorShape output_shape;
    OP_REQUIRES_OK(ctx,
                   ctx->op_kernel().MakeShape(shape_tensor, &output_shape));

    OP_REQUIRES(ctx, input_shape.dims() <= output_shape.dims(),
                errors::InvalidArgument(
                    "Rank of input (", input_shape.dims(),
                    ") must be no greater than rank of output shape (",
                    output_shape.dims(), ")."));

    BCast bcast(BCast::FromShape(input_shape), BCast::FromShape(output_shape));

    OP_REQUIRES(ctx, bcast.IsValid(),
                errors::InvalidArgument(
                    "Incompatible shapes: ", input_shape.DebugString(), " vs. ",
                    output_shape.DebugString()));
    OP_REQUIRES(ctx, BCast::ToShape(bcast.output_shape()) == output_shape,
                errors::InvalidArgument("Unable to broadcast tensor of shape ",
                                        input_shape, " to tensor of shape ",
                                        output_shape));

    collapsed_input_shape_ = BCast::ToShape(bcast.x_reshape());
    collapsed_output_shape_ = BCast::ToShape(bcast.y_reshape());

    OP_REQUIRES(ctx, collapsed_output_shape_.dims() <= kNcdhwDimensionCount,
                errors::InvalidArgument(
                    "DML doesn't support more than 5D for BroadcastTo after "
                    "collapsing dimensions together, but the output has ",
                    collapsed_output_shape_.dims(), " dimensions."));
  }

  const TensorShape& GetCollapsedInputShape() const {
    return collapsed_input_shape_;
  }

  const TensorShape& GetCollapsedOutputShape() const {
    return collapsed_output_shape_;
  }

 private:
  TensorShape collapsed_input_shape_;
  TensorShape collapsed_output_shape_;
};

class DmlBroadcastToKernel : public DmlKernel {
 public:
  using InitHelper = BroadcastToInitHelper;

  explicit DmlBroadcastToKernel(DmlKernelConstruction* ctx,
                                const InitHelper* init_helper) {
    CHECK(ctx->GetInputCount() == 2);
    CHECK(ctx->GetOutputCount() == 1);

    DmlKernelTensors tensors;

    const TensorShape input_shape = init_helper->GetCollapsedInputShape();
    const TensorShape output_shape = init_helper->GetCollapsedOutputShape();

    DmlTensorInfo input;
    input.kernel_index = 0;
    input.desc = DmlTensorDesc::Create(ctx->GetInputDataType(0), output_shape,
                                       input_shape);
    tensors.inputs = {input};

    DmlTensorInfo output;
    output.kernel_index = 0;
    output.desc = DmlTensorDesc::Create(ctx->GetOutputDataType(0), output_shape,
                                        output_shape);
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

#define REGISTER_KERNEL(type)                                         \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("BroadcastTo")                                             \
          .Device(DEVICE_DML)                                         \
          .TypeConstraint<type>("T")                                  \
          .TypeConstraint<int32>("Tidx")                              \
          .HostMemory("shape"),                                       \
      DmlKernelWrapper<DmlBroadcastToKernel,                          \
                       GetOutputShapeFromDimsTensorHelper<int32, 1>>) \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("BroadcastTo")                                             \
          .Device(DEVICE_DML)                                         \
          .TypeConstraint<type>("T")                                  \
          .TypeConstraint<int64>("Tidx")                              \
          .HostMemory("shape"),                                       \
      DmlKernelWrapper<DmlBroadcastToKernel,                          \
                       GetOutputShapeFromDimsTensorHelper<int64, 1>>)

// TODO(b/25387198): A special kernel exists for int32 (see broadcast_to_op.cc).
TF_CALL_DML_ALL_TYPES_EXCEPT_INT32(REGISTER_KERNEL);
#undef REGISTER_KERNEL

}  // namespace tensorflow