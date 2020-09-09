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
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/dml_kernel_wrapper.h"
#include "tensorflow/core/kernels/dml_ops_common.h"

namespace tensorflow {

template <typename Tidx>
class CumsumInitHelper : public InitializationHelper {
 public:
  struct Attributes {
    explicit Attributes(OpKernelConstruction* ctx) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("reverse", &reverse));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("exclusive", &exclusive));
    }

    bool reverse;
    bool exclusive;
  };

  CumsumInitHelper(OpKernelContext* ctx, std::shared_ptr<const Attributes> attr)
      : attr_(attr) {
    const Tensor& input = ctx->input(0);
    const Tensor& tensor_axis = ctx->input(1);

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(tensor_axis.shape()),
                errors::InvalidArgument("ScanOp: axis must be a scalar, not ",
                                        tensor_axis.shape().DebugString()));

    const Tidx axis_arg =
        internal::SubtleMustCopy(tensor_axis.scalar<Tidx>()());
    axis_ = (axis_arg < 0) ? input.dims() + axis_arg : axis_arg;
    OP_REQUIRES(ctx, FastBoundsCheck(axis_, input.dims()),
                errors::InvalidArgument(
                    "ScanOp: Expected scan axis in the range [", -input.dims(),
                    ", ", input.dims(), "), but got ", axis_));
  }

  bool IsReverse() const { return attr_->reverse; }
  bool IsExclusive() const { return attr_->exclusive; }
  int64 GetAxis() const { return axis_; }

 private:
  const std::shared_ptr<const Attributes> attr_;
  int64 axis_;
};

template <typename Tidx>
class DmlCumsumKernel : public DmlKernel {
 public:
  using InitHelper = CumsumInitHelper<Tidx>;

  explicit DmlCumsumKernel(DmlKernelConstruction* ctx,
                           const InitHelper* init_helper) {
    DCHECK(ctx->GetInputCount() == 2);
    DCHECK(ctx->GetOutputCount() == 1);

    const TensorShape& original_input_shape = ctx->GetInputTensorShape(0);

    int64 axis = init_helper->GetAxis();
    DML_AXIS_DIRECTION axis_direction = init_helper->IsReverse()
                                            ? DML_AXIS_DIRECTION_DECREASING
                                            : DML_AXIS_DIRECTION_INCREASING;

    // Collapse the dimensions to the left and to the right of the axis together
    int left_dim_size = 1;
    for (int i = 0; i < axis; ++i) {
      left_dim_size *= original_input_shape.dim_size(i);
    }

    int right_dim_size = 1;
    for (int i = axis + 1; i < original_input_shape.dims(); ++i) {
      right_dim_size *= original_input_shape.dim_size(i);
    }

    int axis_dim_size = original_input_shape.dim_size(axis);

    TensorShape tensor_shape({1, left_dim_size, axis_dim_size, right_dim_size});

    DmlTensorInfo input;
    input.kernel_index = 0;
    input.desc = DmlTensorDesc::Create(ctx->GetInputDataType(0), tensor_shape,
                                       tensor_shape);

    DmlTensorInfo output;
    output.kernel_index = 0;
    output.desc = DmlTensorDesc::Create(ctx->GetOutputDataType(0), tensor_shape,
                                        tensor_shape);

    DmlKernelTensors tensors;
    tensors.inputs = {input};
    tensors.outputs = {output};

    auto inputs = GetDmlTensorDescs(tensors.inputs);
    auto outputs = GetDmlTensorDescs(tensors.outputs);

    // The non-axis dimensions have already been collapsed together, so the dml
    // axis is always "2"
    constexpr uint32_t dml_axis = 2;

    DML_CUMULATIVE_SUMMATION_OPERATOR_DESC cumsum_desc = {};
    cumsum_desc.InputTensor = &inputs[0];
    cumsum_desc.OutputTensor = &outputs[0];
    cumsum_desc.Axis = dml_axis;
    cumsum_desc.AxisDirection = axis_direction;
    cumsum_desc.HasExclusiveSum = init_helper->IsExclusive();

    DML_OPERATOR_DESC op_desc = {DML_OPERATOR_CUMULATIVE_SUMMATION,
                                 &cumsum_desc};
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

#define REGISTER_KERNELS(type)                                                \
  REGISTER_KERNEL_BUILDER(Name("Cumsum")                                      \
                              .Device(DEVICE_DML)                             \
                              .TypeConstraint<type>("T")                      \
                              .TypeConstraint<int32>("Tidx")                  \
                              .HostMemory("axis"),                            \
                          DmlKernelWrapper<DmlCumsumKernel<int32>,            \
                                           GetOutputShapeAsInputShapeHelper>) \
  REGISTER_KERNEL_BUILDER(Name("Cumsum")                                      \
                              .Device(DEVICE_DML)                             \
                              .TypeConstraint<type>("T")                      \
                              .TypeConstraint<int64>("Tidx")                  \
                              .HostMemory("axis"),                            \
                          DmlKernelWrapper<DmlCumsumKernel<int64>,            \
                                           GetOutputShapeAsInputShapeHelper>)
TF_CALL_float(REGISTER_KERNELS);
TF_CALL_int64(REGISTER_KERNELS);
TF_CALL_uint64(REGISTER_KERNELS);
TF_CALL_int32(REGISTER_KERNELS);
TF_CALL_uint32(REGISTER_KERNELS);
TF_CALL_half(REGISTER_KERNELS);
#undef REGISTER_KERNELS

}  // namespace tensorflow