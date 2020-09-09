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

class PackInitHelper : public InitializationHelper {
 public:
  struct Attributes {
    explicit Attributes(OpKernelConstruction* ctx) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("axis", &axis));
    }

    int axis;
  };

  PackInitHelper(OpKernelContext* ctx, std::shared_ptr<const Attributes> attr)
      : attr_(attr) {
    OpInputList values;
    OP_REQUIRES_OK(ctx, ctx->input_list("values", &values));
    CHECK(values.size() > 0);
    input_shape_ = values[0].shape();

    int output_dims = input_shape_.dims() + 1;
    positive_axis_ = attr_->axis < 0 ? attr_->axis + output_dims : attr_->axis;

    OP_REQUIRES(ctx, 0 <= positive_axis_ && positive_axis_ < output_dims,
                errors::InvalidArgument("axis = ", attr_->axis, " not in [",
                                        -output_dims, ", ", output_dims, ")"));

    input_count_ = values.size();
    const TensorShape& first_input_shape = values[0].shape();

    // Verify that all input shapes match
    for (uint32_t i = 1; i < values.size(); i++) {
      const TensorShape& input_shape = values[i].shape();

      OP_REQUIRES(ctx, first_input_shape.IsSameSize(input_shape),
                  errors::InvalidArgument(
                      "Shapes of all inputs must match: values[0].shape = ",
                      first_input_shape.DebugString(), " != values[", i,
                      "].shape = ", input_shape.DebugString()));
    }
  }

  int GetAxis() const { return positive_axis_; }
  int GetInputCount() const { return input_count_; }
  const TensorShape& GetInputShape() const { return input_shape_; }

 private:
  const std::shared_ptr<const Attributes> attr_;
  int positive_axis_;
  int input_count_;
  TensorShape input_shape_;
};

using InitHelper = PackInitHelper;

class PackShapeHelper : public ShapeHelper {
 public:
  std::vector<TensorShape> GetOutputShapes(
      OpKernelContext* ctx,
      const InitializationHelper* initialization_helper) const override {
    auto init_helper = static_cast<const InitHelper*>(initialization_helper);
    int axis = init_helper->GetAxis();
    int input_count = init_helper->GetInputCount();
    const TensorShape& input_shape = init_helper->GetInputShape();

    TensorShape output_shape(input_shape);
    output_shape.InsertDim(axis, input_count);

    return {std::move(output_shape)};
  }
};

class DmlPackKernel : public DmlKernel {
 public:
  using InitHelper = tensorflow::InitHelper;

  explicit DmlPackKernel(DmlKernelConstruction* ctx,
                         const InitHelper* init_helper) {
    DCHECK(ctx->GetInputCount() > 0);
    DCHECK(ctx->GetOutputCount() == 1);

    TensorShape output_shape = ctx->GetOutputTensorShape(0);
    int axis = init_helper->GetAxis();

    // We can collapse all dimensions to the left together and all dimensions
    // to the right together. This allows us to send tensors with an "unlimited"
    // number of dimensions to DirectML
    int left_dim_size = 1;
    int right_dim_size = 1;

    for (int i = 0; i < axis; ++i) {
      left_dim_size *= output_shape.dim_size(i);
    }

    for (int i = axis + 1; i < output_shape.dims(); ++i) {
      right_dim_size *= output_shape.dim_size(i);
    }

    TensorShape input_shape({left_dim_size, 1, right_dim_size});

    DmlKernelTensors tensors;

    for (uint32_t i = 0; i < ctx->GetInputCount(); ++i) {
      DmlTensorInfo input_info;
      input_info.kernel_index = i;
      input_info.desc = DmlTensorDesc::Create(ctx->GetInputDataType(i),
                                              input_shape, input_shape);

      tensors.inputs.push_back(std::move(input_info));
    }

    int axis_size = output_shape.dim_size(axis);
    output_shape = TensorShape({left_dim_size, axis_size, right_dim_size});

    DmlTensorInfo output;
    output.kernel_index = 0;
    output.desc = DmlTensorDesc::Create(ctx->GetOutputDataType(0), output_shape,
                                        output_shape);
    tensors.outputs = {output};

    auto inputs = GetDmlTensorDescs(tensors.inputs);
    auto outputs = GetDmlTensorDescs(tensors.outputs);

    DML_JOIN_OPERATOR_DESC desc = {};
    desc.Axis = kNchwDimensionCount - 2;
    desc.InputCount = inputs.size();
    desc.InputTensors = inputs.data();
    desc.OutputTensor = outputs.data();

    DML_OPERATOR_DESC op_desc = {DML_OPERATOR_JOIN, &desc};
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

#define REGISTER_KERNEL(type)                                    \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("Pack").Device(DEVICE_DML).TypeConstraint<type>("T"), \
      DmlKernelWrapper<DmlPackKernel, PackShapeHelper>)

// TODO(b/25387198): A special kernel exists for int32 (see pack_op.cc).
TF_CALL_DML_ALL_TYPES_EXCEPT_INT32(REGISTER_KERNEL);
#undef REGISTER_KERNEL

}  // namespace tensorflow