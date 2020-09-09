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

enum AxisArgumentName { NAME_IS_AXIS, NAME_IS_CONCAT_DIM };

template <AxisArgumentName AxisArgName>
class ConcatInitHelper : public InitializationHelper {
 public:
  using Attributes = EmptyAttributes;

  ConcatInitHelper(OpKernelContext* ctx,
                   std::shared_ptr<const Attributes> attr) {
    const Tensor* concat_dim_tensor;
    const char* axis_attribute_name =
        AxisArgName == NAME_IS_AXIS
            ? "axis"
            : AxisArgName == NAME_IS_CONCAT_DIM ? "concat_dim" : "<invalid>";
    OP_REQUIRES_OK(ctx, ctx->input(axis_attribute_name, &concat_dim_tensor));

    OpInputList values;
    OP_REQUIRES_OK(ctx, ctx->input_list("values", &values));
    const int input_dims = values[0].dims();
    first_input_shape_ = values[0].shape();

    CHECK(concat_dim_tensor->shape().dims() == 0);
    int64 concat_dim;
    // In case of ConcatV2, "axis" could be int32 or int64
    if (AxisArgName == NAME_IS_AXIS) {
      CHECK(concat_dim_tensor->dtype() == DT_INT32 ||
            concat_dim_tensor->dtype() == DT_INT64);
    } else {
      CHECK(concat_dim_tensor->dtype() == DT_INT32);
    }
    if (concat_dim_tensor->dtype() == DT_INT32) {
      concat_dim =
          internal::SubtleMustCopy(concat_dim_tensor->scalar<int32>()());
    } else {
      concat_dim =
          internal::SubtleMustCopy(concat_dim_tensor->scalar<int64>()());
    }

    concat_axis_ = concat_dim < 0 ? concat_dim + input_dims : concat_dim;
    OP_REQUIRES(ctx, 0 <= concat_axis_ && concat_axis_ < input_dims,
                errors::InvalidArgument(
                    "ConcatOp : Expected concatenating dimensions in the range "
                    "[",
                    -input_dims, ", ", input_dims, "), but got ", concat_dim));

    output_concat_dim_size_ = first_input_shape_.dim_size(concat_axis_);
    for (int i = 1; i < values.size(); ++i) {
      const auto& in = values[i];
      OP_REQUIRES(
          ctx, in.dims() == input_dims,
          errors::InvalidArgument(
              "ConcatOp : Ranks of all input tensors should match: shape[0] = ",
              first_input_shape_.DebugString(), " vs. shape[", i,
              "] = ", in.shape().DebugString()));
      output_concat_dim_size_ += in.dim_size(concat_axis_);

      for (int j = 0; j < in.dims(); ++j) {
        if (j != concat_axis_) {
          OP_REQUIRES(
              ctx, in.dim_size(j) == first_input_shape_.dim_size(j),
              errors::InvalidArgument(
                  "ConcatOp : Dimensions of inputs should match: shape[0] = ",
                  first_input_shape_.DebugString(), " vs. shape[", i,
                  "] = ", in.shape().DebugString()));
        }
      }
    }
  }

  // Concat is only a no-op if all input tensors (aside from the concat dim
  // tensor) are empty
  bool IsNoOpKernel(
      OpKernelContext* ctx,
      absl::Span<const TensorShape> output_shapes) const override {
    uint32_t concat_dim_tensor_index =
        AxisArgName == NAME_IS_CONCAT_DIM ? 0 : ctx->num_inputs() - 1;

    for (int i = 0; i < ctx->num_inputs(); ++i) {
      if (i == concat_dim_tensor_index) {
        continue;
      }

      if (ctx->input(i).NumElements() != 0) {
        return false;
      }
    }

    return true;
  }

  int64 GetConcatAxis() const { return concat_axis_; }
  int64 GetOutputConcatDimSize() const { return output_concat_dim_size_; }
  const TensorShape& GetFirstInputShape() const { return first_input_shape_; }

 private:
  int64 concat_axis_;
  int64 output_concat_dim_size_;
  TensorShape first_input_shape_;
};

template <AxisArgumentName AxisArgName>
using InitHelper = ConcatInitHelper<AxisArgName>;

template <AxisArgumentName AxisArgName>
class ConcatShapeHelper : public ShapeHelper {
 public:
  std::vector<TensorShape> GetOutputShapes(
      OpKernelContext* ctx,
      const InitializationHelper* initialization_helper) const override {
    auto init_helper =
        static_cast<const InitHelper<AxisArgName>*>(initialization_helper);

    int64 concat_axis = init_helper->GetConcatAxis();
    int64 output_concat_dim = init_helper->GetOutputConcatDimSize();
    const TensorShape& first_input_shape = init_helper->GetFirstInputShape();

    TensorShape output_shape(first_input_shape);
    output_shape.set_dim(concat_axis, output_concat_dim);

    return {std::move(output_shape)};
  }
};

template <AxisArgumentName AxisArgName>
class DmlConcatKernel : public DmlKernel {
 public:
  using InitHelper = InitHelper<AxisArgName>;

  explicit DmlConcatKernel(DmlKernelConstruction* ctx,
                           const InitHelper* init_helper) {
    // We need to concatenate at least 2 tensors, and we need an additional
    // tensor for the concatenation axis
    CHECK(ctx->GetInputCount() >= 3);
    CHECK(ctx->GetOutputCount() == 1);

    uint32_t first_concat_tensor_index;
    uint32_t concat_dim_tensor_index;
    if constexpr (AxisArgName == NAME_IS_CONCAT_DIM) {
      // For Concat, the inputs come AFTER the axis
      first_concat_tensor_index = 1;
      concat_dim_tensor_index = 0;
    } else {
      // For ConcatV2, the inputs come BEFORE the axis
      first_concat_tensor_index = 0;
      concat_dim_tensor_index = ctx->GetInputCount() - 1;
    }

    DmlKernelTensors tensors;
    const TensorShape& first_input_shape = ctx->GetInputTensorShape(0);

    int64 concat_axis = init_helper->GetConcatAxis();

    // We can collapse all dimensions to the left together and all dimensions
    // to the right together. This allows us to send tensors with an "unlimited"
    // number of dimensions to DirectML
    int left_dim_size = 1;
    int right_dim_size = 1;
    TensorShape output_shape = ctx->GetOutputTensorShape(0);

    for (int i = 0; i < concat_axis; ++i) {
      left_dim_size *= output_shape.dim_size(i);
    }

    for (int i = concat_axis + 1; i < output_shape.dims(); ++i) {
      right_dim_size *= output_shape.dim_size(i);
    }

    int axis_size = output_shape.dim_size(concat_axis);
    output_shape = TensorShape({left_dim_size, axis_size, right_dim_size});

    DmlTensorInfo output;
    output.kernel_index = 0;
    output.desc = DmlTensorDesc::Create(ctx->GetOutputDataType(0), output_shape,
                                        output_shape);
    tensors.outputs = {output};

    // DML doesn't support empty tensors, so filter them out when generating the
    // kernel input indices (which is what determines the mapping between kernel
    // inputs and DML op inputs)
    for (uint32_t i = 0; i < ctx->GetInputCount(); ++i) {
      if (i == concat_dim_tensor_index) {
        continue;  // Ignore the concat axis tensor
      }

      if (ctx->GetInputTensorShape(i).num_elements() == 0) {
        // Empty tensor; ignore this input
        continue;
      }

      int axis_dim_size = ctx->GetInputTensorShape(i).dim_size(concat_axis);
      TensorShape tensor_shape({left_dim_size, axis_dim_size, right_dim_size});

      DmlTensorInfo input_info;
      input_info.kernel_index = i;
      input_info.desc = DmlTensorDesc::Create(ctx->GetInputDataType(i),
                                              tensor_shape, tensor_shape);
      tensors.inputs.push_back(std::move(input_info));
    }

    // If all tensors are empty, this kernel should have already been no-op'd
    // earlier
    CHECK(!tensors.inputs.empty());

    auto inputs = GetDmlTensorDescs(tensors.inputs);
    auto outputs = GetDmlTensorDescs(tensors.outputs);

    DML_JOIN_OPERATOR_DESC op_specific_desc = {};
    op_specific_desc.InputCount = inputs.size();
    op_specific_desc.InputTensors = inputs.data();
    op_specific_desc.OutputTensor = outputs.data();
    op_specific_desc.Axis = kNchwDimensionCount - 2;

    DML_OPERATOR_DESC op_desc = {DML_OPERATOR_JOIN, &op_specific_desc};
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

template <AxisArgumentName AxisArgName>
using DmlConcatWrapper = DmlKernelWrapper<DmlConcatKernel<AxisArgName>,
                                          ConcatShapeHelper<AxisArgName>>;

#define REGISTER_KERNEL(type)                                   \
  REGISTER_KERNEL_BUILDER(Name("Concat")                        \
                              .Device(DEVICE_DML)               \
                              .TypeConstraint<type>("T")        \
                              .HostMemory("concat_dim"),        \
                          DmlConcatWrapper<NAME_IS_CONCAT_DIM>) \
  REGISTER_KERNEL_BUILDER(Name("ConcatV2")                      \
                              .Device(DEVICE_DML)               \
                              .TypeConstraint<type>("T")        \
                              .HostMemory("axis"),              \
                          DmlConcatWrapper<NAME_IS_AXIS>)

// TODO(b/25387198): A special kernel exists for int32 (see concat_op.cc).
TF_CALL_DML_ALL_TYPES_EXCEPT_INT32(REGISTER_KERNEL);
#undef REGISTER_KERNEL

}  // namespace tensorflow