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
class MatrixBandPartInitHelper : public InitializationHelper {
 public:
  using Attributes = EmptyAttributes;
  MatrixBandPartInitHelper(OpKernelContext* ctx,
                           std::shared_ptr<const Attributes> attr) {
    const Tensor& input = ctx->input(0);
    const TensorShape& input_shape = input.shape();
    const int64 height = input.dim_size(input.dims() - 2);
    const int64 width = input.dim_size(input.dims() - 1);

    // Preliminary validation of sizes.
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrixOrHigher(input_shape),
                errors::InvalidArgument(
                    "input must be at least 2-dim, received shape: ",
                    input.shape().DebugString()));

    const Tensor& num_lower_in = ctx->input(1);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(num_lower_in.shape()),
                errors::InvalidArgument("num_lower must be scalar, got shape ",
                                        num_lower_in.shape().DebugString()));

    auto as_int64_scalar = [](const Tensor& tensor) -> int64 {
      if (tensor.dtype() == DT_INT32) {
        return tensor.scalar<int32>()();
      } else {
        return tensor.scalar<int64>()();
      }
    };
    const int64 num_lower = as_int64_scalar(num_lower_in);
    OP_REQUIRES(
        ctx, num_lower <= height,
        errors::InvalidArgument(
            "num_lower must be negative or less or equal to number of rows (",
            height, ") got: ", num_lower));

    const Tensor& num_upper_in = ctx->input(2);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(num_upper_in.shape()),
                errors::InvalidArgument("num_upper must be scalar, got shape ",
                                        num_upper_in.shape().DebugString()));
    const int64 num_upper = as_int64_scalar(num_upper_in);
    OP_REQUIRES(ctx, num_upper <= width,
                errors::InvalidArgument("num_upper must be negative or less or "
                                        "equal to number of columns (",
                                        width, ") got: ", num_upper));

    // Although the validation allows num_lower == height and num_upper ==
    // width, they give the same result as num_lower == height - 1 and num_upper
    // == width - 1
    num_lower_ = num_lower < 0 ? height - 1 : std::min(num_lower, height - 1);
    num_upper_ = num_upper < 0 ? width - 1 : std::min(num_upper, width - 1);
  }

  int64 GetNumLower() const { return num_lower_; }
  int64 GetNumUpper() const { return num_upper_; }

 private:
  int64 num_lower_;
  int64 num_upper_;
};

class DmlMatrixBandPartKernel : public DmlKernel {
 public:
  using InitHelper = MatrixBandPartInitHelper;
  DmlMatrixBandPartKernel(DmlKernelConstruction* ctx,
                          const InitHelper* init_helper) {
    const TensorShape input_shape = ctx->GetInputTensorShape(0);
    const int64 height = input_shape.dim_size(input_shape.dims() - 2);
    const int64 width = input_shape.dim_size(input_shape.dims() - 1);
    const int64 batch_size = input_shape.num_elements() / height / width;
    const TensorShape flattened_shape({1, batch_size, height, width});

    DmlTensorInfo input_output_info;
    input_output_info.kernel_index = 0;
    input_output_info.desc = DmlTensorDesc::Create(
        ctx->GetInputDataType(0), flattened_shape, flattened_shape);

    DmlKernelTensors tensors;
    tensors.inputs = {input_output_info};
    tensors.outputs = {input_output_info};

    auto inputs = GetDmlTensorDescs(tensors.inputs);
    const int64 num_lower = init_helper->GetNumLower();
    const int64 num_upper = init_helper->GetNumUpper();
    const bool is_identity = num_lower == height - 1 && num_upper == width - 1;

    if (is_identity) {
      auto outputs = GetDmlTensorDescs(tensors.outputs);

      DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC identity_desc = {};
      identity_desc.InputTensor = &inputs[0];
      identity_desc.OutputTensor = &outputs[0];

      DML_OPERATOR_DESC op_desc = {DML_OPERATOR_ELEMENT_WISE_IDENTITY,
                                   &identity_desc};
      Initialize(ctx, std::move(tensors), op_desc);

      return;
    }

    auto scope = dml::Graph(ctx->GetDmlDevice());
    auto input = dml::InputTensor(scope, 0, inputs[0]);

    // Indices of each column broadcasted to the rows
    auto col_indices = dml::Sequence<int32_t>(
        scope, 0, 1, {1, 1, 1, static_cast<uint32_t>(width)});

    col_indices = dml::Reinterpret(col_indices, input.GetOutputDesc().sizes,
                                   dml::TensorDesc::Dimensions({0, 0, 0, 1}));

    // Indices of each row broadcasted to the columns
    auto row_indices = dml::Sequence<int32_t>(
        scope, 0, 1, {1, 1, static_cast<uint32_t>(height), 1});

    row_indices = dml::Reinterpret(row_indices, input.GetOutputDesc().sizes,
                                   dml::TensorDesc::Dimensions({0, 0, 1, 0}));

    auto width_tensor =
        dml::ScalarTensor<int32_t>(scope, width, input.GetOutputDesc().sizes);
    auto int_zero =
        dml::ScalarTensor<int32_t>(scope, 0, input.GetOutputDesc().sizes);

    auto data_type = GetDmlDataTypeFromTfDataType(ctx->GetInputDataType(0));
    auto zero = dml::Reinterpret(int_zero, data_type);

    // Start is inclusive and end is exclusive
    auto start = dml::Max(int_zero, row_indices - num_lower);
    auto end = dml::Min(width_tensor, (num_upper + 1) + row_indices);

    auto result =
        dml::If(col_indices >= start && col_indices < end, input, zero);

    Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
        scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

    Initialize(ctx, std::move(tensors), compiled_op.Get());
  }
};

#define REGISTER_DML_KERNEL(type)                                              \
  REGISTER_KERNEL_BUILDER(Name("MatrixBandPart")                               \
                              .Device(DEVICE_DML)                              \
                              .TypeConstraint<type>("T")                       \
                              .HostMemory("num_lower")                         \
                              .HostMemory("num_upper"),                        \
                          DmlKernelWrapper<DmlMatrixBandPartKernel,            \
                                           GetOutputShapeAsInputShapeHelper>); \
  REGISTER_KERNEL_BUILDER(Name("BatchMatrixBandPart")                          \
                              .Device(DEVICE_DML)                              \
                              .TypeConstraint<type>("T")                       \
                              .HostMemory("num_lower")                         \
                              .HostMemory("num_upper"),                        \
                          DmlKernelWrapper<DmlMatrixBandPartKernel,            \
                                           GetOutputShapeAsInputShapeHelper>);

TF_CALL_float(REGISTER_DML_KERNEL);
TF_CALL_half(REGISTER_DML_KERNEL);
TF_CALL_bool(REGISTER_DML_KERNEL);
#undef REGISTER_DML_KERNEL
}  // namespace tensorflow