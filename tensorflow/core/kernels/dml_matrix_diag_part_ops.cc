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

template <typename T>
class MatrixDiagPartInitHelper : public InitializationHelper {
 public:
  using Attributes = EmptyAttributes;

  MatrixDiagPartInitHelper(OpKernelContext* ctx,
                           std::shared_ptr<const Attributes> attr) {
    const Tensor& input = ctx->input(0);

    // MatrixDiagPart and MatrixDiagPartV2 both use this OpKernel.
    // MatrixDiagPart only has one input, so we have to check the number of
    // inputs before reading additional parameters in MatrixDiagV2.
    int32 lower_diag_index = 0;
    int32 upper_diag_index = 0;
    T padding_value(0);

    // MatrixDiagPartV2-specific.
    if (ctx->num_inputs() > 1) {
      auto& diag_index = ctx->input(1);
      OP_REQUIRES(ctx,
                  TensorShapeUtils::IsScalar(diag_index.shape()) ||
                      TensorShapeUtils::IsVector(diag_index.shape()),
                  errors::InvalidArgument(
                      "diag_index must be a scalar or vector, received shape: ",
                      diag_index.shape().DebugString()));
      lower_diag_index = diag_index.flat<int32>()(0);
      upper_diag_index = lower_diag_index;
      if (TensorShapeUtils::IsVector(diag_index.shape())) {
        auto diag_index_size = diag_index.dim_size(0);
        OP_REQUIRES(
            ctx, 0 < diag_index_size && diag_index_size <= 2,
            errors::InvalidArgument(
                "diag_index must have only one or two elements, received ",
                diag_index_size, " elements."));
        if (diag_index_size > 1) {
          upper_diag_index = diag_index.flat<int32>()(1);
        }
      }
      padding_value = ctx->input(2).flat<T>()(0);
    }
    const TensorShape& input_shape = input.shape();

    // Preliminary validation of sizes.
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrixOrHigher(input_shape),
                errors::InvalidArgument(
                    "input must be at least 2-dim, received shape: ",
                    input.shape().DebugString()));

    // Make sure lower_diag_index and upper_diag_index is valid.
    const int rank = input_shape.dims();
    const Eigen::Index num_rows = input_shape.dim_size(rank - 2);
    const Eigen::Index num_cols = input_shape.dim_size(rank - 1);
    OP_REQUIRES(  // Checks lower_diag_index == 0 for when matrix shape = 0.
        ctx,
        (-num_rows < lower_diag_index && lower_diag_index < num_cols) ||
            lower_diag_index == 0,
        errors::InvalidArgument(
            "lower_diag_index is out of bound: ", lower_diag_index,
            ". It must be between ", -num_rows, " and ", num_cols));
    OP_REQUIRES(ctx,
                (-num_rows < upper_diag_index && upper_diag_index < num_cols) ||
                    upper_diag_index == 0,
                errors::InvalidArgument(
                    "upper_diag_index is out of bound: ", upper_diag_index,
                    " It must be between ", -num_rows, " and ", num_cols));
    OP_REQUIRES(
        ctx, lower_diag_index <= upper_diag_index,
        errors::InvalidArgument(
            "lower_diag_index must not be larger than upper_diag_index: ",
            lower_diag_index, " > ", upper_diag_index));

    for (int i = 0; i < rank - 2; ++i) {
      output_shape_.AddDim(input_shape.dim_size(i));
    }
    const Eigen::Index num_diags = upper_diag_index - lower_diag_index + 1;
    if (num_diags > 1) output_shape_.AddDim(num_diags);
    const int32 max_diag_len =
        std::min(num_rows + std::min(upper_diag_index, 0),
                 num_cols - std::max(lower_diag_index, 0));
    output_shape_.AddDim(max_diag_len);

    padding_value_ = padding_value;
    lower_diag_index_ = lower_diag_index;
    upper_diag_index_ = upper_diag_index;
  }

  TensorShape GetOutputShape() const { return output_shape_; }
  int32 GetLowerDiagIndex() const { return lower_diag_index_; }
  int32 GetUpperDiagIndex() const { return upper_diag_index_; }
  T GetPaddingValue() const { return padding_value_; }

 private:
  TensorShape output_shape_;
  T padding_value_;
  int32 lower_diag_index_;
  int32 upper_diag_index_;
};

template <typename T>
class MatrixDiagPartShapeHelper : public ShapeHelper {
 public:
  std::vector<TensorShape> GetOutputShapes(
      OpKernelContext* ctx,
      const InitializationHelper* initialization_helper) const override {
    auto init_helper =
        static_cast<const MatrixDiagPartInitHelper<T>*>(initialization_helper);

    return {init_helper->GetOutputShape()};
  }
};

template <typename T>
class DmlMatrixDiagPartKernel : public DmlKernel {
 public:
  using InitHelper = MatrixDiagPartInitHelper<T>;

  DmlMatrixDiagPartKernel(DmlKernelConstruction* ctx,
                          const InitHelper* init_helper) {
    const TensorShape& in_shape = ctx->GetInputTensorShape(0);
    int32 k_min = init_helper->GetLowerDiagIndex();
    int32 k_max = init_helper->GetUpperDiagIndex();

    // Fast path for MatrixDiag and MatrixDiagV2 when k=0, num_rows=num_cols
    const bool is_square_matrix = in_shape.dim_size(in_shape.dims() - 2) ==
                                  in_shape.dim_size(in_shape.dims() - 1);

    const bool use_fast_path = is_square_matrix && k_min == 0 && k_max == 0;

    if (use_fast_path) {
      ExtractDiagPartFromSimpleMatrix(ctx, init_helper);
    } else {
      ExtractDiagPartFromComplexMatrix(ctx, init_helper);
    }
  }

 private:
  void ExtractDiagPartFromSimpleMatrix(DmlKernelConstruction* ctx,
                                       const InitHelper* init_helper) {
    const TensorShape& input_shape = ctx->GetInputTensorShape(0);
    const TensorShape& output_shape = ctx->GetOutputTensorShape(0);

    uint32_t batch_size = 1u;
    for (int i = 0; i < input_shape.dims() - 2; ++i) {
      batch_size *= input_shape.dim_size(i);
    }

    auto elem_count_per_batch =
        static_cast<uint32_t>(output_shape.num_elements() / batch_size);
    auto input_height =
        static_cast<uint32_t>(input_shape.dim_size(input_shape.dims() - 2));
    auto input_width =
        static_cast<uint32_t>(input_shape.dim_size(input_shape.dims() - 1));

    // Flatten the output batches of vectors
    TensorShape flattened_output_shape(
        {batch_size, 1, 1, elem_count_per_batch});

    auto dtype_tf = ctx->GetInputDataType(0);
    auto dtype_dml = GetDmlDataTypeFromTfDataType(dtype_tf);

    // Flatten the input into a vector and use strides to skip over zeros
    uint32_t input_sizes[] = {batch_size, 1, 1, elem_count_per_batch};
    uint32_t input_strides[] = {
        input_height * input_width,
        0,
        0,
        (elem_count_per_batch + 1),
    };

    DmlTensorInfo input;
    input.kernel_index = 0;
    input.desc = DmlTensorDesc(dtype_dml, input_sizes, input_strides);

    DmlTensorInfo output;
    output.kernel_index = 0;
    output.desc =
        DmlTensorDesc::Create(ctx->GetOutputDataType(0), flattened_output_shape,
                              flattened_output_shape);

    DmlKernelTensors tensors;
    tensors.inputs = {input};
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

  void ExtractDiagPartFromComplexMatrix(DmlKernelConstruction* ctx,
                                        const InitHelper* init_helper) {
    const TensorShape& input_shape = ctx->GetInputTensorShape(0);
    const TensorShape& output_shape = ctx->GetOutputTensorShape(0);
    uint32_t xlen = input_shape.dim_size(input_shape.dims() - 1);
    uint32_t ylen = input_shape.dim_size(input_shape.dims() - 2);
    uint32_t leading_dims_size = input_shape.num_elements() / xlen / ylen;
    dml::TensorDesc::Dimensions m_shape({1, leading_dims_size, ylen, xlen});

    int32 k0 = init_helper->GetLowerDiagIndex();
    int32 k1 = init_helper->GetUpperDiagIndex();

    uint32_t out_cols = output_shape.dim_size(output_shape.dims() - 1);
    uint32_t out_rows =
        k0 == k1 ? 1 : output_shape.dim_size(output_shape.dims() - 2);
    uint32_t out_leading_dim_size =
        output_shape.num_elements() / out_cols / out_rows;

    dml::TensorDesc::Dimensions flattened_out_shape(
        {1, out_leading_dim_size, out_rows, out_cols});

    int32 xlenp = xlen + 1;
    int32 stride = xlenp + 1;
    int32 xmax = xlen * xlenp + xlenp - 1;
    int32 ymax = xlenp * ylen - 1;

    DmlTensorInfo input;
    input.kernel_index = 0;
    input.desc =
        DmlTensorDesc::Create(ctx->GetInputDataType(0), m_shape, m_shape);

    DmlTensorInfo output;
    output.kernel_index = 0;
    output.desc = DmlTensorDesc::Create(
        ctx->GetOutputDataType(0), flattened_out_shape, flattened_out_shape);

    DmlKernelTensors tensors;
    tensors.inputs = {input};
    tensors.outputs = {output};

    auto inputs = GetDmlTensorDescs(tensors.inputs);
    auto scope = dml::Graph(ctx->GetDmlDevice());
    auto m = dml::InputTensor(scope, 0, inputs[0]);

    float padding_value = static_cast<float>(init_helper->GetPaddingValue());
    auto m_padded = dml::Padding(m, DML_PADDING_MODE_CONSTANT, padding_value,
                                 {0, 0, 0, 0}, {0, 0, 1, 1});

    auto m2 = dml::Reinterpret(
        m_padded, {1, 1, leading_dims_size, (ylen + 1) * (xlen + 1)}, {});

    uint32_t minxy = std::min(xlen, ylen);

    auto diag_distances =
        dml::Sequence<int32>(scope, 0, stride, {1, 1, 1, minxy});

    dml::Expression diags_indices;

    // Starting indices for super diagonals
    int32 xstart_end = std::max(0, k0) - 1;
    int32 xdiag_size = k1 - xstart_end;
    dml::Expression xdiags;

    if (xdiag_size > 0) {
      dml::TensorDesc::Dimensions broadcast_sizes(
          {1, 1, static_cast<uint32_t>(xdiag_size), minxy});
      dml::TensorDesc::Dimensions xstart_sizes(
          {1, 1, static_cast<uint32_t>(xdiag_size), 1});

      auto xstart = dml::Sequence<int32>(scope, k1, -1, xstart_sizes);
      xstart = dml::Reinterpret(xstart, broadcast_sizes,
                                dml::TensorDesc::Dimensions({0, 0, 1, 0}));

      auto xmax_sequence =
          dml::Sequence<int32>(scope, xmax - k1 * xlenp, xlenp, xstart_sizes);
      xmax_sequence =
          dml::Reinterpret(xmax_sequence, broadcast_sizes,
                           dml::TensorDesc::Dimensions({0, 0, 1, 0}));

      auto broadcasted_diag_distances =
          dml::Reinterpret(diag_distances, broadcast_sizes,
                           dml::TensorDesc::Dimensions({0, 0, 0, 1}));

      xdiags = dml::Min(xstart + broadcasted_diag_distances, xmax_sequence);
      diags_indices = xdiags;
    }

    // Starting indices for sub diagonals
    int32 ystart_begin = -std::min(-1, k1);
    int32 ydiag_size = 1 - k0 - ystart_begin;
    dml::Expression ydiags;

    if (ydiag_size > 0) {
      dml::TensorDesc::Dimensions broadcast_sizes(
          {1, 1, static_cast<uint32_t>(ydiag_size), minxy});
      dml::TensorDesc::Dimensions ystart_sizes(
          {1, 1, static_cast<uint32_t>(ydiag_size), 1});

      auto ystart = dml::Sequence<int32>(scope, ystart_begin * xlenp, xlenp,
                                         ystart_sizes);
      ystart = dml::Reinterpret(ystart, broadcast_sizes,
                                dml::TensorDesc::Dimensions({0, 0, 1, 0}));

      auto ymax_scalar = dml::ScalarTensor<int32>(scope, ymax, ystart_sizes);
      ymax_scalar = dml::Reinterpret(ymax_scalar, broadcast_sizes,
                                     dml::TensorDesc::Dimensions({0, 0, 1, 0}));

      auto broadcasted_diag_distances =
          dml::Reinterpret(diag_distances, broadcast_sizes,
                           dml::TensorDesc::Dimensions({0, 0, 0, 1}));

      ydiags = dml::Min(ystart + broadcasted_diag_distances, ymax_scalar);
      diags_indices = ydiags;
    }

    if (xdiag_size > 0 && ydiag_size > 0) {
      diags_indices = dml::Join({xdiags, ydiags}, 2);
    }

    // Reshape into a single row
    diags_indices =
        dml::Reinterpret(diags_indices, {1, 1, 1, out_rows * out_cols}, {});

    // Broadcast to all batches
    diags_indices = dml::Reinterpret(
        diags_indices, {1, 1, leading_dims_size, out_rows * out_cols},
        dml::TensorDesc::Dimensions({0, 0, 0, 1}));

    auto diags = dml::GatherElements(m2, diags_indices, 3);

    Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
        scope.Compile(DML_EXECUTION_FLAG_NONE, {diags});

    Initialize(ctx, std::move(tensors), compiled_op.Get());
  }
};  // namespace tensorflow

#define REGISTER_DML_KERNEL(T)                                               \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("MatrixDiagPart").Device(DEVICE_DML).TypeConstraint<T>("T"),      \
      DmlKernelWrapper<DmlMatrixDiagPartKernel<T>,                           \
                       MatrixDiagPartShapeHelper<T>>);                       \
  REGISTER_KERNEL_BUILDER(Name("MatrixDiagPartV2")                           \
                              .Device(DEVICE_DML)                            \
                              .TypeConstraint<T>("T")                        \
                              .HostMemory("k")                               \
                              .HostMemory("padding_value"),                  \
                          DmlKernelWrapper<DmlMatrixDiagPartKernel<T>,       \
                                           MatrixDiagPartShapeHelper<T>>);   \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("BatchMatrixDiagPart").Device(DEVICE_DML).TypeConstraint<T>("T"), \
      DmlKernelWrapper<DmlMatrixDiagPartKernel<T>,                           \
                       MatrixDiagPartShapeHelper<T>>);

TF_CALL_half(REGISTER_DML_KERNEL);
TF_CALL_float(REGISTER_DML_KERNEL);
TF_CALL_bool(REGISTER_DML_KERNEL)
}  // namespace tensorflow