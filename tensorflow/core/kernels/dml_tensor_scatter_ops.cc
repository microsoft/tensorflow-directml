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

// Returns true if the three tensors have valid number of elements
// If shape_input has 0 elements, then we need to have indices and updates with
// exactly 0 elements too, otherwise we should error. If indices has 0 elements
// then updates should also have 0 elements, otherwise we should error.
static bool ValidEmptyOutputShape(int64 num_inputs, int64 num_indices,
                                  int64 num_updates) {
  if (num_indices == 0 && num_updates == 0) {
    return true;  // regardless of num_inputs ?= 0, covers both cases
  }
  // now we want all 3 tensors to have values
  return (num_inputs != 0 && num_indices != 0 && num_updates != 0);
}

static Status ValidateUpdateShape(const TensorShape& params_shape,
                                  const Tensor& indices,
                                  const Tensor& updates) {
  const int64 slice_dim =
      (indices.dims() > 1) ? indices.dim_size(indices.dims() - 1) : 1;
  const int64 batch_dim = (indices.dims() > 1) ? indices.dims() - 1 : 1;

  auto shape_err = [&]() {
    return errors::InvalidArgument(
        "Must have updates.shape = indices.shape[:batch_dim] + ",
        "params_shape[slice_dim:], got updates.shape: ",
        updates.shape().DebugString(),
        ", indices.shape: ", indices.shape().DebugString(),
        ", params_shape: ", params_shape.DebugString(),
        ", slice_dim: ", slice_dim, ", and batch_dim: ", batch_dim);
  };

  if (updates.dims() < batch_dim) return shape_err();
  if (params_shape.dims() < slice_dim + (updates.dims() - batch_dim)) {
    return shape_err();
  }
  if (updates.dims() != batch_dim + params_shape.dims() - slice_dim) {
    return shape_err();
  }
  for (int d = 0; d < batch_dim; ++d) {
    if (updates.dim_size(d) != indices.dim_size(d)) return shape_err();
  }
  for (int d = 0; d < updates.dims() - batch_dim; ++d) {
    if (updates.dim_size(d + batch_dim) !=
        params_shape.dim_size(d + slice_dim)) {
      return shape_err();
    }
  }
  return Status::OK();
}

template <typename Index>
static Status ValidateInputs(const TensorShape& params_shape,
                             const Tensor& indices, const Tensor& updates) {
  const TensorShape& indices_shape(indices.shape());
  const TensorShape& updates_shape(updates.shape());

  if (!TensorShapeUtils::IsVectorOrHigher(params_shape)) {
    return errors::InvalidArgument("Output must be at least 1-D, ",
                                   "got shape: ", params_shape.DebugString());
  }

  if (!ValidEmptyOutputShape(params_shape.num_elements(),
                             indices_shape.num_elements(),
                             updates_shape.num_elements())) {
    return errors::InvalidArgument(
        "Indices and updates specified for empty output.  indices shape: ",
        indices.shape().DebugString());
  }

  if (updates.dim_size(0) != indices.dim_size(0)) {
    return errors::InvalidArgument(
        "The outermost dimension of updates and indices ",
        "must match. Got indices.shape ", indices_shape.DebugString(),
        ", updates.shape ", updates_shape.DebugString());
  }
  TF_RETURN_IF_ERROR(ValidateUpdateShape(params_shape, indices, updates));

  // Check that we have enough index space
  const int64 N_big = indices.NumElements();
  if (N_big > std::numeric_limits<Index>::max()) {
    return errors::InvalidArgument("indices has too many elements for ",
                                   DataTypeString(DataTypeToEnum<Index>::v()),
                                   " indexing: ", N_big, " > ",
                                   std::numeric_limits<Index>::max());
  }
  if (params_shape.dim_size(0) > std::numeric_limits<Index>::max()) {
    return errors::InvalidArgument("params_shape[0] too large for ",
                                   DataTypeString(DataTypeToEnum<Index>::v()),
                                   " indexing: ", params_shape.dim_size(0),
                                   " > ", std::numeric_limits<Index>::max());
  }

  // Calculate the number of dimensions in indices
  int64 slice_dim = (indices_shape.dims() > 1)
                        ? indices_shape.dim_size(indices_shape.dims() - 1)
                        : 1;

  // Calculate the number of elements that make up each slice of our updated
  // tensor. This allows us to work with flattened tensors and copy over whole
  // slices at a time.
  Index total_nd = params_shape.dims();

  int64 slice_size = 1;
  for (int64 i = slice_dim; i < total_nd; ++i) {
    slice_size *= params_shape.dim_size(i);
  }

  if (slice_size > std::numeric_limits<Index>::max()) {
    return errors::InvalidArgument(
        "slice size is too large for indexing: ", slice_size, " > ",
        std::numeric_limits<Index>::max());
  }

  if (slice_dim > 7) {
    return errors::InvalidArgument(
        "Only indices.shape[-1] values between 0 and 7 "
        "are currently supported.  Requested rank: ",
        slice_dim);
  }

  return Status::OK();
}

template <typename Index>
class TensorScatterInitHelper : public InitializationHelper {
 public:
  using Attributes = EmptyAttributes;

  explicit TensorScatterInitHelper(OpKernelContext* ctx,
                                   std::shared_ptr<const Attributes> attr) {
    const Tensor& input = ctx->input(0);
    const Tensor& indices = ctx->input(1);
    const Tensor& updates = ctx->input(2);

    OP_REQUIRES(ctx, indices.shape().dims() >= 1,
                errors::InvalidArgument(
                    "Indices shape must have rank at least one. Found:",
                    indices.shape().DebugString()));
    OP_REQUIRES(ctx, updates.shape().dims() >= 1,
                errors::InvalidArgument(
                    "Updates shape must have rank at least one. Found:",
                    updates.shape().DebugString()));

    TensorShape shape = input.shape();

    OP_REQUIRES(ctx,
                ValidEmptyOutputShape(shape.num_elements(),
                                      indices.shape().num_elements(),
                                      updates.shape().num_elements()),
                errors::InvalidArgument(
                    "Indices and updates specified for empty output shape"));

    const int64 outer_dims = indices.shape().dims() - 1;

    for (int i = 0; i < outer_dims; ++i) {
      OP_REQUIRES(ctx,
                  indices.shape().dim_size(i) == updates.shape().dim_size(i),
                  errors::InvalidArgument(
                      "Outer dimensions of indices and update must match. "
                      "Indices shape: ",
                      indices.shape().DebugString(),
                      ", updates shape:", updates.shape().DebugString()));
    }

    const int64 ix = indices.shape().dim_size(outer_dims);
    OP_REQUIRES(
        ctx, updates.shape().dims() - outer_dims == shape.dims() - ix,
        errors::InvalidArgument("Inner dimensions of output shape must match "
                                "inner dimensions of updates shape. Output: ",
                                shape.DebugString(),
                                " updates: ", updates.shape().DebugString()));
    for (int i = 0; i + outer_dims < updates.shape().dims(); ++i) {
      OP_REQUIRES(
          ctx,
          updates.shape().dim_size(i + outer_dims) == shape.dim_size(ix + i),
          errors::InvalidArgument(
              "The inner ", shape.dims() - ix,
              " dimensions of output.shape=", shape.DebugString(),
              " must match the inner ", updates.shape().dims() - outer_dims,
              " dimensions of updates.shape=", updates.shape().DebugString()));
    }

    OP_REQUIRES_OK(ctx, ValidateInputs<Index>(shape, indices, updates));
  }
};

template <typename Index>
class DmlTensorScatterUpdateKernel : public DmlKernel {
 public:
  using InitHelper = TensorScatterInitHelper<Index>;

  DmlTensorScatterUpdateKernel(DmlKernelConstruction* ctx,
                               const InitHelper* init_helper) {
    const TensorShape& in_out_shape = ctx->GetInputTensorShape(0);
    const TensorShape& indices_shape = ctx->GetInputTensorShape(1);

    DmlKernelTensors tensors = GetTensorInfos(ctx, DmlKernelParams{});
    auto input_descs = GetDmlTensorDescs(tensors.inputs);
    auto output_descs = GetDmlTensorDescs(tensors.outputs);

    DML_SCATTER_ND_OPERATOR_DESC scatter_nd_desc = {};
    scatter_nd_desc.InputTensor = &input_descs[0];
    scatter_nd_desc.IndicesTensor = &input_descs[1];
    scatter_nd_desc.UpdatesTensor = &input_descs[2];
    scatter_nd_desc.OutputTensor = &output_descs[0];
    scatter_nd_desc.InputDimensionCount = in_out_shape.dims();
    scatter_nd_desc.IndicesDimensionCount = indices_shape.dims();

    DML_OPERATOR_DESC op_desc = {DML_OPERATOR_SCATTER_ND, &scatter_nd_desc};
    Initialize(ctx, std::move(tensors), op_desc);
  }
};

template <typename Index, typename BinaryOp>
class DmlTensorScatterBinaryKernel : public DmlKernel {
 public:
  using InitHelper = TensorScatterInitHelper<Index>;

  DmlTensorScatterBinaryKernel(DmlKernelConstruction* ctx,
                               const InitHelper* init_helper) {
    const TensorShape& in_out_shape = ctx->GetInputTensorShape(0);
    const TensorShape& indices_shape = ctx->GetInputTensorShape(1);
    const TensorShape& updates_shape = ctx->GetInputTensorShape(2);

    DmlTensorInfo in_out_tensor;
    in_out_tensor.desc = DmlTensorDesc::Create(ctx->GetInputDataType(0),
                                               in_out_shape, in_out_shape);

    DmlTensorInfo indices_tensor;
    indices_tensor.desc = DmlTensorDesc::Create(ctx->GetInputDataType(1),
                                                indices_shape, indices_shape);

    DmlTensorInfo updates_tensor;
    updates_tensor.desc = DmlTensorDesc::Create(ctx->GetInputDataType(2),
                                                updates_shape, updates_shape);

    DmlTensorInfo empty_tensor;
    empty_tensor.desc = DmlTensorDesc::Create(ctx->GetInputDataType(0),
                                              in_out_shape, TensorShape({1}));

    DmlKernelTensors tensors;
    tensors.inputs = {in_out_tensor, indices_tensor, updates_tensor,
                      empty_tensor};
    tensors.outputs = {in_out_tensor};

    auto dml_dtype = GetDmlDataTypeFromTfDataType(ctx->GetOutputDataType(0));
    constexpr uint32_t in_dim_count = 1;
    constexpr uint32_t in_size = 1;
    constexpr uint32_t in_stride = 1;
    empty_buffer_size_ =
        DMLCalcBufferTensorSize(dml_dtype, in_dim_count, &in_size, &in_stride);

    auto inputs = GetDmlTensorDescs(tensors.inputs);
    auto graph = dml::Graph(ctx->GetDmlDevice());
    auto input = dml::InputTensor(graph, 0, inputs[0]);
    auto indices = dml::InputTensor(graph, 1, inputs[1]);
    auto updates = dml::InputTensor(graph, 2, inputs[2]);
    auto empty_input = dml::InputTensor(graph, 3, inputs[3]);

    // First, perform the scatter on an empty tensor
    auto result = dml::ScatterND(empty_input, indices, updates,
                                 in_out_shape.dims(), indices_shape.dims());

    // Then, perform the binary operation on the scattered tensor and the
    // original input tensor
    result = BinaryOp()(input, result);

    Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
        graph.Compile(DML_EXECUTION_FLAG_NONE, {result});

    Initialize(ctx, std::move(tensors), compiled_op.Get());
  }

  StatusOr<DmlGpuEvent> Compute(DmlKernelContext* ctx) const override {
    D3D12BufferRegion input_buffer =
        ctx->CreateBufferForTensor(ctx->GetInputTensor(0));

    D3D12BufferRegion indices_buffer =
        ctx->CreateBufferForTensor(ctx->GetInputTensor(1));

    D3D12BufferRegion updates_buffer =
        ctx->CreateBufferForTensor(ctx->GetInputTensor(2));

    D3D12BufferRegion output_buffer =
        ctx->CreateBufferForTensor(*ctx->GetOutputTensor(0));

    DmlBuffer empty_buffer = ctx->AllocateDefaultBuffer(empty_buffer_size_);

    const absl::optional<DML_BUFFER_BINDING> input_bindings[4] = {
        input_buffer.GetBufferBinding(),
        indices_buffer.GetBufferBinding(),
        updates_buffer.GetBufferBinding(),
        empty_buffer.GetBufferBinding(),
    };

    const absl::optional<DML_BUFFER_BINDING> output_bindings[1] = {
        output_buffer.GetBufferBinding(),
    };

    ctx->ZeroBuffer(empty_buffer.Resource(), empty_buffer.Offset(),
                    empty_buffer.SizeInBytes());

    return ctx->ExecuteOperator(GetCompiledOp(), GetPersistentResourceBinding(),
                                input_bindings, output_bindings);
  }

 private:
  uint64_t empty_buffer_size_ = 0;
};

#define DML_REGISTER_KERNELS(type)                                          \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("TensorScatterUpdate")                                           \
          .Device(DEVICE_DML)                                               \
          .TypeConstraint<type>("T")                                        \
          .TypeConstraint<int32>("Tindices"),                               \
      DmlKernelWrapper<DmlTensorScatterUpdateKernel<int32>,                 \
                       GetOutputShapeAsInputShapeHelper>)                   \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("TensorScatterUpdate")                                           \
          .Device(DEVICE_DML)                                               \
          .TypeConstraint<type>("T")                                        \
          .TypeConstraint<int64>("Tindices"),                               \
      DmlKernelWrapper<DmlTensorScatterUpdateKernel<int64>,                 \
                       GetOutputShapeAsInputShapeHelper>)                   \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("TensorScatterAdd")                                              \
          .Device(DEVICE_DML)                                               \
          .TypeConstraint<type>("T")                                        \
          .TypeConstraint<int32>("Tindices"),                               \
      DmlKernelWrapper<                                                     \
          DmlTensorScatterBinaryKernel<int32, std::plus<dml::Expression>>,  \
          GetOutputShapeAsInputShapeHelper>)                                \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("TensorScatterAdd")                                              \
          .Device(DEVICE_DML)                                               \
          .TypeConstraint<type>("T")                                        \
          .TypeConstraint<int64>("Tindices"),                               \
      DmlKernelWrapper<                                                     \
          DmlTensorScatterBinaryKernel<int64, std::plus<dml::Expression>>,  \
          GetOutputShapeAsInputShapeHelper>)                                \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("TensorScatterSub")                                              \
          .Device(DEVICE_DML)                                               \
          .TypeConstraint<type>("T")                                        \
          .TypeConstraint<int32>("Tindices"),                               \
      DmlKernelWrapper<                                                     \
          DmlTensorScatterBinaryKernel<int32, std::minus<dml::Expression>>, \
          GetOutputShapeAsInputShapeHelper>)                                \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("TensorScatterSub")                                              \
          .Device(DEVICE_DML)                                               \
          .TypeConstraint<type>("T")                                        \
          .TypeConstraint<int64>("Tindices"),                               \
      DmlKernelWrapper<                                                     \
          DmlTensorScatterBinaryKernel<int64, std::minus<dml::Expression>>, \
          GetOutputShapeAsInputShapeHelper>)

TF_CALL_float(DML_REGISTER_KERNELS);
#undef DML_REGISTER_KERNELS

}  // namespace tensorflow