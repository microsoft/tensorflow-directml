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
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/kernels/dml_kernel_wrapper.h"
#include "tensorflow/core/kernels/dml_ops_common.h"
#include "tensorflow/core/lib/gtl/cleanup.h"

namespace tensorflow {

template <typename TIndex>
class GatherNdInitHelper : public InitializationHelper {
 public:
  using Attributes = EmptyAttributes;

  explicit GatherNdInitHelper(OpKernelContext* ctx,
                              std::shared_ptr<const Attributes> attr) {
    if (ctx->input(0).dtype() == DT_RESOURCE) {
      OP_REQUIRES_OK(
          ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &params_resource_));
      params_resource_->mu()->lock_shared();
      locked_ = true;
    }

    const Tensor params = GetParamsTensor(ctx);
    const Tensor& indices = ctx->input(1);

    OP_REQUIRES(ctx, TensorShapeUtils::IsVectorOrHigher(params.shape()),
                errors::InvalidArgument("params must be at least a vector"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVectorOrHigher(indices.shape()),
                errors::InvalidArgument("indices must be at least a vector"));
    OP_REQUIRES(
        ctx, indices.dim_size(indices.dims() - 1) <= params.dims(),
        errors::InvalidArgument(
            "index innermost dimension length must be <= params rank; saw: ",
            indices.dim_size(indices.dims() - 1), " vs. ", params.dims()));

    const TensorShape& indices_shape(indices.shape());
    const int64 indices_nd = indices_shape.dim_size(indices_shape.dims() - 1);

    // Check that we have enough index space
    int64 N_big = 1;
    for (int i = 0; i < indices_shape.dims() - 1; ++i) {
      N_big *= indices_shape.dim_size(i);
    }
    OP_REQUIRES(ctx, N_big <= std::numeric_limits<int>::max(),
                errors::InvalidArgument(
                    "indices has too many elements for int indexing: ", N_big,
                    " > ", std::numeric_limits<int>::max()));
    OP_REQUIRES(
        ctx, params.NumElements() <= std::numeric_limits<TIndex>::max(),
        errors::InvalidArgument("params.NumElements() too large for ",
                                DataTypeString(DataTypeToEnum<TIndex>::v()),
                                " indexing: ", params.NumElements(), " > ",
                                std::numeric_limits<TIndex>::max()));

    // The result shape is
    //   indices.shape[:-1] + params.shape[indices.shape[-1]:]
    TIndex N_result = 1;
    for (int i = 0; i < indices_shape.dims() - 1; ++i) {
      N_result *= indices_shape.dim_size(i);
    }

    const TensorShape& params_shape(params.shape());
    TIndex total_nd = params_shape.dims();

    TensorShape result_shape(indices_shape);
    result_shape.RemoveLastDims(1);

    int64 slice_size_big = 1;
    for (TIndex i = indices_nd; i < total_nd; ++i) {
      slice_size_big *= params_shape.dim_size(i);
      result_shape.AddDim(params_shape.dim_size(i));
    }

    OP_REQUIRES(ctx, slice_size_big <= std::numeric_limits<TIndex>::max(),
                errors::InvalidArgument(
                    "slice size is too large for indexing: ", slice_size_big,
                    " > ", std::numeric_limits<TIndex>::max()));

    OP_REQUIRES(
        ctx, indices_nd < 8,
        errors::InvalidArgument("Only indices.shape[-1] values between 1 and 7 "
                                "are currently supported.  Requested rank: ",
                                indices_nd));

    // If params are empty but indices are nonempty, then it's the caller's
    // mistake and we should return an error.
    OP_REQUIRES(
        ctx,
        indices_shape.num_elements() == 0 || params_shape.num_elements() != 0,
        errors::InvalidArgument("Requested more than 0 entries, but "
                                "params is empty.  Params shape: ",
                                params_shape.DebugString()));

    output_shape_ = std::move(result_shape);
  }

  bool IsNoOpKernel(OpKernelContext* ctx,
                    absl::Span<const TensorShape> output_shapes) const final {
    const Tensor params = GetParamsTensor(ctx);
    const Tensor& indices = ctx->input(1);

    int64 indices_leading_dims = 1;
    for (int i = 0; i < indices.dims() - 1; ++i) {
      indices_leading_dims *= indices.dim_size(i);
    }

    if (indices_leading_dims == 0) {
      return true;
    }

    int64 last_indices_dim = indices.dim_size(indices.dims() - 1);

    return params.NumElements() == 0 && last_indices_dim == 0;
  }

  TensorShape GetOutputShape() const { return output_shape_; }

  Tensor GetParamsTensor(OpKernelContext* ctx) const {
    if (ctx->input(0).dtype() == DT_RESOURCE) {
      return params_resource_ ? *params_resource_->tensor()
                              : ctx->mutable_input(0, false);
    } else {
      return ctx->input(0);
    }
  }

  void Unlock() const {
    if (params_resource_ && locked_) {
      params_resource_->mu()->unlock_shared();
      locked_ = false;
    }
  }

  virtual ~GatherNdInitHelper() { Unlock(); }

 private:
  TensorShape output_shape_;
  core::RefCountPtr<Var> params_resource_;
  mutable bool locked_ = false;
};

template <typename TIndex>
class GatherNdShapeHelper : public ShapeHelper {
 public:
  std::vector<TensorShape> GetOutputShapes(
      OpKernelContext* ctx,
      const InitializationHelper* initialization_helper) const override {
    auto init_helper =
        static_cast<const GatherNdInitHelper<TIndex>*>(initialization_helper);

    return {init_helper->GetOutputShape()};
  }
};

template <typename TIndex>
class DmlGatherNdKernel : public DmlKernel {
 public:
  using InitHelper = GatherNdInitHelper<TIndex>;

  DmlGatherNdKernel(DmlKernelConstruction* ctx, const InitHelper* init_helper) {
    const Tensor params =
        init_helper->GetParamsTensor(ctx->GetOpKernelContext());

    const TensorShape& params_shape = params.shape();
    const TensorShape& indices_shape = ctx->GetInputTensorShape(1);
    const TensorShape& output_shape = ctx->GetOutputTensorShape(0);

    // Flatten the indices into a matrix
    int64 last_indices_dim = indices_shape.dim_size(indices_shape.dims() - 1);

    int64 indices_leading_dims = 1;
    for (int i = 0; i < indices_shape.dims() - 1; ++i) {
      indices_leading_dims *= indices_shape.dim_size(i);
    }

    // Flatten the params into a tensor of rank last_indices_dim + 1
    TensorShape flat_params_shape;

    for (int i = 0; i < last_indices_dim; ++i) {
      if (i < params_shape.dims()) {
        flat_params_shape.AddDim(params_shape.dim_size(i));
      } else {
        flat_params_shape.AddDim(1);
      }
    }

    int64 last_params_dim = 1;
    for (int i = last_indices_dim; i < params_shape.dims(); ++i) {
      last_params_dim *= params_shape.dim_size(i);
    }

    flat_params_shape.AddDim(last_params_dim);

    TensorShape flat_indices_shape({
        indices_leading_dims,
        last_indices_dim,
    });

    // Flatten the output shape
    TensorShape flat_output_shape({
        indices_leading_dims,
        last_params_dim,
    });

    int missing_dims = flat_params_shape.dims() - flat_output_shape.dims();

    for (int i = 0; i < missing_dims; ++i) {
      flat_indices_shape.InsertDim(0, 1);
      flat_output_shape.InsertDim(0, 1);
    }

    DmlTensorInfo params_tensor;
    params_tensor.kernel_index = 0;
    params_tensor.desc = DmlTensorDesc::Create(
        params.dtype(), flat_params_shape, flat_params_shape);

    DmlTensorInfo output_tensor;
    output_tensor.kernel_index = 0;
    output_tensor.desc = DmlTensorDesc::Create(
        ctx->GetOutputDataType(0), flat_output_shape, flat_output_shape);

    auto scope = dml::Graph(ctx->GetDmlDevice());

    // When the indices' last dimension is 0, we gather the entire tensor
    if (last_indices_dim == 0) {
      DmlKernelTensors tensors;
      tensors.inputs = {params_tensor};
      tensors.outputs = {output_tensor};

      auto input_descs = GetDmlTensorDescs(tensors.inputs);
      auto params = dml::InputTensor(scope, 0, input_descs[0]);
      auto result = dml::Tile(
          params, {static_cast<uint32_t>(indices_leading_dims), 1, 1, 1});

      // TFDML #24881131
      if (Is64BitSignedIntegerType(ctx->GetOutputDataType(0))) {
        result = dml::ConvertInt32ToInt64(result);
      }

      Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
          scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

      Initialize(ctx, std::move(tensors), compiled_op.Get());
    } else {
      DmlTensorInfo indices_tensor;
      indices_tensor.kernel_index = 1;
      indices_tensor.desc = DmlTensorDesc::Create(
          ctx->GetInputDataType(1), flat_indices_shape, flat_indices_shape);

      DmlKernelTensors tensors;
      tensors.inputs = {params_tensor, indices_tensor};
      tensors.outputs = {output_tensor};

      auto input_descs = GetDmlTensorDescs(tensors.inputs);
      auto params = dml::InputTensor(scope, 0, input_descs[0]);
      auto indices = dml::InputTensor(scope, 1, input_descs[1]);

      auto result = dml::GatherND(params, indices, flat_params_shape.dims(),
                                  flat_indices_shape.dims(), 0);

      // TFDML #24881131
      if (Is64BitSignedIntegerType(ctx->GetOutputDataType(0))) {
        result = dml::ConvertInt32ToInt64(result);
      }

      Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
          scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

      Initialize(ctx, std::move(tensors), compiled_op.Get());
    }
  }

  StatusOr<DmlGpuEvent> Compute(DmlKernelContext* ctx) const override {
    auto init_helper = ctx->GetInitializationHelper<InitHelper>();

    auto lock_cleanup =
        gtl::MakeCleanup([init_helper] { init_helper->Unlock(); });

    const Tensor params_tensor =
        init_helper->GetParamsTensor(ctx->GetOpKernelContext());

    const Tensor& indices_tensor = ctx->GetInputTensor(1);
    int64 last_indices_dim = indices_tensor.dim_size(indices_tensor.dims() - 1);

    // Create input buffers
    absl::InlinedVector<D3D12BufferRegion, 2> input_buffers;
    input_buffers.push_back(ctx->CreateBufferForTensor(params_tensor));

    // Create input bindings
    absl::InlinedVector<absl::optional<DML_BUFFER_BINDING>, 2> input_bindings;
    input_bindings.push_back(input_buffers[0].GetBufferBinding());

    // When last_indices_dim == 0, we use a tile instead of a gather, and
    // therefore we don't need a second input
    if (last_indices_dim != 0) {
      input_buffers.push_back(ctx->CreateBufferForTensor(indices_tensor));
      input_bindings.push_back(input_buffers[1].GetBufferBinding());
    }

    DmlGpuEvent gpu_event;
    absl::InlinedVector<absl::optional<DML_BUFFER_BINDING>, 1> output_bindings;

    Tensor* output = ctx->GetOutputTensor(0);

    D3D12BufferRegion output_buffer = ctx->CreateBufferForTensor(*output);

    output_bindings.push_back(output_buffer.GetBufferBinding());

    auto status_or_event =
        DmlKernel::Compute(ctx, input_bindings, output_bindings);
    if (!status_or_event.ok()) {
      return status_or_event;
    }

    gpu_event = ctx->InsertUavBarrier();
    return gpu_event;
  }
};

template <typename TIndex>
using DmlGatherNdWrapper =
    DmlKernelWrapper<DmlGatherNdKernel<TIndex>, GatherNdShapeHelper<TIndex>>;

#define DML_REGISTER_KERNELS(type)                                \
  REGISTER_KERNEL_BUILDER(Name("GatherNd")                        \
                              .Device(DEVICE_DML)                 \
                              .TypeConstraint<type>("Tparams")    \
                              .TypeConstraint<int32>("Tindices"), \
                          DmlGatherNdWrapper<int32>)              \
  REGISTER_KERNEL_BUILDER(Name("GatherNd")                        \
                              .Device(DEVICE_DML)                 \
                              .TypeConstraint<type>("Tparams")    \
                              .TypeConstraint<int64>("Tindices"), \
                          DmlGatherNdWrapper<int64>)

TF_CALL_float(DML_REGISTER_KERNELS);
TF_CALL_half(DML_REGISTER_KERNELS);
TF_CALL_int32(DML_REGISTER_KERNELS);
TF_CALL_int64(DML_REGISTER_KERNELS);
#undef DML_REGISTER_KERNELS

#define DML_REGISTER_KERNELS(type)                                \
  REGISTER_KERNEL_BUILDER(Name("ResourceGatherNd")                \
                              .Device(DEVICE_DML)                 \
                              .HostMemory("resource")             \
                              .TypeConstraint<type>("dtype")      \
                              .TypeConstraint<int32>("Tindices"), \
                          DmlGatherNdWrapper<int32>)              \
  REGISTER_KERNEL_BUILDER(Name("ResourceGatherNd")                \
                              .Device(DEVICE_DML)                 \
                              .HostMemory("resource")             \
                              .TypeConstraint<type>("dtype")      \
                              .TypeConstraint<int64>("Tindices"), \
                          DmlGatherNdWrapper<int64>)

TF_CALL_float(DML_REGISTER_KERNELS);
TF_CALL_half(DML_REGISTER_KERNELS);
#undef DML_REGISTER_KERNELS

}  // namespace tensorflow
