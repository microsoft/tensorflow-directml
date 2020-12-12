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
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/kernels/dml_kernel_wrapper.h"
#include "tensorflow/core/kernels/dml_ops_common.h"

namespace tensorflow {

template <typename TIndex>
class GatherInitializationHelper : public InitializationHelper {
 public:
  struct Attributes {
    explicit Attributes(OpKernelConstruction* ctx) {
      // Set attr->batch_dims to 0 if the attribute does not exist.
      if (!ctx->GetAttr("batch_dims", &batch_dims).ok()) {
        batch_dims = 0;
      }
    }

    int32 batch_dims;
  };

  GatherInitializationHelper(OpKernelContext* ctx,
                             std::shared_ptr<const Attributes> attr) {
    if (ctx->input(0).dtype() == DT_RESOURCE) {
      OP_REQUIRES_OK(
          ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &params_resource_));
      params_resource_->mu()->lock_shared();
    }

    const Tensor params = GetParamsTensor(ctx);
    const Tensor& indices = ctx->input(1);

    OP_REQUIRES(
        ctx, TensorShapeUtils::IsVectorOrHigher(params.shape()),
        errors::InvalidArgument("params must be at least 1 dimensional"));

    positive_batch_dims_ = attr->batch_dims < 0
                               ? indices.dims() + attr->batch_dims
                               : attr->batch_dims;

    if (ctx->num_inputs() == 3) {
      const Tensor& axis_tensor = ctx->input(2);

      OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(axis_tensor.shape()),
                  errors::InvalidArgument("axis must be scalar"));

      if (axis_tensor.dtype() == DT_INT32) {
        axis_ = axis_tensor.scalar<int32>()();
      } else if (axis_tensor.dtype() == DT_INT64) {
        axis_ = axis_tensor.scalar<int64>()();
      } else {
        OP_REQUIRES(ctx, false,
                    errors::InvalidArgument("axis must be int32 or int64."));
      }
    } else {
      axis_ = positive_batch_dims_;
    }

    OP_REQUIRES(
        ctx, axis_ >= -params.dims() && axis_ < params.dims(),
        errors::InvalidArgument("Expected axis in the range [", -params.dims(),
                                ", ", params.dims(), "), but got ", axis_));

    if (axis_ < 0) {
      axis_ += params.dims();
    }

    if (positive_batch_dims_ != 0) {
      OP_REQUIRES(ctx,
                  positive_batch_dims_ >= -indices.dims() &&
                      positive_batch_dims_ < indices.dims(),
                  errors::InvalidArgument("Expected batch_dims in the range [",
                                          -indices.dims(), ", ", indices.dims(),
                                          "), but got ", positive_batch_dims_));

      OP_REQUIRES(ctx, positive_batch_dims_ < params.dims(),
                  errors::InvalidArgument("batch_dims (", positive_batch_dims_,
                                          ") must be less than rank(params) (",
                                          params.dims(), ")."));

      OP_REQUIRES(ctx, axis_ >= positive_batch_dims_,
                  errors::InvalidArgument("batch_dims (", positive_batch_dims_,
                                          ") must be less than or equal to ",
                                          "axis (", axis_, ")."));
    }

    // Check that we have enough index space
    OP_REQUIRES(
        ctx, params.dim_size(axis_) <= std::numeric_limits<TIndex>::max(),
        errors::InvalidArgument("params.shape[", axis_, "] too large for ",
                                DataTypeString(DataTypeToEnum<TIndex>::v()),
                                " indexing: ", params.dim_size(axis_), " > ",
                                std::numeric_limits<TIndex>::max()));
  }

  int32 GetBatchDims() const { return positive_batch_dims_; }
  int64 GetAxis() const { return axis_; }

  const Tensor GetParamsTensor(OpKernelContext* ctx) const {
    return ctx->input(0).dtype() == DT_RESOURCE ? *params_resource_->tensor()
                                                : ctx->input(0);
  }

  void Unlock() const {
    if (params_resource_) {
      params_resource_->mu()->unlock_shared();
    }
  }

 private:
  int64 axis_;
  int32 positive_batch_dims_;
  core::RefCountPtr<Var> params_resource_;
};

template <typename TIndex>
class GatherShapeHelper : public ShapeHelper {
 public:
  std::vector<TensorShape> GetOutputShapes(
      OpKernelContext* ctx,
      const InitializationHelper* initialization_helper) const override {
    auto init_helper = static_cast<const GatherInitializationHelper<TIndex>*>(
        initialization_helper);

    const Tensor params = init_helper->GetParamsTensor(ctx);
    const Tensor& indices = ctx->input(1);

    // The result shape is params.shape[:axis] + indices.shape[batch_dims:] +
    // params.shape[axis + 1:].
    TensorShape output_shape;

    int32 batch_dims = init_helper->GetBatchDims();
    int64 axis = init_helper->GetAxis();

    for (int i = 0; i < batch_dims; ++i) {
      output_shape.AddDim(params.dim_size(i));
    }
    for (int i = batch_dims; i < axis; ++i) {
      output_shape.AddDim(params.dim_size(i));
    }
    for (int i = batch_dims; i < indices.dims(); ++i) {
      output_shape.AddDim(indices.dim_size(i));
    }
    for (int i = axis + 1; i < params.dims(); ++i) {
      output_shape.AddDim(params.dim_size(i));
    }

    return {std::move(output_shape)};
  }
};

struct SimpleGather {
  dml::TensorDesc::Dimensions params_shape;
  dml::TensorDesc::Dimensions indices_shape;
  dml::TensorDesc::Dimensions output_shape;
  uint32_t gather_axis;
  uint32_t index_dimensions;
};

SimpleGather SimplifyGather(const TensorShape& params_shape,
                            const TensorShape& indices_shape, int64 axis,
                            int32 batch_dims) {
  // Collapse the batch dimensions together
  uint32_t collapsed_batch_dims = 1;
  for (int i = 0; i < batch_dims; ++i) {
    collapsed_batch_dims *= params_shape.dim_size(i);
  }

  // Collapse the non-batch dimensions to the left of the axis together
  uint32_t left_collapsed_dims = 1;
  for (int i = batch_dims; i < axis; ++i) {
    left_collapsed_dims *= params_shape.dim_size(i);
  }

  // Collapse all non-batch dimensions in Indices together
  uint32_t collapsed_indices_elements = 1;
  for (int i = batch_dims; i < indices_shape.dims(); ++i) {
    collapsed_indices_elements *= indices_shape.dim_size(i);
  }

  // Collapse the dimensions to the right of the axis together
  uint32_t right_collapsed_dims = 1;
  for (int i = axis + 1; i < params_shape.dims(); ++i) {
    right_collapsed_dims *= params_shape.dim_size(i);
  }

  uint32_t gather_dims = params_shape.dim_size(axis);

  SimpleGather desc = {};
  desc.params_shape = {collapsed_batch_dims, left_collapsed_dims, gather_dims,
                       right_collapsed_dims};
  desc.gather_axis = 2;

  if (batch_dims < indices_shape.dims()) {
    desc.indices_shape = {1, 1, collapsed_batch_dims,
                          collapsed_indices_elements};
    desc.output_shape = {collapsed_batch_dims, left_collapsed_dims,
                         collapsed_indices_elements, right_collapsed_dims};
    desc.index_dimensions = 1;
  } else {
    desc.indices_shape = {1, 1, 1, collapsed_batch_dims};
    desc.output_shape = {1, collapsed_batch_dims, left_collapsed_dims,
                         right_collapsed_dims};
    desc.index_dimensions = 0;
  }

  return desc;
}

template <typename TIndex>
class DmlGatherKernel : public DmlKernel {
 public:
  using InitHelper = GatherInitializationHelper<TIndex>;

  explicit DmlGatherKernel(
      DmlKernelConstruction* ctx,
      const GatherInitializationHelper<TIndex>* init_helper) {
    CHECK(ctx->GetInputCount() == 2 || ctx->GetInputCount() == 3);
    CHECK(ctx->GetOutputCount() == 1);

    const Tensor params_tensor =
        init_helper->GetParamsTensor(ctx->GetOpKernelContext());

    const TensorShape& indices_shape = ctx->GetInputTensorShape(1);
    int32 batch_dims = init_helper->GetBatchDims();
    int64 axis = init_helper->GetAxis();

    SimpleGather simple_gather =
        SimplifyGather(params_tensor.shape(), indices_shape, axis, batch_dims);

    DmlTensorInfo params_input;
    params_input.kernel_index = 0;
    params_input.desc =
        DmlTensorDesc::Create(params_tensor.dtype(), simple_gather.params_shape,
                              simple_gather.params_shape);

    DmlTensorInfo indices_input;
    indices_input.kernel_index = 1;
    indices_input.desc = DmlTensorDesc::Create(ctx->GetInputDataType(1),
                                               simple_gather.indices_shape,
                                               simple_gather.indices_shape);

    DmlTensorInfo output;
    output.kernel_index = 0;
    output.desc = DmlTensorDesc::Create(ctx->GetOutputDataType(0),
                                        simple_gather.output_shape,
                                        simple_gather.output_shape);

    DmlKernelTensors tensors;
    tensors.inputs = {params_input, indices_input};
    tensors.outputs = {output};

    auto inputs = GetDmlTensorDescs(tensors.inputs);

    // Create a custom tensor policy to correctly pad out strides for int64
    // emulation
    dml::TensorPolicy out_policy = dml::TensorPolicy::Default();
    if (Is64BitIntegerType(ctx->GetInputDataType(0))) {
      out_policy = GetEmulatedInt64TensorPolicy();
    }

    auto scope = dml::Graph(ctx->GetDmlDevice(), out_policy);
    auto input_tensor = dml::InputTensor(scope, 0, inputs[0]);
    auto indices_tensor = dml::InputTensor(scope, 1, inputs[1]);

    // DML's Gather only supports uint32 for the indices tensor, but TF models
    // use either int64 or int32. int64 already gets converted to uint32 with
    // the strides hack
    // (TFDML #24881131), so we
    // only need to reinterpret the int32 data to uint32 here.
    if (indices_tensor.GetOutputDesc().dataType == DML_TENSOR_DATA_TYPE_INT32) {
      indices_tensor =
          dml::Reinterpret(indices_tensor, DML_TENSOR_DATA_TYPE_UINT32);
    }

    auto result =
        dml::Gather(input_tensor, indices_tensor, simple_gather.gather_axis,
                    simple_gather.index_dimensions);

    Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
        scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

    Initialize(ctx, std::move(tensors), compiled_op.Get());
  }

  StatusOr<DmlGpuEvent> Compute(DmlKernelContext* ctx) const override {
    // Currently, 64-bit integers in DML are emulated using 32-bit integers
    // using striding to emulate a larger type. Because we can't guarantee that
    // our output tensor's memory is zero'd, we need to do so manually prior to
    // running running gather.
    Tensor* output = ctx->GetOutputTensor(0);

    auto init_helper = ctx->GetInitializationHelper<InitHelper>();

    D3D12BufferRegion input_buffers[] = {
        ctx->CreateBufferForTensor(
            init_helper->GetParamsTensor(ctx->GetOpKernelContext())),
        ctx->CreateBufferForTensor(ctx->GetInputTensor(1)),
    };

    D3D12BufferRegion output_buffers[] = {ctx->CreateBufferForTensor(*output)};

    if (Is64BitIntegerType(output->dtype())) {
      ctx->ZeroBuffer(output_buffers[0]);
    }

    // Create bindings
    auto input_bindings = dml_util::GetBufferBindings(input_buffers);
    auto output_bindings = dml_util::GetBufferBindings(output_buffers);

    DmlGpuEvent gpu_event =
        ctx->ExecuteOperator(GetCompiledOp(), GetPersistentResourceBinding(),
                             input_bindings, output_bindings);

    init_helper->Unlock();
    return gpu_event;
  }
};

template <typename TIndex>
using DmlGatherWrapper =
    DmlKernelWrapper<DmlGatherKernel<TIndex>, GatherShapeHelper<TIndex>>;

template <typename TIndex>
using DmlResourceGatherWrapper =
    DmlKernelWrapper<DmlGatherKernel<TIndex>, GatherShapeHelper<TIndex>,
                     DmlKernelCachePolicy::Never>;

#define DML_REGISTER_KERNELS(type)                                \
  REGISTER_KERNEL_BUILDER(Name("Gather")                          \
                              .Device(DEVICE_DML)                 \
                              .TypeConstraint<type>("Tparams")    \
                              .TypeConstraint<int32>("Tindices"), \
                          DmlGatherWrapper<int32>)                \
  REGISTER_KERNEL_BUILDER(Name("GatherV2")                        \
                              .Device(DEVICE_DML)                 \
                              .TypeConstraint<type>("Tparams")    \
                              .TypeConstraint<int32>("Tindices")  \
                              .HostMemory("axis"),                \
                          DmlGatherWrapper<int32>)                \
  REGISTER_KERNEL_BUILDER(Name("Gather")                          \
                              .Device(DEVICE_DML)                 \
                              .TypeConstraint<type>("Tparams")    \
                              .TypeConstraint<int64>("Tindices"), \
                          DmlGatherWrapper<int64>)                \
  REGISTER_KERNEL_BUILDER(Name("GatherV2")                        \
                              .Device(DEVICE_DML)                 \
                              .TypeConstraint<type>("Tparams")    \
                              .TypeConstraint<int64>("Tindices")  \
                              .HostMemory("axis"),                \
                          DmlGatherWrapper<int64>)                \
  REGISTER_KERNEL_BUILDER(Name("ResourceGather")                  \
                              .Device(DEVICE_DML)                 \
                              .HostMemory("resource")             \
                              .TypeConstraint<type>("dtype")      \
                              .TypeConstraint<int32>("Tindices"), \
                          DmlResourceGatherWrapper<int32>)        \
  REGISTER_KERNEL_BUILDER(Name("ResourceGather")                  \
                              .Device(DEVICE_DML)                 \
                              .HostMemory("resource")             \
                              .TypeConstraint<type>("dtype")      \
                              .TypeConstraint<int64>("Tindices"), \
                          DmlResourceGatherWrapper<int64>)

TF_CALL_DML_ALL_TYPES(DML_REGISTER_KERNELS);
#undef DML_REGISTER_KERNELS

}  // namespace tensorflow
