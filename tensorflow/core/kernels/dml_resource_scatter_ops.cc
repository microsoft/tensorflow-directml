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
#include "tensorflow/core/lib/gtl/cleanup.h"

namespace tensorflow {

static Status ValidateRefScatter(const Tensor& params, const Tensor& indices,
                                 const Tensor& updates) {
  if (!params.IsInitialized()) {
    return errors::FailedPrecondition("Null ref for params");
  }

  return Status::OK();
}

static Status ValidateResourceScatter(const Tensor& indices,
                                      const Tensor& updates) {
  int64 num_updates = updates.NumElements();
  int64 num_indices = indices.NumElements();
  if (num_indices > 0 && !TensorShapeUtils::IsScalar(updates.shape()) &&
      num_updates % num_indices != 0) {
    return errors::InvalidArgument(
        "shape of indices (", indices.shape().DebugString(),
        ") is not compatible with the shape of updates (",
        updates.shape().DebugString(), ")");
  }

  return Status::OK();
}

template <typename Index>
static Status ValidateCommonScatter(const Tensor& params,
                                    const Tensor& indices) {
  // Check that we have enough index space
  const int64 N_big = indices.NumElements();

  if (N_big > std::numeric_limits<Index>::max()) {
    return errors::InvalidArgument("indices has too many elements for ",
                                   DataTypeString(DataTypeToEnum<Index>::v()),
                                   " indexing: ", N_big, " > ",
                                   std::numeric_limits<Index>::max());
  }

  const Index N = static_cast<Index>(indices.NumElements());

  if (params.dim_size(0) > std::numeric_limits<Index>::max()) {
    return errors::InvalidArgument("params.shape[0] too large for ",
                                   DataTypeString(DataTypeToEnum<Index>::v()),
                                   " indexing: ", params.dim_size(0), " > ",
                                   std::numeric_limits<Index>::max());
  }

  return Status::OK();
}

template <typename Index>
class ResourceScatterNDInitHelper : public InitializationHelper {
 public:
  using Attributes = EmptyAttributes;

  explicit ResourceScatterNDInitHelper(OpKernelContext* ctx,
                                       std::shared_ptr<const Attributes> attr) {
    DCHECK(ctx->input_is_ref(0) || ctx->input(0).dtype() == DT_RESOURCE);

    if (!ctx->input_is_ref(0)) {
      OP_REQUIRES_OK(
          ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &params_resource_));
      params_resource_->mu()->lock_shared();
      locked_ = true;
    }

    const Tensor params = GetParamsTensor(ctx);
    const Tensor& indices = ctx->input(1);
    const Tensor& updates = ctx->input(2);

    if (ctx->input_is_ref(0)) {
      OP_REQUIRES_OK(ctx, ValidateRefScatter(params, indices, updates));
    }

    OP_REQUIRES_OK(ctx, ValidateCommonScatter<Index>(params, indices));

    if (!ctx->input_is_ref(0)) {
      OP_REQUIRES_OK(ctx, ValidateResourceScatter(indices, updates));
    }
  }

  Tensor GetParamsTensor(OpKernelContext* ctx) const {
    DCHECK(ctx->input_is_ref(0) || ctx->input(0).dtype() == DT_RESOURCE);

    return params_resource_ ? *params_resource_->tensor()
                            : ctx->mutable_input(0, false);
  }

  void Unlock() const {
    if (params_resource_ && locked_) {
      params_resource_->mu()->unlock_shared();
      locked_ = false;
    }
  }

  virtual ~ResourceScatterNDInitHelper() { Unlock(); }

 private:
  core::RefCountPtr<Var> params_resource_;
  mutable bool locked_ = false;
};

template <typename Index>
class DmlResourceScatterNDUpdateKernel : public DmlKernel {
 public:
  using InitHelper = ResourceScatterNDInitHelper<Index>;

  DmlResourceScatterNDUpdateKernel(DmlKernelConstruction* ctx,
                                   const InitHelper* init_helper) {
    const Tensor params_tensor =
        init_helper->GetParamsTensor(ctx->GetOpKernelContext());

    const TensorShape& in_out_shape = params_tensor.shape();
    const TensorShape& indices_shape = ctx->GetInputTensorShape(1);
    const TensorShape& updates_shape = ctx->GetInputTensorShape(2);

    DmlTensorInfo in_out_tensor;
    in_out_tensor.desc = DmlTensorDesc::Create(params_tensor.dtype(),
                                               in_out_shape, in_out_shape);

    DmlTensorInfo indices_tensor;
    indices_tensor.desc = DmlTensorDesc::Create(ctx->GetInputDataType(1),
                                                indices_shape, indices_shape);

    DmlTensorInfo updates_tensor;
    updates_tensor.desc = DmlTensorDesc::Create(ctx->GetInputDataType(2),
                                                updates_shape, updates_shape);

    DmlKernelTensors tensors;
    tensors.inputs = {in_out_tensor, indices_tensor, updates_tensor};
    tensors.outputs = {in_out_tensor};

    if (params_tensor.dtype() != DT_RESOURCE) {
      // The input ref and the output ref must refer to the same memory
      tensors.output_refs_forwarding = {0};
    }

    auto inputs = GetDmlTensorDescs(tensors.inputs);
    auto graph = dml::Graph(ctx->GetDmlDevice());
    auto input = dml::InputTensor(graph, 0, inputs[0]);
    auto indices = dml::InputTensor(graph, 1, inputs[1]);
    auto updates = dml::InputTensor(graph, 2, inputs[2]);

    // First, perform the scatter on an empty tensor
    auto result = dml::ScatterND(input, indices, updates, in_out_shape.dims(),
                                 indices_shape.dims());

    Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
        graph.Compile(DML_EXECUTION_FLAG_NONE, {result});

    Initialize(ctx, std::move(tensors), compiled_op.Get());
  }

  StatusOr<DmlGpuEvent> Compute(DmlKernelContext* ctx) const override {
    auto init_helper = ctx->GetInitializationHelper<InitHelper>();

    auto lock_cleanup =
        gtl::MakeCleanup([init_helper] { init_helper->Unlock(); });

    const Tensor params_tensor =
        init_helper->GetParamsTensor(ctx->GetOpKernelContext());

    // Create input buffers
    D3D12BufferRegion input_buffers[] = {
        ctx->CreateBufferForTensor(params_tensor),
        ctx->CreateBufferForTensor(ctx->GetInputTensor(1)),
        ctx->CreateBufferForTensor(ctx->GetInputTensor(2)),
    };

    // Create input bindings
    absl::optional<DML_BUFFER_BINDING> input_bindings[] = {
        input_buffers[0].GetBufferBinding(),
        input_buffers[1].GetBufferBinding(),
        input_buffers[2].GetBufferBinding(),
    };

    DmlGpuEvent gpu_event;

    absl::optional<DML_BUFFER_BINDING> output_bindings[] = {
        input_bindings[0],
    };

    auto status_or_event =
        DmlKernel::Compute(ctx, input_bindings, output_bindings);
    if (!status_or_event.ok()) {
      return status_or_event;
    }

    gpu_event = status_or_event.ValueOrDie();

    return gpu_event;
  }
};

template <typename Index, typename BinaryOp>
class DmlResourceScatterNDBinaryKernel : public DmlKernel {
 public:
  using InitHelper = ResourceScatterNDInitHelper<Index>;

  DmlResourceScatterNDBinaryKernel(DmlKernelConstruction* ctx,
                                   const InitHelper* init_helper) {
    const Tensor params_tensor =
        init_helper->GetParamsTensor(ctx->GetOpKernelContext());

    const TensorShape& in_out_shape = params_tensor.shape();
    const TensorShape& indices_shape = ctx->GetInputTensorShape(1);
    const TensorShape& updates_shape = ctx->GetInputTensorShape(2);

    DmlTensorInfo in_out_tensor;
    in_out_tensor.desc = DmlTensorDesc::Create(params_tensor.dtype(),
                                               in_out_shape, in_out_shape);

    DmlTensorInfo indices_tensor;
    indices_tensor.desc = DmlTensorDesc::Create(ctx->GetInputDataType(1),
                                                indices_shape, indices_shape);

    DmlTensorInfo updates_tensor;
    updates_tensor.desc = DmlTensorDesc::Create(ctx->GetInputDataType(2),
                                                updates_shape, updates_shape);

    DmlTensorInfo empty_tensor;
    empty_tensor.desc = DmlTensorDesc::Create(params_tensor.dtype(),
                                              in_out_shape, in_out_shape);

    DmlKernelTensors tensors;
    tensors.inputs = {in_out_tensor, indices_tensor, updates_tensor,
                      empty_tensor};
    tensors.outputs = {in_out_tensor};

    if (params_tensor.dtype() != DT_RESOURCE) {
      // The input ref and the output ref must refer to the same memory
      tensors.output_refs_forwarding = {0};
    }

    auto dml_dtype = GetDmlDataTypeFromTfDataType(params_tensor.dtype());
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
    auto init_helper = ctx->GetInitializationHelper<InitHelper>();

    auto lock_cleanup =
        gtl::MakeCleanup([init_helper] { init_helper->Unlock(); });

    const Tensor params_tensor =
        init_helper->GetParamsTensor(ctx->GetOpKernelContext());

    DmlBuffer empty_buffer = ctx->AllocateDefaultBuffer(empty_buffer_size_);
    // Create input buffers
    D3D12BufferRegion input_buffers[] = {
        ctx->CreateBufferForTensor(params_tensor),
        ctx->CreateBufferForTensor(ctx->GetInputTensor(1)),
        ctx->CreateBufferForTensor(ctx->GetInputTensor(2)),
    };

    // Create input bindings
    absl::optional<DML_BUFFER_BINDING> input_bindings[] = {
        input_buffers[0].GetBufferBinding(),
        input_buffers[1].GetBufferBinding(),
        input_buffers[2].GetBufferBinding(),
        empty_buffer.GetBufferBinding(),
    };

    ctx->ZeroBuffer(empty_buffer.Resource(), empty_buffer.Offset(),
                    empty_buffer.SizeInBytes());

    DmlGpuEvent gpu_event;
    DmlBuffer output_buffer =
        ctx->AllocateDefaultBuffer(input_buffers[0].SizeInBytes());

    absl::optional<DML_BUFFER_BINDING> output_bindings[] = {
        output_buffer.GetBufferBinding(),
    };

    auto status_or_event =
        DmlKernel::Compute(ctx, input_bindings, output_bindings);
    if (!status_or_event.ok()) {
      return status_or_event;
    }

    ctx->CopyBufferToBuffer(input_buffers[0].Resource(),
                            input_buffers[0].Offset(), output_buffer.Resource(),
                            output_buffer.Offset(),
                            output_buffer.SizeInBytes());

    gpu_event = ctx->InsertUavBarrier();
    return gpu_event;
  }

 private:
  uint64_t empty_buffer_size_ = 0;
};

#define DML_REGISTER_KERNELS(type)                                             \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("ScatterNdUpdate")                                                  \
          .Device(DEVICE_DML)                                                  \
          .TypeConstraint<type>("T")                                           \
          .TypeConstraint<int32>("Tindices"),                                  \
      DmlKernelWrapper<DmlResourceScatterNDUpdateKernel<int32>,                \
                       GetOutputShapeAsInputShapeHelper>)                      \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("ScatterNdUpdate")                                                  \
          .Device(DEVICE_DML)                                                  \
          .TypeConstraint<type>("T")                                           \
          .TypeConstraint<int64>("Tindices"),                                  \
      DmlKernelWrapper<DmlResourceScatterNDUpdateKernel<int64>,                \
                       GetOutputShapeAsInputShapeHelper>)                      \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("ResourceScatterNdUpdate")                                          \
          .Device(DEVICE_DML)                                                  \
          .HostMemory("ref")                                                   \
          .TypeConstraint<type>("T")                                           \
          .TypeConstraint<int32>("Tindices"),                                  \
      DmlKernelWrapper<DmlResourceScatterNDUpdateKernel<int32>,                \
                       NoOutputShapeHelper, DmlKernelCachePolicy::Never>)      \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("ResourceScatterNdUpdate")                                          \
          .Device(DEVICE_DML)                                                  \
          .HostMemory("ref")                                                   \
          .TypeConstraint<type>("T")                                           \
          .TypeConstraint<int64>("Tindices"),                                  \
      DmlKernelWrapper<DmlResourceScatterNDUpdateKernel<int64>,                \
                       NoOutputShapeHelper, DmlKernelCachePolicy::Never>)      \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("ScatterNdAdd")                                                     \
          .Device(DEVICE_DML)                                                  \
          .TypeConstraint<type>("T")                                           \
          .TypeConstraint<int32>("Tindices"),                                  \
      DmlKernelWrapper<                                                        \
          DmlResourceScatterNDBinaryKernel<int32, std::plus<dml::Expression>>, \
          GetOutputShapeAsInputShapeHelper>)                                   \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("ScatterNdAdd")                                                     \
          .Device(DEVICE_DML)                                                  \
          .TypeConstraint<type>("T")                                           \
          .TypeConstraint<int64>("Tindices"),                                  \
      DmlKernelWrapper<                                                        \
          DmlResourceScatterNDBinaryKernel<int64, std::plus<dml::Expression>>, \
          GetOutputShapeAsInputShapeHelper>)                                   \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("ScatterNdNonAliasingAdd")                                          \
          .Device(DEVICE_DML)                                                  \
          .TypeConstraint<type>("T")                                           \
          .TypeConstraint<int32>("Tindices"),                                  \
      DmlKernelWrapper<                                                        \
          DmlResourceScatterNDBinaryKernel<int32, std::plus<dml::Expression>>, \
          GetOutputShapeAsInputShapeHelper>)                                   \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("ScatterNdNonAliasingAdd")                                          \
          .Device(DEVICE_DML)                                                  \
          .TypeConstraint<type>("T")                                           \
          .TypeConstraint<int64>("Tindices"),                                  \
      DmlKernelWrapper<                                                        \
          DmlResourceScatterNDBinaryKernel<int64, std::plus<dml::Expression>>, \
          GetOutputShapeAsInputShapeHelper>)                                   \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("ScatterNdSub")                                                     \
          .Device(DEVICE_DML)                                                  \
          .TypeConstraint<type>("T")                                           \
          .TypeConstraint<int32>("Tindices"),                                  \
      DmlKernelWrapper<DmlResourceScatterNDBinaryKernel<                       \
                           int32, std::minus<dml::Expression>>,                \
                       GetOutputShapeAsInputShapeHelper>)                      \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("ScatterNdSub")                                                     \
          .Device(DEVICE_DML)                                                  \
          .TypeConstraint<type>("T")                                           \
          .TypeConstraint<int64>("Tindices"),                                  \
      DmlKernelWrapper<DmlResourceScatterNDBinaryKernel<                       \
                           int64, std::minus<dml::Expression>>,                \
                       GetOutputShapeAsInputShapeHelper>)                      \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("ResourceScatterNdAdd")                                             \
          .HostMemory("ref")                                                   \
          .Device(DEVICE_DML)                                                  \
          .TypeConstraint<type>("T")                                           \
          .TypeConstraint<int32>("Tindices"),                                  \
      DmlKernelWrapper<                                                        \
          DmlResourceScatterNDBinaryKernel<int32, std::plus<dml::Expression>>, \
          NoOutputShapeHelper, DmlKernelCachePolicy::Never>)                   \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("ResourceScatterNdAdd")                                             \
          .HostMemory("ref")                                                   \
          .Device(DEVICE_DML)                                                  \
          .TypeConstraint<type>("T")                                           \
          .TypeConstraint<int64>("Tindices"),                                  \
      DmlKernelWrapper<                                                        \
          DmlResourceScatterNDBinaryKernel<int64, std::plus<dml::Expression>>, \
          NoOutputShapeHelper, DmlKernelCachePolicy::Never>)                   \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("ResourceScatterNdSub")                                             \
          .Device(DEVICE_DML)                                                  \
          .HostMemory("ref")                                                   \
          .TypeConstraint<type>("T")                                           \
          .TypeConstraint<int32>("Tindices"),                                  \
      DmlKernelWrapper<DmlResourceScatterNDBinaryKernel<                       \
                           int32, std::minus<dml::Expression>>,                \
                       NoOutputShapeHelper, DmlKernelCachePolicy::Never>)      \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("ResourceScatterNdSub")                                             \
          .Device(DEVICE_DML)                                                  \
          .HostMemory("ref")                                                   \
          .TypeConstraint<type>("T")                                           \
          .TypeConstraint<int64>("Tindices"),                                  \
      DmlKernelWrapper<DmlResourceScatterNDBinaryKernel<                       \
                           int64, std::minus<dml::Expression>>,                \
                       NoOutputShapeHelper, DmlKernelCachePolicy::Never>)

TF_CALL_float(DML_REGISTER_KERNELS);
#undef DML_REGISTER_KERNELS

}  // namespace tensorflow
