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
static Status ValidateCommonScatter(const Tensor& params, const Tensor& indices,
                                    const Tensor& updates) {
  if (!TensorShapeUtils::IsVectorOrHigher(params.shape())) {
    return errors::InvalidArgument("Output must be at least 1-D, ",
                                   "got shape: ", params.shape().DebugString());
  }

  if (!ValidEmptyOutputShape(params.NumElements(), indices.NumElements(),
                             updates.NumElements())) {
    return errors::InvalidArgument(
        "Indices and updates specified for empty output.  indices shape: ",
        indices.shape().DebugString());
  }

  if (updates.dim_size(0) != indices.dim_size(0)) {
    return errors::InvalidArgument(
        "The outermost dimension of updates and indices ",
        "must match. Got indices.shape ", indices.shape().DebugString(),
        ", updates.shape ", updates.shape().DebugString());
  }

  TF_RETURN_IF_ERROR(ValidateUpdateShape(params.shape(), indices, updates));

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
    if (ctx->input_is_ref(0) || ctx->input(0).dtype() == DT_RESOURCE) {
      isTensorInput_ = false;
    }

    if (!ctx->input_is_ref(0) && ctx->input(0).dtype() == DT_RESOURCE) {
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

    OP_REQUIRES_OK(ctx, ValidateCommonScatter<Index>(params, indices, updates));
  }

  Tensor GetParamsTensor(OpKernelContext* ctx) const {
    if (isTensorInput_) {
      return ctx->input(0);
    } else {
      return params_resource_ ? *params_resource_->tensor()
                              : ctx->mutable_input(0, false);
    }
  }

  void Unlock() const {
    if (params_resource_ && locked_) {
      params_resource_->mu()->unlock_shared();
      locked_ = false;
    }
  }

  bool IsTensorInput() const { return isTensorInput_; }

  virtual ~ResourceScatterNDInitHelper() { Unlock(); }

 private:
  core::RefCountPtr<Var> params_resource_;
  mutable bool locked_ = false;
  mutable bool isTensorInput_ = true;
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

    if (ctx->GetOpKernelContext()->input_is_ref(0)) {
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

// For arithmetic ScatterNd operations, TensorFlow supports duplicate indices so
// we can't use DirectML's ScatterNd. For now, we can use the graph as a
// workaround but we should revisit it in the future and add a DirectML API if
// we get signals that this implementation is a bottleneck.
template <typename BinaryOperation, DML_REDUCE_FUNCTION reduce_function,
          typename TParams>
struct ScatterNdBinaryOperation {
  dml::Expression operator()(dml::Graph& scope, dml::Expression params,
                             dml::Expression indices, dml::Expression updates,
                             dml::Expression strides, bool int64_indices) {
    // First, compute the 1D version of the indices as if we were indexing into
    // a 1D array
    const auto broadcasted_strides = dml::Reinterpret(
        strides, indices.GetOutputDesc().sizes,
        dml::TensorDesc::Dimensions({0, 0, 0, int64_indices ? 2u : 1u}));

    const auto global_indices = dml::Reduce(indices * broadcasted_strides,
                                            DML_REDUCE_FUNCTION_SUM, {3});

    const auto params_sizes = params.GetOutputDesc().sizes;
    const uint32_t row_count = params_sizes[2];
    const dml::TensorDesc::Dimensions row_indices_sizes({1, 1, row_count, 1});

    const auto indices_dtype = global_indices.GetOutputDesc().dataType;

    const auto row_indices = dml::FillValueSequence(
        scope, row_indices_sizes, indices_dtype,
        dml::ScalarUnion(0, indices_dtype), dml::ScalarUnion(1, indices_dtype));

    const auto indices_sizes = indices.GetOutputDesc().sizes;
    const dml::TensorDesc::Dimensions broadcasted_sizes({
        1,
        indices_sizes[2],
        row_count,
        params_sizes[3],
    });

    const auto broadcasted_row_indices =
        dml::Reinterpret(row_indices, broadcasted_sizes,
                         dml::TensorDesc::Dimensions({0, 0, 1, 0}));

    const auto broadcasted_indices =
        dml::Reinterpret(global_indices, broadcasted_sizes,
                         dml::TensorDesc::Dimensions({0, 1, 0, 0}));

    const auto updates_sizes = updates.GetOutputDesc().sizes;
    const auto broadcasted_updates = dml::Reinterpret(
        updates, broadcasted_sizes,
        dml::TensorDesc::Dimensions({0, updates_sizes[3], 0, 1}));

    const auto identity =
        dml::ScalarTensor<TParams>(scope, TParams(0), broadcasted_sizes);

    const auto sparse_updates =
        dml::If(broadcasted_indices == broadcasted_row_indices,
                broadcasted_updates, identity);

    const auto reduced_updates =
        dml::Reduce(sparse_updates, reduce_function, {1});

    const auto result = BinaryOperation()(params, reduced_updates);

    return result;
  }
};

template <typename Index, typename BinaryOp>
class DmlResourceScatterNDBinaryKernel : public DmlKernel {
  absl::optional<DmlBuffer> strides_buffer_;

 public:
  using InitHelper = ResourceScatterNDInitHelper<Index>;

  DmlResourceScatterNDBinaryKernel(DmlKernelConstruction* ctx,
                                   const InitHelper* init_helper) {
    const Tensor params_tensor =
        init_helper->GetParamsTensor(ctx->GetOpKernelContext());

    const TensorShape& in_out_shape = params_tensor.shape();
    const TensorShape& indices_shape = ctx->GetInputTensorShape(1);
    const TensorShape& updates_shape = ctx->GetInputTensorShape(2);

    const int64 indices_last_dim =
        indices_shape.dim_size(indices_shape.dims() - 1);

    const TensorShape flat_indices_shape = {
        indices_shape.num_elements() / indices_last_dim,
        indices_last_dim,
    };

    const int64 slice_dim =
        (indices_shape.dims() > 1)
            ? indices_shape.dim_size(indices_shape.dims() - 1)
            : 1;

    int64 slice_size = 1;
    for (int64 i = slice_dim; i < in_out_shape.dims(); ++i) {
      slice_size *= in_out_shape.dim_size(i);
    }

    const int64 safe_slice_dim = (slice_dim < 1) ? 1 : slice_dim;
    const int64 num_updates = indices_shape.num_elements() / safe_slice_dim;

    const TensorShape flat_updates_shape = {
        num_updates,
        slice_size,
    };

    const TensorShape flat_in_out_shape = {
        in_out_shape.num_elements() / slice_size,
        slice_size,
    };

    const TensorShape strides_shape = {indices_last_dim};

    const DataType indices_dtype = ctx->GetInputDataType(1);

    DmlTensorInfo in_out_tensor;
    in_out_tensor.desc = DmlTensorDesc::Create(
        params_tensor.dtype(), flat_in_out_shape, flat_in_out_shape);

    DmlTensorInfo indices_tensor;
    indices_tensor.desc = DmlTensorDesc::Create(
        indices_dtype, flat_indices_shape, flat_indices_shape);

    DmlTensorInfo updates_tensor;
    updates_tensor.desc = DmlTensorDesc::Create(
        ctx->GetInputDataType(2), flat_updates_shape, flat_updates_shape);

    DmlTensorInfo strides_tensor;
    strides_tensor.desc =
        DmlTensorDesc::Create(indices_dtype, strides_shape, strides_shape);

    DmlKernelTensors tensors;
    tensors.inputs = {in_out_tensor, indices_tensor, updates_tensor,
                      strides_tensor};
    tensors.outputs = {in_out_tensor};

    if (ctx->GetOpKernelContext()->input_is_ref(0)) {
      // The input ref and the output ref must refer to the same memory
      tensors.output_refs_forwarding = {0};
    }

    auto inputs = GetDmlTensorDescs(tensors.inputs);
    auto graph = dml::Graph(ctx->GetDmlDevice());
    auto input = dml::InputTensor(graph, 0, inputs[0]);
    auto indices = dml::InputTensor(graph, 1, inputs[1]);
    auto updates = dml::InputTensor(graph, 2, inputs[2]);
    auto strides = dml::InputTensor(graph, 3, inputs[3]);
    auto result = BinaryOp()(graph, input, indices, updates, strides,
                             Is64BitIntegerType(indices_dtype));

    const uint32_t buffer_size = indices_last_dim * DataTypeSize(indices_dtype);
    strides_buffer_ = ctx->AllocateDefaultBuffer(buffer_size);

    OP_REQUIRES(ctx->GetOpKernelContext(), strides_buffer_,
                errors::ResourceExhausted("OOM when allocating a buffer of ",
                                          buffer_size, " bytes"));

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

    const Tensor indices_tensor = ctx->GetInputTensor(1);

    const int64 indices_last_dim =
        indices_tensor.dim_size(indices_tensor.dims() - 1);

    absl::InlinedVector<Index, 8> strides(indices_last_dim);
    Index stride = 1;

    for (int i = indices_last_dim - 1; i >= 0; --i) {
      strides[i] = stride;
      stride *= params_tensor.dim_size(i);
    }

    auto byte_ptr = reinterpret_cast<const uint8_t*>(strides.data());
    auto byte_span = absl::MakeSpan(byte_ptr, strides.size() * sizeof(Index));

    const auto status_or_event = ctx->CopyHostToBuffer(
        strides_buffer_->Resource(), strides_buffer_->Offset(), byte_span);

    TF_RETURN_IF_ERROR(status_or_event.status());

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
        strides_buffer_->GetBufferBinding(),
    };

    DmlGpuEvent gpu_event;
    absl::InlinedVector<absl::optional<DML_BUFFER_BINDING>, 1> output_bindings;
    const bool isTensorInput = init_helper->IsTensorInput();
    if (isTensorInput) {
      D3D12BufferRegion output_buffer =
          ctx->CreateBufferForTensor(*ctx->GetOutputTensor(0));
      output_bindings.push_back(output_buffer.GetBufferBinding());

      auto status_or_event =
          DmlKernel::Compute(ctx, input_bindings, output_bindings);
      if (!status_or_event.ok()) {
        return status_or_event;
      }

    } else {
      DmlBuffer output_buffer =
          ctx->AllocateDefaultBuffer(input_buffers[0].SizeInBytes());
      output_bindings.push_back(output_buffer.GetBufferBinding());

      auto status_or_event =
          DmlKernel::Compute(ctx, input_bindings, output_bindings);
      if (!status_or_event.ok()) {
        return status_or_event;
      }
      ctx->CopyBufferToBuffer(input_buffers[0].Resource(),
                              input_buffers[0].Offset(),
                              output_buffer.Resource(), output_buffer.Offset(),
                              output_buffer.SizeInBytes());
    }
    gpu_event = ctx->InsertUavBarrier();
    return gpu_event;
  }
};

template <typename type>
using ScatterNdPlusOp = ScatterNdBinaryOperation<std::plus<dml::Expression>,
                                                 DML_REDUCE_FUNCTION_SUM, type>;

template <typename type>
using ScatterNdMinusOp =
    ScatterNdBinaryOperation<std::minus<dml::Expression>,
                             DML_REDUCE_FUNCTION_SUM, type>;

#define DML_REGISTER_KERNELS(type)                                         \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("ScatterNdUpdate")                                              \
          .Device(DEVICE_DML)                                              \
          .TypeConstraint<type>("T")                                       \
          .TypeConstraint<int32>("Tindices"),                              \
      DmlKernelWrapper<DmlResourceScatterNDUpdateKernel<int32>,            \
                       GetOutputShapeAsInputShapeHelper>)                  \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("ScatterNdUpdate")                                              \
          .Device(DEVICE_DML)                                              \
          .TypeConstraint<type>("T")                                       \
          .TypeConstraint<int64>("Tindices"),                              \
      DmlKernelWrapper<DmlResourceScatterNDUpdateKernel<int64>,            \
                       GetOutputShapeAsInputShapeHelper>)                  \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("ResourceScatterNdUpdate")                                      \
          .Device(DEVICE_DML)                                              \
          .HostMemory("ref")                                               \
          .TypeConstraint<type>("T")                                       \
          .TypeConstraint<int32>("Tindices"),                              \
      DmlKernelWrapper<DmlResourceScatterNDUpdateKernel<int32>,            \
                       NoOutputShapeHelper, DmlKernelCachePolicy::Never>)  \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("ResourceScatterNdUpdate")                                      \
          .Device(DEVICE_DML)                                              \
          .HostMemory("ref")                                               \
          .TypeConstraint<type>("T")                                       \
          .TypeConstraint<int64>("Tindices"),                              \
      DmlKernelWrapper<DmlResourceScatterNDUpdateKernel<int64>,            \
                       NoOutputShapeHelper, DmlKernelCachePolicy::Never>)  \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("ScatterNdAdd")                                                 \
          .Device(DEVICE_DML)                                              \
          .TypeConstraint<type>("T")                                       \
          .TypeConstraint<int32>("Tindices"),                              \
      DmlKernelWrapper<                                                    \
          DmlResourceScatterNDBinaryKernel<int32, ScatterNdPlusOp<type>>,  \
          GetOutputShapeAsInputShapeHelper>)                               \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("ScatterNdAdd")                                                 \
          .Device(DEVICE_DML)                                              \
          .TypeConstraint<type>("T")                                       \
          .TypeConstraint<int64>("Tindices"),                              \
      DmlKernelWrapper<                                                    \
          DmlResourceScatterNDBinaryKernel<int64, ScatterNdPlusOp<type>>,  \
          GetOutputShapeAsInputShapeHelper>)                               \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("ScatterNdNonAliasingAdd")                                      \
          .Device(DEVICE_DML)                                              \
          .TypeConstraint<type>("T")                                       \
          .TypeConstraint<int32>("Tindices"),                              \
      DmlKernelWrapper<                                                    \
          DmlResourceScatterNDBinaryKernel<int32, ScatterNdPlusOp<type>>,  \
          GetOutputShapeAsInputShapeHelper>)                               \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("ScatterNdNonAliasingAdd")                                      \
          .Device(DEVICE_DML)                                              \
          .TypeConstraint<type>("T")                                       \
          .TypeConstraint<int64>("Tindices"),                              \
      DmlKernelWrapper<                                                    \
          DmlResourceScatterNDBinaryKernel<int64, ScatterNdPlusOp<type>>,  \
          GetOutputShapeAsInputShapeHelper>)                               \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("ScatterNdSub")                                                 \
          .Device(DEVICE_DML)                                              \
          .TypeConstraint<type>("T")                                       \
          .TypeConstraint<int32>("Tindices"),                              \
      DmlKernelWrapper<                                                    \
          DmlResourceScatterNDBinaryKernel<int32, ScatterNdMinusOp<type>>, \
          GetOutputShapeAsInputShapeHelper>)                               \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("ScatterNdSub")                                                 \
          .Device(DEVICE_DML)                                              \
          .TypeConstraint<type>("T")                                       \
          .TypeConstraint<int64>("Tindices"),                              \
      DmlKernelWrapper<                                                    \
          DmlResourceScatterNDBinaryKernel<int64, ScatterNdMinusOp<type>>, \
          GetOutputShapeAsInputShapeHelper>)                               \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("ResourceScatterNdAdd")                                         \
          .HostMemory("ref")                                               \
          .Device(DEVICE_DML)                                              \
          .TypeConstraint<type>("T")                                       \
          .TypeConstraint<int32>("Tindices"),                              \
      DmlKernelWrapper<                                                    \
          DmlResourceScatterNDBinaryKernel<int32, ScatterNdPlusOp<type>>,  \
          NoOutputShapeHelper, DmlKernelCachePolicy::Never>)               \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("ResourceScatterNdAdd")                                         \
          .HostMemory("ref")                                               \
          .Device(DEVICE_DML)                                              \
          .TypeConstraint<type>("T")                                       \
          .TypeConstraint<int64>("Tindices"),                              \
      DmlKernelWrapper<                                                    \
          DmlResourceScatterNDBinaryKernel<int64, ScatterNdPlusOp<type>>,  \
          NoOutputShapeHelper, DmlKernelCachePolicy::Never>)               \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("ResourceScatterNdSub")                                         \
          .Device(DEVICE_DML)                                              \
          .HostMemory("ref")                                               \
          .TypeConstraint<type>("T")                                       \
          .TypeConstraint<int32>("Tindices"),                              \
      DmlKernelWrapper<                                                    \
          DmlResourceScatterNDBinaryKernel<int32, ScatterNdMinusOp<type>>, \
          NoOutputShapeHelper, DmlKernelCachePolicy::Never>)               \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("ResourceScatterNdSub")                                         \
          .Device(DEVICE_DML)                                              \
          .HostMemory("ref")                                               \
          .TypeConstraint<type>("T")                                       \
          .TypeConstraint<int64>("Tindices"),                              \
      DmlKernelWrapper<                                                    \
          DmlResourceScatterNDBinaryKernel<int64, ScatterNdMinusOp<type>>, \
          NoOutputShapeHelper, DmlKernelCachePolicy::Never>)

TF_CALL_int32(DML_REGISTER_KERNELS);
TF_CALL_half(DML_REGISTER_KERNELS);
TF_CALL_float(DML_REGISTER_KERNELS);
#undef DML_REGISTER_KERNELS

}  // namespace tensorflow
