/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.
Portions Copyright (ctx) Microsoft Corporation.

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

#include <numeric>

#include "tensorflow/core/common_runtime/dml/dml_operator_helper.h"
#include "tensorflow/core/common_runtime/dml/dml_util.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/kernels/dml_kernel_wrapper.h"
#include "tensorflow/core/kernels/dml_ops_common.h"

namespace tensorflow {

// Check whether updates.shape = indices.shape + params.shape[1:]
static bool ValidRefShapes(const Tensor& params, const Tensor& updates,
                           const Tensor& indices) {
  if (updates.dims() == 0) return true;
  if (updates.dims() != indices.dims() + params.dims() - 1) return false;
  for (int d = 0; d < indices.dims(); d++) {
    if (updates.dim_size(d) != indices.dim_size(d)) {
      return false;
    }
  }
  for (int d = 1; d < params.dims(); d++) {
    if (params.dim_size(d) != updates.dim_size(d - 1 + indices.dims())) {
      return false;
    }
  }
  return true;
}

static Status ValidateRefScatter(const Tensor& params, const Tensor& indices,
                                 const Tensor& updates) {
  if (!params.IsInitialized()) {
    return errors::FailedPrecondition("Null ref for params");
  }

  if (!TensorShapeUtils::IsVectorOrHigher(params.shape())) {
    return errors::InvalidArgument("params must be at least 1-D, got shape ",
                                   params.shape().DebugString());
  }

  if (!ValidRefShapes(params, updates, indices)) {
    return errors::InvalidArgument(
        "Must have updates.shape = indices.shape + "
        "params.shape[1:] or updates.shape = [], got ",
        "updates.shape ", updates.shape().DebugString(), ", indices.shape ",
        indices.shape().DebugString(), ", params.shape ",
        params.shape().DebugString());
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
class ScatterUpdateInitializationHelper : public InitializationHelper {
 public:
  using Attributes = EmptyAttributes;

  ScatterUpdateInitializationHelper(OpKernelContext* ctx,
                                    std::shared_ptr<const Attributes> attr) {
    DCHECK(ctx->input_is_ref(0) || ctx->input(0).dtype() == DT_RESOURCE);

    if (!ctx->input_is_ref(0)) {
      OP_REQUIRES_OK(
          ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &params_resource_));
      params_resource_->mu()->lock_shared();
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
    if (params_resource_) {
      params_resource_->mu()->unlock_shared();
    }
  }

 private:
  core::RefCountPtr<Var> params_resource_;
};  // namespace tensorflow

struct ScatterUpdateOperation {
  static constexpr bool inplace_allowed = true;

  dml::Expression operator()(dml::Graph& scope, dml::Expression params,
                             dml::Expression indices, dml::Expression updates,
                             uint32_t scatter_axis, bool int64_indices,
                             bool scalar_updates) {
    return dml::ScatterElements(params, indices, updates, scatter_axis);
  }
};

struct BinaryMinOperation {
  dml::Expression operator()(dml::Expression a, dml::Expression b) {
    return dml::Min(a, b);
  }
};

struct BinaryMaxOperation {
  dml::Expression operator()(dml::Expression a, dml::Expression b) {
    return dml::Max(a, b);
  }
};

template <typename BinaryOperation, typename TParams>
static constexpr TParams BinaryOperationIdentityValue() {
  if (std::is_same<BinaryOperation, std::multiplies<dml::Expression>>::value) {
    return static_cast<TParams>(1);
  }

  if (std::is_same<BinaryOperation, std::divides<dml::Expression>>::value) {
    return static_cast<TParams>(1);
  }

  if (std::is_same<BinaryOperation, std::plus<dml::Expression>>::value) {
    return static_cast<TParams>(0);
  }

  if (std::is_same<BinaryOperation, std::minus<dml::Expression>>::value) {
    return static_cast<TParams>(0);
  }

  if (std::is_same<BinaryOperation, BinaryMinOperation>::value) {
    return std::numeric_limits<TParams>::max();
  }

  if (std::is_same<BinaryOperation, BinaryMaxOperation>::value) {
    return std::numeric_limits<TParams>::lowest();
  }
}

// For arithmetic Scatter operations, TensorFlow supports duplicate indices so
// we can't use DirectML's Scatter. For now, we can use the graph as a
// workaround but we should revisit it in the future and add a DirectML API if
// we get signals that this implementation is a bottleneck.
template <typename BinaryOperation, DML_REDUCE_FUNCTION reduce_function,
          typename TParams>
struct ScatterBinaryOperation {
  static constexpr bool inplace_allowed = false;

  dml::Expression operator()(dml::Graph& scope, dml::Expression params,
                             dml::Expression indices, dml::Expression updates,
                             uint32_t scatter_axis, bool int64_indices,
                             bool scalar_updates) {
    auto params_sizes = params.GetOutputDesc().sizes;
    uint32_t row_count = params_sizes[scatter_axis];

    dml::TensorDesc::Dimensions row_indices_sizes({1, 1, row_count, 1});

    auto row_indices = dml::FillValueSequence(
        scope, row_indices_sizes, indices.GetOutputDesc().dataType,
        dml::ScalarUnion(0, indices.GetOutputDesc().dataType),
        dml::ScalarUnion(1, indices.GetOutputDesc().dataType));

    auto indices_sizes = indices.GetOutputDesc().sizes;
    dml::TensorDesc::Dimensions broadcasted_sizes({
        1,
        indices_sizes[2],
        row_count,
        params_sizes[3],
    });

    auto broadcasted_row_indices =
        dml::Reinterpret(row_indices, broadcasted_sizes,
                         dml::TensorDesc::Dimensions({0, 0, 1, 0}));

    uint32_t indices_stride_multiplier = int64_indices ? 2 : 1;
    auto broadcasted_indices = dml::Reinterpret(
        indices, broadcasted_sizes,
        dml::TensorDesc::Dimensions({0, indices_stride_multiplier, 0, 0}));

    dml::Expression broadcasted_updates =
        scalar_updates
            ? dml::Reinterpret(updates, broadcasted_sizes,
                               dml::TensorDesc::Dimensions({0, 0, 0, 0}))
            : dml::Reinterpret(
                  updates, broadcasted_sizes,
                  dml::TensorDesc::Dimensions({0, indices_sizes[3], 0, 1}));

    constexpr TParams identity_value =
        BinaryOperationIdentityValue<BinaryOperation, TParams>();

    auto identity =
        dml::ScalarTensor<TParams>(scope, identity_value, broadcasted_sizes);

    auto sparse_updates =
        dml::If(broadcasted_indices == broadcasted_row_indices,
                broadcasted_updates, identity);

    auto reduced_updates = dml::Reduce(sparse_updates, reduce_function, {1});
    auto result = BinaryOperation()(params, reduced_updates);

    return result;
  }
};

template <typename Index, typename ScatterOp>
class DmlScatterUpdateKernel : public DmlKernel {
 public:
  using InitHelper = ScatterUpdateInitializationHelper<Index>;

  explicit DmlScatterUpdateKernel(DmlKernelConstruction* ctx,
                                  const InitHelper* init_helper) {
    const Tensor params_tensor =
        init_helper->GetParamsTensor(ctx->GetOpKernelContext());

    const TensorShape& params_shape = params_tensor.shape();
    const TensorShape& indices_shape = ctx->GetInputTensorShape(1);
    const TensorShape& updates_shape = ctx->GetInputTensorShape(2);
    bool scalar_updates = TensorShapeUtils::IsScalar(updates_shape);

    const TensorShape flat_params_shape({
        params_shape.dim_size(0),
        params_shape.num_elements() / params_shape.dim_size(0),
    });

    const TensorShape flat_indices_shape({
        indices_shape.num_elements(),
        params_shape.num_elements() / params_shape.dim_size(0),
    });

    const TensorShape non_broadcast_flat_indices_shape({
        indices_shape.num_elements(),
        1,
    });

    const TensorShape flat_updates_shape({
        indices_shape.num_elements(),
        params_shape.num_elements() / params_shape.dim_size(0),
    });

    const TensorShape& non_broadcast_flat_updates_shape =
        scalar_updates ? updates_shape : flat_updates_shape;

    DmlTensorInfo input_tensor_info;
    input_tensor_info.kernel_index = 0;
    input_tensor_info.desc = DmlTensorDesc::Create(
        params_tensor.dtype(), flat_params_shape, flat_params_shape);

    DmlTensorInfo indices_tensor_info;
    indices_tensor_info.kernel_index = 1;
    indices_tensor_info.desc =
        DmlTensorDesc::Create(ctx->GetInputDataType(1), flat_indices_shape,
                              non_broadcast_flat_indices_shape);

    DmlTensorInfo updates_tensor_info;
    updates_tensor_info.kernel_index = 2;
    updates_tensor_info.desc =
        DmlTensorDesc::Create(ctx->GetInputDataType(2), flat_updates_shape,
                              non_broadcast_flat_updates_shape);

    DmlTensorInfo output_tensor_info;
    output_tensor_info.kernel_index = 0;
    output_tensor_info.desc = DmlTensorDesc::Create(params_tensor.dtype(),
                                                    params_shape, params_shape);

    DmlKernelTensors tensors;
    tensors.inputs = {
        input_tensor_info,
        indices_tensor_info,
        updates_tensor_info,
    };

    tensors.outputs = {output_tensor_info};

    if (params_tensor.dtype() != DT_RESOURCE) {
      // The input ref and the output ref must refer to the same memory
      tensors.output_refs_forwarding = {0};
    }

    auto inputs = GetDmlTensorDescs(tensors.inputs);
    auto scope = dml::Graph(ctx->GetDmlDevice());
    auto params = dml::InputTensor(scope, 0, inputs[0]);
    auto indices = dml::InputTensor(scope, 1, inputs[1]);
    auto updates = dml::InputTensor(scope, 2, inputs[2]);

    const uint32_t scatter_axis =
        params.GetOutputDesc().sizes.size() - flat_params_shape.dims();
    auto result = ScatterOp()(scope, params, indices, updates, scatter_axis,
                              Is64BitIntegerType(ctx->GetInputDataType(1)),
                              scalar_updates);

    Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
        scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

    Initialize(ctx, std::move(tensors), compiled_op.Get());
  }

  StatusOr<DmlGpuEvent> Compute(DmlKernelContext* ctx) const override {
    auto init_helper = ctx->GetInitializationHelper<InitHelper>();

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

    if (ScatterOp::inplace_allowed) {
      absl::optional<DML_BUFFER_BINDING> output_bindings[] = {
          input_bindings[0],
      };

      gpu_event =
          ctx->ExecuteOperator(GetCompiledOp(), GetPersistentResourceBinding(),
                               input_bindings, output_bindings);
    } else {
      DmlBuffer output_buffer =
          ctx->AllocateDefaultBuffer(input_buffers[0].SizeInBytes());

      absl::optional<DML_BUFFER_BINDING> output_bindings[] = {
          output_buffer.GetBufferBinding(),
      };

      ctx->ExecuteOperator(GetCompiledOp(), GetPersistentResourceBinding(),
                           input_bindings, output_bindings);

      ctx->CopyBufferToBuffer(input_buffers[0].Resource(),
                              input_buffers[0].Offset(),
                              output_buffer.Resource(), output_buffer.Offset(),
                              output_buffer.SizeInBytes());

      gpu_event = ctx->InsertUavBarrier();
    }

    init_helper->Unlock();
    return gpu_event;
  }
};

#define REGISTER_SCATTER_KERNEL_INDEX(type, name, op, index_type) \
  REGISTER_KERNEL_BUILDER(                                        \
      Name(name)                                                  \
          .Device(DEVICE_DML)                                     \
          .TypeConstraint<type>("T")                              \
          .TypeConstraint<index_type>("Tindices"),                \
      DmlKernelWrapper<DmlScatterUpdateKernel<index_type, op>,    \
                       GetOutputShapeAsInputShapeHelper>)

#define REGISTER_SCATTER_KERNEL(type, name, op)         \
  REGISTER_SCATTER_KERNEL_INDEX(type, name, op, int32); \
  REGISTER_SCATTER_KERNEL_INDEX(type, name, op, int64);

#define REGISTER_RESOURCE_SCATTER_KERNEL_INDEX(type, name, op, index_type) \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name(name)                                                           \
          .Device(DEVICE_DML)                                              \
          .HostMemory("resource")                                          \
          .TypeConstraint<type>("dtype")                                   \
          .TypeConstraint<index_type>("Tindices"),                         \
      DmlKernelWrapper<DmlScatterUpdateKernel<index_type, op>,             \
                       NoOutputShapeHelper, DmlKernelCachePolicy::Never>)

#define REGISTER_RESOURCE_SCATTER_KERNEL(type, name, op)         \
  REGISTER_RESOURCE_SCATTER_KERNEL_INDEX(type, name, op, int32); \
  REGISTER_RESOURCE_SCATTER_KERNEL_INDEX(type, name, op, int64);

template <typename type>
using ScatterPlusOp = ScatterBinaryOperation<std::plus<dml::Expression>,
                                             DML_REDUCE_FUNCTION_SUM, type>;

template <typename type>
using ScatterMinusOp = ScatterBinaryOperation<std::minus<dml::Expression>,
                                              DML_REDUCE_FUNCTION_SUM, type>;

template <typename type>
using ScatterMulOp = ScatterBinaryOperation<std::multiplies<dml::Expression>,
                                            DML_REDUCE_FUNCTION_MULTIPLY, type>;

template <typename type>
using ScatterDivOp = ScatterBinaryOperation<std::divides<dml::Expression>,
                                            DML_REDUCE_FUNCTION_MULTIPLY, type>;

template <typename type>
using ScatterMinOp =
    ScatterBinaryOperation<BinaryMinOperation, DML_REDUCE_FUNCTION_MIN, type>;

template <typename type>
using ScatterMaxOp =
    ScatterBinaryOperation<BinaryMaxOperation, DML_REDUCE_FUNCTION_MAX, type>;

#define REGISTER_DML_KERNEL(type)                                         \
  REGISTER_SCATTER_KERNEL(type, "ScatterUpdate", ScatterUpdateOperation); \
  REGISTER_SCATTER_KERNEL(type, "ScatterAdd", ScatterPlusOp<type>);       \
  REGISTER_SCATTER_KERNEL(type, "ScatterSub", ScatterMinusOp<type>);      \
  REGISTER_SCATTER_KERNEL(type, "ScatterMul", ScatterMulOp<type>);        \
  REGISTER_SCATTER_KERNEL(type, "ScatterDiv", ScatterDivOp<type>);        \
  REGISTER_SCATTER_KERNEL(type, "ScatterMin", ScatterMinOp<type>);        \
  REGISTER_SCATTER_KERNEL(type, "ScatterMax", ScatterMaxOp<type>);        \
  REGISTER_RESOURCE_SCATTER_KERNEL(type, "ResourceScatterUpdate",         \
                                   ScatterUpdateOperation);               \
  REGISTER_RESOURCE_SCATTER_KERNEL(type, "ResourceScatterAdd",            \
                                   ScatterPlusOp<type>);                  \
  REGISTER_RESOURCE_SCATTER_KERNEL(type, "ResourceScatterSub",            \
                                   ScatterMinusOp<type>);                 \
  REGISTER_RESOURCE_SCATTER_KERNEL(type, "ResourceScatterMul",            \
                                   ScatterMulOp<type>);                   \
  REGISTER_RESOURCE_SCATTER_KERNEL(type, "ResourceScatterDiv",            \
                                   ScatterDivOp<type>);                   \
  REGISTER_RESOURCE_SCATTER_KERNEL(type, "ResourceScatterMin",            \
                                   ScatterMinOp<type>);                   \
  REGISTER_RESOURCE_SCATTER_KERNEL(type, "ResourceScatterMax",            \
                                   ScatterMaxOp<type>);

// We register the subset of types that the GPU device registers for these
// operators, which is why half is not included
TF_CALL_float(REGISTER_DML_KERNEL);
#undef REGISTER_DML_KERNEL
#undef REGISTER_SCATTER_KERNEL
#undef REGISTER_SCATTER_KERNEL_INDEX
#undef REGISTER_RESOURCE_SCATTER_KERNEL
#undef REGISTER_RESOURCE_SCATTER_KERNEL_INDEX

}  // namespace tensorflow
