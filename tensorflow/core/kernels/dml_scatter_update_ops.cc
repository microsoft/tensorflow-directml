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

#include "tensorflow/core/common_runtime/dml/dml_operator_helper.h"
#include "tensorflow/core/common_runtime/dml/dml_util.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/kernels/dml_kernel_wrapper.h"
#include "tensorflow/core/kernels/dml_ops_common.h"

namespace tensorflow {

// Check whether updates.shape = indices.shape + params.shape[1:]
static bool ValidShapes(const Tensor& params, const Tensor& updates,
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

static Status DoValidationChecking(const Tensor& params, const Tensor& indices,
                                   const Tensor& updates) {
  if (params.dtype() != DT_RESOURCE && !params.IsInitialized()) {
    return errors::FailedPrecondition("Null ref for params");
  }

  if (!TensorShapeUtils::IsVectorOrHigher(params.shape())) {
    return errors::InvalidArgument("params must be at least 1-D, got shape ",
                                   params.shape().DebugString());
  }

  if (!ValidShapes(params, updates, indices)) {
    return errors::InvalidArgument(
        "Must have updates.shape = indices.shape + "
        "params.shape[1:] or updates.shape = [], got ",
        "updates.shape ", updates.shape().DebugString(), ", indices.shape ",
        indices.shape().DebugString(), ", params.shape ",
        params.shape().DebugString());
  }

  return Status::OK();
}

template <typename Index>
class ScatterUpdateInitializationHelper : public InitializationHelper {
 public:
  using Attributes = EmptyAttributes;

  ScatterUpdateInitializationHelper(OpKernelContext* ctx,
                                    std::shared_ptr<const Attributes> attr) {
    if (ctx->input(0).dtype() == DT_RESOURCE) {
      OP_REQUIRES_OK(
          ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &params_resource_));
      params_resource_->mu()->lock_shared();
    }

    const Tensor& params = GetParamsTensor(ctx);
    const Tensor& indices = ctx->input(1);
    const Tensor& updates = ctx->input(2);
    OP_REQUIRES_OK(ctx, DoValidationChecking(params, indices, updates));

    // Check that we have enough index space
    const int64 N_big = indices.NumElements();
    OP_REQUIRES(
        ctx, N_big <= std::numeric_limits<Index>::max(),
        errors::InvalidArgument("indices has too many elements for ",
                                DataTypeString(DataTypeToEnum<Index>::v()),
                                " indexing: ", N_big, " > ",
                                std::numeric_limits<Index>::max()));
    const Index N = static_cast<Index>(indices.NumElements());
    OP_REQUIRES(
        ctx, params.dim_size(0) <= std::numeric_limits<Index>::max(),
        errors::InvalidArgument("params.shape[0] too large for ",
                                DataTypeString(DataTypeToEnum<Index>::v()),
                                " indexing: ", params.dim_size(0), " > ",
                                std::numeric_limits<Index>::max()));
  }

  const Tensor& GetParamsTensor(OpKernelContext* ctx) const {
    return ctx->input(0).dtype() == DT_RESOURCE ? *params_resource_->tensor()
                                                : ctx->input(0);
  }

  void Unlock() const { params_resource_->mu()->unlock_shared(); }

 private:
  core::RefCountPtr<Var> params_resource_;
};  // namespace tensorflow

struct ScatterUpdateOperation {
  static constexpr bool inplace_allowed = true;

  dml::Expression operator()(dml::Graph& scope, dml::Expression params,
                             dml::Expression indices, dml::Expression updates,
                             uint32_t scatter_axis) {
    return dml::ScatterElements(params, indices, updates, scatter_axis);
  }
};

template <typename TParams>
struct ScatterMinOperation {
  static constexpr bool inplace_allowed = false;

  dml::Expression operator()(dml::Graph& scope, dml::Expression params,
                             dml::Expression indices, dml::Expression updates,
                             uint32_t scatter_axis) {
    TParams identity_value = std::numeric_limits<TParams>::max();
    auto identity = dml::ScalarTensor<TParams>(scope, identity_value,
                                               params.GetOutputDesc().sizes);
    auto sparse_updates =
        dml::ScatterElements(identity, indices, updates, scatter_axis);

    return dml::Min(params, sparse_updates);
  }
};

template <typename TParams>
struct ScatterMaxOperation {
  static constexpr bool inplace_allowed = false;

  dml::Expression operator()(dml::Graph& scope, dml::Expression params,
                             dml::Expression indices, dml::Expression updates,
                             uint32_t scatter_axis) {
    constexpr TParams identity_value = std::numeric_limits<TParams>::min();
    auto identity = dml::ScalarTensor<TParams>(scope, identity_value,
                                               params.GetOutputDesc().sizes);
    auto sparse_updates =
        dml::ScatterElements(identity, indices, updates, scatter_axis);

    return dml::Max(params, sparse_updates);
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
}

template <typename BinaryOperation, typename TParams>
struct ScatterBinaryOperation {
  static constexpr bool inplace_allowed = false;

  dml::Expression operator()(dml::Graph& scope, dml::Expression params,
                             dml::Expression indices, dml::Expression updates,
                             uint32_t scatter_axis) {
    constexpr TParams identity_value =
        BinaryOperationIdentityValue<BinaryOperation, TParams>();
    auto identity = dml::ScalarTensor<TParams>(scope, identity_value,
                                               params.GetOutputDesc().sizes);

    auto sparse_updates =
        dml::ScatterElements(identity, indices, updates, scatter_axis);

    return BinaryOperation()(params, sparse_updates);
  }
};

template <typename Index, typename ScatterOp>
class DmlScatterUpdateKernel : public DmlKernel {
 public:
  using InitHelper = ScatterUpdateInitializationHelper<Index>;

  explicit DmlScatterUpdateKernel(DmlKernelConstruction* ctx,
                                  const InitHelper* init_helper) {
    const Tensor& params_tensor =
        init_helper->GetParamsTensor(ctx->GetOpKernelContext());

    const TensorShape& params_shape = params_tensor.shape();
    const TensorShape& indices_shape = ctx->GetInputTensorShape(1);
    const TensorShape& updates_shape = ctx->GetInputTensorShape(2);

    const TensorShape flat_params_shape({
          params_shape.dim_size(0),
          params_shape.num_elements() / params_shape.dim_size(0),
      });

    DmlKernelTensors tensors;
    {
      DmlTensorInfo input_tensor_info;
      input_tensor_info.kernel_index = 0;
      input_tensor_info.desc = DmlTensorDesc::Create(
          params_tensor.dtype(), flat_params_shape, flat_params_shape);
      tensors.inputs.push_back(std::move(input_tensor_info));
    }

    {
      const TensorShape non_broadcastflat_indices_shape({
          indices_shape.num_elements(),
          1,
      });

      const TensorShape flat_indices_shape({
          indices_shape.num_elements(),
          updates_shape.num_elements() / indices_shape.num_elements(),
      });

      DmlTensorInfo indices_tensor_info;
      indices_tensor_info.kernel_index = 1;
      indices_tensor_info.desc =
          DmlTensorDesc::Create(ctx->GetInputDataType(1), flat_indices_shape,
                                non_broadcastflat_indices_shape);
      tensors.inputs.push_back(std::move(indices_tensor_info));
    }

    {
      const TensorShape flat_updates_shape({
          indices_shape.num_elements(),
          updates_shape.num_elements() / indices_shape.num_elements(),
      });

      DmlTensorInfo updates_tensor_info;
      updates_tensor_info.kernel_index = 2;
      updates_tensor_info.desc = DmlTensorDesc::Create(
          ctx->GetInputDataType(2), flat_updates_shape, flat_updates_shape);
      tensors.inputs.push_back(std::move(updates_tensor_info));
    }

    {
      DmlTensorInfo output_tensor_info;
      output_tensor_info.kernel_index = 0;
      output_tensor_info.desc = DmlTensorDesc::Create(
          params_tensor.dtype(), params_shape, params_shape);
      tensors.outputs.push_back(std::move(output_tensor_info));
    }

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
    auto result = ScatterOp()(scope, params, indices, updates, scatter_axis);

    Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
        scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

    Initialize(ctx, std::move(tensors), compiled_op.Get());
  }

  StatusOr<DmlGpuEvent> Compute(DmlKernelContext* ctx) const override {
    auto init_helper = ctx->GetInitializationHelper<InitHelper>();

    const Tensor& params_tensor =
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
using ScatterPlusOp = ScatterBinaryOperation<std::plus<dml::Expression>, type>;

template <typename type>
using ScatterMinusOp =
    ScatterBinaryOperation<std::minus<dml::Expression>, type>;

template <typename type>
using ScatterMulOp =
    ScatterBinaryOperation<std::multiplies<dml::Expression>, type>;

template <typename type>
using ScatterDivOp =
    ScatterBinaryOperation<std::divides<dml::Expression>, type>;

#define REGISTER_DML_KERNEL(type)                                         \
  REGISTER_SCATTER_KERNEL(type, "ScatterUpdate", ScatterUpdateOperation); \
  REGISTER_SCATTER_KERNEL(type, "ScatterAdd", ScatterPlusOp<type>);       \
  REGISTER_SCATTER_KERNEL(type, "ScatterSub", ScatterMinusOp<type>);      \
  REGISTER_SCATTER_KERNEL(type, "ScatterMul", ScatterMulOp<type>);        \
  REGISTER_SCATTER_KERNEL(type, "ScatterDiv", ScatterDivOp<type>);        \
  REGISTER_SCATTER_KERNEL(type, "ScatterMin", ScatterMinOperation<type>); \
  REGISTER_SCATTER_KERNEL(type, "ScatterMax", ScatterMaxOperation<type>); \
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
                                   ScatterMinOperation<type>);            \
  REGISTER_RESOURCE_SCATTER_KERNEL(type, "ResourceScatterMax",            \
                                   ScatterMaxOperation<type>);

// We register the subset of types that the GPU device registers for these
// operators, which is why half is not included
TF_CALL_float(REGISTER_DML_KERNEL);
#undef REGISTER_DML_KERNEL
#undef REGISTER_SCATTER_KERNEL
#undef REGISTER_SCATTER_KERNEL_INDEX
#undef REGISTER_RESOURCE_SCATTER_KERNEL
#undef REGISTER_RESOURCE_SCATTER_KERNEL_INDEX

}  // namespace tensorflow
