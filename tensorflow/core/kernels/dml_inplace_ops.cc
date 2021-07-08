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

class InplaceInitializationHelper : public InitializationHelper {
 public:
  using Attributes = EmptyAttributes;

  InplaceInitializationHelper(OpKernelContext* ctx,
                              std::shared_ptr<const Attributes> attr) {
    auto x = ctx->input(0);
    auto i = ctx->input(1);
    auto v = ctx->input(2);

    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(i.shape()),
                errors::InvalidArgument("i must be a vector. ",
                                        i.shape().DebugString()));
    OP_REQUIRES(ctx, x.dims() == v.dims(),
                errors::InvalidArgument(
                    "x and v shape doesn't match (ranks differ): ",
                    x.shape().DebugString(), " vs. ", v.shape().DebugString()));
    for (int i = 1; i < x.dims(); ++i) {
      OP_REQUIRES(
          ctx, x.dim_size(i) == v.dim_size(i),
          errors::InvalidArgument("x and v shape doesn't match at index ", i,
                                  " : ", x.shape().DebugString(), " vs. ",
                                  v.shape().DebugString()));
    }
    OP_REQUIRES(ctx, i.dim_size(0) == v.dim_size(0),
                errors::InvalidArgument(
                    "i and x shape doesn't match at index 0: ",
                    i.shape().DebugString(), " vs. ", v.shape().DebugString()));
  }
};

template <typename Expression>
class DmlInplaceKernel : public DmlKernel {
 public:
  using InitHelper = InplaceInitializationHelper;

  explicit DmlInplaceKernel(DmlKernelConstruction* ctx,
                            const InitHelper* init_helper) {
    const TensorShape& input_shape = ctx->GetInputTensorShape(0);
    const TensorShape& indices_shape = ctx->GetInputTensorShape(1);
    const TensorShape& updates_shape = ctx->GetInputTensorShape(2);

    const TensorShape collapsed_input_output_shape = {
        1,
        1,
        input_shape.dim_size(0),
        input_shape.num_elements() / input_shape.dim_size(0),
    };

    const TensorShape collapsed_indices_shape = {
        1,
        1,
        indices_shape.num_elements(),
        1,
    };

    const TensorShape collapsed_updates_shape = {
        1,
        1,
        updates_shape.dim_size(0),
        updates_shape.num_elements() / updates_shape.dim_size(0),
    };

    DmlTensorInfo input_output;
    input_output.kernel_index = 0;
    input_output.desc = DmlTensorDesc::Create(ctx->GetInputDataType(0),
                                              collapsed_input_output_shape,
                                              collapsed_input_output_shape);

    DmlTensorInfo indices;
    indices.kernel_index = 1;
    indices.desc =
        DmlTensorDesc::Create(ctx->GetInputDataType(1), collapsed_updates_shape,
                              collapsed_indices_shape);

    DmlTensorInfo updates;
    updates.kernel_index = 2;
    updates.desc =
        DmlTensorDesc::Create(ctx->GetInputDataType(2), collapsed_updates_shape,
                              collapsed_updates_shape);

    DmlKernelTensors tensors;
    tensors.inputs = {input_output, indices, updates};
    tensors.outputs = {input_output};

    const auto inputs = GetDmlTensorDescs(tensors.inputs);
    auto scope = dml::Graph(ctx->GetDmlDevice());
    const auto input_tensor = dml::InputTensor(scope, 0, inputs[0]);
    const auto indices_tensor = dml::InputTensor(scope, 1, inputs[1]);
    const auto updates_tensor = dml::InputTensor(scope, 2, inputs[2]);
    auto result =
        Expression()(scope, input_tensor, indices_tensor, updates_tensor);

    // TFDML #24881131
    if (Is64BitSignedIntegerType(ctx->GetOutputDataType(0))) {
      result = dml::ConvertInt32ToInt64(result);
    }

    Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
        scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

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

    const absl::optional<DML_BUFFER_BINDING> input_bindings[3] = {
        input_buffer.GetBufferBinding(),
        indices_buffer.GetBufferBinding(),
        updates_buffer.GetBufferBinding(),
    };

    const absl::optional<DML_BUFFER_BINDING> output_bindings[1] = {
        output_buffer.GetBufferBinding(),
    };

    auto status_or_gpu_event =
        DmlKernel::Compute(ctx, input_bindings, output_bindings);

    if (!status_or_gpu_event.ok()) {
      return status_or_gpu_event;
    }

    ctx->CopyBufferToBuffer(input_buffer.Resource(), input_buffer.Offset(),
                            output_buffer.Resource(), output_buffer.Offset(),
                            input_buffer.SizeInBytes());

    return ctx->InsertUavBarrier();
  }
};

struct InplaceUpdateFunctor {
  dml::Expression operator()(dml::Graph& scope, const dml::Expression& input,
                             const dml::Expression& indices,
                             const dml::Expression& updates) {
    return dml::ScatterElements(input, indices, updates, 2);
  }
};

struct InplaceAddFunctor {
  dml::Expression operator()(dml::Graph& scope, const dml::Expression& input,
                             const dml::Expression& indices,
                             const dml::Expression& updates) {
    const auto zero = dml::ZeroTensor(scope, input.GetOutputDesc().dataType,
                                      input.GetOutputDesc().sizes);

    const auto scattered_updates =
        dml::ScatterElements(zero, indices, updates, 2);
    const auto result = input + scattered_updates;
    return result;
  }
};

struct InplaceSubFunctor {
  dml::Expression operator()(dml::Graph& scope, const dml::Expression& input,
                             const dml::Expression& indices,
                             const dml::Expression& updates) {
    const auto zero = dml::ZeroTensor(scope, input.GetOutputDesc().dataType,
                                      input.GetOutputDesc().sizes);

    const auto scattered_updates =
        dml::ScatterElements(zero, indices, updates, 2);
    const auto result = input - scattered_updates;
    return result;
  }
};

#define DML_REGISTER_KERNELS(type)                                        \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("InplaceUpdate").Device(DEVICE_DML).TypeConstraint<type>("T"), \
      DmlKernelWrapper<DmlInplaceKernel<InplaceUpdateFunctor>,            \
                       GetOutputShapeAsInputShapeHelper>)                 \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("InplaceAdd").Device(DEVICE_DML).TypeConstraint<type>("T"),    \
      DmlKernelWrapper<DmlInplaceKernel<InplaceAddFunctor>,               \
                       GetOutputShapeAsInputShapeHelper>)                 \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("InplaceSub").Device(DEVICE_DML).TypeConstraint<type>("T"),    \
      DmlKernelWrapper<DmlInplaceKernel<InplaceSubFunctor>,               \
                       GetOutputShapeAsInputShapeHelper>)

TF_CALL_float(DML_REGISTER_KERNELS);
TF_CALL_half(DML_REGISTER_KERNELS);
TF_CALL_int64(DML_REGISTER_KERNELS);
#undef DML_REGISTER_KERNELS

}  // namespace tensorflow
