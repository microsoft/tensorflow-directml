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

struct SimpleTernary {
  dml::TensorDesc::Dimensions cond_shape;
  dml::TensorDesc::Dimensions then_shape;
  dml::TensorDesc::Dimensions else_shape;
  dml::TensorDesc::Dimensions output_shape;
};

// Attempts to coalesce dimensions that are not broadcast to work with DML's 4D
// tensor limitation. Requires all shapes to have the same rank.
absl::optional<SimpleTernary> SimplifyTernary(const TensorShape& cond_shape,
                                              const TensorShape& then_shape,
                                              const TensorShape& else_shape,
                                              const TensorShape& output_shape,
                                              uint32_t output_size = 4) {
  DCHECK(cond_shape.dims() == then_shape.dims());
  DCHECK(cond_shape.dims() == else_shape.dims());
  DCHECK(cond_shape.dims() == output_shape.dims());

  SimpleTernary simplified = {};
  simplified.cond_shape.resize(output_size, 1);
  simplified.then_shape.resize(output_size, 1);
  simplified.else_shape.resize(output_size, 1);
  simplified.output_shape.resize(output_size, 1);

  uint32_t coalesced = 1;
  int write_dim = output_size - 1;

  // This requires all shapes have the same rank. A dimension may be coalesced
  // in all shapes if no shape is broadcast for that dimension; by definition,
  // this means all shapes have the same size for that dimension. Build up the
  // coalesced size until a dimension is reached that cannot be coalesced, then
  // write out the coalesced dimension and the non-coalesced dimension.
  for (int read_dim = output_shape.dims() - 1; read_dim >= 0; read_dim--) {
    auto cond_size = static_cast<uint32_t>(cond_shape.dim_size(read_dim));
    auto then_size = static_cast<uint32_t>(then_shape.dim_size(read_dim));
    auto else_size = static_cast<uint32_t>(else_shape.dim_size(read_dim));
    auto output_size = static_cast<uint32_t>(output_shape.dim_size(read_dim));

    if (output_size == cond_size && output_size == then_size &&
        output_size == else_size) {
      coalesced *= output_size;
    } else {
      if (coalesced > 1) {
        if (write_dim < 0) {
          return absl::nullopt;
        }
        simplified.cond_shape[write_dim] = coalesced;
        simplified.then_shape[write_dim] = coalesced;
        simplified.else_shape[write_dim] = coalesced;
        simplified.output_shape[write_dim] = coalesced;
        coalesced = 1;
        write_dim--;
      }
      if (write_dim < 0) {
        return absl::nullopt;
      }
      simplified.cond_shape[write_dim] = cond_size;
      simplified.then_shape[write_dim] = then_size;
      simplified.else_shape[write_dim] = else_size;
      simplified.output_shape[write_dim] = output_size;
      write_dim--;
    }
  }

  if (coalesced > 1) {
    if (write_dim < 0) {
      return absl::nullopt;
    }
    simplified.cond_shape[write_dim] = coalesced;
    simplified.then_shape[write_dim] = coalesced;
    simplified.else_shape[write_dim] = coalesced;
    simplified.output_shape[write_dim] = coalesced;
  }

  // TODO: TFDML #25934615
  // Consider improving to handle groups of adjacent dims that can be coaleseced
  // because their products are equal or 1.

  return simplified;
}

class TernaryInitHelper : public InitializationHelper {
 public:
  using Attributes = EmptyAttributes;

  TernaryInitHelper(OpKernelContext* ctx,
                    std::shared_ptr<const Attributes> attr) {
    TensorShape cond_shape = ctx->input(0).shape();
    TensorShape then_shape = ctx->input(1).shape();
    TensorShape else_shape = ctx->input(2).shape();

    // Select has a special broadcasting rule when the condition tensor is a
    // vector to broadcast it forward instead of going backward. In this case,
    // it can just absorb the broadcasted dimensions of "then" and "else".
    TensorShape output_shape =
        TensorShapeUtils::IsVector(cond_shape)
            ? BroadcastTensorShapes({then_shape, else_shape})
            : BroadcastTensorShapes({then_shape, else_shape, cond_shape});

    // Broadcasting works a bit differently for the Condition tensor. We have
    // to broadcast forward instead of backwards.
    int32 cond_missing_dims = output_shape.dims() - cond_shape.dims();
    for (int32 i = 0; i < cond_missing_dims; ++i) {
      cond_shape.AddDim(1);
    }

    int32 then_missing_dims = output_shape.dims() - then_shape.dims();
    for (int32 i = 0; i < then_missing_dims; ++i) {
      then_shape.InsertDim(0, 1);
    }

    int32 else_missing_dims = output_shape.dims() - else_shape.dims();
    for (int32 i = 0; i < else_missing_dims; ++i) {
      else_shape.InsertDim(0, 1);
    }

    simple_ternary_ =
        SimplifyTernary(cond_shape, then_shape, else_shape, output_shape);

    OP_REQUIRES(
        ctx, simple_ternary_,
        errors::InvalidArgument(
            "DML doesn't support more than 4 dimensions for Select after "
            "collapsing non-broadcast dimensions together, but could "
            "not simplify the given shape to 4D."));

    broadcasted_output_shape_ = std::move(output_shape);
  }

  const absl::optional<SimpleTernary>& GetSimpleTernary() const {
    return simple_ternary_;
  }

  const TensorShape& GetBroadcastedOutputShape() const {
    return broadcasted_output_shape_;
  }

 private:
  absl::optional<SimpleTernary> simple_ternary_;
  TensorShape broadcasted_output_shape_;
};

using InitHelper = TernaryInitHelper;

class GetSelectOutputShapeHelper : public ShapeHelper {
 public:
  std::vector<TensorShape> GetOutputShapes(
      OpKernelContext* ctx,
      const InitializationHelper* initialization_helper) const override {
    auto init_helper = static_cast<const InitHelper*>(initialization_helper);
    return {init_helper->GetBroadcastedOutputShape()};
  }
};

class DmlTernaryKernel : public DmlKernel {
 public:
  using InitHelper = tensorflow::InitHelper;

  explicit DmlTernaryKernel(DmlKernelConstruction* ctx,
                            const InitHelper* init_helper) {
    const auto& simple_ternary = init_helper->GetSimpleTernary();

    DmlTensorInfo cond_input;
    cond_input.kernel_index = 0;
    cond_input.desc = DmlTensorDesc::Create(ctx->GetInputDataType(0),
                                            simple_ternary->output_shape,
                                            simple_ternary->cond_shape);

    DmlTensorInfo then_input;
    then_input.kernel_index = 1;
    then_input.desc = DmlTensorDesc::Create(ctx->GetInputDataType(1),
                                            simple_ternary->output_shape,
                                            simple_ternary->then_shape);

    DmlTensorInfo else_input;
    else_input.kernel_index = 2;
    else_input.desc = DmlTensorDesc::Create(ctx->GetInputDataType(2),
                                            simple_ternary->output_shape,
                                            simple_ternary->else_shape);

    DmlTensorInfo output;
    output.kernel_index = 0;
    output.desc = DmlTensorDesc::Create(ctx->GetOutputDataType(0),
                                        simple_ternary->output_shape,
                                        simple_ternary->output_shape);

    DmlKernelTensors tensors;
    tensors.inputs = {cond_input, then_input, else_input};
    tensors.outputs = {output};

    auto inputs = GetDmlTensorDescs(tensors.inputs);
    auto outputs = GetDmlTensorDescs(tensors.outputs);

    dml::TensorPolicy out_policy = dml::TensorPolicy::Default();
    if (Is64BitIntegerType(ctx->GetOutputDataType(0))) {
      out_policy = GetEmulatedInt64TensorPolicy();
    }

    auto scope = dml::Graph(ctx->GetDmlDevice(), out_policy);
    auto cond_tensor = dml::InputTensor(scope, 0, inputs[0]);
    auto then_tensor = dml::InputTensor(scope, 1, inputs[1]);
    auto else_tensor = dml::InputTensor(scope, 2, inputs[2]);
    auto result = dml::If(cond_tensor, then_tensor, else_tensor);

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

    if (Is64BitIntegerType(output->dtype())) {
      ctx->ZeroBuffer(ctx->CreateBufferForTensor(*output));
    }

    return DmlKernel::Compute(ctx);
  }
};

#define DML_REGISTER_KERNEL(type)                                      \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("Select").Device(DEVICE_DML).TypeConstraint<type>("T"),     \
      DmlKernelWrapper<DmlTernaryKernel, GetSelectOutputShapeHelper>); \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("SelectV2").Device(DEVICE_DML).TypeConstraint<type>("T"),   \
      DmlKernelWrapper<DmlTernaryKernel, GetSelectOutputShapeHelper>);
TF_CALL_DML_ALL_TYPES(DML_REGISTER_KERNEL);
#undef DML_REGISTER_KERNEL

}  // namespace tensorflow