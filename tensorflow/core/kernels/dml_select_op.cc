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
#include "tensorflow/core/util/bcast.h"

namespace tensorflow {

struct SimpleTernary {
  dml::TensorDesc::Dimensions cond_shape;
  dml::TensorDesc::Dimensions then_shape;
  dml::TensorDesc::Dimensions else_shape;
  dml::TensorDesc::Dimensions output_shape;
};

// Attempts to coalesce dimensions that are not broadcast to work with DML's 8D
// tensor limitation. Requires all shapes to have the same rank.
absl::optional<SimpleTernary> SimplifyTernary(const TensorShape& cond_shape,
                                              const TensorShape& then_shape,
                                              const TensorShape& else_shape,
                                              const TensorShape& output_shape,
                                              uint32_t output_size = 8) {
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

class BaseSelectInitHelper : public InitializationHelper {
 public:
  using Attributes = EmptyAttributes;
  virtual const SimpleTernary& GetSimpleTernary() const = 0;
  virtual const TensorShape& GetBroadcastedOutputShape() const = 0;
};

class SelectInitHelper : public BaseSelectInitHelper {
 public:
  SelectInitHelper(OpKernelContext* ctx,
                   std::shared_ptr<const Attributes> attr) {
    TensorShape cond_shape = ctx->input(0).shape();
    TensorShape then_shape = ctx->input(1).shape();
    TensorShape else_shape = ctx->input(2).shape();

    if (TensorShapeUtils::IsScalar(cond_shape)) {
      OP_REQUIRES_OK(
          ctx, ComputeScalarSelectShapes(cond_shape, then_shape, else_shape));
    } else {
      bool broadcasting = (TensorShapeUtils::IsVector(cond_shape) &&
                           !TensorShapeUtils::IsVector(then_shape));
      if (broadcasting) {
        OP_REQUIRES_OK(
            ctx, ComputeBatchSelectShapes(cond_shape, then_shape, else_shape));
      } else {
        OP_REQUIRES_OK(ctx, ComputeElementWiseSelectShapes(
                                cond_shape, then_shape, else_shape));
      }
    }

    broadcasted_output_shape_ = then_shape;
  }

  const SimpleTernary& GetSimpleTernary() const final {
    return simple_ternary_;
  }

  const TensorShape& GetBroadcastedOutputShape() const final {
    return broadcasted_output_shape_;
  }

 private:
  Status ComputeScalarSelectShapes(const TensorShape& cond_shape,
                                   const TensorShape& then_shape,
                                   const TensorShape& else_shape) {
    if (!then_shape.IsSameSize(else_shape)) {
      return errors::InvalidArgument(
          "'then' and 'else' must have the same size.  but received: ",
          then_shape.DebugString(), " vs. ", else_shape.DebugString());
    }

    const uint32_t batch_size = static_cast<uint32_t>(then_shape.dim_size(0));
    const uint32_t outer_dims_size =
        then_shape.num_elements() / then_shape.dim_size(0);

    const uint32_t num_elements = then_shape.num_elements();
    simple_ternary_.cond_shape = {1};
    simple_ternary_.then_shape = {num_elements};
    simple_ternary_.else_shape = {num_elements};
    simple_ternary_.output_shape = {num_elements};

    return Status::OK();
  }

  Status ComputeBatchSelectShapes(const TensorShape& cond_shape,
                                  const TensorShape& then_shape,
                                  const TensorShape& else_shape) {
    if (!TensorShapeUtils::IsVector(cond_shape)) {
      return errors::InvalidArgument("'cond' must be a vector, but saw shape: ",
                                     cond_shape.DebugString());
    }

    if (!TensorShapeUtils::IsVectorOrHigher(then_shape)) {
      return errors::InvalidArgument(
          "'then' must be at least a vector, but saw shape: ",
          then_shape.DebugString());
    }

    if (then_shape.dim_size(0) != cond_shape.num_elements()) {
      return errors::InvalidArgument(
          "Number of batches of 'then' must match size of 'cond', but saw: ",
          then_shape.dim_size(0), " vs. ", cond_shape.num_elements());
    }

    if (!then_shape.IsSameSize(else_shape)) {
      return errors::InvalidArgument(
          "'then' and 'else' must have the same size.  but received: ",
          then_shape.DebugString(), " vs. ", else_shape.DebugString());
    }

    const uint32_t batch_size = static_cast<uint32_t>(then_shape.dim_size(0));
    TensorShape flat_outer_shape = ComputeFlatOuterDims(then_shape, 2);

    dml::TensorDesc::Dimensions simple_shape({
        batch_size,
        static_cast<uint32_t>(flat_outer_shape.dim_size(1)),
    });

    simple_ternary_.cond_shape = {batch_size, 1};
    simple_ternary_.then_shape = simple_shape;
    simple_ternary_.else_shape = simple_shape;
    simple_ternary_.output_shape = simple_shape;

    return Status::OK();
  }

  Status ComputeElementWiseSelectShapes(const TensorShape& cond_shape,
                                        const TensorShape& then_shape,
                                        const TensorShape& else_shape) {
    if (!cond_shape.IsSameSize(then_shape) ||
        !cond_shape.IsSameSize(else_shape)) {
      return errors::InvalidArgument(
          "'cond', 'then' and 'else' must have the same size when 'cond' is "
          "not a vector.  but received: ",
          cond_shape.DebugString(), " vs. ", then_shape.DebugString(), " vs. ",
          else_shape.DebugString());
    }

    const uint32_t num_elements = cond_shape.num_elements();
    simple_ternary_.cond_shape = {num_elements};
    simple_ternary_.then_shape = {num_elements};
    simple_ternary_.else_shape = {num_elements};
    simple_ternary_.output_shape = {num_elements};

    return Status::OK();
  }

  SimpleTernary simple_ternary_;
  TensorShape broadcasted_output_shape_;
  TensorShape coalesced_output_shape_;
};

class SelectV2InitHelper : public BaseSelectInitHelper {
 public:
  SelectV2InitHelper(OpKernelContext* ctx,
                     std::shared_ptr<const Attributes> attr) {
    TensorShape cond_shape = ctx->input(0).shape();
    TensorShape then_shape = ctx->input(1).shape();
    TensorShape else_shape = ctx->input(2).shape();

    // The `cond`, `then`, and `else` are broadcastable (bcast.IsValid()),
    // This matches the behavior of numpy.

    // Combine `then` and `else`.
    BCast then_else_bcast(BCast::FromShape(then_shape),
                          BCast::FromShape(else_shape), false);
    OP_REQUIRES(ctx, then_else_bcast.IsValid(),
                errors::InvalidArgument("then ", then_shape.DebugString(),
                                        " and else ", else_shape.DebugString(),
                                        " must be broadcastable"));
    // Combine `cond` with `then` and `else`.
    BCast bcast(
        BCast::FromShape(cond_shape),
        BCast::FromShape(BCast::ToShape(then_else_bcast.output_shape())),
        false);
    OP_REQUIRES(ctx, bcast.IsValid(),
                errors::InvalidArgument("condition ", cond_shape.DebugString(),
                                        ", then ", then_shape.DebugString(),
                                        ", and else ", else_shape.DebugString(),
                                        " must be broadcastable"));

    // Broadcast `cond`, `then` and `else` to combined shape,
    // in order to obtain the reshape.
    BCast cond_bcast(BCast::FromShape(BCast::ToShape(bcast.output_shape())),
                     BCast::FromShape(cond_shape), false);
    BCast then_bcast(BCast::FromShape(BCast::ToShape(bcast.output_shape())),
                     BCast::FromShape(then_shape), false);
    BCast else_bcast(BCast::FromShape(BCast::ToShape(bcast.output_shape())),
                     BCast::FromShape(else_shape), false);
    OP_REQUIRES(
        ctx,
        cond_bcast.IsValid() && then_bcast.IsValid() && else_bcast.IsValid(),
        errors::InvalidArgument("condition ", cond_shape.DebugString(),
                                ", then ", then_shape.DebugString(),
                                ", and else ", else_shape.DebugString(),
                                " must be broadcastable"));

    // Combined shape should be the final shape.
    OP_REQUIRES(
        ctx,
        cond_bcast.output_shape() == bcast.output_shape() &&
            then_bcast.output_shape() == bcast.output_shape() &&
            else_bcast.output_shape() == bcast.output_shape(),
        errors::InvalidArgument("condition ", cond_shape.DebugString(),
                                ", then ", then_shape.DebugString(),
                                ", and else ", else_shape.DebugString(),
                                " must be broadcastable to the same shape"));

    broadcasted_output_shape_ = BCast::ToShape(bcast.output_shape());

    absl::optional<SimpleTernary> simple_ternary = SimplifyTernary(
        BCast::ToShape(cond_bcast.y_reshape()),
        BCast::ToShape(then_bcast.y_reshape()),
        BCast::ToShape(else_bcast.y_reshape()), broadcasted_output_shape_);

    OP_REQUIRES(
        ctx, broadcasted_output_shape_.dims() <= 8,
        errors::InvalidArgument(
            "DML doesn't support more than 8 dimensions for Select after "
            "collapsing non-broadcast dimensions together, but could "
            "not simplify the given shape to 8D."));

    simple_ternary_ = *simple_ternary;
  }

  const SimpleTernary& GetSimpleTernary() const final {
    return simple_ternary_;
  }

  const TensorShape& GetBroadcastedOutputShape() const final {
    return broadcasted_output_shape_;
  }

 private:
  SimpleTernary simple_ternary_;
  TensorShape broadcasted_output_shape_;
};

class GetSelectOutputShapeHelper : public ShapeHelper {
 public:
  std::vector<TensorShape> GetOutputShapes(
      OpKernelContext* ctx,
      const InitializationHelper* initialization_helper) const override {
    auto init_helper =
        static_cast<const BaseSelectInitHelper*>(initialization_helper);
    return {init_helper->GetBroadcastedOutputShape()};
  }
};

template <typename TInitHelper>
class DmlTernaryKernel : public DmlKernel {
 public:
  using InitHelper = TInitHelper;

  explicit DmlTernaryKernel(DmlKernelConstruction* ctx,
                            const InitHelper* init_helper) {
    const SimpleTernary& simple_ternary = init_helper->GetSimpleTernary();

    DmlTensorInfo cond_input;
    cond_input.kernel_index = 0;
    cond_input.desc = DmlTensorDesc::Create(ctx->GetInputDataType(0),
                                            simple_ternary.output_shape,
                                            simple_ternary.cond_shape);

    DmlTensorInfo then_input;
    then_input.kernel_index = 1;
    then_input.desc = DmlTensorDesc::Create(ctx->GetInputDataType(1),
                                            simple_ternary.output_shape,
                                            simple_ternary.then_shape);

    DmlTensorInfo else_input;
    else_input.kernel_index = 2;
    else_input.desc = DmlTensorDesc::Create(ctx->GetInputDataType(2),
                                            simple_ternary.output_shape,
                                            simple_ternary.else_shape);

    DmlTensorInfo output;
    output.kernel_index = 0;
    output.desc = DmlTensorDesc::Create(ctx->GetOutputDataType(0),
                                        simple_ternary.output_shape,
                                        simple_ternary.output_shape);

    DmlKernelTensors tensors;
    tensors.inputs = {cond_input, then_input, else_input};
    tensors.outputs = {output};

    auto inputs = GetDmlTensorDescs(tensors.inputs);
    auto scope = dml::Graph(ctx->GetDmlDevice());
    auto cond_tensor = dml::InputTensor(scope, 0, inputs[0]);
    auto then_tensor = dml::InputTensor(scope, 1, inputs[1]);
    auto else_tensor = dml::InputTensor(scope, 2, inputs[2]);
    auto result = dml::If(cond_tensor, then_tensor, else_tensor);

    Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
        scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

    Initialize(ctx, std::move(tensors), compiled_op.Get());
  }
};

#define DML_REGISTER_KERNEL(type)                                    \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("Select").Device(DEVICE_DML).TypeConstraint<type>("T"),   \
      DmlKernelWrapper<DmlTernaryKernel<SelectInitHelper>,           \
                       GetSelectOutputShapeHelper>);                 \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("SelectV2").Device(DEVICE_DML).TypeConstraint<type>("T"), \
      DmlKernelWrapper<DmlTernaryKernel<SelectV2InitHelper>,         \
                       GetSelectOutputShapeHelper>);

TF_CALL_half(DML_REGISTER_KERNEL);
TF_CALL_float(DML_REGISTER_KERNEL);
TF_CALL_bool(DML_REGISTER_KERNEL);
TF_CALL_int32(DML_REGISTER_KERNEL);
TF_CALL_int64(DML_REGISTER_KERNEL);
#undef DML_REGISTER_KERNEL

}  // namespace tensorflow