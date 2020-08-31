/* Copyright (c) Microsoft Corporation.

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

#include "tensorflow/core/util/bcast.h"

namespace tensorflow {

std::vector<TensorShape> GetOutputShapeAsInputShapeHelper::GetOutputShapes(
    OpKernelContext* ctx,
    const InitializationHelper* initialization_helper) const {
  const Tensor& input_tensor =
      ctx->input_is_ref(0) ? ctx->mutable_input(0, false) : ctx->input(0);
  return {input_tensor.shape()};
}

static TensorShape BroadcastTensorShape(const TensorShape& input_shape_0,
                                        const TensorShape& input_shape_1) {
  if (input_shape_0 == input_shape_1) {
    return input_shape_0;
  }

  const auto output_rank = std::max(input_shape_0.dims(), input_shape_1.dims());
  TensorShape output_shape;
  for (int i = 0; i < output_rank; ++i) {
    output_shape.AddDim(0);
  }

  // Walk backwards through both input shapes and broadcast each dimension
  int in_dim_0_idx = input_shape_0.dims() - 1;
  int in_dim_1_idx = input_shape_1.dims() - 1;
  for (int out_dim_idx = output_rank - 1; out_dim_idx >= 0; --out_dim_idx) {
    int64 in_dim_0 = 1;
    if (in_dim_0_idx >= 0) {
      in_dim_0 = input_shape_0.dim_size(in_dim_0_idx);
      --in_dim_0_idx;
    }

    int64 in_dim_1 = 1;
    if (in_dim_1_idx >= 0) {
      in_dim_1 = input_shape_1.dim_size(in_dim_1_idx);
      --in_dim_1_idx;
    }

    CHECK((in_dim_0 == in_dim_1) || (in_dim_0 == 1) || (in_dim_1 == 1));
    int64 broadcasted_dim = std::max(in_dim_0, in_dim_1);
    CHECK(broadcasted_dim >= 0);

    // Special case - you can't broadcast a zero dimension (the dimension stays
    // zero)
    if (in_dim_0 == 0 || in_dim_1 == 0) {
      broadcasted_dim = 0;
    }

    output_shape.set_dim(out_dim_idx, broadcasted_dim);
  }

  return output_shape;
}

TensorShape BroadcastTensorShapes(absl::Span<const TensorShape> shapes) {
  CHECK(!shapes.empty());

  TensorShape accumulated_shape = shapes[0];

  for (const TensorShape& shape : shapes) {
    accumulated_shape = BroadcastTensorShape(accumulated_shape, shape);
  }

  return accumulated_shape;
}

BroadcastedOutputShapeInitHelper::BroadcastedOutputShapeInitHelper(
    OpKernelContext* ctx, std::shared_ptr<const Attributes> attr) {
  constexpr bool fewer_dims_optimization = false;

  for (int i = 0; i < ctx->num_inputs(); ++i) {
    TensorShape input_shape = ctx->input_is_ref(i)
                                  ? ctx->mutable_input(i, false).shape()
                                  : ctx->input(i).shape();

    BCast bcast_helper(BCast::FromShape(broadcasted_shape_),
                       BCast::FromShape(input_shape), fewer_dims_optimization);

    OP_REQUIRES(ctx, bcast_helper.IsValid(),
                errors::InvalidArgument(
                    "Incompatible shapes: ", broadcasted_shape_.DebugString(),
                    " vs. ", input_shape.DebugString()));

    broadcasted_shape_ = BCast::ToShape(bcast_helper.output_shape());
  }
}

std::vector<TensorShape> GetBroadcastedOutputShapeHelper::GetOutputShapes(
    OpKernelContext* ctx,
    const InitializationHelper* initialization_helper) const {
  auto init_helper = static_cast<const InitHelper*>(initialization_helper);
  return {init_helper->GetBroadcastedShape()};
}

std::vector<TensorShape> BatchNormShapeHelper::GetOutputShapes(
    OpKernelContext* ctx,
    const InitializationHelper* initialization_helper) const {
  // _FusedBatchNormEx can have 6 inputs
  CHECK(ctx->num_inputs() == 5 || ctx->num_inputs() == 6);
  CHECK(ctx->num_outputs() == 5 || ctx->num_outputs() == 6);

  if (ctx->num_outputs() == 5) {
    // FusedBatchNorm/FusedBatchNormV2 case

    // The shape of the normalized output matches the input tensor, and the
    // computed/saved mean/variance tensors match the shape of the scale tensor
    // (which is 1D, and the same size as the input tensor's C dimension)
    return {
        ctx->input(0).shape(), ctx->input(1).shape(), ctx->input(1).shape(),
        ctx->input(1).shape(), ctx->input(1).shape(),
    };
  } else {
    // FusedBatchNormV3 has an additional output tensor (which we don't actually
    // use, so give it an empty shape)
    return {
        ctx->input(0).shape(), ctx->input(1).shape(), ctx->input(1).shape(),
        ctx->input(1).shape(), ctx->input(1).shape(), TensorShape(),
    };
  }
}

std::vector<TensorShape> BatchNormGradShapeHelper::GetOutputShapes(
    OpKernelContext* ctx,
    const InitializationHelper* initialization_helper) const {
  CHECK(ctx->num_inputs() == 5 || ctx->num_inputs() == 6);
  CHECK(ctx->num_outputs() == 5);

  const TensorShape& x_shape = ctx->input(0).shape();
  const TensorShape& scale_shape = ctx->input(2).shape();

  // x_backprop, scale_backprop, offset_backprop, unused, unused
  // scale_backprop and offset_backprop are both 1D and have the same shape.
  return {
      x_shape, scale_shape, scale_shape, TensorShape(), TensorShape(),
  };
}

}  // namespace tensorflow