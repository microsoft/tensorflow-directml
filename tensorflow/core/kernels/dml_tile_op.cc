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

template <typename Tmultiples>
class TileShapeHelper : public ShapeHelper {
 public:
  std::vector<TensorShape> GetOutputShapes(
      OpKernelContext* ctx,
      const InitializationHelper* initialization_helper) const override {
    const Tensor& input = ctx->input(0);
    const Tensor& multiples = ctx->input(1);
    int input_dims = input.shape().dims();

    const gtl::ArraySlice<Tmultiples> multiples_array(
        multiples.flat<Tmultiples>().data(), input_dims);

    TensorShape output_shape;
    for (int i = 0; i < input_dims; ++i) {
      CHECK(multiples_array[i] >= 0);
      output_shape.AddDim(input.dim_size(i) * multiples_array[i]);
    }

    return {std::move(output_shape)};
  }
};

struct SimpleTile {
  bool is_identity_op;
  dml::TensorDesc::Dimensions input_shape;
  dml::TensorDesc::Dimensions output_shape;
  absl::InlinedVector<uint32_t, 4> repeats;
};

// Attempts to reduce the tile into a more compact representation that
// works with DML's 4D tensor limitation.
template <typename Tmultiples>
absl::optional<SimpleTile> SimplifyTile(const TensorShape& input_shape,
                                        const Tensor& multiples_tensor,
                                        uint32_t output_size = 4) {
  SimpleTile desc = {};
  desc.input_shape.resize(output_size, 1);
  desc.output_shape.resize(output_size, 1);
  desc.repeats.resize(output_size, 1);
  int simple_shape_dim = output_size - 1;

  uint32_t total_elements = 1;
  uint32_t total_repeat = 1;

  auto multiples = multiples_tensor.flat<Tmultiples>();
  DCHECK(multiples.size() == input_shape.dims());

  for (int i = input_shape.dims() - 1; i >= 0; i--) {
    if (simple_shape_dim < 0) {
      // There's no more space to write simplified dims.
      return absl::nullopt;
    }

    uint32_t size = static_cast<uint32_t>(input_shape.dim_size(i));
    uint32_t repeat = static_cast<uint32_t>(multiples(i));

    // Coalesce adjacent dims when possible.
    int coalesced_dims = 0;
    for (int j = i - 1; j >= 0; j--) {
      uint32_t next_size = static_cast<uint32_t>(input_shape.dim_size(j));
      uint32_t next_repeat = static_cast<uint32_t>(multiples(j));
      if (next_size == 1 || repeat == 1) {
        size *= next_size;
        repeat *= next_repeat;
        coalesced_dims++;
      } else {
        break;
      }
    }
    i -= coalesced_dims;

    desc.input_shape[simple_shape_dim] = size;
    desc.output_shape[simple_shape_dim] = size * repeat;
    desc.repeats[simple_shape_dim] = repeat;
    simple_shape_dim--;

    total_elements *= size;
    total_repeat *= repeat;
  }

  if (total_repeat == 1 || total_elements == 1) {
    desc.is_identity_op = true;
  }

  // NOTE: one additional way to tile of an arbitrary input shape is to
  // recursively tile up to 4 dims at a time. For example, consider the
  // 7D shape [2,2,2,2,3,4,5] with multiples [2,2,2,2,2,2,2].
  // None of the dims in this example can be coalesced, but this can be
  // handled by tiling multiple times:
  //
  // Tile 1: input shape = [2,3,4,5], repeats = [2,2,2,2], output = [4,6,8,10].
  // Tile 2: input shape = [2,2,2,1920], repeats = [2,2,2,1], output =
  // [4,4,4,1920].
  //
  // Each tile step handles up to 4 dimensions of the original input, and its
  // output is reinterpreted as single flattened dimnsion in the next tile step.
  // Note that the lowest-order dimension of the subsequent tiles always has a
  // repeat of 1, so it can coalesce with the next dimension. In short, it's
  // possible to handle an arbitrary rank N input with an upper bound of N/4 DML
  // tile ops in a graph.

  return desc;
}

template <typename Tmultiples>
class TileInitializationHelper : public InitializationHelper {
 public:
  using Attributes = EmptyAttributes;

  bool IsNoOpKernel(
      OpKernelContext* ctx,
      absl::Span<const TensorShape> output_shapes) const override {
    if (ctx->input(0).NumElements() == 0) {
      return true;
    }

    const gtl::ArraySlice<Tmultiples> multiples_array(
        ctx->input(1).flat<Tmultiples>().data(), ctx->input(0).dims());

    for (int i = 0; i < multiples_array.size(); ++i) {
      if (multiples_array[i] == 0) {
        return true;
      }
    }

    return false;
  }

  TileInitializationHelper(OpKernelContext* ctx,
                           std::shared_ptr<const Attributes> attr) {
    const TensorShape& input_shape = ctx->input(0).shape();
    const Tensor& multiples = ctx->input(1);

    // DML only supports tiling 4D tensors. Attempt to simplify into 4D.
    simple_tile_ =
        SimplifyTile<Tmultiples>(input_shape, multiples, kNchwDimensionCount);
    OP_REQUIRES(ctx, simple_tile_,
                errors::InvalidArgument(
                    "DML doesn't support more than 4 dimensions for Tile after "
                    "collapsing non-repeatable dimensions together, but could "
                    "not simplify the given shape to 4D."));
  }

  const absl::optional<SimpleTile>& GetSimpleTile() const {
    return simple_tile_;
  }

 private:
  absl::optional<SimpleTile> simple_tile_;
};

template <typename Tmultiples>
class DmlTileKernel : public DmlKernel {
 public:
  using InitHelper = TileInitializationHelper<Tmultiples>;

  explicit DmlTileKernel(DmlKernelConstruction* ctx,
                         const InitHelper* init_helper) {
    CHECK(ctx->GetInputCount() == 2);
    CHECK(ctx->GetOutputCount() == 1);

    auto simple_tile = init_helper->GetSimpleTile();
    const auto& in_shape = simple_tile->input_shape;
    const auto& out_shape = simple_tile->output_shape;
    auto dtype = ctx->GetInputDataType(0);

    if (simple_tile->is_identity_op) {
      DmlTensorInfo input;
      input.kernel_index = 0;
      input.desc = DmlTensorDesc::Create(dtype, out_shape, in_shape);

      DmlTensorInfo output;
      output.kernel_index = 0;
      output.desc = DmlTensorDesc::Create(dtype, out_shape, out_shape);

      DmlKernelTensors tensors;
      tensors.inputs = {input};
      tensors.outputs = {output};

      auto inputs = GetDmlTensorDescs(tensors.inputs);
      auto outputs = GetDmlTensorDescs(tensors.outputs);

      DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC desc = {};
      desc.InputTensor = inputs.data();
      desc.OutputTensor = outputs.data();

      DML_OPERATOR_DESC op_desc = {DML_OPERATOR_ELEMENT_WISE_IDENTITY, &desc};
      Initialize(ctx, std::move(tensors), op_desc);
    } else {
      DmlTensorInfo input;
      input.kernel_index = 0;
      input.desc = DmlTensorDesc::Create(dtype, in_shape, in_shape);

      DmlTensorInfo output;
      output.kernel_index = 0;
      output.desc = DmlTensorDesc::Create(dtype, out_shape, out_shape);

      DmlKernelTensors tensors;
      tensors.inputs = {input};
      tensors.outputs = {output};

      auto inputs = GetDmlTensorDescs(tensors.inputs);
      auto outputs = GetDmlTensorDescs(tensors.outputs);

      DML_TILE_OPERATOR_DESC tile_desc = {};
      tile_desc.InputTensor = inputs.data();
      tile_desc.OutputTensor = outputs.data();
      tile_desc.RepeatsCount = simple_tile->repeats.size();
      tile_desc.Repeats = simple_tile->repeats.data();

      DML_OPERATOR_DESC op_desc = {DML_OPERATOR_TILE, &tile_desc};
      Initialize(ctx, std::move(tensors), op_desc);
    }
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

template <typename Tmultiples>
using DmlTileWrapper =
    DmlKernelWrapper<DmlTileKernel<Tmultiples>, TileShapeHelper<Tmultiples>>;

#define DML_REGISTER_KERNEL(type)                                  \
  REGISTER_KERNEL_BUILDER(Name("Tile")                             \
                              .Device(DEVICE_DML)                  \
                              .TypeConstraint<type>("T")           \
                              .TypeConstraint<int32>("Tmultiples") \
                              .HostMemory("multiples"),            \
                          DmlTileWrapper<int32>);                  \
  REGISTER_KERNEL_BUILDER(Name("Tile")                             \
                              .Device(DEVICE_DML)                  \
                              .TypeConstraint<type>("T")           \
                              .TypeConstraint<int64>("Tmultiples") \
                              .HostMemory("multiples"),            \
                          DmlTileWrapper<int64>);

TF_CALL_DML_ALL_TYPES(DML_REGISTER_KERNEL);
#undef DML_REGISTER_KERNEL

}  // namespace tensorflow