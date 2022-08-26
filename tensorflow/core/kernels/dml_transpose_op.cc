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
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/dml_kernel_wrapper.h"
#include "tensorflow/core/kernels/dml_ops_common.h"

namespace tensorflow {

struct SimpleTranspose {
  TensorShape input_shape;
  TensorShape output_shape;
  absl::InlinedVector<int, 8> permutations;
};

template <typename TPerm>
static SimpleTranspose SimplifyTranspose(const TensorShape& input_shape,
                                         const Tensor& perm_tensor) {
  auto perm_values = perm_tensor.flat<TPerm>();

  absl::InlinedVector<int64, 8> simple_input_sizes(input_shape.dims());
  absl::InlinedVector<int, 8> input_to_perm_map(input_shape.dims());
  int input_index = static_cast<int>(perm_values(0));
  int dim_size = input_shape.dim_size(input_index);
  input_to_perm_map[input_index] = 0;

  // Merge all adjacent dimensions that will still be adjacent after the
  // transpose operation (i.e. increasing sequences with a step of 1)
  for (int64 i = 1; i < perm_tensor.NumElements(); ++i) {
    int prev_index = static_cast<int>(perm_values(i - 1));
    int index = static_cast<int>(perm_values(i));
    input_to_perm_map[index] = i;

    if (index == prev_index + 1) {
      dim_size *= input_shape.dim_size(index);

      // Set the dimension to -1 to signal that it should be removed later
      simple_input_sizes[index] = -1;
    } else {
      simple_input_sizes[input_index] = dim_size;
      dim_size = input_shape.dim_size(index);
      input_index = index;
    }
  }

  simple_input_sizes[input_index] = dim_size;

  // Shift collapsed input dimensions to the left and adjust the permutation
  // indices accordingly
  int input_left_shift = 0;
  absl::InlinedVector<int, 8> simple_permutations(simple_input_sizes.size());
  for (int i = 0; i < simple_input_sizes.size(); ++i) {
    int perm_index = input_to_perm_map[i];

    if (simple_input_sizes[i] == -1) {
      input_left_shift++;

      // Set the dimension to -1 to signal that it should be removed later
      simple_permutations[perm_index] = -1;
    } else {
      int new_index = i - input_left_shift;
      simple_input_sizes[new_index] = simple_input_sizes[i];
      simple_permutations[perm_index] = new_index;
    }
  }

  simple_input_sizes.resize(simple_input_sizes.size() - input_left_shift);

  // Shift the permutations to the left
  int perm_left_shift = 0;
  for (int i = 0; i < simple_permutations.size(); ++i) {
    if (simple_permutations[i] == -1) {
      perm_left_shift++;
    } else {
      simple_permutations[i - perm_left_shift] = simple_permutations[i];
    }
  }

  simple_permutations.resize(simple_permutations.size() - perm_left_shift);

  // Finally, create the output shape
  TensorShape simple_output_shape;
  for (int i = 0; i < simple_permutations.size(); ++i) {
    int dim_size = simple_input_sizes[simple_permutations[i]];
    simple_output_shape.AddDim(dim_size);
  }

  SimpleTranspose simple_transpose;
  simple_transpose.input_shape = TensorShape(simple_input_sizes);
  simple_transpose.output_shape = std::move(simple_output_shape);
  simple_transpose.permutations = std::move(simple_permutations);

  return simple_transpose;
}

template <typename TPerm>
static std::vector<TensorShape> GetOutputShapesHelper(OpKernelContext* ctx) {
  const TensorShape& input_shape = ctx->input(0).shape();
  const Tensor& perm_tensor = ctx->input(1);

  TensorShape output_shape(input_shape);

  auto perm_values = perm_tensor.flat<TPerm>();

  for (int64 i = 0; i < perm_tensor.NumElements(); ++i) {
    TPerm input_dim_index = perm_values(i);
    CHECK(input_dim_index < input_shape.dims());
    output_shape.set_dim(i, input_shape.dim_size(input_dim_index));
  }

  return {std::move(output_shape)};
}

static dml::TensorStrides ComputeStrides(const TensorShape& input_shape,
                                         absl::Span<const int> permutations,
                                         uint32_t expected_dim_count) {
  DCHECK(input_shape.dims() == permutations.size());

  int leading_dims = permutations.size() < expected_dim_count
                         ? expected_dim_count - permutations.size()
                         : 0;

  dml::TensorStrides output_strides(leading_dims + permutations.size(), 1);

  constexpr uint32_t stride = 1;

  for (int64 i = permutations.size() - 1; i >= 0; --i) {
    int input_dim_index = permutations[i];
    DCHECK(input_dim_index < output_strides.size());

    output_strides[leading_dims + input_dim_index] = stride;
    stride *= input_shape.dim_size(input_dim_index);
  }

  return output_strides;
}

class TransposeInitHelper : public InitializationHelper {
 public:
  using Attributes = EmptyAttributes;

  TransposeInitHelper(OpKernelContext* ctx,
                      std::shared_ptr<const Attributes> attr) {
    const TensorShape& input_shape = ctx->input(0).shape();
    const Tensor& perm_tensor = ctx->input(1);

    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(perm_tensor.shape()),
                errors::InvalidArgument("perm must be a vector, not ",
                                        perm_tensor.shape().DebugString()));

    OP_REQUIRES(ctx, input_shape.dims() == perm_tensor.NumElements(),
                errors::InvalidArgument("transpose expects a vector of size ",
                                        input_shape.dims(),
                                        ". But input(1) is a vector of size ",
                                        perm_tensor.NumElements()));

    assert(perm_tensor.dtype() == DT_INT32 || perm_tensor.dtype() == DT_INT64);
    simple_transpose_ =
        perm_tensor.dtype() == DT_INT32
            ? SimplifyTranspose<int32>(input_shape, perm_tensor)
            : SimplifyTranspose<int64>(input_shape, perm_tensor);

    OP_REQUIRES(ctx, simple_transpose_.input_shape.dims() <= 8,
                errors::InvalidArgument(
                    "DML doesn't support more than 8D for Transpose, but ",
                    simple_transpose_.input_shape.dims(),
                    " dimensions were provided."));
  }

  SimpleTranspose GetSimpleTranspose() const { return simple_transpose_; }

 private:
  SimpleTranspose simple_transpose_;
};

class TransposeShapeHelper : public ShapeHelper {
 public:
  std::vector<TensorShape> GetOutputShapes(
      OpKernelContext* ctx,
      const InitializationHelper* initialization_helper) const override {
    const Tensor& perm_tensor = ctx->input(1);

    assert(perm_tensor.dtype() == DT_INT32 || perm_tensor.dtype() == DT_INT64);
    return perm_tensor.dtype() == DT_INT32 ? GetOutputShapesHelper<int32>(ctx)
                                           : GetOutputShapesHelper<int64>(ctx);
  }
};

class DmlTransposeKernel : public DmlKernel {
 public:
  using InitHelper = TransposeInitHelper;

  explicit DmlTransposeKernel(DmlKernelConstruction* ctx,
                              const InitHelper* init_helper) {
    CHECK(ctx->GetInputCount() == 2);
    CHECK(ctx->GetOutputCount() == 1);

    auto simple_transpose = init_helper->GetSimpleTranspose();

    DmlTensorInfo input;
    input.kernel_index = 0;
    input.desc = DmlTensorDesc::Create(ctx->GetInputDataType(0),
                                       simple_transpose.input_shape,
                                       simple_transpose.input_shape);

    DmlTensorInfo output;
    output.kernel_index = 0;
    output.desc = DmlTensorDesc::Create(ctx->GetOutputDataType(0),
                                        simple_transpose.output_shape,
                                        simple_transpose.output_shape);

    const auto out_policy = dml::TensorPolicy(
        [ctx, simple_transpose](DML_TENSOR_DATA_TYPE dataType,
                                DML_TENSOR_FLAGS flags,
                                dml::Span<const uint32_t> sizes) {
          uint32_t dimension_count = static_cast<uint32_t>(sizes.size());

          dml::TensorProperties props = {};
          props.strides =
              ComputeStrides(simple_transpose.input_shape,
                             simple_transpose.permutations, dimension_count);

          props.guaranteedBaseOffsetAlignment = 0;

          props.totalTensorSizeInBytes = DMLCalcBufferTensorSize(
              dataType, dimension_count, sizes.data(), props.strides->data());

          return props;
        });

    DmlKernelTensors tensors;
    tensors.inputs = {input};
    tensors.outputs = {output};

    auto inputs = GetDmlTensorDescs(tensors.inputs);
    auto scope = dml::Graph(ctx->GetDmlDevice(), out_policy);
    auto result = dml::Identity(dml::InputTensor(scope, 0, inputs[0]));

    Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
        scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

    Initialize(ctx, std::move(tensors), compiled_op.Get());
  }
};

#define REGISTER_KERNEL(type)        \
  REGISTER_KERNEL_BUILDER(           \
      Name("Transpose")              \
          .Device(DEVICE_DML)        \
          .TypeConstraint<type>("T") \
          .HostMemory("perm"),       \
      DmlKernelWrapper<DmlTransposeKernel, TransposeShapeHelper>);

TF_CALL_float(REGISTER_KERNEL);
TF_CALL_half(REGISTER_KERNEL);
TF_CALL_bool(REGISTER_KERNEL);
TF_CALL_int64(REGISTER_KERNEL);
TF_CALL_int32(REGISTER_KERNEL);
TF_CALL_uint16(REGISTER_KERNEL);
TF_CALL_int16(REGISTER_KERNEL);
TF_CALL_uint8(REGISTER_KERNEL);
TF_CALL_int8(REGISTER_KERNEL);

#undef REGISTER_KERNEL

}  // namespace tensorflow