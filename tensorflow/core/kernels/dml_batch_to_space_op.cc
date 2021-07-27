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

#include <numeric>

#include "tensorflow/core/common_runtime/dml/dml_operator_helper.h"
#include "tensorflow/core/common_runtime/dml/dml_util.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/dml_kernel_wrapper.h"
#include "tensorflow/core/kernels/dml_ops_common.h"

namespace tensorflow {

// DML can only permute up to 8 dimensions, and BatchToSpace needs to permute
// blockDims * 2 + 2 dimensions
constexpr int kMaxSpaceToBatchBlockDims = 3;

class BaseBatchToSpaceInitHelper : public InitializationHelper {
 public:
  const TensorShape& GetInternalInputShape() const {
    return internal_input_shape_;
  }

  const TensorShape& GetInternalOutputShape() const {
    return internal_output_shape_;
  }

  absl::Span<const int64> GetInternalBlockSizes() const {
    return internal_block_sizes_;
  }

  absl::Span<const int64> GetInternalCrops() const { return internal_crops_; }

  TensorShape GetOutputShape() const { return output_shape_; }
  int GetInternalBlockDims() const { return internal_block_dims_; }

 protected:
  void Initialize(OpKernelContext* ctx, const Tensor& orig_crops,
                  absl::Span<const int64> block_shape) {
    const Tensor& orig_input_tensor = ctx->input(0);
    const int input_dims = orig_input_tensor.dims();
    const int block_dims = block_shape.size();

    OP_REQUIRES(
        ctx, orig_input_tensor.dims() >= 1 + block_dims,
        errors::InvalidArgument("input rank should be >= ", 1 + block_dims,
                                " instead of ", orig_input_tensor.dims()));

    OP_REQUIRES(ctx,
                TensorShapeUtils::IsMatrix(orig_crops.shape()) &&
                    block_dims == orig_crops.dim_size(0) &&
                    2 == orig_crops.dim_size(1),
                errors::InvalidArgument("crops should have shape [", block_dims,
                                        ", 2] instead of ",
                                        orig_crops.shape().DebugString()));
    // To avoid out-of-bounds access in the case that the block_shape and/or
    // crops tensors are concurrently modified, we must copy the values.
    auto crops = IntTensorToVec<int64>(orig_crops);

    // Determine the length of the prefix of block dims that can be combined
    // into the batch dimension due to having no padding and block_shape=1.
    int removed_prefix_block_dims = 0;
    for (; removed_prefix_block_dims < block_dims;
         ++removed_prefix_block_dims) {
      const int dim = removed_prefix_block_dims;
      if (crops[2 * dim] != 0 || crops[2 * dim + 1] != 0 ||
          block_shape[dim] != 1) {
        break;
      }
    }

    // Determine the length of the suffix of block dims that can be combined
    // into the depth dimension due to having no padding and block_shape=1.
    int removed_suffix_block_dims = 0;
    for (; removed_suffix_block_dims < block_dims - removed_prefix_block_dims;
         ++removed_suffix_block_dims) {
      const int dim = block_dims - 1 - removed_suffix_block_dims;
      if (crops[2 * dim] != 0 || crops[2 * dim + 1] != 0 ||
          block_shape[dim] != 1) {
        break;
      }
    }

    // Compute the product of the block_shape values.
    int64 block_shape_product = 1;
    for (int block_dim = 0; block_dim < block_dims; ++block_dim) {
      block_shape_product *= block_shape[block_dim];
    }
    OP_REQUIRES(
        ctx, block_shape_product > 0,
        errors::InvalidArgument("Product of block sizes must be positive, got ",
                                block_shape_product));

    const int64 orig_input_batch_size = orig_input_tensor.dim_size(0);
    OP_REQUIRES(ctx, orig_input_batch_size % block_shape_product == 0,
                errors::InvalidArgument(
                    "Input batch dimension (", orig_input_batch_size,
                    ") is not divisible by product of block sizes (",
                    block_shape_product, ")"));

    const int internal_block_dims =
        block_dims - removed_prefix_block_dims - removed_suffix_block_dims;
    OP_REQUIRES(ctx, internal_block_dims <= kMaxSpaceToBatchBlockDims,
                errors::InvalidArgument(
                    "Maximum number of non-combined block dimensions is ",
                    internal_block_dims, " but must not exceed ",
                    kMaxSpaceToBatchBlockDims));

    // For the purpose of computing the result, the input will be treated as
    // having this shape, of rank 2 + internal_block_dims.
    TensorShape internal_input_shape;

    // For the purpose of computing the result, the output will be treated as
    // having this shape, of rank 2 + internal_block_dims.
    TensorShape internal_output_shape;

    // The actual output shape exposed to callers.
    TensorShape external_output_shape;

    external_output_shape.AddDim(orig_input_batch_size / block_shape_product);

    int64 input_batch_size = orig_input_batch_size;
    for (int block_dim = 0; block_dim < removed_prefix_block_dims;
         ++block_dim) {
      const int64 size = orig_input_tensor.dim_size(block_dim + 1);
      input_batch_size *= size;
      external_output_shape.AddDim(size);
    }
    internal_input_shape.AddDim(input_batch_size);
    internal_output_shape.AddDim(input_batch_size / block_shape_product);

    for (int block_dim = removed_prefix_block_dims;
         block_dim < block_dims - removed_suffix_block_dims; ++block_dim) {
      const int64 crop_start = crops[2 * block_dim],
                  crop_end = crops[2 * block_dim + 1];
      OP_REQUIRES(ctx, crop_start >= 0 && crop_end >= 0,
                  errors::InvalidArgument("Crops must be non-negative"));
      const int64 input_size = orig_input_tensor.dim_size(block_dim + 1);
      const int64 block_shape_value = block_shape[block_dim];
      const int64 cropped_size =
          input_size * block_shape_value - crop_start - crop_end;
      OP_REQUIRES(
          ctx, cropped_size >= 0,
          errors::InvalidArgument("cropped_shape[", block_dim,
                                  "]=", cropped_size, " must be non-negative"));
      internal_input_shape.AddDim(input_size);
      internal_output_shape.AddDim(cropped_size);
      external_output_shape.AddDim(cropped_size);
    }

    int64 depth = 1;
    for (int dim = block_dims - removed_suffix_block_dims + 1; dim < input_dims;
         ++dim) {
      const int64 size = orig_input_tensor.dim_size(dim);
      external_output_shape.AddDim(size);
      depth *= size;
    }
    internal_input_shape.AddDim(depth);
    internal_output_shape.AddDim(depth);

    internal_input_shape_ = std::move(internal_input_shape);
    internal_output_shape_ = std::move(internal_output_shape);
    output_shape_ = std::move(external_output_shape);
    internal_block_dims_ = internal_block_dims;

    internal_block_sizes_.assign(
        block_shape.begin() + removed_prefix_block_dims,
        block_shape.end() - removed_suffix_block_dims);

    internal_crops_.assign(crops.begin() + removed_prefix_block_dims * 2,
                           crops.end() - removed_suffix_block_dims * 2);
  }

  bool IsNoOpKernel(OpKernelContext* ctx,
                    absl::Span<const TensorShape> output_shapes) const final {
    if (ctx->input(0).NumElements() == 0) return true;
    if (output_shapes[0].num_elements() == 0) return true;
    return false;
  }

 private:
  TensorShape internal_input_shape_;
  TensorShape internal_output_shape_;
  TensorShape output_shape_;
  int internal_block_dims_;
  absl::InlinedVector<int64, 4> internal_block_sizes_;
  absl::InlinedVector<int64, 8> internal_crops_;
};

class BatchToSpaceInitHelper : public BaseBatchToSpaceInitHelper {
 public:
  struct Attributes {
    explicit Attributes(OpKernelConstruction* ctx) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("block_size", &block_size));

      OP_REQUIRES(
          ctx, block_size > 1,
          errors::InvalidArgument("Block size should be > 1: ", block_size));
    }
    int block_size;
  };

  explicit BatchToSpaceInitHelper(OpKernelContext* ctx,
                                  std::shared_ptr<const Attributes> attr) {
    const Tensor& in0 = ctx->input(0);
    const int dims = in0.dims();

    // Check on the input dimensions first.
    // The input is presumed to be [batch, height, width, depth]
    static const int kRequiredDims = 4;
    OP_REQUIRES(ctx, kRequiredDims == dims,
                errors::InvalidArgument("Input rank should be: ", kRequiredDims,
                                        "instead of: ", dims));

    const int64 block_shape[] = {attr->block_size, attr->block_size};

    const Tensor& orig_crops = ctx->input(1);
    Initialize(ctx, orig_crops, block_shape);
  }
};

class BatchToSpaceNdInitHelper : public BaseBatchToSpaceInitHelper {
 public:
  using Attributes = EmptyAttributes;

  BatchToSpaceNdInitHelper(OpKernelContext* ctx,
                           std::shared_ptr<const Attributes> attr) {
    const Tensor& orig_block_shape = ctx->input(1);
    const Tensor& orig_crops = ctx->input(2);

    OP_REQUIRES(
        ctx, TensorShapeUtils::IsVector(orig_block_shape.shape()),
        errors::InvalidArgument("block_shape rank should be 1 instead of ",
                                orig_block_shape.dims()));

    auto block_shape = IntTensorToVec<int64>(orig_block_shape);
    Initialize(ctx, orig_crops, block_shape);
  }
};

class BatchToSpaceShapeHelper : public ShapeHelper {
 public:
  std::vector<TensorShape> GetOutputShapes(
      OpKernelContext* ctx,
      const InitializationHelper* initialization_helper) const override {
    auto init_helper =
        static_cast<const BaseBatchToSpaceInitHelper*>(initialization_helper);
    return {init_helper->GetOutputShape()};
  }
};

template <typename TInitHelper>
class DmlBatchToSpaceKernel : public DmlKernel {
 public:
  static_assert(std::is_base_of<BaseBatchToSpaceInitHelper, TInitHelper>::value,
                "TInitHelper must derive from BaseBatchToSpaceInitHelper");

  using InitHelper = TInitHelper;

  DmlBatchToSpaceKernel(DmlKernelConstruction* ctx,
                        const InitHelper* init_helper) {
    const TensorShape& internal_input_shape =
        init_helper->GetInternalInputShape();

    const TensorShape& internal_output_shape =
        init_helper->GetInternalOutputShape();

    int internal_block_dims = init_helper->GetInternalBlockDims();

    DmlTensorInfo input_desc;
    input_desc.kernel_index = 0;
    input_desc.desc = DmlTensorDesc::Create(
        ctx->GetInputDataType(0), internal_input_shape, internal_input_shape);

    DmlTensorInfo output_desc;
    output_desc.kernel_index = 0;
    output_desc.desc =
        DmlTensorDesc::Create(ctx->GetOutputDataType(0), internal_output_shape,
                              internal_output_shape);

    DmlKernelTensors tensors;
    tensors.inputs = {input_desc};
    tensors.outputs = {output_desc};

    auto inputs = GetDmlTensorDescs(tensors.inputs);

    if (internal_block_dims == 0) {
      auto outputs = GetDmlTensorDescs(tensors.outputs);

      DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC identity_desc = {};
      identity_desc.InputTensor = &inputs[0];
      identity_desc.OutputTensor = &outputs[0];

      DML_OPERATOR_DESC op_desc = {DML_OPERATOR_ELEMENT_WISE_IDENTITY,
                                   &identity_desc};
      Initialize(ctx, std::move(tensors), op_desc);
      return;
    }

    auto scope = dml::Graph(ctx->GetDmlDevice());
    auto input = dml::InputTensor(scope, 0, inputs[0]);

    absl::Span<const int64> internal_block_sizes =
        init_helper->GetInternalBlockSizes();

    uint32_t batch_size = internal_input_shape.dim_size(0);
    uint32_t block_shape_product = std::accumulate(internal_block_sizes.begin(),
                                                   internal_block_sizes.end(),
                                                   1, std::multiplies<int64>());

    // Reshape the input into [block_shape[0], ..., block_shape[M-1], batch /
    // prod(block_shape), input_shape[1], ..., input_shape[N-1]]
    dml::TensorDesc::Dimensions reshaped_sizes;
    for (int i = 0; i < internal_block_sizes.size(); ++i) {
      reshaped_sizes.push_back(internal_block_sizes[i]);
    }

    reshaped_sizes.push_back(internal_input_shape.dim_size(0) /
                             block_shape_product);

    for (int i = 1; i < internal_input_shape.dims(); ++i) {
      reshaped_sizes.push_back(internal_input_shape.dim_size(i));
    }

    // Permute the reshaped input into [batch / prod(block_shape),
    // input_shape[1], block_shape[0], ..., input_shape[M], block_shape[M-1],
    // input_shape[M+1], ..., input_shape[N-1]]
    uint32_t stride = 1;
    dml::TensorDesc::Dimensions reshaped_strides(reshaped_sizes.size());
    for (int i = reshaped_sizes.size() - 1; i >= 0; --i) {
      reshaped_strides[i] = stride;
      stride *= reshaped_sizes[i];
    }

    dml::TensorDesc::Dimensions perm_strides;
    perm_strides.reserve(reshaped_strides.size());

    dml::TensorDesc::Dimensions perm_sizes;
    perm_sizes.reserve(reshaped_strides.size());

    perm_strides.push_back(reshaped_strides[internal_block_dims]);
    perm_sizes.push_back(reshaped_sizes[internal_block_dims]);

    for (int i = 0; i < internal_block_dims; ++i) {
      int reshaped_index = internal_block_dims + i + 1;
      perm_strides.push_back(reshaped_strides[reshaped_index]);
      perm_sizes.push_back(reshaped_sizes[reshaped_index]);

      perm_strides.push_back(reshaped_strides[i]);
      perm_sizes.push_back(reshaped_sizes[i]);
    }

    for (int i = internal_block_dims * 2 + 1; i < reshaped_sizes.size(); ++i) {
      perm_strides.push_back(reshaped_strides[i]);
      perm_sizes.push_back(reshaped_sizes[i]);
    }

    auto permuted = dml::Reinterpret(input, perm_sizes, perm_strides);
    permuted = dml::Identity(permuted);

    // Reshape permuted into [batch / prod(block_shape), input_shape[1] *
    // block_shape[0], ..., input_shape[M] * block_shape[M-1], input_shape[M+1],
    // ..., input_shape[N-1]]
    dml::TensorDesc::Dimensions perm_reshaped_sizes;
    perm_reshaped_sizes.reserve(perm_sizes.size() - internal_block_dims / 2);
    perm_reshaped_sizes.push_back(perm_sizes.front());

    for (int i = 1; i <= internal_block_dims * 2; i += 2) {
      uint32_t new_size = perm_sizes[i] * perm_sizes[i + 1];
      perm_reshaped_sizes.push_back(new_size);
    }

    for (int i = internal_block_dims * 2 + 1; i < perm_sizes.size(); ++i) {
      perm_reshaped_sizes.push_back(perm_sizes[i]);
    }

    auto permuted_reshaped =
        dml::Reinterpret(permuted, perm_reshaped_sizes, {});

    // Finally, slice the appropriate dimensions
    dml::TensorDesc::Dimensions slice_offsets(perm_reshaped_sizes.size());
    dml::TensorDesc::Dimensions slice_sizes = perm_reshaped_sizes;
    absl::InlinedVector<int32_t, 4> slice_strides(perm_reshaped_sizes.size(), 1);

    absl::Span<const int64> internal_crops = init_helper->GetInternalCrops();

    for (int i = 0; i < internal_block_dims; ++i) {
      int64 start_crop = internal_crops[i * 2];
      int64 end_crop = internal_crops[i * 2 + 1];
      slice_offsets[i + 1] = start_crop;
      slice_sizes[i + 1] = perm_reshaped_sizes[i + 1] - start_crop - end_crop;
    }

    auto result = dml::Slice(permuted_reshaped, slice_offsets, slice_sizes,
                             slice_strides);

    Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
        scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

    Initialize(ctx, std::move(tensors), compiled_op.Get());
  }
};

#define REGISTER_DML_KERNEL(T)                                          \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("BatchToSpaceND")                                            \
          .Device(DEVICE_DML)                                           \
          .TypeConstraint<T>("T")                                       \
          .HostMemory("block_shape")                                    \
          .HostMemory("crops"),                                         \
      DmlKernelWrapper<DmlBatchToSpaceKernel<BatchToSpaceNdInitHelper>, \
                       BatchToSpaceShapeHelper>);                       \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("BatchToSpace")                                              \
          .Device(DEVICE_DML)                                           \
          .TypeConstraint<T>("T")                                       \
          .HostMemory("crops"),                                         \
      DmlKernelWrapper<DmlBatchToSpaceKernel<BatchToSpaceInitHelper>,   \
                       BatchToSpaceShapeHelper>);

TF_CALL_float(REGISTER_DML_KERNEL);
TF_CALL_half(REGISTER_DML_KERNEL);
#undef REGISTER_DML_KERNEL

}  // namespace tensorflow
