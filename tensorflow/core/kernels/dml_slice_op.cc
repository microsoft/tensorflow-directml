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

static absl::InlinedVector<uint32_t, 5> GetSliceSizes(
    const Tensor& size_tensor, const TensorShape& input_shape,
    absl::Span<const uint32_t> offsets) {
  absl::InlinedVector<int64, 5> sizes = IntTensorToVec<int64>(size_tensor);

  // DML takes uint32 values for the sizes
  absl::InlinedVector<uint32_t, 5> uint32_sizes;

  for (int i = 0; i < input_shape.dims(); ++i) {
    if (sizes[i] == -1) {
      // A sizes[i] of -1 means "all elements from offsets[i] to dim_size(i)".
      uint32_sizes.push_back(
          static_cast<uint32_t>(input_shape.dim_size(i) - offsets[i]));
    } else {
      uint32_sizes.push_back(static_cast<uint32_t>(sizes[i]));
    }
  }

  return uint32_sizes;
}

class SliceShapeHelper : public ShapeHelper {
 public:
  std::vector<TensorShape> GetOutputShapes(
      OpKernelContext* ctx,
      const InitializationHelper* initialization_helper) const override {
    TensorShape output_shape;

    const Tensor& input_tensor = ctx->input(0);
    const Tensor& offset_tensor = ctx->input(1);
    const Tensor& size_tensor = ctx->input(2);

    auto offsets = IntTensorToVec<uint32_t>(offset_tensor);
    auto sizes = GetSliceSizes(size_tensor, input_tensor.shape(), offsets);

    for (int i = 0; i < input_tensor.dims(); ++i) {
      int64 slice_offset = offsets[i];
      int64 slice_size = sizes[i];
      if (input_tensor.dim_size(i) == 0) {
        CHECK(slice_offset == 0 && slice_size == 0);
      } else {
        CHECK(0 <= slice_offset && slice_offset <= input_tensor.dim_size(i));
        CHECK(0 <= slice_size &&
              slice_offset + slice_size <= input_tensor.dim_size(i));
      }
      output_shape.AddDim(slice_size);
    }

    return {std::move(output_shape)};
  }
};

class DmlSliceKernel : public DmlKernel {
 public:
  using InitHelper = NoOpInitializationHelper;

  explicit DmlSliceKernel(DmlKernelConstruction* ctx,
                          const InitHelper* init_helper) {
    CHECK(ctx->GetInputCount() == 3);
    CHECK(ctx->GetOutputCount() == 1);

    const TensorShape& input_shape = ctx->GetInputTensorShape(0);
    const Tensor& offset_tensor = ctx->GetConstantInputTensor(1);
    const Tensor& size_tensor = ctx->GetConstantInputTensor(2);

    const int input_dims = input_shape.dims();
    CHECK(offset_tensor.shape().dims() == 1);
    CHECK(offset_tensor.NumElements() == input_dims);
    CHECK(size_tensor.shape().dims() == 1);
    CHECK(size_tensor.NumElements() == input_dims);

    auto offsets = IntTensorToVec<uint32_t>(offset_tensor);
    auto sizes = GetSliceSizes(size_tensor, input_shape, offsets);

    TensorShape output_shape;

    bool is_identity = true;

    // If the offset of any dimension is not zero or the slice size is not
    // the size of the dimension, it is not an identity slice.
    for (int i = 0; i < input_dims; ++i) {
      if (offsets[i] != 0 || sizes[i] != input_shape.dim_size(i)) {
        is_identity = false;
        break;
      }
    }

    DmlKernelParams params;
    params.output_shape = ctx->GetOutputTensorShape(0);
    params.kernel_input_indices = {0};

    DmlKernelTensors tensors = GetTensorInfos(ctx, params);
    auto inputs = GetDmlTensorDescs(tensors.inputs);
    auto outputs = GetDmlTensorDescs(tensors.outputs);

    if (is_identity) {
      DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC identity_desc = {};
      identity_desc.InputTensor = inputs.data();
      identity_desc.OutputTensor = outputs.data();

      DML_OPERATOR_DESC op_desc = {DML_OPERATOR_ELEMENT_WISE_IDENTITY,
                                   &identity_desc};
      Initialize(ctx, std::move(tensors), op_desc);
    } else {
      // Pad the sizes and offsets to match DML's 4D minimum requirement
      if (sizes.size() < kNchwDimensionCount) {
        uint32_t pad_amount = kNchwDimensionCount - sizes.size();
        sizes.insert(sizes.begin(), pad_amount, 1);
        offsets.insert(offsets.begin(), pad_amount, 0);
      }

      absl::InlinedVector<uint32_t, 5> strides(sizes.size(), 1);

      DML_SLICE_OPERATOR_DESC slice_desc = {};
      slice_desc.InputTensor = inputs.data();
      slice_desc.OutputTensor = outputs.data();
      slice_desc.Sizes = sizes.data();
      slice_desc.Strides = strides.data();
      slice_desc.Offsets = offsets.data();
      slice_desc.DimensionCount = sizes.size();

      DML_OPERATOR_DESC op_desc = {DML_OPERATOR_SLICE, &slice_desc};
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

#define REGISTER_KERNELS(type)                                                \
  REGISTER_KERNEL_BUILDER(Name("Slice")                                       \
                              .Device(DEVICE_DML)                             \
                              .TypeConstraint<type>("T")                      \
                              .TypeConstraint<int32>("Index")                 \
                              .HostMemory("begin")                            \
                              .HostMemory("size"),                            \
                          DmlKernelWrapper<DmlSliceKernel, SliceShapeHelper>) \
  REGISTER_KERNEL_BUILDER(Name("Slice")                                       \
                              .Device(DEVICE_DML)                             \
                              .TypeConstraint<type>("T")                      \
                              .TypeConstraint<int64>("Index")                 \
                              .HostMemory("begin")                            \
                              .HostMemory("size"),                            \
                          DmlKernelWrapper<DmlSliceKernel, SliceShapeHelper>)
// TODO(b/25387198): A special kernel exists for int32 (see slice_op.cc).
TF_CALL_DML_ALL_TYPES_EXCEPT_INT32(REGISTER_KERNELS);
#undef REGISTER_KERNELS

}  // namespace tensorflow