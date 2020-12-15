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
#include "tensorflow/core/util/strided_slice_op.h"

namespace tensorflow {

struct SimplifiedSlice {
  dml::TensorDesc::Dimensions input_sizes;
  dml::TensorDesc::Dimensions input_strides;
  dml::TensorDesc::Dimensions output_sizes;
  dml::SmallVector<uint32_t, 5> window_offset;
  dml::SmallVector<uint32_t, 5> window_sizes;
  dml::SmallVector<int32_t, 5> window_strides;
};

template <typename T>
void ShiftDim(T& vec, int shift_amount, uint32_t dim_count) {
  std::rotate(vec.begin(), vec.begin() + shift_amount, vec.end());
  vec.resize(dim_count);
}

// This helper may simplify an N-dimensional slice to a lower rank slice by
// coalescing dimensions that meet the following criteria:
// - Dimensions with size 1 are always coalesced.
// - Adjacent dimensions that are fully included in the slice are always
// coalesced.
// - A higher-order dimension that is partially included in the slice, and has
// no offset/stride, will be
//   merged with lower-order dimensions that are fully included in the slice.
static absl::optional<SimplifiedSlice> SimplifySlice(
    const TensorShape& input_shape,
    const gtl::InlinedVector<int64, 4>& canonical_begins,
    const gtl::InlinedVector<int64, 4>& canonical_ends,
    const gtl::InlinedVector<int64, 4>& strides, uint32_t min_output_size = 4,
    uint32_t max_output_size = 5) {
  assert(input_shape.dims() == begins.size());
  assert(input_shape.dims() == ends.size());
  assert(input_shape.dims() == strides.size());
  assert(max_output_size > 0);

  SimplifiedSlice desc = {};
  desc.input_sizes.resize(max_output_size, 1);
  desc.input_strides.resize(max_output_size, 1);
  desc.output_sizes.resize(max_output_size, 1);
  desc.window_offset.resize(max_output_size, 0);
  desc.window_sizes.resize(max_output_size, 1);
  desc.window_strides.resize(max_output_size, 1);

  int current_dim = max_output_size - 1;

  // Insertion becomes a no-op if the shape cannot be simplified into the
  // requested max_output_size.
  auto InsertDim = [&](uint32_t input_size, uint32_t input_stride,
                       uint32_t output_size, uint32_t window_offset,
                       uint32_t window_size, int32_t window_stride) {
    if (current_dim >= 0) {
      desc.input_sizes[current_dim] = input_size;
      desc.input_strides[current_dim] = input_stride;
      desc.output_sizes[current_dim] = output_size;
      desc.window_offset[current_dim] = window_offset;
      desc.window_sizes[current_dim] = window_size;
      desc.window_strides[current_dim] = window_stride;
    }
    current_dim--;
  };

  uint32_t coalesced = 1;
  uint32_t total_stride = 1;

  for (int i = input_shape.dims() - 1; i >= 0; i--) {
    const uint32_t input_size = input_shape.dim_size(i);
    const int32_t window_stride = static_cast<int32_t>(strides[i]);

    // Here, begin and end contain the canonical values. This means that they
    // cannot be negative when strides are positive. When strides are negative,
    // end can only be positive or -1. See the ValidateStridedSliceOp function
    // in strided_slice_op.cc for reference.
    const int64 begin = canonical_begins[i];
    const int64 end = canonical_ends[i];
    CHECK(end >= -1);

    uint32_t window_offset, window_size, output_size;
    if (window_stride > 0) {
      window_offset = begin;
      window_size = end - begin;
      output_size = 1 + (window_size - 1) / window_stride;
    } else {
      window_offset = end + 1;  // +1 to convert exclusive to inclusive
      window_size = begin - end;
      output_size = 1 + (window_size - 1) / -window_stride;
    }

    if (input_size == output_size && window_stride > 0) {
      // The dimension can be collapsed, since all of its elements are included
      // in the slice. However, coalescing can only be performed if the elements
      // are read in order (i.e. stride is positive).
      coalesced *= input_size;
    } else {
      if (begin == 0 && window_stride == 1 && coalesced > 1) {
        // The current dim is merged with all previously collapsed dims.This is
        // only possible because slicing of the current dim emits elements
        // adjacent to the previously collapsed dims. Some of the tail elements
        // in the current dim won't be included in the slice, but they can be
        // skipped by padding the input strides to account for the extra
        // physical elements.
        InsertDim(
            /*inputSize    */ coalesced * input_size,
            /*inputStride  */ total_stride,
            /*outputSize   */ coalesced * output_size,
            /*windowOffset */ 0,
            /*windowSize   */ coalesced * output_size,
            /*windowStride */ 1);
        total_stride *= coalesced * input_size;
      } else {
        // The current dim cannot be merged at all, so (up to) two dims are
        // inserted: the previously collapsed dims, if any, and a separate dim
        // for the non-contiguous current dim.
        if (coalesced > 1) {
          InsertDim(
              /*inputSize    */ coalesced,
              /*inputStride  */ total_stride,
              /*outputSize   */ coalesced,
              /*windowOffset */ 0,
              /*windowSize   */ coalesced,
              /*windowStride */ 1);
          total_stride *= coalesced;
        }
        InsertDim(
            /*inputSize    */ input_size,
            /*inputStride  */ total_stride,
            /*outputSize   */ output_size,
            /*windowOffset */ window_offset,
            /*windowSize   */ window_size,
            /*windowStride */ window_stride);
        total_stride *= input_size;
      }
      coalesced = 1;
    }
  }

  if (coalesced > 1) {
    InsertDim(
        /*inputSize    */ coalesced,
        /*inputStride  */ total_stride,
        /*outputSize   */ coalesced,
        /*windowOffset */ 0,
        /*windowSize   */ coalesced,
        /*windowStride */ 1);
    total_stride *= coalesced;
  }

  // current_dim is the index of the next dim to write; if it's -1, then all
  // max_output_size dims have been filled (0 dims remain). Anything larger
  // than -1 indicates padding.
  int dims_remaining = current_dim + 1;
  if (dims_remaining < 0) {
    return absl::nullopt;
  } else {
    for (int i = current_dim; i >= 0; i--) {
      desc.input_strides[current_dim--] = total_stride;
    }

    // DML is (in general) faster with fewer dims, so shift values left if there
    // are leading padding dims. No need to use 5D shader if 4D is possible.
    int max_shift = max_output_size - min_output_size;
    int shift_amount = std::min<int>(max_shift, dims_remaining);
    uint32_t dim_count = max_output_size - shift_amount;

    ShiftDim(desc.input_sizes, shift_amount, dim_count);
    ShiftDim(desc.input_strides, shift_amount, dim_count);
    ShiftDim(desc.output_sizes, shift_amount, dim_count);
    ShiftDim(desc.window_offset, shift_amount, dim_count);
    ShiftDim(desc.window_sizes, shift_amount, dim_count);
    ShiftDim(desc.window_strides, shift_amount, dim_count);
  }

  return desc;
}

class StridedSliceInitHelper : public InitializationHelper {
 public:
  struct Attributes {
    explicit Attributes(OpKernelConstruction* ctx) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("begin_mask", &begin_mask));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("end_mask", &end_mask));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("ellipsis_mask", &ellipsis_mask));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("new_axis_mask", &new_axis_mask));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("shrink_axis_mask", &shrink_axis_mask));
    }

    int32 begin_mask, end_mask;
    int32 ellipsis_mask, new_axis_mask, shrink_axis_mask;
  };

  // StridedSlice is only a no-op if the first input or its output is empty
  bool IsNoOpKernel(
      OpKernelContext* ctx,
      absl::Span<const TensorShape> output_shapes) const override {
    return ctx->input(0).NumElements() == 0 ||
           output_shapes[0].num_elements() == 0;
  }

  StridedSliceInitHelper(OpKernelContext* ctx,
                         std::shared_ptr<const Attributes> attr) {
    TensorShape processing_shape;
    bool slice_dim0 = true;
    bool is_simple_slice = true;
    gtl::InlinedVector<int64, 4> begin;
    gtl::InlinedVector<int64, 4> end;
    gtl::InlinedVector<int64, 4> strides;

    // StridedSliceGrad has a 5th tensor for dy.
    bool is_grad_op = ctx->num_inputs() == 5;

    // StridedSliceGrad stores shape in a 1D host tensor.
    TensorShape input_shape;
    if (is_grad_op) {
      const Tensor& input_shape_tensor = ctx->input(0);
      OP_REQUIRES(
          ctx, input_shape_tensor.dims() == 1,
          errors::InvalidArgument("shape must be 1-D, got shape.shape = ",
                                  input_shape_tensor.shape().DebugString()));
      if (input_shape_tensor.dtype() == DT_INT32) {
        OP_REQUIRES_OK(ctx, TensorShapeUtils::MakeShape(
                                input_shape_tensor.vec<int32>(), &input_shape));
      } else if (input_shape_tensor.dtype() == DT_INT64) {
        OP_REQUIRES_OK(ctx, TensorShapeUtils::MakeShape(
                                input_shape_tensor.vec<int64>(), &input_shape));
      } else {
        LOG(FATAL) << "shape must have type int32 or int64.";
      }
    } else {
      input_shape = ctx->input(0).shape();
    }

    OP_REQUIRES_OK(
        ctx, ValidateStridedSliceOp(
                 &ctx->input(1), &ctx->input(2), ctx->input(3), input_shape,
                 attr->begin_mask, attr->end_mask, attr->ellipsis_mask,
                 attr->new_axis_mask, attr->shrink_axis_mask, &processing_shape,
                 &output_shape_, &is_identity_, &is_simple_slice, &slice_dim0,
                 &begin, &end, &strides));

    // Check to make sure dy is consistent with the original slice.
    if (is_grad_op) {
      TensorShape dy_shape = ctx->input(4).shape();
      OP_REQUIRES(
          ctx, output_shape_ == dy_shape,
          errors::InvalidArgument("shape of dy was ", dy_shape.DebugString(),
                                  " instead of ", output_shape_.DebugString()));
      output_shape_ = input_shape;
    }

    // Attempt to simplify the slice into a lower-rank slice.
    simple_slice_ = SimplifySlice(input_shape, begin, end, strides);
    if (!simple_slice_) {
      OP_REQUIRES(
          ctx, simple_slice_,
          errors::InvalidArgument("DML only support slicing up to 5D inputs, "
                                  "but received ",
                                  input_shape.dims()));
    }
  }

  const TensorShape& GetOutputShape() const { return output_shape_; }
  const bool IsIdentity() const { return is_identity_; }

  const absl::optional<SimplifiedSlice>& GetSimplifiedSlice() const {
    return simple_slice_;
  }

 private:
  TensorShape output_shape_;
  absl::optional<SimplifiedSlice> simple_slice_;
  bool is_identity_;
};

using InitHelper = StridedSliceInitHelper;

class StridedSliceShapeHelper : public ShapeHelper {
 public:
  std::vector<TensorShape> GetOutputShapes(
      OpKernelContext* ctx,
      const InitializationHelper* initialization_helper) const override {
    auto init_helper = static_cast<const InitHelper*>(initialization_helper);
    return {init_helper->GetOutputShape()};
  }
};

class DmlStridedSliceKernel : public DmlKernel {
 public:
  using InitHelper = tensorflow::InitHelper;

  explicit DmlStridedSliceKernel(DmlKernelConstruction* ctx,
                                 const InitHelper* init_helper) {
    CHECK(ctx->GetInputCount() == 4);
    CHECK(ctx->GetOutputCount() == 1);

    auto simple_slice = init_helper->GetSimplifiedSlice();
    auto dtype_tf = ctx->GetInputDataType(0);
    DML_TENSOR_DATA_TYPE dtype_dml = DML_TENSOR_DATA_TYPE_UNKNOWN;

    // TODO #24881131: 64-bit data support should be revisited
    // TFDML #24881131
    uint64_t end_padding_in_bytes = 0;
    dml::TensorDesc::Dimensions output_strides(
        simple_slice->output_sizes.size());
    uint32_t stride = 1;
    for (int i = simple_slice->output_sizes.size() - 1; i >= 0; i--) {
      output_strides[i] = stride;
      stride *= simple_slice->output_sizes[i];
    }
    if (Is64BitIntegerType(dtype_tf)) {
      dtype_dml = DML_TENSOR_DATA_TYPE_UINT32;
      for (auto& stride : simple_slice->input_strides) {
        stride *= 2;
      }
      for (int i = simple_slice->output_sizes.size() - 1; i >= 0; i--) {
        output_strides[i] *= 2;
      }
      end_padding_in_bytes = sizeof(uint32_t);
    } else {
      dtype_dml = GetDmlDataTypeFromTfDataType(dtype_tf);
    }

    DmlTensorInfo input;
    input.kernel_index = 0;
    input.desc =
        DmlTensorDesc{dtype_dml, simple_slice->input_sizes,
                      simple_slice->input_strides, 0, end_padding_in_bytes};

    DmlTensorInfo output;
    output.kernel_index = 0;
    output.desc = DmlTensorDesc{dtype_dml, simple_slice->output_sizes,
                                output_strides, 0, end_padding_in_bytes};

    DmlKernelTensors tensors;
    tensors.inputs = {input};
    tensors.outputs = {output};

    auto inputs = GetDmlTensorDescs(tensors.inputs);
    auto outputs = GetDmlTensorDescs(tensors.outputs);

    if (init_helper->IsIdentity()) {
      DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC identity_desc = {};
      identity_desc.InputTensor = inputs.data();
      identity_desc.OutputTensor = outputs.data();

      DML_OPERATOR_DESC op_desc = {DML_OPERATOR_ELEMENT_WISE_IDENTITY,
                                   &identity_desc};
      Initialize(ctx, std::move(tensors), op_desc);
    } else {
      DML_SLICE1_OPERATOR_DESC slice_desc = {};
      slice_desc.InputTensor = inputs.data();
      slice_desc.OutputTensor = outputs.data();
      slice_desc.InputWindowSizes = simple_slice->window_sizes.data();
      slice_desc.InputWindowStrides = simple_slice->window_strides.data();
      slice_desc.InputWindowOffsets = simple_slice->window_offset.data();
      slice_desc.DimensionCount = simple_slice->input_sizes.size();

      DML_OPERATOR_DESC op_desc = {DML_OPERATOR_SLICE1, &slice_desc};
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

#define REGISTER_KERNELS(type)       \
  REGISTER_KERNEL_BUILDER(           \
      Name("StridedSlice")           \
          .Device(DEVICE_DML)        \
          .TypeConstraint<type>("T") \
          .HostMemory("begin")       \
          .HostMemory("end")         \
          .HostMemory("strides"),    \
      DmlKernelWrapper<DmlStridedSliceKernel, StridedSliceShapeHelper>)
// TODO(b/25387198): A special kernel exists for int32 (see
// strided_slice_op.cc).
TF_CALL_DML_ALL_TYPES_EXCEPT_INT32(REGISTER_KERNELS);
#undef REGISTER_KERNELS

// ----------------------------------------
// StridedSliceGrad
// ----------------------------------------

class DmlStridedSliceGradKernel : public DmlKernel {
 public:
  using InitHelper = tensorflow::InitHelper;

  explicit DmlStridedSliceGradKernel(DmlKernelConstruction* ctx,
                                     const InitHelper* init_helper) {
    CHECK(ctx->GetInputCount() == 5);
    CHECK(ctx->GetOutputCount() == 1);

    auto simple_slice = init_helper->GetSimplifiedSlice();
    auto dtype_tf = ctx->GetInputDataType(4);
    DML_TENSOR_DATA_TYPE dtype_dml = DML_TENSOR_DATA_TYPE_UNKNOWN;

    // TODO #24881131: 64-bit data support should be revisited
    // TFDML #24881131
    uint64_t end_padding_in_bytes = 0;
    dml::TensorDesc::Dimensions output_strides(
        simple_slice->output_sizes.size());
    uint32_t stride = 1;
    for (int i = simple_slice->output_sizes.size() - 1; i >= 0; i--) {
      output_strides[i] = stride;
      stride *= simple_slice->output_sizes[i];
    }
    if (Is64BitIntegerType(dtype_tf)) {
      dtype_dml = DML_TENSOR_DATA_TYPE_UINT32;
      for (auto& stride : simple_slice->input_strides) {
        stride *= 2;
      }
      for (int i = simple_slice->output_sizes.size() - 1; i >= 0; i--) {
        output_strides[i] *= 2;
      }
      end_padding_in_bytes = sizeof(uint32_t);
    } else {
      dtype_dml = GetDmlDataTypeFromTfDataType(dtype_tf);
    }

    DmlTensorInfo input;
    input.kernel_index = 4;
    input.desc = DmlTensorDesc{dtype_dml, simple_slice->output_sizes,
                               output_strides, 0, end_padding_in_bytes};

    DmlTensorInfo output;
    output.kernel_index = 0;
    output.desc =
        DmlTensorDesc{dtype_dml, simple_slice->input_sizes,
                      simple_slice->input_strides, 0, end_padding_in_bytes};

    DmlKernelTensors tensors;
    tensors.inputs = {input};
    tensors.outputs = {output};

    auto inputs = GetDmlTensorDescs(tensors.inputs);
    auto outputs = GetDmlTensorDescs(tensors.outputs);

    if (init_helper->IsIdentity()) {
      DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC identity_desc = {};
      identity_desc.InputTensor = inputs.data();
      identity_desc.OutputTensor = outputs.data();

      DML_OPERATOR_DESC op_desc = {DML_OPERATOR_ELEMENT_WISE_IDENTITY,
                                   &identity_desc};
      Initialize(ctx, std::move(tensors), op_desc);
    } else {
      DML_SLICE_GRAD_OPERATOR_DESC slice_desc = {};
      slice_desc.InputGradientTensor = inputs.data();
      slice_desc.OutputGradientTensor = outputs.data();
      slice_desc.InputWindowSizes = simple_slice->window_sizes.data();
      slice_desc.InputWindowStrides = simple_slice->window_strides.data();
      slice_desc.InputWindowOffsets = simple_slice->window_offset.data();
      slice_desc.DimensionCount = simple_slice->input_sizes.size();

      DML_OPERATOR_DESC op_desc = {DML_OPERATOR_SLICE_GRAD, &slice_desc};
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

#define REGISTER_KERNELS(type)       \
  REGISTER_KERNEL_BUILDER(           \
      Name("StridedSliceGrad")       \
          .Device(DEVICE_DML)        \
          .TypeConstraint<type>("T") \
          .HostMemory("shape")       \
          .HostMemory("begin")       \
          .HostMemory("end")         \
          .HostMemory("strides"),    \
      DmlKernelWrapper<DmlStridedSliceGradKernel, StridedSliceShapeHelper>)
// TODO(b/25387198): A special kernel exists for int32 (see
// strided_slice_op.cc).
TF_CALL_DML_ALL_TYPES_EXCEPT_INT32(REGISTER_KERNELS);
#undef REGISTER_KERNELS

}  // namespace tensorflow