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
#include "tensorflow/core/lib/gtl/cleanup.h"
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
    uint32_t max_output_size = 8) {
  assert(input_shape.dims() == canonical_begins.size());
  assert(input_shape.dims() == canonical_ends.size());
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

  bool IsNoOpKernel(
      OpKernelContext* ctx,
      absl::Span<const TensorShape> output_shapes) const override {
    const bool is_grad_op = ctx->num_inputs() == 5;

    if (is_grad_op) {
      // For StridedSliceGrad, the last input is the input gradient
      return ctx->input(4).NumElements() == 0 ||
             output_shapes[0].num_elements() == 0;
    }

    // StridedSlice is only a no-op if the first input or its output is empty
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
          errors::InvalidArgument("DML only support slicing up to 8D inputs, "
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
    const DML_TENSOR_DATA_TYPE dtype_dml =
        GetDmlDataTypeFromTfDataType(dtype_tf);

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
      for (auto& stride : simple_slice->input_strides) {
        stride *= 2;
      }
      for (int i = simple_slice->output_sizes.size() - 1; i >= 0; i--) {
        output_strides[i] *= 2;
      }
      end_padding_in_bytes = sizeof(uint32_t);
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

    auto scope = dml::Graph(ctx->GetDmlDevice());
    auto inputs = GetDmlTensorDescs(tensors.inputs);
    auto result = dml::InputTensor(scope, 0, inputs[0]);

    if (init_helper->IsIdentity()) {
      result = dml::Identity(result);
    } else {
      result =
          dml::Slice(result, simple_slice->window_offset,
                     simple_slice->window_sizes, simple_slice->window_strides);
    }

    // TFDML #24881131
    if (Is64BitSignedIntegerType(ctx->GetOutputDataType(0))) {
      result = dml::ConvertInt32ToInt64(result);
    }

    Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
        scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

    Initialize(ctx, std::move(tensors), compiled_op.Get());
  }
};

#define REGISTER_KERNEL(type)        \
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
TF_CALL_half(REGISTER_KERNEL);
TF_CALL_float(REGISTER_KERNEL);
TF_CALL_bool(REGISTER_KERNEL);
TF_CALL_int8(REGISTER_KERNEL);
TF_CALL_int64(REGISTER_KERNEL);
#undef REGISTER_KERNEL

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
    const DML_TENSOR_DATA_TYPE dtype_dml =
        GetDmlDataTypeFromTfDataType(dtype_tf);

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
      for (auto& stride : simple_slice->input_strides) {
        stride *= 2;
      }
      for (int i = simple_slice->output_sizes.size() - 1; i >= 0; i--) {
        output_strides[i] *= 2;
      }
      end_padding_in_bytes = sizeof(uint32_t);
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

    auto scope = dml::Graph(ctx->GetDmlDevice());
    auto inputs = GetDmlTensorDescs(tensors.inputs);
    auto result = dml::InputTensor(scope, 0, inputs[0]);

    if (init_helper->IsIdentity()) {
      result = dml::Identity(result);
    } else {
      result = dml::SliceGrad(
          result, simple_slice->input_sizes, simple_slice->window_offset,
          simple_slice->window_sizes, simple_slice->window_strides);
    }

    // TFDML #24881131
    if (Is64BitSignedIntegerType(ctx->GetOutputDataType(0))) {
      result = dml::ConvertInt32ToInt64(result);
    }

    Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
        scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

    Initialize(ctx, std::move(tensors), compiled_op.Get());
  }
};

#define REGISTER_KERNEL(type)        \
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
TF_CALL_half(REGISTER_KERNEL);
TF_CALL_float(REGISTER_KERNEL);
TF_CALL_bool(REGISTER_KERNEL);
TF_CALL_int8(REGISTER_KERNEL);
TF_CALL_int64(REGISTER_KERNEL);
#undef REGISTER_KERNEL

class StridedSliceAssignInitHelper : public InitializationHelper {
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

  bool IsNoOpKernel(
      OpKernelContext* ctx,
      absl::Span<const TensorShape> output_shapes) const override {
    if (!output_shapes.empty() && output_shapes[0].num_elements() == 0) {
      return true;
    }

    if (ctx->input(4).NumElements() == 0) {
      return true;
    }

    auto lock_cleanup = gtl::MakeCleanup([this] { Unlock(); });
    const Tensor input_tensor = GetInputTensor(ctx);

    if (input_tensor.NumElements() == 0) {
      return true;
    }

    return false;
  }

  StridedSliceAssignInitHelper(OpKernelContext* ctx,
                               std::shared_ptr<const Attributes> attr) {
    DCHECK(ctx->input_is_ref(0) || ctx->input(0).dtype() == DT_RESOURCE);

    if (ctx->input(0).dtype() == DT_RESOURCE) {
      OP_REQUIRES_OK(
          ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &input_resource_));
      input_resource_->mu()->lock_shared();
      locked_ = true;
    }

    const Tensor input = GetInputTensor(ctx);

    TensorShape processing_shape;
    bool slice_dim0 = true;
    bool is_simple_slice = true;
    gtl::InlinedVector<int64, 4> begin;
    gtl::InlinedVector<int64, 4> end;
    gtl::InlinedVector<int64, 4> strides;

    TensorShape input_shape = input.shape();
    TensorShape final_shape;

    OP_REQUIRES_OK(
        ctx, ValidateStridedSliceOp(
                 &ctx->input(1), &ctx->input(2), ctx->input(3), input_shape,
                 attr->begin_mask, attr->end_mask, attr->ellipsis_mask,
                 attr->new_axis_mask, attr->shrink_axis_mask, &processing_shape,
                 &final_shape, &is_identity_, &is_simple_slice, &slice_dim0,
                 &begin, &end, &strides));

    if (processing_shape.num_elements()) {
      TensorShape values_shape = ctx->input(4).shape();
      OP_REQUIRES(
          ctx, final_shape == values_shape,
          errors::Unimplemented(
              "sliced l-value shape ", final_shape.DebugString(),
              " does not match r-value shape ", values_shape.DebugString(),
              ". Automatic broadcasting not ", "yet implemented."));
    }

    // Attempt to simplify the slice into a lower-rank slice.
    simple_slice_ = SimplifySlice(input_shape, begin, end, strides);
    if (!simple_slice_) {
      OP_REQUIRES(
          ctx, simple_slice_,
          errors::InvalidArgument("DML only support slicing up to 8D inputs, "
                                  "but received ",
                                  input_shape.dims()));
    }
  }

  Tensor GetInputTensor(OpKernelContext* ctx) const {
    DCHECK(ctx->input_is_ref(0) || ctx->input(0).dtype() == DT_RESOURCE);

    return input_resource_ ? *input_resource_->tensor()
                           : ctx->mutable_input(0, false);
  }

  void Unlock() const {
    if (input_resource_ && locked_) {
      input_resource_->mu()->unlock_shared();
      locked_ = false;
    }
  }

  const absl::optional<SimplifiedSlice>& GetSimplifiedSlice() const {
    return simple_slice_;
  }

  const bool IsIdentity() const { return is_identity_; }

 private:
  absl::optional<SimplifiedSlice> simple_slice_;
  core::RefCountPtr<Var> input_resource_;
  mutable bool locked_ = false;
  bool is_identity_;
};

class DmlStridedSliceAssignKernel : public DmlKernel {
 public:
  using InitHelper = StridedSliceAssignInitHelper;

  explicit DmlStridedSliceAssignKernel(DmlKernelConstruction* ctx,
                                       const InitHelper* init_helper) {
    const Tensor input = init_helper->GetInputTensor(ctx->GetOpKernelContext());
    const TensorShape& input_shape = input.shape();
    const TensorShape& updates_shape = ctx->GetInputTensorShape(4);

    auto simple_slice = init_helper->GetSimplifiedSlice();
    auto dtype_tf = ctx->GetInputDataType(4);

    dml::TensorDimensions collapsed_input_sizes = {
        1,
        1,
        1,
        static_cast<uint32_t>(input_shape.num_elements()),
    };

    dml::TensorDimensions collapsed_updates_sizes = {
        1,
        1,
        1,
        static_cast<uint32_t>(updates_shape.num_elements()),
    };

    DmlTensorInfo updates;
    updates.kernel_index = 4;
    updates.desc = DmlTensorDesc::Create(dtype_tf, collapsed_updates_sizes,
                                         collapsed_updates_sizes);

    DmlTensorInfo output;
    output.kernel_index = 0;
    output.desc = DmlTensorDesc::Create(dtype_tf, collapsed_input_sizes,
                                        collapsed_input_sizes);

    DmlKernelTensors tensors;
    tensors.inputs = {updates};

    if (!init_helper->IsIdentity()) {
      DmlTensorInfo original_input;
      original_input.kernel_index = 0;
      original_input.desc = DmlTensorDesc::Create(
          dtype_tf, collapsed_input_sizes, collapsed_input_sizes);

      tensors.inputs.push_back(original_input);
    }

    tensors.outputs = {output};

    if (input.dtype() != DT_RESOURCE) {
      // The input ref and the output ref must refer to the same memory
      tensors.output_refs_forwarding = {0};
    }

    auto scope = dml::Graph(ctx->GetDmlDevice());
    auto inputs = GetDmlTensorDescs(tensors.inputs);
    auto updates_tensor = dml::InputTensor(scope, 0, inputs[0]);

    dml::Expression result;

    if (init_helper->IsIdentity()) {
      result = dml::Identity(updates_tensor);
    } else {
      auto original_input_tensor = dml::InputTensor(scope, 1, inputs[1]);

      auto indices_start = dml::ScalarUnion(0, DML_TENSOR_DATA_TYPE_UINT32);
      auto indices_delta = dml::ScalarUnion(1, DML_TENSOR_DATA_TYPE_UINT32);

      auto indices = dml::FillValueSequence(scope, simple_slice->input_sizes,
                                            DML_TENSOR_DATA_TYPE_UINT32,
                                            indices_start, indices_delta);

      auto sliced_indices =
          dml::Slice(indices, simple_slice->window_offset,
                     simple_slice->window_sizes, simple_slice->window_strides);

      sliced_indices =
          dml::Reinterpret(sliced_indices, collapsed_updates_sizes, {});

      result = dml::ScatterElements(original_input_tensor, sliced_indices,
                                    updates_tensor, 3);
    }

    // TFDML #24881131
    if (Is64BitSignedIntegerType(ctx->GetInputDataType(4))) {
      result = dml::ConvertInt32ToInt64(result);
    }

    Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
        scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

    Initialize(ctx, std::move(tensors), compiled_op.Get());
  }

  StatusOr<DmlGpuEvent> Compute(DmlKernelContext* ctx) const override {
    auto init_helper = ctx->GetInitializationHelper<InitHelper>();

    auto lock_cleanup =
        gtl::MakeCleanup([init_helper] { init_helper->Unlock(); });

    const Tensor input_tensor =
        init_helper->GetInputTensor(ctx->GetOpKernelContext());

    // Identity can be done in-place
    if (init_helper->IsIdentity()) {
      D3D12BufferRegion input_buffer =
          ctx->CreateBufferForTensor(ctx->GetInputTensor(4));

      D3D12BufferRegion output_buffer =
          ctx->CreateBufferForTensor(input_tensor);

      absl::optional<DML_BUFFER_BINDING> input_bindings[] = {
          input_buffer.GetBufferBinding(),
      };

      absl::optional<DML_BUFFER_BINDING> output_bindings[] = {
          output_buffer.GetBufferBinding(),
      };

      return DmlKernel::Compute(ctx, input_bindings, output_bindings);
    }

    // Create input buffers
    D3D12BufferRegion input_buffers[] = {
        ctx->CreateBufferForTensor(ctx->GetInputTensor(4)),
        ctx->CreateBufferForTensor(input_tensor),
    };

    // Create input bindings
    absl::optional<DML_BUFFER_BINDING> input_bindings[] = {
        input_buffers[0].GetBufferBinding(),
        input_buffers[1].GetBufferBinding(),
    };

    DmlBuffer output_buffer =
        ctx->AllocateDefaultBuffer(input_buffers[1].SizeInBytes());

    absl::optional<DML_BUFFER_BINDING> output_bindings[] = {
        output_buffer.GetBufferBinding(),
    };

    auto status_or_event =
        DmlKernel::Compute(ctx, input_bindings, output_bindings);
    if (!status_or_event.ok()) {
      return status_or_event;
    }

    ctx->CopyBufferToBuffer(input_buffers[1].Resource(),
                            input_buffers[1].Offset(), output_buffer.Resource(),
                            output_buffer.Offset(),
                            output_buffer.SizeInBytes());

    return ctx->InsertUavBarrier();
  }
};

#define REGISTER_KERNEL(type)                                                 \
  REGISTER_KERNEL_BUILDER(Name("StridedSliceAssign")                          \
                              .Device(DEVICE_DML)                             \
                              .TypeConstraint<type>("T")                      \
                              .HostMemory("begin")                            \
                              .HostMemory("end")                              \
                              .HostMemory("strides"),                         \
                          DmlKernelWrapper<DmlStridedSliceAssignKernel,       \
                                           GetOutputShapeAsInputShapeHelper>) \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("ResourceStridedSliceAssign")                                      \
          .Device(DEVICE_DML)                                                 \
          .TypeConstraint<type>("T")                                          \
          .HostMemory("ref")                                                  \
          .HostMemory("begin")                                                \
          .HostMemory("end")                                                  \
          .HostMemory("strides"),                                             \
      DmlKernelWrapper<DmlStridedSliceAssignKernel, NoOutputShapeHelper,      \
                       DmlKernelCachePolicy::Never>)                          \
// TODO(b/25387198): A special kernel exists for int32 (see
// strided_slice_op.cc).
TF_CALL_half(REGISTER_KERNEL);
TF_CALL_float(REGISTER_KERNEL);
TF_CALL_bool(REGISTER_KERNEL);
TF_CALL_int8(REGISTER_KERNEL);
TF_CALL_int64(REGISTER_KERNEL);
#undef REGISTER_KERNEL

}  // namespace tensorflow