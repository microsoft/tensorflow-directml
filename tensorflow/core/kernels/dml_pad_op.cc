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
#include "tensorflow/core/util/mirror_pad_mode.h"

namespace tensorflow {

struct SimplePad {
  absl::InlinedVector<uint32_t, 4> in_shape;
  absl::InlinedVector<uint32_t, 4> out_shape;
  absl::InlinedVector<uint32_t, 4> start_padding;
  absl::InlinedVector<uint32_t, 4> end_padding;
};

// Coalesces padded dimensions with all contiguous non-padded dimensions that
// follow. For example:
//
// original input shape = [2,2,2,3]
// original paddings = [[1,1],[0,0],[1,2],[0,0]]
// original output shape = [4,2,5,3]
//
// coalesced input shape = [4,6]
// coalesced paddings = [[2,2],[3,6]]
// coalesced output shape = [8,15]
//
// This is more aggressive than the implementation in the CPU kernel, which only
// collapses adjacent non-padded dimensions (none in the above example). This
// function also inflates the shapes and paddings to meet DML requirements,
// converts paddings to two separate arrays, and casts signed integers to
// unsigned.
template <typename Tpadding>
absl::optional<SimplePad> SimplifyPad(const TensorShape& input_shape,
                                      const Tensor& paddings_tensor,
                                      size_t min_output_size = 4,
                                      size_t max_output_size = 5) {
  auto paddings = paddings_tensor.matrix<Tpadding>();
  DCHECK(input_shape.dims() == paddings.dimension(0));

  SimplePad simple_pad = {};

  int i = 0;
  while (i < paddings.dimension(0)) {
    auto size = static_cast<uint32_t>(input_shape.dim_size(i));
    auto start_pad = static_cast<uint32_t>(paddings(i, 0));
    auto end_pad = static_cast<uint32_t>(paddings(i, 1));

    // Coalesce subsequent non-padded dims into the current dim.
    int j = i + 1;
    while (j < paddings.dimension(0) && paddings(j, 0) == 0 &&
           paddings(j, 1) == 0) {
      auto other_dim_size = static_cast<uint32_t>(input_shape.dim_size(j));
      size *= other_dim_size;
      start_pad *= other_dim_size;
      end_pad *= other_dim_size;
      j++;
    }
    i = j;

    simple_pad.in_shape.push_back(size);
    simple_pad.out_shape.push_back(size + start_pad + end_pad);
    simple_pad.start_padding.push_back(start_pad);
    simple_pad.end_padding.push_back(end_pad);
  }

  if (simple_pad.in_shape.size() > max_output_size) {
    return absl::nullopt;
  }

  // Inflate DML shapes/pads to the minimum required size.
  for (size_t i = simple_pad.in_shape.size(); i < min_output_size; i++) {
    simple_pad.in_shape.insert(simple_pad.in_shape.begin(), 1);
    simple_pad.out_shape.insert(simple_pad.out_shape.begin(), 1);
    simple_pad.start_padding.insert(simple_pad.start_padding.begin(), 0);
    simple_pad.end_padding.insert(simple_pad.end_padding.begin(), 0);
  }

  return simple_pad;
}

template <typename T, typename Tpadding>
class PadInitHelper : public InitializationHelper {
 public:
  struct Attributes {
    explicit Attributes(OpKernelConstruction* ctx) {
      MirrorPadMode mode;
      if (ctx->GetAttr("mode", &mode).ok()) {
        switch (mode) {
          case MirrorPadMode::SYMMETRIC:
            padding_mode = DML_PADDING_MODE_SYMMETRIC;
            break;
          case MirrorPadMode::REFLECT:
            padding_mode = DML_PADDING_MODE_REFLECTION;
            break;
          default:
            OP_REQUIRES(ctx, false,
                        errors::InvalidArgument(
                            "mode must be either REFLECT or SYMMETRIC."));
        }
      } else {
        padding_mode = DML_PADDING_MODE_CONSTANT;
      }
    }

    DML_PADDING_MODE padding_mode;
  };

  PadInitHelper(OpKernelContext* ctx, std::shared_ptr<const Attributes> attr)
      : padding_mode_(attr->padding_mode) {
    const Tensor& input = ctx->input(0);
    const Tensor& paddings = ctx->input(1);

    const int dims = input.dims();
    static const int kMinDims = 0;
    static const int kMaxDims = 6;
    OP_REQUIRES(ctx, kMinDims <= dims && dims <= kMaxDims,
                errors::Unimplemented("inputs rank not in [", kMinDims, ",",
                                      kMaxDims, "]: ", dims));

    OP_REQUIRES(
        ctx,
        TensorShapeUtils::IsMatrix(paddings.shape()) &&
            paddings.dim_size(1) == 2,
        errors::InvalidArgument("paddings must be a matrix with 2 columns: ",
                                paddings.shape().DebugString()));

    const int fixed_dims = (ctx->op_kernel().allow_legacy_scalars() &&
                            dims == 0 && paddings.dim_size(0) == 1)
                               ? 1
                               : dims;

    OP_REQUIRES(
        ctx, dims == paddings.dim_size(0),
        errors::InvalidArgument(
            "The first dimension of paddings must be the rank of inputs",
            paddings.shape().DebugString(), " ", input.shape().DebugString()));

    pad_value_ = T();
    if (ctx->num_inputs() == 3) {
      const Tensor& constant_values = ctx->input(2);
      OP_REQUIRES(
          ctx, TensorShapeUtils::IsScalar(constant_values.shape()),
          errors::InvalidArgument("constant_values must be a scalar. Found: ",
                                  constant_values.shape().DebugString()));
      pad_value_ = ctx->input(2).scalar<T>()();
    }

    typename TTypes<Tpadding>::ConstMatrix pads = paddings.matrix<Tpadding>();

    for (int d = 0; d < fixed_dims; ++d) {
      const Tpadding before_d = pads(d, 0);
      const Tpadding after_d = pads(d, 1);
      OP_REQUIRES(ctx, before_d >= 0 && after_d >= 0,
                  errors::InvalidArgument("Paddings must be non-negative: ",
                                          before_d, " ", after_d));

      if (padding_mode_ == DML_PADDING_MODE_SYMMETRIC) {
        OP_REQUIRES(
            ctx, before_d <= input.dim_size(d) && after_d <= input.dim_size(d),
            errors::InvalidArgument("paddings must be no greater "
                                    "than the dimension size: ",
                                    before_d, ", ", after_d, " greater than ",
                                    input.dim_size(d)));
      } else if (padding_mode_ == DML_PADDING_MODE_REFLECTION) {
        OP_REQUIRES(
            ctx, before_d < input.dim_size(d) && after_d < input.dim_size(d),
            errors::InvalidArgument("paddings must be less than"
                                    " the dimension size: ",
                                    before_d, ", ", after_d, " not less than ",
                                    input.dim_size(d)));
      }

      const int64 size_d =
          (ctx->op_kernel().allow_legacy_scalars() && d == input.dims())
              ? 1
              : input.dim_size(d);
      output_shape_.AddDim(before_d + size_d + after_d);
    }

    simple_pad_ = SimplifyPad<Tpadding>(input.shape(), paddings);
    OP_REQUIRES(ctx, simple_pad_.has_value(),
                errors::InvalidArgument(
                    "DML can only handle up to 5D padding, but the given shape "
                    "and paddings cannot be simplified to 5D."));
  }

  const TensorShape& GetOutputShape() const { return output_shape_; }

  T GetPadValue() const { return pad_value_; }

  DML_PADDING_MODE GetPaddingMode() const { return padding_mode_; }

  const absl::optional<SimplePad>& GetSimplePad() const { return simple_pad_; }

 private:
  TensorShape output_shape_;
  T pad_value_;
  absl::optional<SimplePad> simple_pad_;
  DML_PADDING_MODE padding_mode_;
};

template <typename T, typename Tpadding>
class PadShapeHelper : public ShapeHelper {
 public:
  std::vector<TensorShape> GetOutputShapes(
      OpKernelContext* ctx,
      const InitializationHelper* initialization_helper) const override {
    auto init_helper =
        static_cast<const PadInitHelper<T, Tpadding>*>(initialization_helper);
    return {init_helper->GetOutputShape()};
  }
};

template <typename T, typename Tpadding>
class DmlPadKernel : public DmlKernel {
 public:
  using InitHelper = tensorflow::PadInitHelper<T, Tpadding>;

  explicit DmlPadKernel(DmlKernelConstruction* ctx,
                        const InitHelper* init_helper) {
    auto dtype = ctx->GetInputDataType(0);
    DCHECK(dtype == ctx->GetOutputDataType(0));
    auto pad = init_helper->GetSimplePad();
    DCHECK(pad.has_value());  // Validated in init_helper

    DmlTensorInfo in;
    in.kernel_index = 0;
    in.desc = DmlTensorDesc::Create(dtype, pad->in_shape, pad->in_shape);
    auto in_desc = in.desc.GetDmlDesc();

    DmlTensorInfo out;
    out.kernel_index = 0;
    out.desc = DmlTensorDesc::Create(dtype, pad->out_shape, pad->out_shape);
    auto out_desc = out.desc.GetDmlDesc();

    DmlKernelTensors tensors;
    tensors.inputs = {in};
    tensors.outputs = {out};

    DML_PADDING_OPERATOR_DESC desc = {};
    desc.InputTensor = &in_desc;
    desc.OutputTensor = &out_desc;
    desc.PaddingMode = init_helper->GetPaddingMode();
    desc.PaddingValue = static_cast<float>(init_helper->GetPadValue());
    desc.DimensionCount = pad->in_shape.size();
    desc.StartPadding = pad->start_padding.data();
    desc.EndPadding = pad->end_padding.data();

    DML_OPERATOR_DESC op_desc = {DML_OPERATOR_PADDING, &desc};

    Initialize(ctx, std::move(tensors), op_desc);
  }
};

template <typename T, typename Tpadding>
using PadWrapper =
    DmlKernelWrapper<DmlPadKernel<T, Tpadding>, PadShapeHelper<T, Tpadding>>;

#define REGISTER_PAD_KERNEL(T, Tpadding)                             \
  REGISTER_KERNEL_BUILDER(Name("Pad")                                \
                              .Device(DEVICE_DML)                    \
                              .TypeConstraint<T>("T")                \
                              .TypeConstraint<Tpadding>("Tpaddings") \
                              .HostMemory("paddings"),               \
                          PadWrapper<T, Tpadding>);                  \
  REGISTER_KERNEL_BUILDER(Name("PadV2")                              \
                              .Device(DEVICE_DML)                    \
                              .TypeConstraint<T>("T")                \
                              .TypeConstraint<Tpadding>("Tpaddings") \
                              .HostMemory("paddings")                \
                              .HostMemory("constant_values"),        \
                          PadWrapper<T, Tpadding>);                  \
  REGISTER_KERNEL_BUILDER(Name("MirrorPad")                          \
                              .Device(DEVICE_DML)                    \
                              .TypeConstraint<T>("T")                \
                              .TypeConstraint<Tpadding>("Tpaddings") \
                              .HostMemory("paddings"),               \
                          PadWrapper<T, Tpadding>);

#define REGISTER_PAD_KERNELS(T)  \
  REGISTER_PAD_KERNEL(T, int32); \
  REGISTER_PAD_KERNEL(T, int64);

// T=INT32 is special and uses host memory for all tensors (registered
// along with other devices in pad_op.cc).
TF_CALL_half(REGISTER_PAD_KERNELS);
TF_CALL_float(REGISTER_PAD_KERNELS);
TF_CALL_uint8(REGISTER_PAD_KERNELS);
TF_CALL_uint16(REGISTER_PAD_KERNELS);
TF_CALL_uint32(REGISTER_PAD_KERNELS);
TF_CALL_int8(REGISTER_PAD_KERNELS);
TF_CALL_int16(REGISTER_PAD_KERNELS);

}  // namespace tensorflow