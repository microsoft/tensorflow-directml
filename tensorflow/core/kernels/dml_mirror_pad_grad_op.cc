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

template <typename Tpadding>
class MirrorPadGradInitHelper : public InitializationHelper {
 public:
  struct Attributes {
    explicit Attributes(OpKernelConstruction* ctx) {
      MirrorPadMode mode;
      OP_REQUIRES_OK(ctx, ctx->GetAttr("mode", &mode));

      switch (mode) {
        case MirrorPadMode::SYMMETRIC: {
          offset = 0;
          break;
        }
        case MirrorPadMode::REFLECT: {
          offset = 1;
          break;
        }
        default:
          OP_REQUIRES(ctx, false,
                      errors::InvalidArgument(
                          "mode must be either REFLECT or SYMMETRIC."));
      }
    }

    int offset;
  };

  MirrorPadGradInitHelper(OpKernelContext* ctx,
                          std::shared_ptr<const Attributes> attr)
      : offset_(attr->offset) {
    const Tensor& in0 = ctx->input(0);

    OP_REQUIRES(
        ctx, in0.dims() <= kNcdhwDimensionCount,
        errors::Unimplemented(
            "DML doesn't support more than 5D for MirrorPadGrad, but found ",
            in0.dims()));

    const Tensor& in1 = ctx->input(1);
    const int dims = in0.dims();
    constexpr int kMinDims = 0;
    constexpr int kMaxDims = 5;
    OP_REQUIRES(ctx, kMinDims <= dims && dims <= kMaxDims,
                errors::Unimplemented("inputs rank not in [", kMinDims, ",",
                                      kMaxDims, "]: ", dims));
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsMatrix(in1.shape()) && in1.dim_size(1) == 2,
        errors::InvalidArgument("paddings must be a matrix with 2 columns: ",
                                in1.shape().DebugString()));
    OP_REQUIRES(
        ctx, dims == in1.dim_size(0),
        errors::InvalidArgument(
            "The first dimension of paddings must be the rank of inputs",
            in1.shape().DebugString(), " ", in0.shape().DebugString()));

    // Compute the shape of the output tensor, and allocate it.
    typename TTypes<Tpadding>::ConstMatrix paddings = in1.matrix<Tpadding>();
    for (int d = 0; d < dims; ++d) {
      const Tpadding before = paddings(d, 0);  // Pad before existing elements.
      const Tpadding after = paddings(d, 1);   // Pad after existing elements.
      OP_REQUIRES(ctx, before >= 0 && after >= 0,
                  errors::InvalidArgument(
                      "Paddings must be non-negative: ", before, ", ", after));

      const int64 out_size = in0.dim_size(d) - (before + after);
      if (offset_ == 0) {  // SYMMETRIC mode.
        OP_REQUIRES(ctx, before <= out_size && after <= out_size,
                    errors::InvalidArgument("paddings must be no greater "
                                            "than the output dimension size: ",
                                            before, ", ", after,
                                            " greater than ", out_size));
      } else if (offset_ == 1) {  // REFLECT mode.
        OP_REQUIRES(ctx, before < out_size && after < out_size,
                    errors::InvalidArgument("paddings must be less than"
                                            " the output dimension size: ",
                                            before, ", ", after,
                                            " not less than ", out_size));
      }
      output_shape_.AddDim(out_size);
    }

    is_identity_ = output_shape_ == in0.shape();
  }

  TensorShape GetOutputShape() const { return output_shape_; }
  bool IsIdentity() const { return is_identity_; }
  int GetOffset() const { return offset_; }

 private:
  TensorShape output_shape_;
  int offset_;
  bool is_identity_;
};

template <typename Tpadding>
class MirrorPadGradShapeHelper : public ShapeHelper {
 public:
  std::vector<TensorShape> GetOutputShapes(
      OpKernelContext* ctx,
      const InitializationHelper* initialization_helper) const override {
    auto init_helper = static_cast<const MirrorPadGradInitHelper<Tpadding>*>(
        initialization_helper);
    return {init_helper->GetOutputShape()};
  }
};

template <typename Tpadding>
class DmlMirrorPadGradKernel : public DmlKernel {
 public:
  using InitHelper = tensorflow::MirrorPadGradInitHelper<Tpadding>;

  DmlMirrorPadGradKernel(DmlKernelConstruction* ctx,
                         const InitHelper* init_helper) {
    DmlKernelParams params;
    params.kernel_input_indices = {0};
    DmlKernelTensors tensors = GetTensorInfos(ctx, params);

    auto inputs = GetDmlTensorDescs(tensors.inputs);

    if (init_helper->IsIdentity()) {
      auto outputs = GetDmlTensorDescs(tensors.outputs);

      DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC identity_desc = {};
      identity_desc.InputTensor = inputs.data();
      identity_desc.OutputTensor = outputs.data();

      DML_OPERATOR_DESC op_desc = {DML_OPERATOR_ELEMENT_WISE_IDENTITY,
                                   &identity_desc};
      Initialize(ctx, std::move(tensors), op_desc);
      return;
    }

    auto scope = dml::Graph(ctx->GetDmlDevice());
    auto result = dml::InputTensor(scope, 0, inputs[0]);

    const Tensor& paddings_tensor = ctx->GetConstantInputTensor(1);
    typename TTypes<Tpadding>::ConstMatrix paddings =
        paddings_tensor.matrix<Tpadding>();

    TensorShape dml_input_shape = ctx->GetInputTensorShape(0);

    int pad_dim_start = 0;

    if (dml_input_shape.dims() < kNchwDimensionCount) {
      pad_dim_start = kNchwDimensionCount - dml_input_shape.dims();
      for (int i = 0; i < pad_dim_start; ++i) {
        dml_input_shape.InsertDim(0, 1);
      }
    }

    absl::InlinedVector<int32_t, 4> slice_strides(dml_input_shape.dims(), 1);

    // Perform the slicing logic twice for every dimension (for before and after
    // paddings)
    for (int i = 0; i < dml_input_shape.dims(); ++i) {
      for (int pad_index = 0; pad_index < 2; ++pad_index) {
        const Tpadding padding = i < pad_dim_start
                                     ? Tpadding()
                                     : paddings(i - pad_dim_start, pad_index);

        if (padding > 0) {
          dml::TensorDesc::Dimensions sizes = result.GetOutputDesc().sizes;
          dml::TensorDesc::Dimensions offsets(sizes.size(), 0);

          uint32_t slice_offset = pad_index == 0 ? 0 : sizes[i] - padding;
          offsets[i] = slice_offset;

          dml::TensorDesc::Dimensions slice_sizes = sizes;
          slice_sizes[i] = padding;

          auto slice = dml::Slice(result, offsets, slice_sizes, slice_strides);

          uint32_t needed_padding = sizes[i] - padding * 2;

          // Pad the slice to make it the same size as the rest of the tensor,
          // which allows for elementwise add to be performed.
          if (needed_padding != 0) {
            dml::TensorDesc::Dimensions pad_padding[2];
            pad_padding[0].assign(sizes.size(), 0);
            pad_padding[1].assign(sizes.size(), 0);

            pad_padding[pad_index][i] =
                needed_padding - init_helper->GetOffset();

            pad_padding[1 - pad_index][i] = init_helper->GetOffset();
            slice = dml::Padding(slice, DML_PADDING_MODE_CONSTANT, 0.0,
                                 pad_padding[0], pad_padding[1]);
          }

          // Reverse the values on the axis' direction to correctly fold them on
          // the non-padding part of the tensor
          if (slice.GetOutputDesc().sizes[i] > 1) {
            auto seq_lengths_sizes = slice.GetOutputDesc().sizes;

            DML_SCALAR_UNION seq_lengths_value;
            seq_lengths_value.UInt32 = seq_lengths_sizes[i];

            seq_lengths_sizes[i] = 1;

            auto seq_lengths = dml::FillValueConstant(
                scope, seq_lengths_sizes, DML_TENSOR_DATA_TYPE_UINT32,
                seq_lengths_value);

            slice = dml::ReverseSubsequences(slice, seq_lengths, i);
          }

          dml::TensorDesc::Dimensions sliced_result_offsets(sizes.size(), 0);
          uint32_t sliced_result_offset = pad_index == 0 ? padding : 0;
          sliced_result_offsets[i] = sliced_result_offset;

          auto sliced_result_sizes = result.GetOutputDesc().sizes;
          sliced_result_sizes[i] -= padding;

          // Extract the non-padded part of the input
          result = dml::Slice(result, sliced_result_offsets,
                              sliced_result_sizes, slice_strides);

          // Finally, add the slice to the result
          result += slice;
        }
      }
    }

    Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
        scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

    Initialize(ctx, std::move(tensors), compiled_op.Get());
  }
};

#define REGISTER_DML_KERNEL(T)                                                \
  REGISTER_KERNEL_BUILDER(Name("MirrorPadGrad")                               \
                              .Device(DEVICE_DML)                             \
                              .TypeConstraint<T>("T")                         \
                              .TypeConstraint<int32>("Tpaddings")             \
                              .HostMemory("paddings"),                        \
                          DmlKernelWrapper<DmlMirrorPadGradKernel<int32>,     \
                                           MirrorPadGradShapeHelper<int32>>); \
  REGISTER_KERNEL_BUILDER(Name("MirrorPadGrad")                               \
                              .Device(DEVICE_DML)                             \
                              .TypeConstraint<T>("T")                         \
                              .TypeConstraint<int64>("Tpaddings")             \
                              .HostMemory("paddings"),                        \
                          DmlKernelWrapper<DmlMirrorPadGradKernel<int64>,     \
                                           MirrorPadGradShapeHelper<int64>>);

TF_CALL_half(REGISTER_DML_KERNEL);
TF_CALL_float(REGISTER_DML_KERNEL);

}  // namespace tensorflow
