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
#include "tensorflow/core/kernels/dml_extract_patches_helpers.h"
#include "tensorflow/core/kernels/dml_kernel_wrapper.h"
#include "tensorflow/core/kernels/dml_ops_common.h"

namespace tensorflow {

static inline void ParseAttributeVec4(OpKernelConstruction* ctx,
                                      const string& attr_name,
                                      /*out*/ absl::Span<uint32_t> attr) {
  std::vector<int32> int32_attr;
  OP_REQUIRES_OK(ctx, ctx->GetAttr(attr_name, &int32_attr));
  OP_REQUIRES(
      ctx, (int32_attr)[0] == 1 && (int32_attr)[3] == 1,
      errors::Unimplemented("Only support ", attr_name, " across space."));
  OP_REQUIRES(ctx, (int32_attr)[1] >= 1 && (int32_attr)[2] >= 1,
              errors::OutOfRange(attr_name, " is out of range."));

  std::transform(int32_attr.begin(), int32_attr.end(), attr.begin(),
                 [](int32 val) { return static_cast<uint32_t>(val); });
}

class ExtractImagePatchesInitHelper : public InitializationHelper {
 public:
  struct Attributes {
    explicit Attributes(OpKernelConstruction* ctx) {
      ParseAttributeVec4(ctx, "ksizes", ksizes);
      ParseAttributeVec4(ctx, "strides", strides);
      ParseAttributeVec4(ctx, "rates", rates);
      OP_REQUIRES_OK(ctx, ctx->GetAttr("padding", &padding));
    }

    uint32_t ksizes[4];
    uint32_t strides[4];
    uint32_t rates[4];
    Padding padding;
  };

  explicit ExtractImagePatchesInitHelper(OpKernelContext* ctx,
                                         std::shared_ptr<const Attributes> attr)
      : attr_(std::move(attr)) {
    // Input tensor is of the following dimensions:
    // [ batch, in_rows, in_cols, channels ]
    const Tensor& input = ctx->input(0);
    OP_REQUIRES(ctx, input.dims() == 4,
                errors::InvalidArgument("input must be 4-dimensional",
                                        input.shape().DebugString()));

    const int batch = input.dim_size(0);
    const int in_rows = input.dim_size(1);
    const int in_cols = input.dim_size(2);
    const int depth = input.dim_size(3);

    const int ksize_rows = attr_->ksizes[1];
    const int ksize_cols = attr_->ksizes[2];

    const int stride_rows = attr_->strides[1];
    const int stride_cols = attr_->strides[2];

    const int rate_rows = attr_->rates[1];
    const int rate_cols = attr_->rates[2];

    const int ksize_rows_eff = ksize_rows + (ksize_rows - 1) * (rate_rows - 1);
    const int ksize_cols_eff = ksize_cols + (ksize_cols - 1) * (rate_cols - 1);

    int64 out_rows, out_cols;
    int64 rows_padding_before, cols_padding_before;
    int64 rows_padding_after, cols_padding_after;

    OP_REQUIRES_OK(ctx,
                   GetWindowedOutputSizeVerbose(
                       in_rows, ksize_rows_eff, stride_rows, attr_->padding,
                       &out_rows, &rows_padding_before, &rows_padding_after));
    OP_REQUIRES_OK(ctx,
                   GetWindowedOutputSizeVerbose(
                       in_cols, ksize_cols_eff, stride_cols, attr_->padding,
                       &out_cols, &cols_padding_before, &cols_padding_after));

    start_padding_[0] = 0;
    start_padding_[1] = static_cast<uint32_t>(rows_padding_before);
    start_padding_[2] = static_cast<uint32_t>(cols_padding_before);
    start_padding_[3] = 0;

    end_padding_[0] = 0;
    end_padding_[1] = static_cast<uint32_t>(rows_padding_after);
    end_padding_[2] = static_cast<uint32_t>(cols_padding_after);
    end_padding_[3] = 0;

    output_shape_ = TensorShape({
        batch,
        out_rows,
        out_cols,
        ksize_rows * ksize_cols * depth,
    });
  }

  TensorShape GetOutputShape() const { return output_shape_; }
  absl::Span<const uint32_t> GetStartPadding() const { return start_padding_; }
  absl::Span<const uint32_t> GetEndPadding() const { return end_padding_; }
  absl::Span<const uint32_t> GetWindowSizes() const { return attr_->ksizes; }
  absl::Span<const uint32_t> GetWindowStrides() const { return attr_->strides; }
  absl::Span<const uint32_t> GetWindowRates() const { return attr_->rates; }

 private:
  std::shared_ptr<const Attributes> attr_;
  TensorShape output_shape_;
  uint32_t start_padding_[4];
  uint32_t end_padding_[4];
};

class ExtractImagePatchesShapeHelper : public ShapeHelper {
 public:
  std::vector<TensorShape> GetOutputShapes(
      OpKernelContext* ctx,
      const InitializationHelper* initialization_helper) const override {
    auto init_helper = static_cast<const ExtractImagePatchesInitHelper*>(
        initialization_helper);
    return {init_helper->GetOutputShape()};
  }
};

class DmlExtractImagePatchesKernel : public DmlKernel {
 public:
  using InitHelper = ExtractImagePatchesInitHelper;
  explicit DmlExtractImagePatchesKernel(DmlKernelConstruction* ctx,
                                        const InitHelper* init_helper) {
    absl::Span<const uint32_t> window_sizes = init_helper->GetWindowSizes();
    absl::Span<const uint32_t> window_strides = init_helper->GetWindowStrides();
    absl::Span<const uint32_t> window_rates = init_helper->GetWindowRates();
    absl::Span<const uint32_t> start_padding = init_helper->GetStartPadding();
    absl::Span<const uint32_t> end_padding = init_helper->GetEndPadding();
    auto output_sizes = NarrowTensorShape<5>(ctx->GetOutputTensorShape(0));

    DmlKernelTensors tensors = GetTensorInfos(ctx, DmlKernelParams{});
    auto inputs = GetDmlTensorDescs(tensors.inputs);
    auto scope = dml::Graph(ctx->GetDmlDevice());
    auto input = dml::InputTensor(scope, 0, inputs[0]);
    auto result = dml::ExtractPatches(scope, input, window_sizes,
                                      window_strides, window_rates,
                                      start_padding, end_padding, output_sizes);

    Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
        scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

    Initialize(ctx, std::move(tensors), compiled_op.Get());
  }
};

#define REGISTER_DML_KERNEL(T)                                               \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("ExtractImagePatches").Device(DEVICE_DML).TypeConstraint<T>("T"), \
      DmlKernelWrapper<DmlExtractImagePatchesKernel,                         \
                       ExtractImagePatchesShapeHelper>);

TF_CALL_float(REGISTER_DML_KERNEL);
TF_CALL_half(REGISTER_DML_KERNEL);
#undef REGISTER_DML_KERNEL

}  // namespace tensorflow