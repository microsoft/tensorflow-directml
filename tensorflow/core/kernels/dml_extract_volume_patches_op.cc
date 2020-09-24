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

static inline void ParseAttributeVec5(OpKernelConstruction* ctx,
                                      const string& attr_name,
                                      /*out*/ absl::Span<uint32_t> attr) {
  std::vector<int32> int32_attr;
  OP_REQUIRES_OK(ctx, ctx->GetAttr(attr_name, &int32_attr));
  OP_REQUIRES(
      ctx, (int32_attr)[0] == 1 && (int32_attr)[4] == 1,
      errors::Unimplemented("Only support ", attr_name, " across space."));
  OP_REQUIRES(
      ctx, (int32_attr)[1] >= 1 && (int32_attr)[2] >= 1 && (int32_attr)[3] >= 1,
      errors::OutOfRange(attr_name, " is out of range."));

  std::transform(int32_attr.begin(), int32_attr.end(), attr.begin(),
                 [](int32 val) { return static_cast<uint32_t>(val); });
}

class ExtractVolumePatchesInitHelper : public InitializationHelper {
 public:
  struct Attributes {
    explicit Attributes(OpKernelConstruction* ctx) {
      ParseAttributeVec5(ctx, "ksizes", ksizes);
      ParseAttributeVec5(ctx, "strides", strides);
      OP_REQUIRES_OK(ctx, ctx->GetAttr("padding", &padding));

      // TODO: Support rates if the CPU/GPU kernels ever support them (still
      // unsupported as of TensorFlow 2.3.0). For now, we set them to 1 to get
      // the same results as the CPU and the GPU.
      rates[0] = 1;
      rates[1] = 1;
      rates[2] = 1;
      rates[3] = 1;
      rates[4] = 1;
    }

    uint32_t ksizes[5];
    uint32_t strides[5];
    uint32_t rates[5];
    Padding padding;
  };

  explicit ExtractVolumePatchesInitHelper(
      OpKernelContext* ctx, std::shared_ptr<const Attributes> attr)
      : attr_(std::move(attr)) {
    // Input tensor is of the following dimensions:
    // [ batch, in_planes, in_rows, in_cols, channels ]
    const Tensor& input = ctx->input(0);
    OP_REQUIRES(ctx, input.dims() == 5,
                errors::InvalidArgument("input must be 5-dimensional",
                                        input.shape().DebugString()));

    const int batch = input.dim_size(0);
    const int in_planes = input.dim_size(1);
    const int in_rows = input.dim_size(2);
    const int in_cols = input.dim_size(3);
    const int depth = input.dim_size(4);

    const int ksize_planes = attr_->ksizes[1];
    const int ksize_rows = attr_->ksizes[2];
    const int ksize_cols = attr_->ksizes[3];

    const int stride_planes = attr_->strides[1];
    const int stride_rows = attr_->strides[2];
    const int stride_cols = attr_->strides[3];

    int64 out_planes, out_rows, out_cols;
    int64 planes_padding_before, rows_padding_before, cols_padding_before;
    int64 planes_padding_after, rows_padding_after, cols_padding_after;

    OP_REQUIRES_OK(
        ctx, GetWindowedOutputSizeVerbose(
                 in_planes, ksize_planes, stride_planes, attr_->padding,
                 &out_planes, &planes_padding_before, &planes_padding_after));
    OP_REQUIRES_OK(
        ctx, GetWindowedOutputSizeVerbose(
                 in_rows, ksize_rows, stride_rows, attr_->padding, &out_rows,
                 &rows_padding_before, &rows_padding_after));
    OP_REQUIRES_OK(
        ctx, GetWindowedOutputSizeVerbose(
                 in_cols, ksize_cols, stride_cols, attr_->padding, &out_cols,
                 &cols_padding_before, &cols_padding_after));

    start_padding_[0] = 0;
    start_padding_[1] = static_cast<uint32_t>(planes_padding_before);
    start_padding_[2] = static_cast<uint32_t>(rows_padding_before);
    start_padding_[3] = static_cast<uint32_t>(cols_padding_before);
    start_padding_[4] = 0;

    end_padding_[0] = 0;
    end_padding_[1] = static_cast<uint32_t>(planes_padding_after);
    end_padding_[2] = static_cast<uint32_t>(rows_padding_after);
    end_padding_[3] = static_cast<uint32_t>(cols_padding_after);
    end_padding_[4] = 0;

    output_shape_ = TensorShape({
        batch,
        out_planes,
        out_rows,
        out_cols,
        ksize_planes * ksize_rows * ksize_cols * depth,
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
  uint32_t start_padding_[5];
  uint32_t end_padding_[5];
};

class ExtractVolumePatchesShapeHelper : public ShapeHelper {
 public:
  std::vector<TensorShape> GetOutputShapes(
      OpKernelContext* ctx,
      const InitializationHelper* initialization_helper) const override {
    auto init_helper = static_cast<const ExtractVolumePatchesInitHelper*>(
        initialization_helper);
    return {init_helper->GetOutputShape()};
  }
};

class DmlExtractVolumePatchesKernel : public DmlKernel {
 public:
  using InitHelper = ExtractVolumePatchesInitHelper;
  explicit DmlExtractVolumePatchesKernel(DmlKernelConstruction* ctx,
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

#define REGISTER_DML_KERNEL(T)                                                \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("ExtractVolumePatches").Device(DEVICE_DML).TypeConstraint<T>("T"), \
      DmlKernelWrapper<DmlExtractVolumePatchesKernel,                         \
                       ExtractVolumePatchesShapeHelper>);

TF_CALL_float(REGISTER_DML_KERNEL);
TF_CALL_half(REGISTER_DML_KERNEL);
#undef REGISTER_DML_KERNEL

}  // namespace tensorflow