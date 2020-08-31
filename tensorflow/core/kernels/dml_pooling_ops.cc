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
#include "tensorflow/core/kernels/pooling_ops_3d.h"
#include "tensorflow/core/kernels/pooling_ops_common.h"

namespace tensorflow {

using Microsoft::WRL::ComPtr;

struct DmlPoolValues {
  absl::InlinedVector<uint32_t, 3> strides;
  absl::InlinedVector<uint32_t, 3> window_size;
  absl::InlinedVector<uint32_t, 3> start_padding;
  absl::InlinedVector<uint32_t, 3> end_padding;
  TensorFormat data_format;
};

class PoolInitHelper : public InitializationHelper {
 public:
  struct Attributes {
    explicit Attributes(OpKernelConstruction* ctx) {
      // AvgPool and MaxPool take 1 input, while MaxPoolV2 takes 3.
      CHECK(ctx->num_inputs() == 1 || ctx->num_inputs() == 3);

      std::string data_format_attr;
      if (ctx->GetAttr("data_format", &data_format_attr).ok()) {
        OP_REQUIRES(ctx, FormatFromString(data_format_attr, &data_format),
                    errors::InvalidArgument("Invalid data format"));
      }

      if (ctx->num_inputs() == 1) {
        ksize.emplace();
        OP_REQUIRES_OK(ctx, ctx->GetAttr("ksize", &ksize.value()));
        CHECK(ksize->size() == kNchwDimensionCount ||
              ksize->size() == kNcdhwDimensionCount);
        for (int i = 0; i < ksize->size(); ++i) {
          CHECK((*ksize)[i] > 0);
        }

        stride.emplace();
        OP_REQUIRES_OK(ctx, ctx->GetAttr("strides", &stride.value()));
        CHECK(stride->size() == ksize->size());

        // Can't pool over the batch dimension
        CHECK((*ksize)[0] == 1 && (*stride)[0] == 1);
      }

      OP_REQUIRES_OK(ctx, ctx->GetAttr("padding", &padding));
    }

    // These attributes aren't present for MaxPoolV2 (because it takes them as
    // input tensors)
    absl::optional<std::vector<int32>> ksize;
    absl::optional<std::vector<int32>> stride;

    Padding padding;
    TensorFormat data_format = FORMAT_NHWC;
  };

  PoolInitHelper(OpKernelContext* ctx, std::shared_ptr<const Attributes> attr)
      : attr_(attr) {}

  const absl::optional<std::vector<int32>>& GetKernelSizes() const {
    return attr_->ksize;
  }

  const absl::optional<std::vector<int32>>& GetKernelStrides() const {
    return attr_->stride;
  }

  Padding GetPadding() const { return attr_->padding; }
  TensorFormat GetDataFormat() const { return attr_->data_format; }

 private:
  const std::shared_ptr<const Attributes> attr_;
};

class MaxPoolGradInitHelper : public InitializationHelper {
 public:
  struct Attributes {
    explicit Attributes(OpKernelConstruction* ctx) {
      // MaxPoolGrad inputs: input, output, inputGradient
      // MaxPoolGradV2 inputs: input, output, inputGradient, ksize, strides
      CHECK(ctx->num_inputs() == 3 || ctx->num_inputs() == 5);

      std::string data_format_attr;
      if (ctx->GetAttr("data_format", &data_format_attr).ok()) {
        OP_REQUIRES(ctx, FormatFromString(data_format_attr, &data_format),
                    errors::InvalidArgument("Invalid data format"));
      }

      if (ctx->num_inputs() == 3) {
        ksize.emplace();
        OP_REQUIRES_OK(ctx, ctx->GetAttr("ksize", &ksize.value()));
        CHECK(ksize->size() == kNchwDimensionCount ||
              ksize->size() == kNcdhwDimensionCount);
        for (int i = 0; i < ksize->size(); ++i) {
          CHECK((*ksize)[i] > 0);
        }

        stride.emplace();
        OP_REQUIRES_OK(ctx, ctx->GetAttr("strides", &stride.value()));
        CHECK(stride->size() == ksize->size());

        // Can't pool over the batch dimension
        CHECK((*ksize)[0] == 1 && (*stride)[0] == 1);
      }

      OP_REQUIRES_OK(ctx, ctx->GetAttr("padding", &padding));
    }

    // These attributes aren't present for MaxPoolGradV2 (because it takes them
    // as input tensors)
    absl::optional<std::vector<int32>> ksize;
    absl::optional<std::vector<int32>> stride;

    Padding padding;
    TensorFormat data_format = FORMAT_NHWC;
  };

  MaxPoolGradInitHelper(OpKernelContext* ctx,
                        std::shared_ptr<const Attributes> attr)
      : attr_(attr) {}

  const absl::optional<std::vector<int32>>& GetKernelSizes() const {
    return attr_->ksize;
  }

  const absl::optional<std::vector<int32>>& GetKernelStrides() const {
    return attr_->stride;
  }

  Padding GetPadding() const { return attr_->padding; }
  TensorFormat GetDataFormat() const { return attr_->data_format; }

 private:
  const std::shared_ptr<const Attributes> attr_;
};

class AvgPoolGradInitHelper : public InitializationHelper {
 public:
  struct Attributes {
    explicit Attributes(OpKernelConstruction* ctx) {
      CHECK(ctx->num_inputs() == 2);

      std::string data_format_attr;
      if (ctx->GetAttr("data_format", &data_format_attr).ok()) {
        OP_REQUIRES(ctx, FormatFromString(data_format_attr, &data_format),
                    errors::InvalidArgument("Invalid data format"));
      }

      ksize.emplace();
      OP_REQUIRES_OK(ctx, ctx->GetAttr("ksize", &ksize.value()));
      CHECK(ksize->size() == kNchwDimensionCount ||
            ksize->size() == kNcdhwDimensionCount);

      for (int i = 0; i < ksize->size(); ++i) {
        CHECK((*ksize)[i] > 0);
      }

      stride.emplace();
      OP_REQUIRES_OK(ctx, ctx->GetAttr("strides", &stride.value()));
      CHECK(stride->size() == ksize->size());

      // Can't pool over the batch dimension
      CHECK((*ksize)[0] == 1 && (*stride)[0] == 1);

      OP_REQUIRES_OK(ctx, ctx->GetAttr("padding", &padding));
    }

    // These attributes aren't present for MaxPoolGradV2 (because it takes them
    // as input tensors)
    absl::optional<std::vector<int32>> ksize;
    absl::optional<std::vector<int32>> stride;

    Padding padding;
    TensorFormat data_format = FORMAT_NHWC;
  };

  AvgPoolGradInitHelper(OpKernelContext* ctx,
                        std::shared_ptr<const Attributes> attr)
      : attr_(attr) {}

  const absl::optional<std::vector<int32>>& GetKernelSizes() const {
    return attr_->ksize;
  }

  const absl::optional<std::vector<int32>>& GetKernelStrides() const {
    return attr_->stride;
  }

  Padding GetPadding() const { return attr_->padding; }
  TensorFormat GetDataFormat() const { return attr_->data_format; }

 private:
  const std::shared_ptr<const Attributes> attr_;
};

class PoolingShapeHelper : public ShapeHelper {
 public:
  std::vector<TensorShape> GetOutputShapes(
      OpKernelContext* ctx,
      const InitializationHelper* initialization_helper) const override {
    std::vector<int32> ksize;
    std::vector<int32> stride;

    auto init_helper =
        static_cast<const PoolInitHelper*>(initialization_helper);

    if (ctx->num_inputs() == 1) {
      DCHECK(init_helper->GetKernelSizes().has_value());
      DCHECK(init_helper->GetKernelStrides().has_value());

      ksize = *init_helper->GetKernelSizes();
      stride = *init_helper->GetKernelStrides();
    } else {
      // For MaxPoolV2, we need to retrieve the kernel sizes/strides from
      // constant CPU input tensors at indices 1 and 2.

      // These tensors must reside in host memory if we want to read them
      CHECK(ctx->input_memory_type(1) == HOST_MEMORY);
      CHECK(ctx->input_memory_type(2) == HOST_MEMORY);
      Tensor ksize_tensor = ctx->input(1);
      Tensor stride_tensor = ctx->input(2);

      static const uint32_t kDimensionCount = 4;

      // The ksize and stride tensors must have 4 or 5 elements each
      CHECK(ksize_tensor.NumElements() == kNchwDimensionCount ||
            ksize_tensor.NumElements() == kNcdhwDimensionCount);
      CHECK(stride_tensor.NumElements() == ksize_tensor.NumElements());

      auto ksize_values = ksize_tensor.flat<int32>();
      auto stride_values = stride_tensor.flat<int32>();

      ksize.assign(ksize_values.data(),
                   ksize_values.data() + ksize_tensor.NumElements());
      stride.assign(stride_values.data(),
                    stride_values.data() + stride_tensor.NumElements());
    }

    TensorShape output_shape;
    Padding padding = init_helper->GetPadding();
    TensorFormat data_format = init_helper->GetDataFormat();

    if (ctx->input(0).shape().dims() == kNcdhwDimensionCount) {
      Pool3dParameters params(ctx, ksize, stride, padding, data_format,
                              ctx->input(0).shape());
      output_shape = params.forward_output_shape();
    } else {
      PoolParameters params(ctx, ksize, stride, padding, data_format,
                            ctx->input(0).shape());
      output_shape = params.forward_output_shape();
    }

    return {std::move(output_shape)};
  }
};

// Helper to get DML API pooling values (window size, strides, and padding) that
// may be either attributes or tensors in TF.
template <typename TInitHelper>
DmlPoolValues GetPoolValuesFromAttributesOrTensors(
    DmlKernelConstruction* ctx, const TInitHelper* init_helper,
    const TensorShape& tensor_in_shape, uint32_t inputCount) {
  // V2 pooling ops take the kernel sizes and strides as input tensors; the
  // others take them as attributes. There may be 2 additional input tensors.
  CHECK(ctx->GetInputCount() == inputCount ||
        ctx->GetInputCount() == (inputCount + 2));
  CHECK(ctx->GetOutputCount() == 1);

  CHECK(tensor_in_shape.dims() == kNchwDimensionCount ||
        tensor_in_shape.dims() == kNcdhwDimensionCount);
  CHECK(tensor_in_shape.dims() == ctx->GetOutputTensorShape(0).dims());

  std::vector<int32> ksize;
  std::vector<int32> stride;

  if (ctx->GetInputCount() == inputCount) {
    // The kernel sizes and strides are provided as attributes, so we can grab
    // them from the shape helper

    DCHECK(init_helper->GetKernelSizes().has_value());
    DCHECK(init_helper->GetKernelStrides().has_value());

    ksize = *init_helper->GetKernelSizes();
    stride = *init_helper->GetKernelStrides();
  } else {
    // For MaxPoolV2/MaxPoolGradV2, we need to retrieve the kernel sizes/strides
    // from constant CPU input tensors (the last two).

    Tensor ksize_tensor = ctx->GetConstantInputTensor(inputCount);
    Tensor stride_tensor = ctx->GetConstantInputTensor(inputCount + 1);

    // The ksize and stride tensors must have as many elements as the rank of
    // the input
    CHECK(ksize_tensor.NumElements() == tensor_in_shape.dims());
    CHECK(stride_tensor.NumElements() == tensor_in_shape.dims());

    auto ksize_values = ksize_tensor.flat<int32>();
    auto stride_values = stride_tensor.flat<int32>();

    ksize.assign(ksize_values.data(),
                 ksize_values.data() + tensor_in_shape.dims());
    stride.assign(stride_values.data(),
                  stride_values.data() + tensor_in_shape.dims());
  }

  Padding padding = init_helper->GetPadding();
  TensorFormat data_format = init_helper->GetDataFormat();

  const int64 tensor_in_rows = GetTensorDim(tensor_in_shape, data_format, 'H');
  const int64 window_rows = GetTensorDim(ksize, data_format, 'H');
  const int64 row_stride = GetTensorDim(stride, data_format, 'H');
  int64 out_height = 0;
  int64 pad_rows_start = 0;
  int64 pad_rows_end = 0;
  TF_CHECK_OK(GetWindowedOutputSizeVerbose(tensor_in_rows, window_rows,
                                           row_stride, padding, &out_height,
                                           &pad_rows_start, &pad_rows_end));

  const int64 tensor_in_cols = GetTensorDim(tensor_in_shape, data_format, 'W');
  const int64 window_cols = GetTensorDim(ksize, data_format, 'W');
  const int64 col_stride = GetTensorDim(stride, data_format, 'W');
  int64 out_width = 0;
  int64 pad_cols_start = 0;
  int64 pad_cols_end = 0;
  TF_CHECK_OK(GetWindowedOutputSizeVerbose(tensor_in_cols, window_cols,
                                           col_stride, padding, &out_width,
                                           &pad_cols_start, &pad_cols_end));

  DmlPoolValues poolValues = {};

  if (tensor_in_shape.dims() == kNcdhwDimensionCount) {
    const int64 tensor_in_depth =
        GetTensorDim(tensor_in_shape, data_format, '0');
    const int64 window_depth = GetTensorDim(ksize, data_format, '0');
    const int64 depth_stride = GetTensorDim(stride, data_format, '0');
    int64 out_depth = 0;
    int64 pad_depth_start = 0;
    int64 pad_depth_end = 0;
    TF_CHECK_OK(GetWindowedOutputSizeVerbose(tensor_in_depth, window_depth,
                                             depth_stride, padding, &out_depth,
                                             &pad_depth_start, &pad_depth_end));

    poolValues.strides.push_back(depth_stride);
    poolValues.window_size.push_back(window_depth);
    poolValues.start_padding.push_back(pad_depth_start);
    poolValues.end_padding.push_back(pad_depth_end);
  }

  poolValues.strides.push_back(row_stride);
  poolValues.strides.push_back(col_stride);
  poolValues.window_size.push_back(window_rows);
  poolValues.window_size.push_back(window_cols);
  poolValues.start_padding.push_back(pad_rows_start);
  poolValues.start_padding.push_back(pad_cols_start);
  poolValues.end_padding.push_back(pad_rows_end);
  poolValues.end_padding.push_back(pad_cols_end);
  poolValues.data_format = data_format;

  return poolValues;
}

// Implements the AvgPool, MaxPool, and MaxPoolV2 ops.
template <DML_OPERATOR_TYPE op_type, typename OperatorDesc>
class DmlPoolingKernel : public DmlKernel {
 public:
  using InitHelper = PoolInitHelper;

  explicit DmlPoolingKernel(DmlKernelConstruction* ctx,
                            const InitHelper* init_helper) {
    const TensorShape& tensor_in_shape = ctx->GetInputTensorShape(0);
    DmlPoolValues poolValues = GetPoolValuesFromAttributesOrTensors(
        ctx, init_helper, tensor_in_shape, 1);

    // Ignore the kernel size/stride tensors, because DML takes them as
    // attributes and not input tensors
    DmlKernelParams params;
    params.kernel_input_indices = {0};

    // The layout of the input/output tensors is determined by the "data_format"
    auto input_output_layout = GetDmlTensorLayout(
        poolValues.data_format, ctx->GetOutputTensorShape(0).dims());

    DmlKernelTensors tensors = GetTensorInfos(ctx, params);
    tensors.inputs[0]->desc =
        CreateTensorDescFromInput(ctx, 0, input_output_layout);
    tensors.outputs[0]->desc =
        CreateTensorDescFromOutput(ctx, 0, input_output_layout);

    auto input_descs = GetDmlTensorDescs(tensors.inputs);
    auto output_descs = GetDmlTensorDescs(tensors.outputs);

    OperatorDesc pooling_desc = {};
    pooling_desc.InputTensor = &input_descs[0];
    pooling_desc.OutputTensor = &output_descs[0];
    pooling_desc.DimensionCount = poolValues.strides.size();
    pooling_desc.Strides = poolValues.strides.data();
    pooling_desc.WindowSize = poolValues.window_size.data();
    pooling_desc.StartPadding = poolValues.start_padding.data();
    pooling_desc.EndPadding = poolValues.end_padding.data();
    // AvgPool in TF never includes padding, so for
    // DML_AVERAGE_POOLING_OPERATOR_DESC::IncludePadding we can just leave it as
    // its default-initialized value (false)

    DML_OPERATOR_DESC op_desc = {op_type, &pooling_desc};
    Initialize(ctx, std::move(tensors), op_desc);
  }
};

class DmlAvgPoolingGradKernel : public DmlKernel {
 public:
  using InitHelper = AvgPoolGradInitHelper;

  explicit DmlAvgPoolingGradKernel(DmlKernelConstruction* ctx,
                                   const InitHelper* init_helper) {
    CHECK(ctx->GetInputCount() == 2);
    CHECK(ctx->GetOutputCount() == 1);

    const TensorShape& tensor_in_shape = ctx->GetOutputTensorShape(0);
    DmlPoolValues poolValues = GetPoolValuesFromAttributesOrTensors(
        ctx, init_helper, tensor_in_shape, ctx->GetInputCount());

    DmlKernelParams params;
    params.kernel_input_indices = {1};

    auto input_output_layout = GetDmlTensorLayout(
        poolValues.data_format, ctx->GetOutputTensorShape(0).dims());

    DmlKernelTensors tensors = GetTensorInfos(ctx, params);
    tensors.inputs[0]->desc =
        CreateTensorDescFromInput(ctx, 1, input_output_layout);
    tensors.outputs[0]->desc =
        CreateTensorDescFromOutput(ctx, 0, input_output_layout);

    auto inputs = GetDmlTensorDescs(tensors.inputs);
    auto outputs = GetDmlTensorDescs(tensors.outputs);

    DML_AVERAGE_POOLING_GRAD_OPERATOR_DESC avg_pooling_grad_desc = {};
    avg_pooling_grad_desc.InputGradientTensor = inputs.data();
    avg_pooling_grad_desc.OutputGradientTensor = outputs.data();
    avg_pooling_grad_desc.DimensionCount = poolValues.strides.size();
    avg_pooling_grad_desc.Strides = poolValues.strides.data();
    avg_pooling_grad_desc.WindowSize = poolValues.window_size.data();
    avg_pooling_grad_desc.StartPadding = poolValues.start_padding.data();
    avg_pooling_grad_desc.EndPadding = poolValues.end_padding.data();
    // AvgPoolGrad in TF never includes padding, so for
    // DML_AVERAGE_POOLING_GRAD_OPERATOR_DESC::IncludePadding we can just leave
    // it as its default-initialized value (false)

    DML_OPERATOR_DESC op_desc = {DML_OPERATOR_AVERAGE_POOLING_GRAD,
                                 &avg_pooling_grad_desc};

    Initialize(ctx, std::move(tensors), op_desc);
  }
};

class DmlMaxPoolGradKernel : public DmlKernel {
 public:
  using InitHelper = MaxPoolGradInitHelper;

  explicit DmlMaxPoolGradKernel(DmlKernelConstruction* ctx,
                                const InitHelper* init_helper) {
    const TensorShape& tensor_in_shape = ctx->GetInputTensorShape(0);
    DmlPoolValues poolValues = GetPoolValuesFromAttributesOrTensors(
        ctx, init_helper, tensor_in_shape, 3);

    // TF doesn't use dilations, but DML needs default values of 1.
    uint32_t dilations[] = {1, 1};

    // Ignore the kernel size/stride tensors, because DML takes them as
    // attributes and not input tensors
    DmlKernelParams params;
    params.kernel_input_indices = {0, 2};

    // The layout of the input/output tensors is determined by the "data_format"
    auto input_output_layout = GetDmlTensorLayout(
        poolValues.data_format, ctx->GetOutputTensorShape(0).dims());

    DmlKernelTensors tensors = GetTensorInfos(ctx, params);
    tensors.inputs[0]->desc =
        CreateTensorDescFromInput(ctx, 0, input_output_layout);
    tensors.inputs[1]->desc =
        CreateTensorDescFromInput(ctx, 2, input_output_layout);
    tensors.outputs[0]->desc =
        CreateTensorDescFromOutput(ctx, 0, input_output_layout);

    auto input_descs = GetDmlTensorDescs(tensors.inputs);
    auto output_descs = GetDmlTensorDescs(tensors.outputs);

    DML_MAX_POOLING_GRAD_OPERATOR_DESC max_pooling_grad_desc = {};
    max_pooling_grad_desc.InputTensor = &input_descs[0];
    max_pooling_grad_desc.InputGradientTensor = &input_descs[1];
    max_pooling_grad_desc.OutputGradientTensor = output_descs.data();
    max_pooling_grad_desc.DimensionCount = poolValues.strides.size();
    max_pooling_grad_desc.Strides = poolValues.strides.data();
    max_pooling_grad_desc.WindowSize = poolValues.window_size.data();
    max_pooling_grad_desc.StartPadding = poolValues.start_padding.data();
    max_pooling_grad_desc.EndPadding = poolValues.end_padding.data();
    max_pooling_grad_desc.Dilations = dilations;

    DML_OPERATOR_DESC op_desc = {DML_OPERATOR_MAX_POOLING_GRAD,
                                 &max_pooling_grad_desc};
    Initialize(ctx, std::move(tensors), op_desc);
  }
};

using DmlAvgPoolKernel = DmlPoolingKernel<DML_OPERATOR_AVERAGE_POOLING,
                                          DML_AVERAGE_POOLING_OPERATOR_DESC>;
using DmlMaxPoolKernel =
    DmlPoolingKernel<DML_OPERATOR_MAX_POOLING, DML_MAX_POOLING_OPERATOR_DESC>;

#define DML_REGISTER_KERNELS(type)                                      \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("AvgPool").Device(DEVICE_DML).TypeConstraint<type>("T"),     \
      DmlKernelWrapper<DmlAvgPoolKernel, PoolingShapeHelper>);          \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("AvgPool3D").Device(DEVICE_DML).TypeConstraint<type>("T"),   \
      DmlKernelWrapper<DmlAvgPoolKernel, PoolingShapeHelper>);          \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("MaxPool").Device(DEVICE_DML).TypeConstraint<type>("T"),     \
      DmlKernelWrapper<DmlMaxPoolKernel, PoolingShapeHelper>);          \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("MaxPool3D").Device(DEVICE_DML).TypeConstraint<type>("T"),   \
      DmlKernelWrapper<DmlMaxPoolKernel, PoolingShapeHelper>);          \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("MaxPoolV2")                                                 \
          .Device(DEVICE_DML)                                           \
          .TypeConstraint<type>("T")                                    \
          .HostMemory("ksize")                                          \
          .HostMemory("strides"),                                       \
      DmlKernelWrapper<DmlMaxPoolKernel, PoolingShapeHelper>);          \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("AvgPoolGrad")                                               \
          .Device(DEVICE_DML)                                           \
          .TypeConstraint<type>("T")                                    \
          .HostMemory("orig_input_shape"),                              \
      DmlKernelWrapper<DmlAvgPoolingGradKernel,                         \
                       GetOutputShapeFromDimsTensorHelper<int32, 0>>);  \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("AvgPool3DGrad")                                             \
          .Device(DEVICE_DML)                                           \
          .TypeConstraint<type>("T")                                    \
          .HostMemory("orig_input_shape"),                              \
      DmlKernelWrapper<DmlAvgPoolingGradKernel,                         \
                       GetOutputShapeFromDimsTensorHelper<int32, 0>>);  \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("MaxPoolGrad").Device(DEVICE_DML).TypeConstraint<type>("T"), \
      DmlKernelWrapper<DmlMaxPoolGradKernel,                            \
                       GetOutputShapeAsInputShapeHelper>);              \
  REGISTER_KERNEL_BUILDER(Name("MaxPoolGradV2")                         \
                              .Device(DEVICE_DML)                       \
                              .TypeConstraint<type>("T")                \
                              .HostMemory("ksize")                      \
                              .HostMemory("strides"),                   \
                          DmlKernelWrapper<DmlMaxPoolGradKernel,        \
                                           GetOutputShapeAsInputShapeHelper>);
TF_CALL_DML_FLOAT_TYPES(DML_REGISTER_KERNELS);
#undef DML_REGISTER_KERNELS

}  // namespace tensorflow
