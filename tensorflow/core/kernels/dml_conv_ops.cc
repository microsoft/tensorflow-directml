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
#include "tensorflow/core/kernels/conv_grad_ops.h"
#include "tensorflow/core/kernels/conv_ops.h"
#include "tensorflow/core/kernels/dml_kernel_wrapper.h"
#include "tensorflow/core/kernels/dml_ops_common.h"
#include "tensorflow/core/kernels/fused_eigen_output_kernels.h"

namespace tensorflow {

using Microsoft::WRL::ComPtr;

class BaseConv2DInitHelper : public InitializationHelper {
 public:
  virtual TensorFormat GetDataFormat() const = 0;
  virtual int64 GetBatch() const = 0;
  virtual int64 GetOutRows() const = 0;
  virtual int64 GetOutCols() const = 0;
  virtual int64 GetOutDepth() const = 0;
};

class DepthwiseConv2DNativeInitHelper : public BaseConv2DInitHelper {
 public:
  struct Attributes {
    explicit Attributes(OpKernelConstruction* ctx) {
      std::vector<int32> strides;
      std::vector<int32> dilations;
      OP_REQUIRES_OK(ctx, ctx->GetAttr("strides", &strides));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("dilations", &dilations));
      std::string data_format_attr;
      OP_REQUIRES_OK(ctx, ctx->GetAttr("data_format", &data_format_attr));
      OP_REQUIRES(ctx, FormatFromString(data_format_attr, &data_format),
                  errors::InvalidArgument("Invalid data format"));

      OP_REQUIRES(ctx, strides.size() == 4,
                  errors::InvalidArgument("Sliding window strides field must "
                                          "specify 4 dimensions"));

      OP_REQUIRES(ctx, dilations.size() == 4,
                  errors::InvalidArgument("Sliding window dilations field must "
                                          "specify 4 dimensions"));

      stride_h = GetTensorDim(strides, data_format, 'H');
      stride_w = GetTensorDim(strides, data_format, 'W');
      const int64 stride_n = GetTensorDim(strides, data_format, 'N');
      const int64 stride_c = GetTensorDim(strides, data_format, 'C');

      OP_REQUIRES(ctx, (stride_n == 1 && stride_c == 1),
                  errors::InvalidArgument(
                      "Current implementation does not yet support "
                      "strides in the batch and depth dimensions."));

      dilation_h = GetTensorDim(dilations, data_format, 'H');
      dilation_w = GetTensorDim(dilations, data_format, 'W');
      const int64 dilation_n = GetTensorDim(strides, data_format, 'N');
      const int64 dilation_c = GetTensorDim(strides, data_format, 'C');

      OP_REQUIRES(ctx, (dilation_n == 1 && dilation_c == 1),
                  errors::InvalidArgument(
                      "Current implementation does not yet support "
                      "strides in the batch and depth dimensions."));

      OP_REQUIRES_OK(ctx, ctx->GetAttr("padding", &padding));
    }

    TensorFormat data_format;
    Padding padding;
    int32 stride_h;
    int32 stride_w;
    int32 dilation_h;
    int32 dilation_w;
  };

  DepthwiseConv2DNativeInitHelper(OpKernelContext* ctx,
                                  std::shared_ptr<const Attributes> attr)
      : attr_(attr) {
    // Input tensor is of the following dimensions:
    // [ batch, in_rows, in_cols, in_depth ]
    const Tensor& input = ctx->input(0);

    // Input filter is of the following dimensions:
    // [ filter_rows, filter_cols, in_depth, depth_multiplier]
    const Tensor& filter = ctx->input(1);

    // For 2D convolution, there should be 4 dimensions.
    OP_REQUIRES(ctx, input.dims() == 4,
                errors::InvalidArgument("input must be 4-dimensional",
                                        input.shape().DebugString()));
    OP_REQUIRES(ctx, filter.dims() == 4,
                errors::InvalidArgument("filter must be 4-dimensional: ",
                                        filter.shape().DebugString()));

    // in_depth for input and filter must match.
    in_depth_ = GetTensorDim(input, attr->data_format, 'C');
    OP_REQUIRES(ctx, in_depth_ == filter.dim_size(2),
                errors::InvalidArgument(
                    "input and filter must have the same depth: ", in_depth_,
                    " vs ", filter.dim_size(2)));

    group_count_ = filter.dim_size(2);

    // The last dimension for filter is depth multiplier.
    int32 depth_multiplier = filter.dim_size(3);

    // The output depth is input depth x depth multipler
    out_depth_ = in_depth_ * depth_multiplier;

    const int64 input_rows_raw = GetTensorDim(input, attr->data_format, 'H');
    OP_REQUIRES(
        ctx, FastBoundsCheck(input_rows_raw, std::numeric_limits<int32>::max()),
        errors::InvalidArgument("Input rows too large"));
    const int32 input_rows = static_cast<int32>(input_rows_raw);
    filter_rows_ = filter.dim_size(0);

    const int64 input_cols_raw = GetTensorDim(input, attr->data_format, 'W');
    OP_REQUIRES(
        ctx, FastBoundsCheck(input_cols_raw, std::numeric_limits<int32>::max()),
        errors::InvalidArgument("Input cols too large"));
    const int32 input_cols = static_cast<int32>(input_cols_raw);
    filter_cols_ = filter.dim_size(1);

    // The first dimension for input is batch.
    batch_ = input.dim_size(0);

    OP_REQUIRES_OK(ctx, GetWindowedOutputSizeVerboseV2(
                            input_rows, filter_rows_, attr->dilation_h,
                            attr_->stride_h, attr->padding, &out_rows_,
                            &pad_rows_before_, &pad_rows_after_));

    OP_REQUIRES_OK(ctx, GetWindowedOutputSizeVerboseV2(
                            input_cols, filter_cols_, attr_->dilation_w,
                            attr_->stride_w, attr->padding, &out_cols_,
                            &pad_cols_before_, &pad_cols_after_));
  }

  TensorFormat GetDataFormat() const final { return attr_->data_format; }
  int64 GetBatch() const final { return batch_; }
  int64 GetOutRows() const final { return out_rows_; }
  int64 GetOutCols() const final { return out_cols_; }
  int64 GetInDepth() const { return in_depth_; }
  int64 GetOutDepth() const final { return out_depth_; }
  int32 GetStrideH() const { return attr_->stride_h; }
  int32 GetStrideW() const { return attr_->stride_w; }
  int32 GetDilationH() const { return attr_->dilation_h; }
  int32 GetDilationW() const { return attr_->dilation_w; }
  int32 GetFilterRows() const { return filter_rows_; }
  int32 GetFilterCols() const { return filter_cols_; }
  int32 GetGroupCount() const { return group_count_; }
  int64 GetPadRowsBefore() const { return pad_rows_before_; }
  int64 GetPadColsBefore() const { return pad_cols_before_; }
  int64 GetPadRowsAfter() const { return pad_rows_after_; }
  int64 GetPadColsAfter() const { return pad_cols_after_; }

 private:
  const std::shared_ptr<const Attributes> attr_;
  int32 filter_rows_;
  int32 filter_cols_;
  int64 batch_;
  int64 out_rows_;
  int64 out_cols_;
  int64 in_depth_;
  int64 out_depth_;
  int32 group_count_;
  int64 pad_rows_before_;
  int64 pad_cols_before_;
  int64 pad_rows_after_;
  int64 pad_cols_after_;
};

class ConvInitHelper : public BaseConv2DInitHelper {
 public:
  struct Attributes {
    explicit Attributes(OpKernelConstruction* ctx) {
      OP_REQUIRES_OK(ctx, InitConv2DParameters(ctx, &params));
    }

    Conv2DParameters params;
  };

  ConvInitHelper(OpKernelContext* ctx, std::shared_ptr<const Attributes> attr)
      : attr_(attr) {
    const Tensor& input = ctx->input(0);
    const Tensor& filter = ctx->input(1);

    OP_REQUIRES_OK(
        ctx, ComputeConv2DDimension(attr->params, input, filter, &dimensions_));
  }

  const Conv2DParameters& GetParams() const { return attr_->params; }
  TensorFormat GetDataFormat() const final { return attr_->params.data_format; }
  int64 GetBatch() const final { return dimensions_.batch; }
  int64 GetOutRows() const final { return dimensions_.out_rows; }
  int64 GetOutCols() const final { return dimensions_.out_cols; }
  int64 GetOutDepth() const final { return dimensions_.out_depth; }

 private:
  const std::shared_ptr<const Attributes> attr_;
  Conv2DDimensions dimensions_;
};

class FusedConvInitHelper : public ConvInitHelper {
 public:
  struct Attributes : public ConvInitHelper::Attributes {
    explicit Attributes(OpKernelConstruction* ctx)
        : ConvInitHelper::Attributes(ctx) {
      using FCT = FusedComputationType;
      std::vector<FusedComputationPattern> patterns = {
          {FCT::kBiasAdd, {"BiasAdd"}},
          {FCT::kBiasAddWithRelu, {"BiasAdd", "Relu"}},
          {FCT::kBiasAddWithRelu6, {"BiasAdd", "Relu6"}},
          {FCT::kBiasAddWithElu, {"BiasAdd", "Elu"}},
          {FCT::kFusedBatchNorm, {"FusedBatchNorm"}},
          {FCT::kFusedBatchNormWithRelu, {"FusedBatchNorm", "Relu"}},
          {FCT::kFusedBatchNormWithRelu6, {"FusedBatchNorm", "Relu6"}},
          {FCT::kFusedBatchNormWithElu, {"FusedBatchNorm", "Elu"}},
      };

      OP_REQUIRES_OK(ctx,
                     InitializeFusedComputation(ctx, "DmlFusedConv2d", patterns,
                                                &fused_computation_type,
                                                &fused_computation_args));
    }

    FusedComputationType fused_computation_type;
    FusedComputationArgs fused_computation_args;
  };

  FusedConvInitHelper(OpKernelContext* ctx,
                      std::shared_ptr<const Attributes> attr)
      : ConvInitHelper(ctx, attr), attr_(attr) {}

  FusedComputationType GetFusedComputationType() const {
    return attr_->fused_computation_type;
  }

  FusedComputationArgs GetFusedComputationArgs() const {
    return attr_->fused_computation_args;
  }

 private:
  const std::shared_ptr<const Attributes> attr_;
};

class Conv2DGradInitHelper : public InitializationHelper {
 public:
  struct Attributes {
    explicit Attributes(OpKernelConstruction* ctx) {
      OP_REQUIRES_OK(ctx, InitConv2DParameters(ctx, &params));
    }

    Conv2DParameters params;
  };

  Conv2DGradInitHelper(OpKernelContext* ctx,
                       std::shared_ptr<const Attributes> attr)
      : attr_(attr) {}

  const Conv2DParameters& GetParams() const { return attr_->params; }

 private:
  const std::shared_ptr<const Attributes> attr_;
};

class ConvShapeHelper : public ShapeHelper {
 public:
  std::vector<TensorShape> GetOutputShapes(
      OpKernelContext* ctx,
      const InitializationHelper* initialization_helper) const override {
    auto init_helper =
        static_cast<const BaseConv2DInitHelper*>(initialization_helper);

    TensorShape out_shape =
        ShapeFromFormat(init_helper->GetDataFormat(), init_helper->GetBatch(),
                        init_helper->GetOutRows(), init_helper->GetOutCols(),
                        init_helper->GetOutDepth());

    return {std::move(out_shape)};
  }
};

class DmlDepthwiseConv2DNativeKernel : public DmlKernel {
 public:
  using InitHelper = DepthwiseConv2DNativeInitHelper;

  explicit DmlDepthwiseConv2DNativeKernel(DmlKernelConstruction* ctx,
                                          const InitHelper* init_helper) {
    DCHECK(ctx->GetInputCount() == 2);
    DCHECK(ctx->GetOutputCount() == 1);

    DCHECK(ctx->GetInputTensorShape(0).dims() == kNchwDimensionCount);
    DCHECK(ctx->GetInputTensorShape(1).dims() == kNchwDimensionCount);
    DCHECK(ctx->GetOutputTensorShape(0).dims() == kNchwDimensionCount);

    uint32_t strides[] = {static_cast<uint32_t>(init_helper->GetStrideH()),
                          static_cast<uint32_t>(init_helper->GetStrideW())};
    uint32_t dilations[] = {static_cast<uint32_t>(init_helper->GetDilationH()),
                            static_cast<uint32_t>(init_helper->GetDilationW())};
    uint32_t start_padding[] = {
        static_cast<uint32_t>(init_helper->GetPadRowsBefore()),
        static_cast<uint32_t>(init_helper->GetPadColsBefore())};
    uint32_t end_padding[] = {
        static_cast<uint32_t>(init_helper->GetPadRowsAfter()),
        static_cast<uint32_t>(init_helper->GetPadColsAfter())};
    uint32_t output_padding[] = {0, 0};
    uint32_t group_count = static_cast<uint32_t>(init_helper->GetGroupCount());

    DmlKernelParams params;
    params.kernel_input_indices = {
        0, 1,
        absl::nullopt  // We don't use the DML bias tensor
    };

    using namespace DmlTensorAxes;

    // The dimensions of the filter tensor are laid out a little differently
    // than what DML expects
    auto filter_layout = {H, W, C, N};

    // The layout of the input/output tensors is determined by the "data_format"
    // attribute
    auto input_output_layout =
        GetDmlTensorLayout(init_helper->GetDataFormat(), kNchwDimensionCount);

    DmlKernelTensors tensors = GetTensorInfos(ctx, params);
    tensors.inputs[0]->desc =
        CreateTensorDescFromInput(ctx, 0, input_output_layout);

    TensorShape filter_shape = {
        init_helper->GetFilterRows(), init_helper->GetFilterCols(),
        init_helper->GetInDepth() / group_count, init_helper->GetOutDepth()};

    tensors.inputs[1]->desc = DmlTensorDesc::Create(
        ctx->GetInputDataType(1), filter_shape, filter_shape, filter_layout);

    tensors.outputs[0]->desc =
        CreateTensorDescFromOutput(ctx, 0, input_output_layout);

    auto input_descs = GetDmlTensorDescs(tensors.inputs);
    auto output_descs = GetDmlTensorDescs(tensors.outputs);

    DML_CONVOLUTION_OPERATOR_DESC conv_desc = {};
    conv_desc.InputTensor = &input_descs[0];
    conv_desc.FilterTensor = &input_descs[1];
    conv_desc.BiasTensor = nullptr;
    conv_desc.OutputTensor = &output_descs[0];
    conv_desc.Mode = DML_CONVOLUTION_MODE_CROSS_CORRELATION;
    conv_desc.Direction = DML_CONVOLUTION_DIRECTION_FORWARD;
    conv_desc.DimensionCount = kNchwSpatialDimensionCount;
    conv_desc.Strides = strides;
    conv_desc.Dilations = dilations;
    conv_desc.StartPadding = start_padding;
    conv_desc.EndPadding = end_padding;
    conv_desc.OutputPadding = output_padding;
    conv_desc.GroupCount = group_count;
    conv_desc.FusedActivation = nullptr;

    DML_OPERATOR_DESC op_desc = {DML_OPERATOR_CONVOLUTION, &conv_desc};
    Initialize(ctx, std::move(tensors), op_desc);
  }
};

#define DML_REGISTER_KERNEL(type)     \
  REGISTER_KERNEL_BUILDER(            \
      Name("DepthwiseConv2dNative")   \
          .Device(DEVICE_DML)         \
          .TypeConstraint<type>("T"), \
      DmlKernelWrapper<DmlDepthwiseConv2DNativeKernel, ConvShapeHelper>);
TF_CALL_DML_FLOAT_TYPES(DML_REGISTER_KERNEL);
#undef DML_REGISTER_KERNEL

class DmlConv2DKernel : public DmlKernel {
 public:
  using InitHelper = ConvInitHelper;

  explicit DmlConv2DKernel(DmlKernelConstruction* ctx,
                           const InitHelper* init_helper) {
    CHECK(ctx->GetInputCount() == 2);
    CHECK(ctx->GetOutputCount() == 1);

    // 2D conv requires 4D tensors
    static const uint32_t kDimensionCount = 4;
    static const uint32_t kSpatialDimensionCount = 2;

    CHECK(ctx->GetInputTensorShape(0).dims() == kDimensionCount);
    CHECK(ctx->GetInputTensorShape(1).dims() == kDimensionCount);
    CHECK(ctx->GetOutputTensorShape(0).dims() == kDimensionCount);

    const Conv2DParameters& conv_params = init_helper->GetParams();

    Conv2DDimensions conv_dims;
    TF_CHECK_OK(ComputeConv2DDimension(conv_params, ctx->GetInputTensorShape(0),
                                       ctx->GetInputTensorShape(1),
                                       &conv_dims));

    uint32_t strides[] = {static_cast<uint32_t>(conv_dims.stride_rows),
                          static_cast<uint32_t>(conv_dims.stride_cols)};
    uint32_t dilations[] = {static_cast<uint32_t>(conv_dims.dilation_rows),
                            static_cast<uint32_t>(conv_dims.dilation_cols)};
    uint32_t start_padding[] = {
        static_cast<uint32_t>(conv_dims.pad_rows_before),
        static_cast<uint32_t>(conv_dims.pad_cols_before)};
    uint32_t end_padding[] = {static_cast<uint32_t>(conv_dims.pad_rows_after),
                              static_cast<uint32_t>(conv_dims.pad_cols_after)};
    uint32_t output_padding[] = {0, 0};
    uint32_t group_count = conv_dims.in_depth / conv_dims.patch_depth;

    DmlKernelParams params;
    params.kernel_input_indices = {
        0, 1,
        absl::nullopt  // We don't use the DML bias tensor
    };

    using namespace DmlTensorAxes;

    // The dimensions of the filter tensor are laid out a little differently
    // than what DML expects
    auto filter_layout = {H, W, C, N};

    // The layout of the input/output tensors is determined by the "data_format"
    // attribute
    auto input_output_layout =
        GetDmlTensorLayout(conv_params.data_format, kDimensionCount);

    DmlKernelTensors tensors = GetTensorInfos(ctx, params);
    tensors.inputs[0]->desc =
        CreateTensorDescFromInput(ctx, 0, input_output_layout);
    tensors.inputs[1]->desc = CreateTensorDescFromInput(ctx, 1, filter_layout);
    tensors.outputs[0]->desc =
        CreateTensorDescFromOutput(ctx, 0, input_output_layout);

    auto input_descs = GetDmlTensorDescs(tensors.inputs);
    auto output_descs = GetDmlTensorDescs(tensors.outputs);

    DML_CONVOLUTION_OPERATOR_DESC conv_desc = {};
    conv_desc.InputTensor = &input_descs[0];
    conv_desc.FilterTensor = &input_descs[1];
    conv_desc.BiasTensor = nullptr;
    conv_desc.OutputTensor = &output_descs[0];
    conv_desc.Mode = DML_CONVOLUTION_MODE_CROSS_CORRELATION;
    conv_desc.Direction = DML_CONVOLUTION_DIRECTION_FORWARD;
    conv_desc.DimensionCount = kSpatialDimensionCount;
    conv_desc.Strides = strides;
    conv_desc.Dilations = dilations;
    conv_desc.StartPadding = start_padding;
    conv_desc.EndPadding = end_padding;
    conv_desc.OutputPadding = output_padding;
    conv_desc.GroupCount = group_count;
    conv_desc.FusedActivation = nullptr;

    DML_OPERATOR_DESC op_desc = {DML_OPERATOR_CONVOLUTION, &conv_desc};
    Initialize(ctx, std::move(tensors), op_desc);
  }
};

#define DML_REGISTER_KERNEL(type)                                  \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("Conv2D").Device(DEVICE_DML).TypeConstraint<type>("T"), \
      DmlKernelWrapper<DmlConv2DKernel, ConvShapeHelper>);
TF_CALL_DML_FLOAT_TYPES(DML_REGISTER_KERNEL);
#undef DML_REGISTER_KERNEL

template <typename T>
class DmlFusedConv2DKernel : public DmlKernel {
 public:
  using InitHelper = FusedConvInitHelper;

  explicit DmlFusedConv2DKernel(DmlKernelConstruction* ctx,
                                const InitHelper* init_helper) {
    CHECK(ctx->GetInputCount() > 2);
    CHECK(ctx->GetOutputCount() == 1);

    // 2D conv requires 4D tensors
    static const uint32_t kDimensionCount = 4;
    static const uint32_t kSpatialDimensionCount = 2;

    CHECK(ctx->GetInputTensorShape(0).dims() == kDimensionCount);
    CHECK(ctx->GetInputTensorShape(1).dims() == kDimensionCount);
    CHECK(ctx->GetOutputTensorShape(0).dims() == kDimensionCount);

    const Conv2DParameters& conv_params = init_helper->GetParams();
    const auto fused_computation_type = init_helper->GetFusedComputationType();
    const auto fused_computation_args = init_helper->GetFusedComputationArgs();

    Conv2DDimensions conv_dims;
    TF_CHECK_OK(ComputeConv2DDimension(conv_params, ctx->GetInputTensorShape(0),
                                       ctx->GetInputTensorShape(1),
                                       &conv_dims));

    uint32_t strides[] = {static_cast<uint32_t>(conv_dims.stride_rows),
                          static_cast<uint32_t>(conv_dims.stride_cols)};
    uint32_t dilations[] = {static_cast<uint32_t>(conv_dims.dilation_rows),
                            static_cast<uint32_t>(conv_dims.dilation_cols)};
    uint32_t start_padding[] = {
        static_cast<uint32_t>(conv_dims.pad_rows_before),
        static_cast<uint32_t>(conv_dims.pad_cols_before)};
    uint32_t end_padding[] = {static_cast<uint32_t>(conv_dims.pad_rows_after),
                              static_cast<uint32_t>(conv_dims.pad_cols_after)};
    uint32_t output_padding[] = {0, 0};
    uint32_t group_count =
        static_cast<uint32_t>(conv_dims.in_depth / conv_dims.patch_depth);

    DmlKernelParams params;

    using namespace DmlTensorAxes;

    // The dimensions of the filter tensor are laid out a little differently
    // than what DML expects
    auto filter_layout = {H, W, C, N};

    // The layout of the input/output tensors is determined by the "data_format"
    // attribute
    auto input_output_layout =
        GetDmlTensorLayout(conv_params.data_format, kDimensionCount);

    DmlKernelTensors tensors = GetTensorInfos(ctx, params);
    tensors.inputs[0]->desc =
        CreateTensorDescFromInput(ctx, 0, input_output_layout);
    tensors.inputs[1]->desc = CreateTensorDescFromInput(ctx, 1, filter_layout);
    tensors.outputs[0]->desc =
        CreateTensorDescFromOutput(ctx, 0, input_output_layout);

    auto input_descs = GetDmlTensorDescs(tensors.inputs);
    auto output_descs = GetDmlTensorDescs(tensors.outputs);

    auto scope = dml::Graph(ctx->GetDmlDevice(),
                            GetDmlXTensorPolicy(conv_params.data_format));
    auto input = dml::InputTensor(scope, 0, input_descs[0]);
    auto filter = dml::InputTensor(scope, 1, input_descs[1]);

    Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op;

    if (BiasAddArgs<T>::IsSupported(fused_computation_type)) {
      CHECK(ctx->GetInputCount() == 3);

      const TensorShape& bias_tensor_shape = ctx->GetInputTensorShape(2);
      // bias must be 1-dimensional
      CHECK(bias_tensor_shape.dims() == 1);
      uint32_t bias_size = bias_tensor_shape.dim_size(0);

      // dml expects bias to be 4d tensor
      dml::TensorDesc::Dimensions bias_sizes{1, bias_size, 1, 1};
      dml::TensorDesc::Dimensions bias_strides{bias_size, 1, 0, 0};
      auto bias = dml::Reinterpret(dml::InputTensor(scope, 2, input_descs[2]),
                                   bias_sizes, bias_strides);
      switch (fused_computation_type) {
        case FusedComputationType::kBiasAdd: {
          auto conv2d = dml::Convolution(
              input, filter, bias, DML_CONVOLUTION_MODE_CROSS_CORRELATION,
              DML_CONVOLUTION_DIRECTION_FORWARD, strides, dilations,
              start_padding, end_padding, output_padding, group_count);
          compiled_op = scope.Compile(DML_EXECUTION_FLAG_NONE, {conv2d});
        } break;
        case FusedComputationType::kBiasAddWithRelu: {
          auto conv2d = dml::Convolution(
              input, filter, bias, DML_CONVOLUTION_MODE_CROSS_CORRELATION,
              DML_CONVOLUTION_DIRECTION_FORWARD, strides, dilations,
              start_padding, end_padding, output_padding, group_count,
              dml::FusedActivation::Relu());
          compiled_op = scope.Compile(DML_EXECUTION_FLAG_NONE, {conv2d});
        } break;
        case FusedComputationType::kBiasAddWithRelu6: {
          auto conv2d = dml::Convolution(
              input, filter, bias, DML_CONVOLUTION_MODE_CROSS_CORRELATION,
              DML_CONVOLUTION_DIRECTION_FORWARD, strides, dilations,
              start_padding, end_padding, output_padding, group_count);
          auto relu6 = dml::ActivationRelu6(conv2d);
          compiled_op = scope.Compile(DML_EXECUTION_FLAG_NONE, {relu6});
        } break;
        case FusedComputationType::kBiasAddWithElu: {
          auto conv2d = dml::Convolution(
              input, filter, bias, DML_CONVOLUTION_MODE_CROSS_CORRELATION,
              DML_CONVOLUTION_DIRECTION_FORWARD, strides, dilations,
              start_padding, end_padding, output_padding, group_count,
              dml::FusedActivation::Elu(1.0f));
          compiled_op = scope.Compile(DML_EXECUTION_FLAG_NONE, {conv2d});
        } break;
        default:
          CHECK(false);
      }
    } else if (FusedBatchNormArgs<T>::IsSupported(fused_computation_type)) {
      CHECK(ctx->GetInputCount() == 6);
      const TensorShape& scale_tensor_shape = ctx->GetInputTensorShape(2);
      const TensorShape& offset_tensor_shape = ctx->GetInputTensorShape(3);
      const TensorShape& mean_tensor_shape = ctx->GetInputTensorShape(4);
      const TensorShape& variance_tensor_shape = ctx->GetInputTensorShape(5);

      // all arguments must be 1-dimensional
      CHECK(scale_tensor_shape.dims() == 1);
      CHECK(offset_tensor_shape.dims() == 1);
      CHECK(mean_tensor_shape.dims() == 1);
      CHECK(variance_tensor_shape.dims() == 1);

      input_descs[2] = CreateTensorDescFromInput(ctx, 2, {C}).GetDmlDesc();
      input_descs[3] = CreateTensorDescFromInput(ctx, 3, {C}).GetDmlDesc();
      input_descs[4] = CreateTensorDescFromInput(ctx, 4, {C}).GetDmlDesc();
      input_descs[5] = CreateTensorDescFromInput(ctx, 5, {C}).GetDmlDesc();

      auto scale = dml::InputTensor(scope, 2, input_descs[2]);
      auto offset = dml::InputTensor(scope, 3, input_descs[3]);
      auto mean = dml::InputTensor(scope, 4, input_descs[4]);
      auto variance = dml::InputTensor(scope, 5, input_descs[5]);

      auto conv2d = dml::Convolution(
          input, filter, absl::nullopt, DML_CONVOLUTION_MODE_CROSS_CORRELATION,
          DML_CONVOLUTION_DIRECTION_FORWARD, strides, dilations, start_padding,
          end_padding, output_padding, group_count);
      switch (fused_computation_type) {
        case FusedComputationType::kFusedBatchNorm: {
          auto batch_norm =
              dml::BatchNormalization(conv2d, mean, variance, scale, offset,
                                      true, fused_computation_args.epsilon);
          compiled_op = scope.Compile(DML_EXECUTION_FLAG_NONE, {batch_norm});
        } break;
        case FusedComputationType::kFusedBatchNormWithRelu: {
          auto batch_norm = dml::BatchNormalization(
              conv2d, mean, variance, scale, offset, true,
              fused_computation_args.epsilon, dml::FusedActivation::Relu());
          compiled_op = scope.Compile(DML_EXECUTION_FLAG_NONE, {batch_norm});
        } break;
        case FusedComputationType::kFusedBatchNormWithRelu6: {
          auto batch_norm =
              dml::BatchNormalization(conv2d, mean, variance, scale, offset,
                                      true, fused_computation_args.epsilon);
          auto relu6 = dml::ActivationRelu6(batch_norm);
          compiled_op = scope.Compile(DML_EXECUTION_FLAG_NONE, {relu6});
        } break;
        case FusedComputationType::kFusedBatchNormWithElu: {
          auto batch_norm = dml::BatchNormalization(
              conv2d, mean, variance, scale, offset, true,
              fused_computation_args.epsilon, dml::FusedActivation::Elu(1.0f));
          compiled_op = scope.Compile(DML_EXECUTION_FLAG_NONE, {batch_norm});
        } break;
        default:
          CHECK(false);
      }
    }

    Initialize(ctx, std::move(tensors), compiled_op.Get());
  }
};

#define DML_REGISTER_KERNEL(type)                                        \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("_FusedConv2D").Device(DEVICE_DML).TypeConstraint<type>("T"), \
      DmlKernelWrapper<DmlFusedConv2DKernel<type>, ConvShapeHelper>);
// FusedConv2D only supports float32
TF_CALL_float(DML_REGISTER_KERNEL);
#undef DML_REGISTER_KERNEL

class DmlConv2DBackpropInputKernel : public DmlKernel {
 public:
  using InitHelper = Conv2DGradInitHelper;

  explicit DmlConv2DBackpropInputKernel(DmlKernelConstruction* ctx,
                                        const InitHelper* init_helper) {
    CHECK(ctx->GetInputCount() == 3);
    CHECK(ctx->GetOutputCount() == 1);

    // 2D conv requires 4D tensors
    static const uint32_t kDimensionCount = 4;
    static const uint32_t kSpatialDimensionCount = 2;

    // Tensor 0 is a 1-d vector of input shapes
    CHECK(ctx->GetInputTensorShape(0).dims() == 1);
    CHECK(ctx->GetInputTensorShape(1).dims() == kDimensionCount);
    CHECK(ctx->GetInputTensorShape(2).dims() == kDimensionCount);
    CHECK(ctx->GetOutputTensorShape(0).dims() == kDimensionCount);

    auto& input_sizes = ctx->GetConstantInputTensor(0);
    TensorShape input_shape;
    TF_CHECK_OK(
        TensorShapeUtils::MakeShape(input_sizes.vec<int32>(), &input_shape));

    DmlKernelParams params;
    params.kernel_input_indices = {
        2, 1,
        absl::nullopt  // We don't use the DML bias tensor
    };

    DmlKernelTensors tensors = GetTensorInfos(ctx, params);
    const Conv2DParameters& conv_params = init_helper->GetParams();

    Conv2DDimensions conv_dims;
    TF_CHECK_OK(ComputeConv2DDimension(
        conv_params, input_shape, ctx->GetInputTensorShape(1), &conv_dims));

    uint32_t strides[] = {static_cast<uint32_t>(conv_dims.stride_rows),
                          static_cast<uint32_t>(conv_dims.stride_cols)};
    uint32_t dilations[] = {static_cast<uint32_t>(conv_dims.dilation_rows),
                            static_cast<uint32_t>(conv_dims.dilation_cols)};
    uint32_t start_padding[] = {
        static_cast<uint32_t>(conv_dims.pad_rows_before),
        static_cast<uint32_t>(conv_dims.pad_cols_before)};
    uint32_t end_padding[] = {static_cast<uint32_t>(conv_dims.pad_rows_after),
                              static_cast<uint32_t>(conv_dims.pad_cols_after)};
    uint32_t output_padding[] = {0, 0};
    uint32_t group_count =
        static_cast<uint32_t>(conv_dims.in_depth / conv_dims.patch_depth);

    using namespace DmlTensorAxes;

    // The dimensions of the filter tensor are laid out a little differently
    // than what DML expects.
    auto filter_layout = {H, W, C, N};

    // The layout of the input/output tensors is determined by the
    // "data_format" attribute
    auto input_output_layout =
        GetDmlTensorLayout(conv_params.data_format, kDimensionCount);

    tensors.inputs[0]->desc =
        CreateTensorDescFromInput(ctx, 2, input_output_layout);
    tensors.inputs[1]->desc = CreateTensorDescFromInput(ctx, 1, filter_layout);
    tensors.outputs[0]->desc =
        CreateTensorDescFromOutput(ctx, 0, input_output_layout);

    auto input_descs = GetDmlTensorDescs(tensors.inputs);
    auto output_descs = GetDmlTensorDescs(tensors.outputs);

    DML_CONVOLUTION_OPERATOR_DESC conv_desc = {};
    conv_desc.InputTensor = &input_descs[0];
    conv_desc.FilterTensor = &input_descs[1];
    conv_desc.BiasTensor = nullptr;
    conv_desc.OutputTensor = &output_descs[0];

    // Note DML_CONVOLUTION_MODE_CROSS_CORRELATION automatically rotates
    // filter 180 when operating in DML_CONVOLUTION_DIRECTION_BACKWARD.
    // Hence we do not need to specify DML_CONVOLUTION_MODE_CONVOLUTION here.
    conv_desc.Mode = DML_CONVOLUTION_MODE_CROSS_CORRELATION;
    conv_desc.Direction = DML_CONVOLUTION_DIRECTION_BACKWARD;
    conv_desc.DimensionCount = kSpatialDimensionCount;
    conv_desc.Strides = strides;
    conv_desc.Dilations = dilations;
    conv_desc.StartPadding = start_padding;
    conv_desc.EndPadding = end_padding;
    conv_desc.OutputPadding = output_padding;
    conv_desc.GroupCount = group_count;
    conv_desc.FusedActivation = nullptr;

    DML_OPERATOR_DESC op_desc = {DML_OPERATOR_CONVOLUTION, &conv_desc};
    Initialize(ctx, std::move(tensors), op_desc);
  }
};

#define DML_REGISTER_KERNEL(type)                    \
  REGISTER_KERNEL_BUILDER(                           \
      Name("Conv2DBackpropInput")                    \
          .Device(DEVICE_DML)                        \
          .TypeConstraint<type>("T")                 \
          .HostMemory("input_sizes"),                \
      DmlKernelWrapper<DmlConv2DBackpropInputKernel, \
                       GetOutputShapeFromDimsTensorHelper<int32, 0>>);
TF_CALL_DML_FLOAT_TYPES(DML_REGISTER_KERNEL);
#undef DML_REGISTER_KERNEL

class DmlConv2DBackpropFilterKernel : public DmlKernel {
 public:
  using InitHelper = Conv2DGradInitHelper;

  explicit DmlConv2DBackpropFilterKernel(
      DmlKernelConstruction* ctx, const Conv2DGradInitHelper* init_helper) {
    CHECK(ctx->GetInputCount() == 3);
    CHECK(ctx->GetOutputCount() == 1);

    // 2D conv requires 4D tensors
    static const uint32_t kDimensionCount = 4;
    static const uint32_t kSpatialDimensionCount = 2;

    CHECK(ctx->GetInputTensorShape(0).dims() == kDimensionCount);
    CHECK(ctx->GetInputTensorShape(1).dims() == 1);
    CHECK(ctx->GetInputTensorShape(2).dims() == kDimensionCount);
    CHECK(ctx->GetOutputTensorShape(0).dims() == kDimensionCount);

    auto& filter_sizes = ctx->GetConstantInputTensor(1);
    TensorShape filter_shape;
    TF_CHECK_OK(
        TensorShapeUtils::MakeShape(filter_sizes.vec<int32>(), &filter_shape));

    const Conv2DParameters& conv_params = init_helper->GetParams();

    Conv2DDimensions conv_dims;
    TF_CHECK_OK(ComputeConv2DDimension(conv_params, ctx->GetInputTensorShape(0),
                                       filter_shape, &conv_dims));

    uint32_t strides[] = {static_cast<uint32_t>(conv_dims.stride_rows),
                          static_cast<uint32_t>(conv_dims.stride_cols)};
    uint32_t dilations[] = {static_cast<uint32_t>(conv_dims.dilation_rows),
                            static_cast<uint32_t>(conv_dims.dilation_cols)};
    uint32_t start_padding[] = {
        static_cast<uint32_t>(conv_dims.pad_rows_before),
        static_cast<uint32_t>(conv_dims.pad_cols_before)};
    uint32_t end_padding[] = {static_cast<uint32_t>(conv_dims.pad_rows_after),
                              static_cast<uint32_t>(conv_dims.pad_cols_after)};
    uint32_t output_padding[] = {0, 0};
    uint32_t group_count =
        static_cast<uint32_t>(conv_dims.in_depth / conv_dims.patch_depth);

    DmlKernelParams params;
    params.kernel_input_indices = {
        0, 2,
        absl::nullopt  // We don't use the DML bias tensor
    };

    using namespace DmlTensorAxes;

    // The dimensions of the filter tensor are laid out a little differently
    // than what DML expects. Note order of C and N
    // are reversed as we convolve with the backprop_output
    // which has N channels, where N is the number of output
    // feature maps in the forward conv2d case.
    auto filter_layout = {H, W, N, C};

    // The layout of the input/output tensors is determined by the
    // "data_format" attribute
    auto input_output_layout =
        GetDmlTensorLayout(conv_params.data_format, kDimensionCount);

    // Swap N and C channels. Note: Channel 'N' of output
    // contains the 'K' feature maps produced in the forward
    // direction. Swapping is required because we
    // convolve the incoming backprop_output gradient containing K
    //  features with the 'K' channelled filter
    DmlTensorAxis axis;
    switch (conv_params.data_format) {
      case FORMAT_NHWC:
        axis = input_output_layout[0];
        input_output_layout[0] = input_output_layout[3];
        input_output_layout[3] = axis;
        break;

      case FORMAT_NCHW:
        axis = input_output_layout[0];
        input_output_layout[0] = input_output_layout[1];
        input_output_layout[1] = axis;
        break;
    }

    DmlKernelTensors tensors = GetTensorInfos(ctx, params);
    tensors.inputs[0]->desc =
        CreateTensorDescFromInput(ctx, 0, input_output_layout);
    tensors.inputs[1]->desc =
        CreateTensorDescFromInput(ctx, 2, input_output_layout);

    // The output tensor gets the filter_layout as we are computing the
    // back-prop for the filter.
    tensors.outputs[0]->desc =
        CreateTensorDescFromOutput(ctx, 0, filter_layout);

    auto input_descs = GetDmlTensorDescs(tensors.inputs);
    auto output_descs = GetDmlTensorDescs(tensors.outputs);

    DML_CONVOLUTION_OPERATOR_DESC conv_desc = {};
    conv_desc.InputTensor = &input_descs[0];
    conv_desc.FilterTensor = &input_descs[1];
    conv_desc.BiasTensor = nullptr;
    conv_desc.OutputTensor = &output_descs[0];
    conv_desc.Mode = DML_CONVOLUTION_MODE_CROSS_CORRELATION;
    conv_desc.Direction = DML_CONVOLUTION_DIRECTION_FORWARD;
    conv_desc.DimensionCount = kSpatialDimensionCount;
    conv_desc.Strides = dilations;
    conv_desc.Dilations = strides;
    conv_desc.StartPadding = start_padding;
    conv_desc.EndPadding = end_padding;
    conv_desc.OutputPadding = output_padding;
    conv_desc.GroupCount = group_count;
    conv_desc.FusedActivation = nullptr;

    DML_OPERATOR_DESC op_desc = {DML_OPERATOR_CONVOLUTION, &conv_desc};
    Initialize(ctx, std::move(tensors), op_desc);
  }
};

#define DML_REGISTER_KERNEL(type)                     \
  REGISTER_KERNEL_BUILDER(                            \
      Name("Conv2DBackpropFilter")                    \
          .Device(DEVICE_DML)                         \
          .TypeConstraint<type>("T")                  \
          .HostMemory("filter_sizes"),                \
      DmlKernelWrapper<DmlConv2DBackpropFilterKernel, \
                       GetOutputShapeFromDimsTensorHelper<int32, 1>>);
TF_CALL_DML_FLOAT_TYPES(DML_REGISTER_KERNEL);
#undef DML_REGISTER_KERNEL

// Depthwise Conv2D has two input tensors (input and filter) so it has two
// gradient ops as well. The initialization logic for both gradient ops is
// the same. Each gradient op receives only one of the original input
// tensors from the forward pass (the "non-backprop tensor"); the gradient op
// receives a 1D host-memory shape tensor for the tensor whose gradients the op
// is computing (the "backprop tensor").
//
// - DepthwiseConv2dNativeBackpropInput : computes gradients w.r.t. input
//   inputs = [input_sizes, filter, out_gradients]
//   BackpropTensorIndex = 0 (input)
//   NonBackpropTensorIndex = 1 (filter)
//
// - DepthwiseConv2dNativeBackpropFilter : computes gradients w.r.t. filter
//   inputs = [input, filter_sizes, out_gradients]
//   BackpropTensorIndex = 1 (filter)
//   NonBackpropTensorIndex = 0 (input)
template <int BackpropTensorIndex>
class DepthwiseConv2DBackpropInitHelper : public InitializationHelper {
 public:
  using Attributes = DepthwiseConv2DNativeInitHelper::Attributes;

  DepthwiseConv2DBackpropInitHelper(OpKernelContext* context,
                                    std::shared_ptr<const Attributes> attr)
      : attr_(attr) {
    constexpr int NonBackpropTensorIndex = !BackpropTensorIndex;

    const char* label = nullptr;
    const char* backprop_sizes_name = nullptr;
    if constexpr (BackpropTensorIndex == 0) {
      label = "Conv2DBackpropInput";
      backprop_sizes_name = "input_sizes";
    } else {
      label = "Conv2DBackpropFilter";
      backprop_sizes_name = "filter_sizes";
    }

    const Tensor& backprop_tensor_shape_tensor =
        context->input(BackpropTensorIndex);
    OP_REQUIRES(
        context,
        TensorShapeUtils::IsVector(backprop_tensor_shape_tensor.shape()),
        errors::InvalidArgument(label, ":", backprop_sizes_name,
                                " input must be 1-dim, not ",
                                backprop_tensor_shape_tensor.dims()));
    TensorShape backprop_tensor_shape;
    const int32* backprop_tensor_shape_data =
        backprop_tensor_shape_tensor.template flat<int32>().data();
    for (int i = 0; i < backprop_tensor_shape_tensor.NumElements(); ++i) {
      OP_REQUIRES(
          context, backprop_tensor_shape_data[i] >= 0,
          errors::InvalidArgument("Dimension ", i, " of ", backprop_sizes_name,
                                  " must be >= 0"));
      backprop_tensor_shape.AddDim(backprop_tensor_shape_data[i]);
    }

    const Tensor& non_backprop_tensor = context->input(NonBackpropTensorIndex);
    const TensorShape& non_backprop_tensor_shape = non_backprop_tensor.shape();

    const TensorShape& input_shape = (BackpropTensorIndex == 0)
                                         ? backprop_tensor_shape
                                         : non_backprop_tensor_shape;
    const TensorShape& filter_shape = (BackpropTensorIndex == 0)
                                          ? non_backprop_tensor_shape
                                          : backprop_tensor_shape;

    const Tensor& out_backprop = context->input(2);

    OP_REQUIRES(
        context, input_shape.dims() == 4,
        errors::InvalidArgument(label, ": input must be 4-dimensional"));
    OP_REQUIRES(
        context, filter_shape.dims() == 4,
        errors::InvalidArgument(label, ": filter must be 4-dimensional"));
    OP_REQUIRES(
        context, out_backprop.dims() == 4,
        errors::InvalidArgument(label, ": out_backprop must be 4-dimensional"));

    const int64 batch_size_raw = input_shape.dim_size(0);
    OP_REQUIRES(
        context,
        FastBoundsCheck(batch_size_raw, std::numeric_limits<uint32_t>::max()),
        errors::InvalidArgument("Batch size too large"));
    OP_REQUIRES(
        context, batch_size_raw == out_backprop.dim_size(0),
        errors::InvalidArgument(
            label, ": input and out_backprop must have the same batch size"));
    batch_size_ = static_cast<uint32_t>(batch_size_raw);

    const int64 input_depth_raw =
        GetTensorDim(input_shape, attr_->data_format, 'C');
    OP_REQUIRES(
        context,
        FastBoundsCheck(input_depth_raw, std::numeric_limits<uint32_t>::max()),
        errors::InvalidArgument("Input depth too large"));
    in_channels_ = static_cast<uint32_t>(input_depth_raw);

    const int64 input_rows_raw =
        GetTensorDim(input_shape, attr_->data_format, 'H');
    OP_REQUIRES(
        context,
        FastBoundsCheck(input_rows_raw, std::numeric_limits<uint32_t>::max()),
        errors::InvalidArgument("Input rows too large"));
    in_height_ = static_cast<uint32_t>(input_rows_raw);

    const int64 input_cols_raw =
        GetTensorDim(input_shape, attr_->data_format, 'W');
    OP_REQUIRES(
        context,
        FastBoundsCheck(input_cols_raw, std::numeric_limits<uint32_t>::max()),
        errors::InvalidArgument("Input cols too large"));
    in_width_ = static_cast<uint32_t>(input_cols_raw);

    const int64 filter_rows_raw = filter_shape.dim_size(0);
    OP_REQUIRES(
        context,
        FastBoundsCheck(filter_rows_raw, std::numeric_limits<uint32_t>::max()),
        errors::InvalidArgument("Filter rows too large"));
    filter_height_ = static_cast<uint32_t>(filter_rows_raw);

    const int64 filter_cols_raw = filter_shape.dim_size(1);
    OP_REQUIRES(
        context,
        FastBoundsCheck(filter_cols_raw, std::numeric_limits<uint32_t>::max()),
        errors::InvalidArgument("Filter cols too large"));
    filter_width_ = static_cast<uint32_t>(filter_cols_raw);

    OP_REQUIRES(context, in_channels_ == filter_shape.dim_size(2),
                errors::InvalidArgument(
                    label, ": input and filter must have the same in_depth"));

    const int64 depth_multiplier = filter_shape.dim_size(3);

    const int64 out_channels_raw =
        GetTensorDim(out_backprop.shape(), attr_->data_format, 'C');
    OP_REQUIRES(
        context,
        FastBoundsCheck(out_channels_raw, std::numeric_limits<uint32_t>::max()),
        errors::InvalidArgument("Output depth too large"));
    OP_REQUIRES(
        context, (depth_multiplier * in_channels_) == out_channels_raw,
        errors::InvalidArgument(
            label, ": depth_multiplier * in_depth not equal to out_depth"));
    out_channels_ = static_cast<uint32_t>(out_channels_raw);

    const int64 output_rows_raw =
        GetTensorDim(out_backprop.shape(), attr_->data_format, 'H');
    OP_REQUIRES(
        context,
        FastBoundsCheck(output_rows_raw, std::numeric_limits<uint32_t>::max()),
        errors::InvalidArgument("Output rows too large"));
    out_height_ = static_cast<uint32_t>(output_rows_raw);

    const int64 output_cols_raw =
        GetTensorDim(out_backprop.shape(), attr_->data_format, 'W');
    OP_REQUIRES(
        context,
        FastBoundsCheck(output_cols_raw, std::numeric_limits<uint32_t>::max()),
        errors::InvalidArgument("Output cols too large"));
    out_width_ = static_cast<uint32_t>(output_cols_raw);

    int64 out_height_calculated;
    int64 pad_rows_before_raw;
    int64 pad_rows_after_raw;
    OP_REQUIRES_OK(context,
                   GetWindowedOutputSizeVerboseV2(
                       in_height_, filter_height_, attr->dilation_h,
                       attr_->stride_h, attr->padding, &out_height_calculated,
                       &pad_rows_before_raw, &pad_rows_after_raw));
    OP_REQUIRES(
        context, out_height_calculated == out_height_,
        errors::InvalidArgument(
            label, ": Number of rows of out_backprop doesn't match computed: ",
            "actual = ", out_height_, ", computed = ", out_height_calculated));
    pad_rows_before_ = static_cast<uint32_t>(pad_rows_before_raw);
    pad_rows_after_ = static_cast<uint32_t>(pad_rows_after_raw);

    int64 out_width_calculated;
    int64 pad_cols_before_raw;
    int64 pad_cols_after_raw;
    OP_REQUIRES_OK(context,
                   GetWindowedOutputSizeVerboseV2(
                       in_width_, filter_width_, attr_->dilation_w,
                       attr_->stride_w, attr->padding, &out_width_calculated,
                       &pad_cols_before_raw, &pad_cols_after_raw));
    OP_REQUIRES(
        context, out_width_calculated == out_width_,
        errors::InvalidArgument(
            label, ": Number of cols of out_backprop doesn't match computed: ",
            "actual = ", out_width_, ", computed = ", out_width_calculated));
    pad_cols_before_ = static_cast<uint32_t>(pad_cols_before_raw);
    pad_cols_after_ = static_cast<uint32_t>(pad_cols_after_raw);
  }

  TensorFormat GetDataFormat() const { return attr_->data_format; }

  uint32_t GetBatchSize() const { return batch_size_; }
  uint32_t GetInChannels() const { return in_channels_; }
  uint32_t GetInHeight() const { return in_height_; }
  uint32_t GetInWidth() const { return in_width_; }
  uint32_t GetFilterHeight() const { return filter_height_; }
  uint32_t GetFilterWidth() const { return filter_width_; }
  uint32_t GetOutChannels() const { return out_channels_; }
  uint32_t GetOutHeight() const { return out_height_; }
  uint32_t GetOutWidth() const { return out_width_; }
  uint32_t GetStrideH() const { return attr_->stride_h; }
  uint32_t GetStrideW() const { return attr_->stride_w; }
  uint32_t GetDilationH() const { return attr_->dilation_h; }
  uint32_t GetDilationW() const { return attr_->dilation_w; }
  uint32_t GetGroupCount() const { return in_channels_; }
  uint32_t GetPadRowsBefore() const { return pad_rows_before_; }
  uint32_t GetPadColsBefore() const { return pad_cols_before_; }
  uint32_t GetPadRowsAfter() const { return pad_rows_after_; }
  uint32_t GetPadColsAfter() const { return pad_cols_after_; }

 private:
  const std::shared_ptr<const Attributes> attr_;
  uint32_t batch_size_;
  uint32_t in_channels_;
  uint32_t in_height_;
  uint32_t in_width_;
  int32 filter_channels_;
  int32 filter_height_;
  int32 filter_width_;
  uint32_t out_channels_;
  uint32_t out_height_;
  uint32_t out_width_;
  uint32_t pad_rows_before_;
  uint32_t pad_cols_before_;
  uint32_t pad_rows_after_;
  uint32_t pad_cols_after_;
};

using DepthwiseConv2DBackpropInputInitHelper =
    DepthwiseConv2DBackpropInitHelper<0>;

class DmlDepthwiseConv2DBackpropFilterKernel : public DmlKernel {
 public:
  using InitHelper = DepthwiseConv2DBackpropInitHelper<1>;

  explicit DmlDepthwiseConv2DBackpropFilterKernel(
      DmlKernelConstruction* ctx, const InitHelper* init_helper) {
    DCHECK(ctx->GetInputCount() == 3);
    DCHECK(ctx->GetOutputCount() == 1);

    uint32_t strides[] = {1, init_helper->GetStrideH(),
                          init_helper->GetStrideW()};
    uint32_t dilations[] = {1, init_helper->GetDilationH(),
                            init_helper->GetDilationW()};
    uint32_t start_padding[] = {0, init_helper->GetPadRowsBefore(),
                                init_helper->GetPadColsBefore()};
    uint32_t end_padding[] = {0, init_helper->GetPadRowsAfter(),
                              init_helper->GetPadColsAfter()};
    uint32_t output_padding[] = {0, 0, 0};

    DmlKernelParams params;
    params.kernel_input_indices = {
        0, 2,
        absl::nullopt  // We don't use the DML bias tensor
    };

    using namespace DmlTensorAxes;

    // The depthwise 2D conv filter backprop calculation can be expressed as a
    // convolution, just like standard 2D convolution, but it requires an extra
    // dimension to correctly reinterpret and transpose dimensions without
    // copying to intermediate tensors.
    DmlTensorLayout input_layout;
    DmlTensorLayout output_backprop_layout;
    TensorShape input_shape;
    TensorShape output_backprop_shape;
    switch (init_helper->GetDataFormat()) {
      case FORMAT_NCHW:
        input_layout = {N, D, C, H, W};
        input_shape = {
            1,                             // N
            init_helper->GetBatchSize(),   // D
            init_helper->GetInChannels(),  // C
            init_helper->GetInHeight(),    // H
            init_helper->GetInWidth(),     // W
        };
        output_backprop_layout = {D, N, C, H, W};
        output_backprop_shape = {
            init_helper->GetBatchSize(),    // D
            init_helper->GetOutChannels(),  // N
            1,                              // C
            init_helper->GetOutHeight(),    // H
            init_helper->GetOutWidth(),     // W
        };
        break;
      case FORMAT_NHWC:
        input_layout = {N, D, H, W, C};
        input_shape = {
            1,                             // N
            init_helper->GetBatchSize(),   // D
            init_helper->GetInHeight(),    // H
            init_helper->GetInWidth(),     // W
            init_helper->GetInChannels(),  // C
        };
        output_backprop_layout = {D, C, H, W, N};
        output_backprop_shape = {
            init_helper->GetBatchSize(),    // D
            1,                              // C
            init_helper->GetOutHeight(),    // H
            init_helper->GetOutWidth(),     // W
            init_helper->GetOutChannels(),  // N
        };
        break;
    }

    auto filter_backprop_layout = {N, H, W, C, D};
    TensorShape filter_backprop_shape = {
        1,                               // N
        init_helper->GetFilterHeight(),  // H
        init_helper->GetFilterWidth(),   // W
        init_helper->GetOutChannels(),   // C
        1                                // D
    };

    DmlKernelTensors tensors = GetTensorInfos(ctx, params);
    tensors.inputs[0]->desc = DmlTensorDesc::Create(
        ctx->GetInputDataType(0), input_shape, input_shape, input_layout);

    tensors.inputs[1]->desc =
        DmlTensorDesc::Create(ctx->GetInputDataType(2), output_backprop_shape,
                              output_backprop_shape, output_backprop_layout);

    tensors.outputs[0]->desc =
        DmlTensorDesc::Create(ctx->GetOutputDataType(0), filter_backprop_shape,
                              filter_backprop_shape, filter_backprop_layout);

    auto input_descs = GetDmlTensorDescs(tensors.inputs);
    auto output_descs = GetDmlTensorDescs(tensors.outputs);

    DML_CONVOLUTION_OPERATOR_DESC conv_desc = {};
    conv_desc.InputTensor = &input_descs[0];
    conv_desc.FilterTensor = &input_descs[1];
    conv_desc.BiasTensor = nullptr;
    conv_desc.OutputTensor = &output_descs[0];
    conv_desc.Mode = DML_CONVOLUTION_MODE_CROSS_CORRELATION;
    conv_desc.Direction = DML_CONVOLUTION_DIRECTION_FORWARD;
    conv_desc.DimensionCount = kNcdhwSpatialDimensionCount;
    conv_desc.Strides = strides;
    conv_desc.Dilations = dilations;
    conv_desc.StartPadding = start_padding;
    conv_desc.EndPadding = end_padding;
    conv_desc.OutputPadding = output_padding;
    conv_desc.GroupCount = init_helper->GetGroupCount();
    conv_desc.FusedActivation = nullptr;

    DML_OPERATOR_DESC op_desc = {DML_OPERATOR_CONVOLUTION, &conv_desc};
    Initialize(ctx, std::move(tensors), op_desc);
  }
};

#define DML_REGISTER_KERNEL(type)                              \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("DepthwiseConv2dNativeBackpropFilter")              \
          .Device(DEVICE_DML)                                  \
          .TypeConstraint<type>("T")                           \
          .HostMemory("filter_sizes"),                         \
      DmlKernelWrapper<DmlDepthwiseConv2DBackpropFilterKernel, \
                       GetOutputShapeFromDimsTensorHelper<int32, 1>>);
TF_CALL_DML_FLOAT_TYPES(DML_REGISTER_KERNEL);
#undef DML_REGISTER_KERNEL

class DmlDepthwiseConv2DBackpropInputKernel : public DmlKernel {
 public:
  using InitHelper = DepthwiseConv2DBackpropInitHelper<0>;

  explicit DmlDepthwiseConv2DBackpropInputKernel(
      DmlKernelConstruction* ctx, const InitHelper* init_helper) {
    DCHECK(ctx->GetInputCount() == 3);
    DCHECK(ctx->GetOutputCount() == 1);

    uint32_t strides[] = {init_helper->GetStrideH(), init_helper->GetStrideW()};
    uint32_t dilations[] = {init_helper->GetDilationH(),
                            init_helper->GetDilationW()};
    uint32_t start_padding[] = {init_helper->GetPadRowsBefore(),
                                init_helper->GetPadColsBefore()};
    uint32_t end_padding[] = {init_helper->GetPadRowsAfter(),
                              init_helper->GetPadColsAfter()};
    uint32_t output_padding[] = {0, 0};
    uint32_t group_count = init_helper->GetGroupCount();

    DmlKernelParams params;
    params.kernel_input_indices = {
        2, 1,
        absl::nullopt  // We don't use the DML bias tensor
    };

    using namespace DmlTensorAxes;

    auto output_layout =
        GetDmlTensorLayout(init_helper->GetDataFormat(), kNchwDimensionCount);

    DmlKernelTensors tensors = GetTensorInfos(ctx, params);

    tensors.inputs[0]->desc = CreateTensorDescFromInput(ctx, 2, output_layout);

    auto filter_layout = {H, W, C, N};
    TensorShape filter_shape = {init_helper->GetFilterHeight(),
                                init_helper->GetFilterWidth(),
                                init_helper->GetInChannels() / group_count,
                                init_helper->GetOutChannels()};

    tensors.inputs[1]->desc = DmlTensorDesc::Create(
        ctx->GetInputDataType(1), filter_shape, filter_shape, filter_layout);

    tensors.outputs[0]->desc =
        CreateTensorDescFromOutput(ctx, 0, output_layout);

    auto input_descs = GetDmlTensorDescs(tensors.inputs);
    auto output_descs = GetDmlTensorDescs(tensors.outputs);

    DML_CONVOLUTION_OPERATOR_DESC conv_desc = {};
    conv_desc.InputTensor = &input_descs[0];
    conv_desc.FilterTensor = &input_descs[1];
    conv_desc.BiasTensor = nullptr;
    conv_desc.OutputTensor = &output_descs[0];
    // Note DML_CONVOLUTION_MODE_CROSS_CORRELATION automatically rotates
    // filter 180 when operating in DML_CONVOLUTION_DIRECTION_BACKWARD.
    // Hence we do not need to specify DML_CONVOLUTION_MODE_CONVOLUTION here.
    conv_desc.Mode = DML_CONVOLUTION_MODE_CROSS_CORRELATION;
    conv_desc.Direction = DML_CONVOLUTION_DIRECTION_BACKWARD;
    conv_desc.DimensionCount = kNchwSpatialDimensionCount;
    conv_desc.Strides = strides;
    conv_desc.Dilations = dilations;
    conv_desc.StartPadding = start_padding;
    conv_desc.EndPadding = end_padding;
    conv_desc.OutputPadding = output_padding;
    conv_desc.GroupCount = group_count;
    conv_desc.FusedActivation = nullptr;

    DML_OPERATOR_DESC op_desc = {DML_OPERATOR_CONVOLUTION, &conv_desc};
    Initialize(ctx, std::move(tensors), op_desc);
  }
};

#define DML_REGISTER_KERNEL(type)                             \
  REGISTER_KERNEL_BUILDER(                                    \
      Name("DepthwiseConv2dNativeBackpropInput")              \
          .Device(DEVICE_DML)                                 \
          .TypeConstraint<type>("T")                          \
          .HostMemory("input_sizes"),                         \
      DmlKernelWrapper<DmlDepthwiseConv2DBackpropInputKernel, \
                       GetOutputShapeFromDimsTensorHelper<int32, 0>>);
TF_CALL_DML_FLOAT_TYPES(DML_REGISTER_KERNEL);
#undef DML_REGISTER_KERNEL

template <bool HasDataFormatAttribute>
struct Conv3DAttributes {
  explicit Conv3DAttributes(OpKernelConstruction* context) {
    if constexpr (HasDataFormatAttribute) {
      string data_format_str;
      OP_REQUIRES_OK(context,
                     context->GetAttr("data_format", &data_format_str));
      OP_REQUIRES(context, FormatFromString(data_format_str, &data_format),
                  errors::InvalidArgument("Invalid data format"));
    } else {
      // V2 conv3d grad ops have a data format attribute but the original ops
      // assume NHWC format.
      data_format = FORMAT_NHWC;
    }

    std::vector<int32> strides;
    OP_REQUIRES_OK(context, context->GetAttr("strides", &strides));
    OP_REQUIRES(context, strides.size() == 5,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 5 dimensions"));

    int32 stride_n = GetTensorDim(strides, data_format, 'N');
    int32 stride_c = GetTensorDim(strides, data_format, 'C');
    stride_d = GetTensorDim(strides, data_format, '0');
    stride_h = GetTensorDim(strides, data_format, '1');
    stride_w = GetTensorDim(strides, data_format, '2');

    OP_REQUIRES(
        context, (stride_n == 1 && stride_c == 1),
        errors::InvalidArgument("Current implementation does not yet support "
                                "strides in the batch and depth dimensions."));
    OP_REQUIRES(
        context, (stride_d > 0 && stride_h > 0 && stride_w > 0),
        errors::InvalidArgument("Spatial strides should be larger than 0."));

    std::vector<int32> dilations;
    OP_REQUIRES_OK(context, context->GetAttr("dilations", &dilations));
    OP_REQUIRES(context, dilations.size() == 5,
                errors::InvalidArgument("Dilation rates field must "
                                        "specify 5 dimensions"));

    int32 dilation_n = GetTensorDim(dilations, data_format, 'N');
    int32 dilation_c = GetTensorDim(dilations, data_format, 'C');
    dilation_d = GetTensorDim(dilations, data_format, '0');
    dilation_h = GetTensorDim(dilations, data_format, '1');
    dilation_w = GetTensorDim(dilations, data_format, '2');

    OP_REQUIRES(context, (dilation_n == 1 && dilation_c == 1),
                errors::InvalidArgument(
                    "Current implementation does not yet support "
                    "dilation rates in the batch and depth dimensions."));
    OP_REQUIRES(
        context, (dilation_d > 0 && dilation_h > 0 && dilation_w > 0),
        errors::InvalidArgument("Dilated rates should be larger than 0."));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding));
  }

  TensorFormat data_format;
  Padding padding;
  int32 stride_d;
  int32 stride_h;
  int32 stride_w;
  int32 dilation_d;
  int32 dilation_h;
  int32 dilation_w;
};

class Conv3DInitHelper : public InitializationHelper {
 public:
  using Attributes = Conv3DAttributes<true>;

  Conv3DInitHelper(OpKernelContext* context,
                   std::shared_ptr<const Attributes> attr)
      : attr_(attr) {
    // Input tensor is of the following dimensions:
    // [ batch, in_z, in_y, in_x, in_channels ]
    const Tensor& input = context->input(0);

    // Input filter is of the following dimensions:
    // [ filter_z, filter_y, filter_x, in_channels, out_channels]
    const Tensor& filter = context->input(1);

    // NOTE: The ordering of the spatial dimensions is arbitrary, but has to be
    // kept consistent between input/filter/output.
    OP_REQUIRES(context, input.dims() == 5,
                errors::InvalidArgument("input must be 5-dimensional"));
    OP_REQUIRES(context, filter.dims() == 5,
                errors::InvalidArgument("filter must be 5-dimensional"));

    const int64 in_batch = GetTensorDim(input, attr->data_format, 'N');
    const int64 in_channels = GetTensorDim(input, attr->data_format, 'C');
    const int64 in_depth = GetTensorDim(input, attr->data_format, '0');
    const int64 in_height = GetTensorDim(input, attr->data_format, '1');
    const int64 in_width = GetTensorDim(input, attr->data_format, '2');

    const int64 filter_depth = filter.dim_size(0);
    const int64 filter_height = filter.dim_size(1);
    const int64 filter_width = filter.dim_size(2);
    const int64 filter_channels = filter.dim_size(3);
    const int64 out_channels = filter.dim_size(4);

    OP_REQUIRES(context, in_channels % filter_channels == 0,
                errors::InvalidArgument(
                    "Input depth must be evenly divisible by filter depth: ",
                    in_channels, " vs ", filter_channels));

    // Dimension order for these arrays is: z, y, x.
    std::array<int64, 3> input_size = {{in_depth, in_height, in_width}};
    std::array<int64, 3> filter_size = {
        {filter_depth, filter_height, filter_width}};
    std::array<int64, 3> dilations = {
        {attr->dilation_d, attr->dilation_h, attr->dilation_w}};
    std::array<int64, 3> strides = {
        {attr->stride_d, attr->stride_h, attr->stride_w}};

    std::array<int64, 3> out, padding;

    OP_REQUIRES_OK(
        context, Get3dOutputSizeV2(input_size, filter_size, dilations, strides,
                                   attr->padding, &out, &padding));

    batch_size_ = static_cast<uint32_t>(in_batch);
    in_channels_ = static_cast<uint32_t>(in_channels);
    in_depth_ = static_cast<uint32_t>(in_depth);
    in_height_ = static_cast<uint32_t>(in_height);
    in_width_ = static_cast<uint32_t>(in_width);
    filter_channels_ = static_cast<uint32_t>(filter_channels);
    filter_depth_ = static_cast<uint32_t>(filter_depth);
    filter_height_ = static_cast<uint32_t>(filter_height);
    filter_width_ = static_cast<uint32_t>(filter_width);
    out_channels_ = static_cast<uint32_t>(out_channels);
    out_depth_ = static_cast<uint32_t>(out[0]);
    out_height_ = static_cast<uint32_t>(out[1]);
    out_width_ = static_cast<uint32_t>(out[2]);
    strides_[0] = attr->stride_d;
    strides_[1] = attr->stride_h;
    strides_[2] = attr->stride_w;
    dilations_[0] = attr->dilation_d;
    dilations_[1] = attr->dilation_h;
    dilations_[2] = attr->dilation_w;
    start_padding_[0] = static_cast<uint32_t>(padding[0]);
    start_padding_[1] = static_cast<uint32_t>(padding[1]);
    start_padding_[2] = static_cast<uint32_t>(padding[2]);
  }

  TensorFormat GetDataFormat() const { return attr_->data_format; }

  uint32_t GetBatchSize() const { return batch_size_; }
  uint32_t GetInChannels() const { return in_channels_; }
  uint32_t GetInDepth() const { return in_depth_; }
  uint32_t GetInHeight() const { return in_height_; }
  uint32_t GetInWidth() const { return in_width_; }
  uint32_t GetFilterChannels() const { return filter_channels_; }
  uint32_t GetFilterDepth() const { return filter_depth_; }
  uint32_t GetFilterHeight() const { return filter_height_; }
  uint32_t GetFilterWidth() const { return filter_width_; }
  uint32_t GetOutChannels() const { return out_channels_; }
  uint32_t GetOutDepth() const { return out_depth_; }
  uint32_t GetOutHeight() const { return out_height_; }
  uint32_t GetOutWidth() const { return out_width_; }
  const std::array<uint32_t, 3>& GetStrides() const { return strides_; }
  const std::array<uint32_t, 3>& GetDilations() const { return dilations_; }
  const std::array<uint32_t, 3>& GetStartPadding() const {
    return start_padding_;
  }
  const std::array<uint32_t, 3>& GetEndPadding() const { return end_padding_; }
  const std::array<uint32_t, 3>& GetOutPadding() const { return out_padding_; }
  uint32_t GetGroupCount() const { return in_channels_ / filter_channels_; }

 private:
  const std::shared_ptr<const Attributes> attr_;
  uint32_t batch_size_;
  uint32_t in_channels_;
  uint32_t in_depth_;
  uint32_t in_height_;
  uint32_t in_width_;
  uint32_t filter_channels_;
  uint32_t filter_depth_;
  uint32_t filter_height_;
  uint32_t filter_width_;
  uint32_t out_channels_;
  uint32_t out_depth_;
  uint32_t out_height_;
  uint32_t out_width_;
  std::array<uint32_t, 3> strides_;
  std::array<uint32_t, 3> dilations_;
  std::array<uint32_t, 3> start_padding_;
  std::array<uint32_t, 3> end_padding_ = {0, 0, 0};
  std::array<uint32_t, 3> out_padding_ = {0, 0, 0};
};

class Conv3DShapeHelper : public ShapeHelper {
 public:
  std::vector<TensorShape> GetOutputShapes(
      OpKernelContext* ctx,
      const InitializationHelper* initialization_helper) const override {
    auto init_helper =
        static_cast<const Conv3DInitHelper*>(initialization_helper);

    TensorShape out_shape = ShapeFromFormat(
        init_helper->GetDataFormat(), init_helper->GetBatchSize(),
        {init_helper->GetOutDepth(), init_helper->GetOutHeight(),
         init_helper->GetOutWidth()},
        init_helper->GetOutChannels());

    return {std::move(out_shape)};
  }
};

class DmlConv3DKernel : public DmlKernel {
 public:
  using InitHelper = Conv3DInitHelper;

  explicit DmlConv3DKernel(DmlKernelConstruction* ctx,
                           const InitHelper* init_helper) {
    CHECK(ctx->GetInputCount() == 2);
    CHECK(ctx->GetOutputCount() == 1);

    // 3D conv requires 5D tensors
    static const uint32_t kDimensionCount = 5;
    static const uint32_t kSpatialDimensionCount = 3;

    CHECK(ctx->GetInputTensorShape(0).dims() == kDimensionCount);
    CHECK(ctx->GetInputTensorShape(1).dims() == kDimensionCount);
    CHECK(ctx->GetOutputTensorShape(0).dims() == kDimensionCount);

    DmlKernelParams params;
    params.kernel_input_indices = {
        0, 1,
        absl::nullopt  // We don't use the DML bias tensor
    };

    using namespace DmlTensorAxes;

    auto filter_layout = {D, H, W, C, N};

    auto input_output_layout =
        GetDmlTensorLayout(init_helper->GetDataFormat(), kDimensionCount);

    DmlKernelTensors tensors = GetTensorInfos(ctx, params);
    tensors.inputs[0]->desc =
        CreateTensorDescFromInput(ctx, 0, input_output_layout);
    tensors.inputs[1]->desc = CreateTensorDescFromInput(ctx, 1, filter_layout);
    tensors.outputs[0]->desc =
        CreateTensorDescFromOutput(ctx, 0, input_output_layout);

    auto input_descs = GetDmlTensorDescs(tensors.inputs);
    auto output_descs = GetDmlTensorDescs(tensors.outputs);

    DML_CONVOLUTION_OPERATOR_DESC conv_desc = {};
    conv_desc.InputTensor = &input_descs[0];
    conv_desc.FilterTensor = &input_descs[1];
    conv_desc.BiasTensor = nullptr;
    conv_desc.OutputTensor = &output_descs[0];
    conv_desc.Mode = DML_CONVOLUTION_MODE_CROSS_CORRELATION;
    conv_desc.Direction = DML_CONVOLUTION_DIRECTION_FORWARD;
    conv_desc.DimensionCount = kSpatialDimensionCount;
    conv_desc.Strides = init_helper->GetStrides().data();
    conv_desc.Dilations = init_helper->GetDilations().data();
    conv_desc.StartPadding = init_helper->GetStartPadding().data();
    conv_desc.EndPadding = init_helper->GetEndPadding().data();
    conv_desc.OutputPadding = init_helper->GetOutPadding().data();
    conv_desc.GroupCount = init_helper->GetGroupCount();
    conv_desc.FusedActivation = nullptr;

    DML_OPERATOR_DESC op_desc = {DML_OPERATOR_CONVOLUTION, &conv_desc};
    Initialize(ctx, std::move(tensors), op_desc);
  }
};

#define DML_REGISTER_KERNEL(type)                                  \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("Conv3D").Device(DEVICE_DML).TypeConstraint<type>("T"), \
      DmlKernelWrapper<DmlConv3DKernel, Conv3DShapeHelper>);
TF_CALL_DML_FLOAT_TYPES(DML_REGISTER_KERNEL);
#undef DML_REGISTER_KERNEL

template <bool HasDataFormatAttribute, bool BackpropInput>
class Conv3DGradInitHelper : public InitializationHelper {
 public:
  using Attributes = Conv3DAttributes<HasDataFormatAttribute>;

  Conv3DGradInitHelper(OpKernelContext* context,
                       std::shared_ptr<const Attributes> attr)
      : attr_(attr) {
    StringPiece label;
    TensorShape input_shape;
    TensorShape filter_shape;
    if constexpr (BackpropInput) {
      label = "Conv3DBackpropInputOp";
      const Tensor& input_sizes = context->input(0);
      OP_REQUIRES_OK(context,
                     context->op_kernel().MakeShape(input_sizes, &input_shape));
      filter_shape = context->input(1).shape();
    } else {
      label = "Conv3DBackpropFilterOp";
      input_shape = context->input(0).shape();
      const Tensor& filter_sizes = context->input(1);
      OP_REQUIRES_OK(
          context, context->op_kernel().MakeShape(filter_sizes, &filter_shape));
    }

    const TensorShape& out_backprop_shape = context->input(2).shape();

    std::vector<int32> strides;
    std::vector<int32> dilations;
    if (attr->data_format == FORMAT_NCHW) {
      strides = {1, 1, attr->stride_d, attr->stride_h, attr->stride_w};
      dilations = {1, 1, attr->dilation_d, attr->dilation_h, attr->dilation_w};
    } else {
      strides = {1, attr->stride_d, attr->stride_h, attr->stride_w, 1};
      dilations = {1, attr->dilation_d, attr->dilation_h, attr->dilation_w, 1};
    }

    ConvBackpropDimensions dims;
    OP_REQUIRES_OK(
        context, ConvBackpropComputeDimensionsV2(
                     label, /*num_spatial_dims=*/3, input_shape, filter_shape,
                     out_backprop_shape, dilations, strides, attr->padding, {},
                     attr->data_format, &dims));

    uint32_t pad_d = static_cast<uint32_t>(dims.SpatialPadding(attr->padding, 0));
    uint32_t pad_h = static_cast<uint32_t>(dims.SpatialPadding(attr->padding, 1));
    uint32_t pad_w = static_cast<uint32_t>(dims.SpatialPadding(attr->padding, 2));

    batch_size_ = static_cast<uint32_t>(dims.batch_size);
    in_channels_ = static_cast<uint32_t>(dims.in_depth);
    in_depth_ = static_cast<uint32_t>(dims.input_size(0));
    in_height_ = static_cast<uint32_t>(dims.input_size(1));
    in_width_ = static_cast<uint32_t>(dims.input_size(2));
    filter_channels_ = static_cast<uint32_t>(filter_shape.dim_size(3));
    filter_depth_ = static_cast<uint32_t>(dims.filter_size(0));
    filter_height_ = static_cast<uint32_t>(dims.filter_size(1));
    filter_width_ = static_cast<uint32_t>(dims.filter_size(2));
    out_channels_ = static_cast<uint32_t>(dims.out_depth);
    out_depth_ = static_cast<uint32_t>(dims.output_size(0));
    out_height_ = static_cast<uint32_t>(dims.output_size(1));
    out_width_ = static_cast<uint32_t>(dims.output_size(2));
    strides_[0] = attr->stride_d;
    strides_[1] = attr->stride_h;
    strides_[2] = attr->stride_w;
    dilations_[0] = attr->dilation_d;
    dilations_[1] = attr->dilation_h;
    dilations_[2] = attr->dilation_w;
    start_padding_[0] = pad_d / 2;
    start_padding_[1] = pad_h / 2;
    start_padding_[2] = pad_w / 2;
    end_padding_[0] = pad_d / 2 + pad_d % 2;
    end_padding_[1] = pad_h / 2 + pad_h % 2;
    end_padding_[2] = pad_w / 2 + pad_w % 2;
  }

  TensorFormat GetDataFormat() const { return attr_->data_format; }

  uint32_t GetBatchSize() const { return batch_size_; }
  uint32_t GetInChannels() const { return in_channels_; }
  uint32_t GetInDepth() const { return in_depth_; }
  uint32_t GetInHeight() const { return in_height_; }
  uint32_t GetInWidth() const { return in_width_; }
  uint32_t GetFilterChannels() const { return filter_channels_; }
  uint32_t GetFilterDepth() const { return filter_depth_; }
  uint32_t GetFilterHeight() const { return filter_height_; }
  uint32_t GetFilterWidth() const { return filter_width_; }
  uint32_t GetOutChannels() const { return out_channels_; }
  uint32_t GetOutDepth() const { return out_depth_; }
  uint32_t GetOutHeight() const { return out_height_; }
  uint32_t GetOutWidth() const { return out_width_; }
  const std::array<uint32_t, 3>& GetStrides() const { return strides_; }
  const std::array<uint32_t, 3>& GetDilations() const { return dilations_; }
  const std::array<uint32_t, 3>& GetStartPadding() const {
    return start_padding_;
  }
  const std::array<uint32_t, 3>& GetEndPadding() const { return end_padding_; }
  const std::array<uint32_t, 3>& GetOutPadding() const { return out_padding_; }
  uint32_t GetGroupCount() const { return in_channels_ / filter_channels_; }

 private:
  const std::shared_ptr<const Attributes> attr_;
  uint32_t batch_size_;
  uint32_t in_channels_;
  uint32_t in_depth_;
  uint32_t in_height_;
  uint32_t in_width_;
  uint32_t filter_channels_;
  uint32_t filter_depth_;
  uint32_t filter_height_;
  uint32_t filter_width_;
  uint32_t out_channels_;
  uint32_t out_depth_;
  uint32_t out_height_;
  uint32_t out_width_;
  std::array<uint32_t, 3> strides_;
  std::array<uint32_t, 3> dilations_;
  std::array<uint32_t, 3> start_padding_;
  std::array<uint32_t, 3> end_padding_;
  std::array<uint32_t, 3> out_padding_ = {0, 0, 0};
};

template <bool HasDataFormatAttribute>
class DmlConv3DBackpropInputKernel : public DmlKernel {
 public:
  using InitHelper = Conv3DGradInitHelper<HasDataFormatAttribute, true>;

  explicit DmlConv3DBackpropInputKernel(DmlKernelConstruction* ctx,
                                        const InitHelper* init_helper) {
    CHECK(ctx->GetInputCount() == 3);
    CHECK(ctx->GetOutputCount() == 1);

    // 3D conv requires 5D tensors
    static const uint32_t kDimensionCount = 5;
    static const uint32_t kSpatialDimensionCount = 3;

    auto& input_sizes = ctx->GetConstantInputTensor(0);
    TensorShape input_shape;
    TF_CHECK_OK(
        TensorShapeUtils::MakeShape(input_sizes.vec<int32>(), &input_shape));

    DmlKernelParams params;
    params.kernel_input_indices = {
        2, 1,
        absl::nullopt  // We don't use the DML bias tensor
    };

    DmlKernelTensors tensors = GetTensorInfos(ctx, params);

    using namespace DmlTensorAxes;

    // The dimensions of the filter tensor are laid out a little differently
    // than what DML expects.
    auto filter_layout = {D, H, W, C, N};

    // The layout of the input/output tensors is determined by the
    // "data_format" attribute
    auto input_output_layout =
        GetDmlTensorLayout(init_helper->GetDataFormat(), kDimensionCount);

    tensors.inputs[0]->desc =
        CreateTensorDescFromInput(ctx, 2, input_output_layout);
    tensors.inputs[1]->desc = CreateTensorDescFromInput(ctx, 1, filter_layout);
    tensors.outputs[0]->desc =
        CreateTensorDescFromOutput(ctx, 0, input_output_layout);

    auto input_descs = GetDmlTensorDescs(tensors.inputs);
    auto output_descs = GetDmlTensorDescs(tensors.outputs);

    DML_CONVOLUTION_OPERATOR_DESC conv_desc = {};
    conv_desc.InputTensor = &input_descs[0];
    conv_desc.FilterTensor = &input_descs[1];
    conv_desc.BiasTensor = nullptr;
    conv_desc.OutputTensor = &output_descs[0];

    // Note DML_CONVOLUTION_MODE_CROSS_CORRELATION automatically rotates
    // filter 180 when operating in DML_CONVOLUTION_DIRECTION_BACKWARD.
    // Hence we do not need to specify DML_CONVOLUTION_MODE_CONVOLUTION here.
    conv_desc.Mode = DML_CONVOLUTION_MODE_CROSS_CORRELATION;
    conv_desc.Direction = DML_CONVOLUTION_DIRECTION_BACKWARD;
    conv_desc.DimensionCount = kSpatialDimensionCount;
    conv_desc.Strides = init_helper->GetStrides().data();
    conv_desc.Dilations = init_helper->GetDilations().data();
    conv_desc.StartPadding = init_helper->GetStartPadding().data();
    conv_desc.EndPadding = init_helper->GetEndPadding().data();
    conv_desc.OutputPadding = init_helper->GetOutPadding().data();
    conv_desc.GroupCount = init_helper->GetGroupCount();
    conv_desc.FusedActivation = nullptr;

    DML_OPERATOR_DESC op_desc = {DML_OPERATOR_CONVOLUTION, &conv_desc};
    Initialize(ctx, std::move(tensors), op_desc);
  }
};

#define DML_REGISTER_KERNEL(type)                               \
  REGISTER_KERNEL_BUILDER(                                      \
      Name("Conv3DBackpropInput")                               \
          .Device(DEVICE_DML)                                   \
          .TypeConstraint<type>("T"),                           \
      DmlKernelWrapper<DmlConv3DBackpropInputKernel<false>,     \
                       GetOutputShapeFromInputShapeHelper<0>>); \
  REGISTER_KERNEL_BUILDER(                                      \
      Name("Conv3DBackpropInputV2")                             \
          .Device(DEVICE_DML)                                   \
          .TypeConstraint<type>("T")                            \
          .HostMemory("input_sizes"),                           \
      DmlKernelWrapper<DmlConv3DBackpropInputKernel<true>,      \
                       GetOutputShapeFromDimsTensorHelper<int32, 0>>);
TF_CALL_DML_FLOAT_TYPES(DML_REGISTER_KERNEL);
#undef DML_REGISTER_KERNEL

template <bool HasDataFormatAttribute>
class DmlConv3DBackpropFilterKernel : public DmlKernel {
 public:
  using InitHelper = Conv3DGradInitHelper<HasDataFormatAttribute, false>;

  explicit DmlConv3DBackpropFilterKernel(DmlKernelConstruction* ctx,
                                         const InitHelper* init_helper) {
    CHECK(ctx->GetInputCount() == 3);
    CHECK(ctx->GetOutputCount() == 1);

    // 2D conv requires 4D tensors
    static const uint32_t kDimensionCount = 5;
    static const uint32_t kSpatialDimensionCount = 3;

    auto& filter_sizes = ctx->GetConstantInputTensor(1);
    TensorShape filter_shape;
    TF_CHECK_OK(
        TensorShapeUtils::MakeShape(filter_sizes.vec<int32>(), &filter_shape));

    DmlKernelParams params;
    params.kernel_input_indices = {
        0, 2,
        absl::nullopt  // We don't use the DML bias tensor
    };

    using namespace DmlTensorAxes;

    // The dimensions of the filter tensor are laid out a little differently
    // than what DML expects. Note order of C and N
    // are reversed as we convolve with the backprop_output
    // which has N channels, where N is the number of output
    // feature maps in the forward conv2d case.
    auto filter_layout = {D, H, W, N, C};

    // The layout of the input/output tensors is determined by the
    // "data_format" attribute
    auto input_output_layout =
        GetDmlTensorLayout(init_helper->GetDataFormat(), kDimensionCount);

    // Swap N and C channels. Note: Channel 'N' of output
    // contains the 'K' feature maps produced in the forward
    // direction. Swapping is required because we
    // convolve the incoming backprop_output gradient containing K
    //  features with the 'K' channelled filter
    switch (init_helper->GetDataFormat()) {
      case FORMAT_NHWC:
        std::swap(input_output_layout[0], input_output_layout[4]);
        break;

      case FORMAT_NCHW:
        std::swap(input_output_layout[0], input_output_layout[1]);
        break;
    }

    DmlKernelTensors tensors = GetTensorInfos(ctx, params);
    tensors.inputs[0]->desc =
        CreateTensorDescFromInput(ctx, 0, input_output_layout);
    tensors.inputs[1]->desc =
        CreateTensorDescFromInput(ctx, 2, input_output_layout);

    // The output tensor gets the filter_layout as we are computing the
    // back-prop for the filter.
    tensors.outputs[0]->desc =
        CreateTensorDescFromOutput(ctx, 0, filter_layout);

    auto input_descs = GetDmlTensorDescs(tensors.inputs);
    auto output_descs = GetDmlTensorDescs(tensors.outputs);

    DML_CONVOLUTION_OPERATOR_DESC conv_desc = {};
    conv_desc.InputTensor = &input_descs[0];
    conv_desc.FilterTensor = &input_descs[1];
    conv_desc.BiasTensor = nullptr;
    conv_desc.OutputTensor = &output_descs[0];
    conv_desc.Mode = DML_CONVOLUTION_MODE_CROSS_CORRELATION;
    conv_desc.Direction = DML_CONVOLUTION_DIRECTION_FORWARD;
    conv_desc.DimensionCount = kSpatialDimensionCount;
    conv_desc.Strides = init_helper->GetStrides().data();
    conv_desc.Dilations = init_helper->GetDilations().data();
    conv_desc.StartPadding = init_helper->GetStartPadding().data();
    conv_desc.EndPadding = init_helper->GetEndPadding().data();
    conv_desc.OutputPadding = init_helper->GetOutPadding().data();
    conv_desc.GroupCount = init_helper->GetGroupCount();
    conv_desc.FusedActivation = nullptr;

    DML_OPERATOR_DESC op_desc = {DML_OPERATOR_CONVOLUTION, &conv_desc};
    Initialize(ctx, std::move(tensors), op_desc);
  }
};

#define DML_REGISTER_KERNEL(type)                               \
  REGISTER_KERNEL_BUILDER(                                      \
      Name("Conv3DBackpropFilter")                              \
          .Device(DEVICE_DML)                                   \
          .TypeConstraint<type>("T"),                           \
      DmlKernelWrapper<DmlConv3DBackpropFilterKernel<false>,    \
                       GetOutputShapeFromInputShapeHelper<1>>); \
  REGISTER_KERNEL_BUILDER(                                      \
      Name("Conv3DBackpropFilterV2")                            \
          .Device(DEVICE_DML)                                   \
          .TypeConstraint<type>("T")                            \
          .HostMemory("filter_sizes"),                          \
      DmlKernelWrapper<DmlConv3DBackpropFilterKernel<true>,     \
                       GetOutputShapeFromDimsTensorHelper<int32, 1>>);
TF_CALL_DML_FLOAT_TYPES(DML_REGISTER_KERNEL);
#undef DML_REGISTER_KERNEL

}  // namespace tensorflow
