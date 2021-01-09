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

    auto scope = dml::Graph(ctx->GetDmlDevice(), GetDmlXTensorPolicy(conv_params.data_format));
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

}  // namespace tensorflow
