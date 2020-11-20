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
#include "tensorflow/core/kernels/fused_batch_norm_op.h"

namespace tensorflow {

class BatchNormInitializationHelper : public InitializationHelper {
 public:
  struct Attributes {
    explicit Attributes(OpKernelConstruction* ctx) {
      std::string tensor_format_attr;
      OP_REQUIRES_OK(ctx, ctx->GetAttr("epsilon", &epsilon));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("is_training", &is_training));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("data_format", &tensor_format_attr));
      OP_REQUIRES(ctx, FormatFromString(tensor_format_attr, &tensor_format),
                  errors::InvalidArgument("Invalid data format"));

      if (ctx->GetAttr("num_side_inputs", &num_side_inputs).ok()) {
        OP_REQUIRES_OK(ctx, ParseActivationMode(ctx, &activation_mode));

        OP_REQUIRES(
            ctx,
            activation_mode ==
                    functor::FusedBatchNormActivationMode::kIdentity ||
                activation_mode == functor::FusedBatchNormActivationMode::kRelu,
            errors::InvalidArgument(
                "FusedBatchNorm only supports Identity and Relu for now."));

        OP_REQUIRES(ctx, num_side_inputs >= 0 && num_side_inputs <= 1,
                    errors::InvalidArgument(
                        "FusedBatchNorm accepts at most one side input."));

        if (num_side_inputs > 0 && is_training) {
          OP_REQUIRES(ctx,
                      activation_mode !=
                          functor::FusedBatchNormActivationMode::kIdentity,
                      errors::InvalidArgument(
                          "Identity activation is not supported with "
                          "non-empty side input"));
        }
      } else {
        num_side_inputs = 0;
        activation_mode = functor::FusedBatchNormActivationMode::kIdentity;
      }
    }

    float epsilon;
    bool is_training;
    TensorFormat tensor_format;
    int num_side_inputs;
    functor::FusedBatchNormActivationMode activation_mode;
  };

  BatchNormInitializationHelper(OpKernelContext* ctx,
                                std::shared_ptr<const Attributes> attr)
      : attr_(std::move(attr)) {
    const Tensor& x = ctx->input(0);
    const Tensor& scale = ctx->input(1);
    const Tensor& offset = ctx->input(2);
    const Tensor& estimated_mean = ctx->input(3);
    const Tensor& estimated_variance = ctx->input(4);

    OP_REQUIRES(ctx, x.dims() == 4,
                errors::InvalidArgument("input must be 4-dimensional",
                                        x.shape().DebugString()));
    OP_REQUIRES(ctx, scale.dims() == 1,
                errors::InvalidArgument("scale must be 1-dimensional",
                                        scale.shape().DebugString()));
    OP_REQUIRES(ctx, offset.dims() == 1,
                errors::InvalidArgument("offset must be 1-dimensional",
                                        offset.shape().DebugString()));
    OP_REQUIRES(ctx, estimated_mean.dims() == 1,
                errors::InvalidArgument("estimated_mean must be 1-dimensional",
                                        estimated_mean.shape().DebugString()));
    OP_REQUIRES(
        ctx, estimated_variance.dims() == 1,
        errors::InvalidArgument("estimated_variance must be 1-dimensional",
                                estimated_variance.shape().DebugString()));
    if (attr_->num_side_inputs > 0) {
      const Tensor& side_input = ctx->input(5);

      OP_REQUIRES(ctx, side_input.shape() == x.shape(),
                  errors::InvalidArgument(
                      "side_input shape must be equal to input shape: ",
                      side_input.shape().DebugString(),
                      " != ", x.shape().DebugString()));
    }
  }

  bool IsNoOpKernel(
      OpKernelContext* ctx,
      absl::Span<const TensorShape> output_shapes) const override {
    // FusedBatchNorm can legitimately have empty tensors depending on whether
    // is_training is true or not. So this kernel is only truly a no-op when the
    // input tensor is empty.
    return (ctx->input(0).NumElements() == 0);
  }

  float GetEpsilon() const { return attr_->epsilon; }
  bool IsTraining() const { return attr_->is_training; }
  TensorFormat GetFormat() const { return attr_->tensor_format; }
  bool AddSideInput() const { return attr_->num_side_inputs > 0; }

  functor::FusedBatchNormActivationMode GetActivationMode() const {
    return attr_->activation_mode;
  }

 private:
  const std::shared_ptr<const Attributes> attr_;
};

class BatchGlobalNormInitializationHelper : public InitializationHelper {
 public:
  struct Attributes {
    explicit Attributes(OpKernelConstruction* ctx) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("variance_epsilon", &variance_epsilon));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("scale_after_normalization",
                                       &scale_after_normalization));
    }
    float variance_epsilon;
    bool scale_after_normalization;
  };

  BatchGlobalNormInitializationHelper(OpKernelContext* ctx,
                                      std::shared_ptr<const Attributes> attr)
      : attr_(std::move(attr)) {
    const Tensor& t = ctx->input(0);
    const Tensor& m = ctx->input(1);
    const Tensor& v = ctx->input(2);
    const Tensor& beta = ctx->input(3);
    const Tensor& gamma = ctx->input(4);
    OP_REQUIRES(ctx, t.dims() == 4,
                errors::InvalidArgument("input must be 4-dimensional",
                                        t.shape().DebugString()));
    OP_REQUIRES(ctx, gamma.dims() == 1,
                errors::InvalidArgument("scale must be 1-dimensional",
                                        gamma.shape().DebugString()));
    OP_REQUIRES(ctx, beta.dims() == 1,
                errors::InvalidArgument("offset must be 1-dimensional",
                                        beta.shape().DebugString()));
    OP_REQUIRES(ctx, m.dims() == 1,
                errors::InvalidArgument("estimated_mean must be 1-dimensional",
                                        m.shape().DebugString()));
    OP_REQUIRES(
        ctx, v.dims() == 1,
        errors::InvalidArgument("estimated_variance must be 1-dimensional",
                                v.shape().DebugString()));
  }

  float GetVarianceEpsilon() const { return attr_->variance_epsilon; }
  bool ScaleAfterNormalization() const {
    return attr_->scale_after_normalization;
  }

 private:
  const std::shared_ptr<const Attributes> attr_;
};

class BatchGlobalNormGradInitializationHelper : public InitializationHelper {
 public:
  struct Attributes {
    explicit Attributes(OpKernelConstruction* ctx) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("variance_epsilon", &variance_epsilon));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("scale_after_normalization",
                                       &scale_after_normalization));
    }
    float variance_epsilon;
    bool scale_after_normalization;
  };

  BatchGlobalNormGradInitializationHelper(
      OpKernelContext* ctx, std::shared_ptr<const Attributes> attr)
      : attr_(std::move(attr)) {
    const Tensor& t = ctx->input(0);
    const Tensor& m = ctx->input(1);
    const Tensor& v = ctx->input(2);
    const Tensor& gamma = ctx->input(3);
    const Tensor& backprop = ctx->input(4);
    OP_REQUIRES(ctx, t.dims() == 4,
                errors::InvalidArgument("input must be 4-dimensional",
                                        t.shape().DebugString()));
    OP_REQUIRES(ctx, gamma.dims() == 1,
                errors::InvalidArgument("scale must be 1-dimensional",
                                        gamma.shape().DebugString()));
    OP_REQUIRES(ctx, backprop.dims() == 4,
                errors::InvalidArgument("offset must be 1-dimensional",
                                        backprop.shape().DebugString()));
    OP_REQUIRES(ctx, m.dims() == 1,
                errors::InvalidArgument("estimated_mean must be 1-dimensional",
                                        m.shape().DebugString()));
    OP_REQUIRES(
        ctx, v.dims() == 1,
        errors::InvalidArgument("estimated_variance must be 1-dimensional",
                                v.shape().DebugString()));
  }

  float GetVarianceEpsilon() const { return attr_->variance_epsilon; }
  bool ScaleAfterNormalization() const {
    return attr_->scale_after_normalization;
  }

 private:
  const std::shared_ptr<const Attributes> attr_;
};

static dml::Expression CreateBatchNormNode(
    dml::Expression x, dml::Expression mean, dml::Expression variance,
    dml::Expression scale, dml::Expression offset, float epsilon,
    functor::FusedBatchNormActivationMode activation_mode) {
  // This should already have been validated in the init helper
  DCHECK(activation_mode == functor::FusedBatchNormActivationMode::kIdentity ||
         activation_mode == functor::FusedBatchNormActivationMode::kRelu);

  auto fused_activation =
      activation_mode == functor::FusedBatchNormActivationMode::kIdentity
          ? dml::FusedActivation::None()
          : dml::FusedActivation::Relu();

  constexpr bool is_spatial = true;
  return dml::BatchNormalization(x, mean, variance, scale, offset, is_spatial,
                                 epsilon, fused_activation);
}

class DmlFusedBatchNormKernel : public DmlKernel {
  enum InputIndex {
    kX,
    kScale,
    kOffset,
    kMean,
    kVariance,
    kSideInput,
  };

 public:
  using InitHelper = BatchNormInitializationHelper;

  explicit DmlFusedBatchNormKernel(
      DmlKernelConstruction* ctx,
      const BatchNormInitializationHelper* init_helper) {
    // FusedBatchNormEx takes an additional side input
    CHECK(ctx->GetInputCount() == 5 || ctx->GetInputCount() == 6);

    // FusedBatchNormV3 returns an additional output
    CHECK(ctx->GetOutputCount() == 5 || ctx->GetOutputCount() == 6);

    float epsilon = init_helper->GetEpsilon();
    bool is_training = init_helper->IsTraining();
    TensorFormat tensor_format = init_helper->GetFormat();

    if (is_training) {
      InitializeForTraining(ctx, epsilon, tensor_format,
                            init_helper->AddSideInput(),
                            init_helper->GetActivationMode());
    } else {
      InitializeForInference(ctx, epsilon, tensor_format,
                             init_helper->AddSideInput(),
                             init_helper->GetActivationMode());
    }
  }

  // Initializes the batch norm kernel for training. In training mode, we don't
  // receive the mean/variance and need to compute it ourselves.
  void InitializeForTraining(
      DmlKernelConstruction* ctx, float epsilon, TensorFormat tensor_format,
      bool add_side_input,
      functor::FusedBatchNormActivationMode activation_mode) {
    DmlKernelParams params;

    // The mean/variance tensors are empty when is_training is set; we need to
    // compute them ourselves
    params.kernel_input_indices = {kX, kScale, kOffset};

    if (add_side_input) {
      params.kernel_input_indices.push_back(kSideInput);
    }

    // Normalized output, computed mean, computed variance, saved mean, saved
    // variance
    params.kernel_output_indices = {0, 1, 2, 3, 4};

    DmlKernelTensors tensors = GetTensorInfos(ctx, params);

    // Input and output tensors have their layout specified by the
    // "data_format" attribute
    DmlTensorLayout input_output_layout =
        GetDmlTensorLayout(tensor_format, kNchwDimensionCount);
    tensors.inputs[0]->desc =
        CreateTensorDescFromInput(ctx, 0, input_output_layout);
    tensors.outputs[0]->desc =
        CreateTensorDescFromOutput(ctx, 0, input_output_layout);

    // Scale and bias are 1D, in the C dimension
    using namespace DmlTensorAxes;
    tensors.inputs[1]->desc = CreateTensorDescFromInput(ctx, 1, {C});
    tensors.inputs[2]->desc = CreateTensorDescFromInput(ctx, 2, {C});

    auto input_descs = GetDmlTensorDescs(tensors.inputs);
    auto output_descs = GetDmlTensorDescs(tensors.outputs);

    // Unfortunately we need to compute the mean/variance ourselves. We can't
    // use DML's built-in MeanVarianceNormalization because we need to return
    // the mean/variance tensors back to TF.

    auto scope =
        dml::Graph(ctx->GetDmlDevice(), GetDmlXTensorPolicy(tensor_format));
    auto x = dml::InputTensor(scope, 0, input_descs[0]);
    auto scale = dml::InputTensor(scope, 1, input_descs[1]);
    auto offset = dml::InputTensor(scope, 2, input_descs[2]);

    DML_TENSOR_DATA_TYPE input_type = x.GetOutputDesc().dataType;
    auto input_sizes = x.GetOutputDesc().sizes;

    // scale and offset are always float32, but the input data type might not
    // be. If that's the case, we need to insert a cast.
    bool is_cast_required = (input_type != scale.GetOutputDesc().dataType);
    if (is_cast_required) {
      scale = dml::Cast(scale, input_type);
      offset = dml::Cast(offset, input_type);
    }

    // We normalize the input for each channel, so the number of elements per
    // normalization is N * H * W
    uint32_t norm_elem_count = input_sizes[0] * input_sizes[2] * input_sizes[3];

    // Compute the mean of the input for each channel. We do this with an
    // AVERAGE reduction across all axes except C.
    auto mean = dml::Reduce(x, DML_REDUCE_FUNCTION_AVERAGE, {0, 2, 3});

    // The strides we need to set to broadcast C across an entire tensor
    dml::TensorDesc::Dimensions broadcast_c_strides = {/*N*/ 0,
                                                       /*C*/ 1,
                                                       /*H*/ 0,
                                                       /*W*/ 0};

    // Broadcasts the C dimension across the entire input tensor
    auto broadcasted_mean =
        dml::Reinterpret(mean, input_sizes, broadcast_c_strides);

    // Compute the variance of the input for each channel.
    auto x_centered = x - broadcasted_mean;
    auto variance =
        dml::Reduce(x_centered, DML_REDUCE_FUNCTION_SUM_SQUARE, {0, 2, 3});
    variance /= norm_elem_count;

    // Apply Bessel's correction to the variance
    float bessel_correction_factor = 1.0f;
    if (norm_elem_count > 1) {  // Prevent division by 0
      bessel_correction_factor = (float)norm_elem_count / (norm_elem_count - 1);
    }

    auto corrected_variance = variance * bessel_correction_factor;

    // If we need to add a side input, we cannot fuse the activation with
    // BatchNorm since the side input needs to be added before the activation
    auto normalized_output = CreateBatchNormNode(
        x, mean, variance, scale, offset, epsilon,
        add_side_input ? functor::FusedBatchNormActivationMode::kIdentity
                       : activation_mode);

    // FusedBatchNormEx can provide a side input that we need to add before
    // activation
    if (add_side_input) {
      auto side_input = dml::InputTensor(scope, 3, input_descs[3]);
      normalized_output += side_input;

      if (activation_mode != functor::FusedBatchNormActivationMode::kIdentity) {
        // Only Relu is supported
        DCHECK(activation_mode == functor::FusedBatchNormActivationMode::kRelu);
        normalized_output = dml::ActivationRelu(normalized_output);
      }
    }

    if (is_cast_required) {
      // These output tensors are defined to always be float32, so insert a cast
      // if necessary
      mean = dml::Cast(mean, DML_TENSOR_DATA_TYPE_FLOAT32);
      corrected_variance =
          dml::Cast(corrected_variance, DML_TENSOR_DATA_TYPE_FLOAT32);
      variance = dml::Cast(variance, DML_TENSOR_DATA_TYPE_FLOAT32);
    }

    auto outputs = {
        normalized_output,
        mean,                // batch_mean
        corrected_variance,  // batch_variance
        mean,                // saved_mean (same as batch_mean)
        variance             // saved_variance
    };

    auto compiled_op = scope.Compile(DML_EXECUTION_FLAG_NONE, outputs);
    Initialize(ctx, std::move(tensors), compiled_op.Get());
  }

  // Initializes the kernel for the is_training=false case.
  void InitializeForInference(
      DmlKernelConstruction* ctx, float epsilon, TensorFormat tensor_format,
      bool add_side_input,
      functor::FusedBatchNormActivationMode activation_mode) {
    DmlKernelParams params;

    // DML's BatchNorm operator takes inputs in a different order than the
    // kernel receives them
    params.kernel_input_indices = {kX, kMean, kVariance, kScale, kOffset};

    if (add_side_input) {
      params.kernel_input_indices.push_back(kSideInput);
    }

    // Normalized output, computed mean, computed variance. We don't use any of
    // the reserved_space outputs for the inference case.
    params.kernel_output_indices = {0, 1, 2};

    DmlKernelTensors tensors = GetTensorInfos(ctx, params);

    // Input and output tensors have their layout specified by the
    // "data_format" attribute
    DmlTensorLayout input_output_layout =
        GetDmlTensorLayout(tensor_format, kNchwDimensionCount);
    tensors.inputs[0]->desc =
        CreateTensorDescFromInput(ctx, 0, input_output_layout);
    tensors.outputs[0]->desc =
        CreateTensorDescFromOutput(ctx, 0, input_output_layout);

    // Mean, variance, scale, and bias are 1D; in the C dimension
    using namespace DmlTensorAxes;
    tensors.inputs[1]->desc = CreateTensorDescFromInput(ctx, 1, {C});
    tensors.inputs[2]->desc = CreateTensorDescFromInput(ctx, 2, {C});
    tensors.inputs[3]->desc = CreateTensorDescFromInput(ctx, 3, {C});
    tensors.inputs[4]->desc = CreateTensorDescFromInput(ctx, 4, {C});

    auto input_descs = GetDmlTensorDescs(tensors.inputs);
    auto output_descs = GetDmlTensorDescs(tensors.outputs);

    auto scope =
        dml::Graph(ctx->GetDmlDevice(), GetDmlXTensorPolicy(tensor_format));
    auto x = dml::InputTensor(scope, 0, input_descs[0]);
    auto mean = dml::InputTensor(scope, 1, input_descs[1]);
    auto variance = dml::InputTensor(scope, 2, input_descs[2]);
    auto scale = dml::InputTensor(scope, 3, input_descs[3]);
    auto offset = dml::InputTensor(scope, 4, input_descs[4]);

    DML_TENSOR_DATA_TYPE input_type = x.GetOutputDesc().dataType;

    // mean/variance in its original datatype
    auto original_mean = mean;
    auto original_variance = variance;

    // scale, offset, mean, and variance are always float32, but the input data
    // type might not be. If that's the case, we need to insert a cast.
    if (input_type != scale.GetOutputDesc().dataType) {
      mean = dml::Cast(mean, input_type);
      variance = dml::Cast(variance, input_type);
      scale = dml::Cast(scale, input_type);
      offset = dml::Cast(offset, input_type);
    }

    // If we need to add a side input, we cannot fuse the activation with
    // BatchNorm since the side input needs to be added before the activation
    auto normalized_output = CreateBatchNormNode(
        x, mean, variance, scale, offset, epsilon,
        add_side_input ? functor::FusedBatchNormActivationMode::kIdentity
                       : activation_mode);

    // FusedBatchNormEx can provide a side input that we need to add before
    // activation
    if (add_side_input) {
      auto side_input = dml::InputTensor(scope, 5, input_descs[5]);
      normalized_output += side_input;

      if (activation_mode != functor::FusedBatchNormActivationMode::kIdentity) {
        // Only Relu is supported
        DCHECK(activation_mode == functor::FusedBatchNormActivationMode::kRelu);
        normalized_output = dml::ActivationRelu(normalized_output);
      }
    }

    // TF requires that we output batch_mean and batch_variance in addition to
    // the normalized output. Since this is the inference case, we just copy the
    // original mean/variance to the batch_mean and batch_variance outputs.
    auto outputs = {
        normalized_output,
        dml::Identity(original_mean),
        dml::Identity(original_variance),
    };

    auto compiled_op = scope.Compile(DML_EXECUTION_FLAG_NONE, outputs);
    Initialize(ctx, std::move(tensors), compiled_op.Get());
  }
};

#define DML_REGISTER_KERNEL(type)                                          \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("FusedBatchNorm").Device(DEVICE_DML).TypeConstraint<type>("T"), \
      DmlKernelWrapper<DmlFusedBatchNormKernel, BatchNormShapeHelper>);
// FusedBatchNorm only supports float32
TF_CALL_float(DML_REGISTER_KERNEL);
#undef DML_REGISTER_KERNEL

#define DML_REGISTER_KERNELS(type)                                      \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("FusedBatchNormV2")                                          \
          .Device(DEVICE_DML)                                           \
          .TypeConstraint<type>("T")                                    \
          .TypeConstraint<float>("U"),                                  \
      DmlKernelWrapper<DmlFusedBatchNormKernel, BatchNormShapeHelper>); \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("FusedBatchNormV3")                                          \
          .Device(DEVICE_DML)                                           \
          .TypeConstraint<type>("T")                                    \
          .TypeConstraint<float>("U"),                                  \
      DmlKernelWrapper<DmlFusedBatchNormKernel, BatchNormShapeHelper>); \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("_FusedBatchNormEx")                                         \
          .Device(DEVICE_DML)                                           \
          .TypeConstraint<type>("T")                                    \
          .TypeConstraint<float>("U"),                                  \
      DmlKernelWrapper<DmlFusedBatchNormKernel, BatchNormShapeHelper>);
TF_CALL_DML_FLOAT_TYPES(DML_REGISTER_KERNELS);
#undef DML_REGISTER_KERNELS

class DmlBatchNormWithGlobalNormalizationKernel : public DmlKernel {
  enum InputIndex {
    kT,
    kM,
    kV,
    kBeta,
    kGamma,
  };

 public:
  using InitHelper = BatchGlobalNormInitializationHelper;
  explicit DmlBatchNormWithGlobalNormalizationKernel(
      DmlKernelConstruction* ctx,
      const BatchGlobalNormInitializationHelper* init_helper) {
    CHECK(ctx->GetInputCount() == 5);
    CHECK(ctx->GetOutputCount() == 1);

    float variance_epsilon = init_helper->GetVarianceEpsilon();
    bool scale_after_normalization = init_helper->ScaleAfterNormalization();

    DmlKernelParams params;

    // DML's BatchNorm operator takes inputs in a different order than the
    // kernel receives them
    if (scale_after_normalization) {
      params.kernel_input_indices = {kT, kM, kV, kGamma, kBeta};
    } else {
      params.kernel_input_indices = {kT, kM, kV, kBeta};
    }
    params.kernel_output_indices = {0};

    DmlKernelTensors tensors = GetTensorInfos(ctx, params);

    auto input_descs = GetDmlTensorDescs(tensors.inputs);
    auto output_descs = GetDmlTensorDescs(tensors.outputs);

    const uint32_t beta_index = scale_after_normalization ? 4 : 3;
    auto scope = dml::Graph(ctx->GetDmlDevice());
    auto t = dml::InputTensor(scope, 0, input_descs[0]);
    auto m = dml::InputTensor(scope, 1, input_descs[1]);
    auto v = dml::InputTensor(scope, 2, input_descs[2]);
    auto beta = dml::InputTensor(scope, beta_index, input_descs[beta_index]);

    // Just use DML's BATCH_NORMALIZATION operator to compute the output.
    dml::Expression normalized_output;
    if (scale_after_normalization) {
      auto gamma = dml::InputTensor(scope, 3, input_descs[3]);
      normalized_output =
          CreateBatchNormNode(t, m, v, gamma, beta, variance_epsilon,
                              functor::FusedBatchNormActivationMode::kIdentity);
    } else {
      auto ones = dml::ScalarTensor(scope, 1.0f, v.GetOutputDesc().sizes);
      normalized_output =
          CreateBatchNormNode(t, m, v, ones, beta, variance_epsilon,
                              functor::FusedBatchNormActivationMode::kIdentity);
    }
    auto outputs = {normalized_output};
    auto compiled_op = scope.Compile(DML_EXECUTION_FLAG_NONE, outputs);
    Initialize(ctx, std::move(tensors), compiled_op.Get());
  }
};

#define DML_REGISTER_KERNEL(type)                                 \
  REGISTER_KERNEL_BUILDER(                                        \
      Name("BatchNormWithGlobalNormalization")                    \
          .Device(DEVICE_DML)                                     \
          .TypeConstraint<type>("T"),                             \
      DmlKernelWrapper<DmlBatchNormWithGlobalNormalizationKernel, \
                       GetOutputShapeAsInputShapeHelper>);
TF_CALL_DML_FLOAT_TYPES(DML_REGISTER_KERNEL);
#undef DML_REGISTER_KERNEL

class DmlFusedBatchNormGradKernel : public DmlKernel {
  enum InputIndex {
    kYBackprop,
    kX,
    kScale,
    kReserveSpace1,
    kReserveSpace2,
    kReserveSpace3,
  };

  enum OutputIndex {
    kXBackprop,
    kScaleBackprop,
    kOffsetBackprop,
  };

 public:
  using InitHelper = NoOpInitializationHelper;

  explicit DmlFusedBatchNormGradKernel(DmlKernelConstruction* ctx,
                                       const InitHelper* init_helper) {
    // FusedBatchNormGradV3 takes an additional input (which we don't use)
    CHECK(ctx->GetInputCount() == 5 || ctx->GetInputCount() == 6);
    CHECK(ctx->GetOutputCount() == 5);

    float epsilon;
    string tensor_format_attr;
    bool is_training;
    TensorFormat tensor_format;
    TF_CHECK_OK(ctx->GetAttr("epsilon", &epsilon));
    TF_CHECK_OK(ctx->GetAttr("data_format", &tensor_format_attr));
    TF_CHECK_OK(ctx->GetAttr("is_training", &is_training));
    CHECK(FormatFromString(tensor_format_attr, &tensor_format));

    DmlKernelParams params;

    // FusedBatchNormGradV3 receives an additional (6th) input which we don't
    // use, so we explicitly only supply 5 indices here
    params.kernel_input_indices = {0, 1, 2, 3, 4};

    // Only the first 3 outputs are used, the remainder are placeholders
    params.kernel_output_indices = {0, 1, 2};

    DmlKernelTensors tensors = GetTensorInfos(ctx, params);

    // y_backprop, x, and x_backprop have their layout specified by the
    // "data_format" attribute. The rest are 1D tensors in the C dimension
    using namespace DmlTensorAxes;
    DmlTensorLayout layout_4D =
        GetDmlTensorLayout(tensor_format, kNchwDimensionCount);
    DmlTensorLayout layout_1D = {C};

    // Inputs
    tensors.inputs[kYBackprop]->desc =
        CreateTensorDescFromInput(ctx, kYBackprop, layout_4D);
    tensors.inputs[kX]->desc = CreateTensorDescFromInput(ctx, kX, layout_4D);
    tensors.inputs[kScale]->desc =
        CreateTensorDescFromInput(ctx, kScale, layout_1D);
    tensors.inputs[kReserveSpace1]->desc =
        CreateTensorDescFromInput(ctx, kReserveSpace1, layout_1D);
    tensors.inputs[kReserveSpace2]->desc =
        CreateTensorDescFromInput(ctx, kReserveSpace2, layout_1D);

    // Outputs
    tensors.outputs[kXBackprop]->desc =
        CreateTensorDescFromOutput(ctx, kXBackprop, layout_4D);
    tensors.outputs[kScaleBackprop]->desc =
        CreateTensorDescFromOutput(ctx, kScaleBackprop, layout_1D);
    tensors.outputs[kOffsetBackprop]->desc =
        CreateTensorDescFromOutput(ctx, kOffsetBackprop, layout_1D);

    auto input_descs = GetDmlTensorDescs(tensors.inputs);
    auto output_descs = GetDmlTensorDescs(tensors.outputs);

    auto scope =
        dml::Graph(ctx->GetDmlDevice(), GetDmlXTensorPolicy(tensor_format));

    auto y_backprop =
        dml::InputTensor(scope, kYBackprop, input_descs[kYBackprop]);
    auto x = dml::InputTensor(scope, kX, input_descs[kX]);
    auto scale = dml::InputTensor(scope, kScale, input_descs[kScale]);
    auto mean =
        dml::InputTensor(scope, kReserveSpace1, input_descs[kReserveSpace1]);
    auto variance =
        dml::InputTensor(scope, kReserveSpace2, input_descs[kReserveSpace2]);

    DML_TENSOR_DATA_TYPE input_type = y_backprop.GetOutputDesc().dataType;
    auto input_sizes = y_backprop.GetOutputDesc().sizes;

    // y_backprop, x, and x_backprop may be float16 or float32, but everything
    // else is always float32. If the types don't match, we need to insert
    // casts.
    bool is_cast_required = (input_type != scale.GetOutputDesc().dataType);

    if (is_cast_required) {
      scale = dml::Cast(scale, input_type);
      mean = dml::Cast(mean, input_type);
      variance = dml::Cast(variance, input_type);
    }

    // The strides we need to set to broadcast C across an entire tensor
    dml::TensorDesc::Dimensions broadcast_c_strides = {/*N*/ 0,
                                                       /*C*/ 1,
                                                       /*H*/ 0,
                                                       /*W*/ 0};

    //
    // Formulae copied from tensorflow/core/kernels/fused_batch_norm_op.cc:
    //
    // x_backprop (training) = scale * rsqrt(variance + epsilon) *
    //              [y_backprop - mean(y_backprop) - (x - mean(x)) *
    //              mean(y_backprop * (x - mean(x))) / (variance + epsilon)]
    //
    // x_backprop (inference) = y_backprop * (scale * rsqrt(pop_var + epsilon))
    //
    // scale_backprop = sum(y_backprop *
    //                  (x - mean(x)) * rsqrt(variance + epsilon))
    // offset_backprop = sum(y_backprop)
    //

    auto variance_e = variance + epsilon;
    auto sqrt_variance_e = dml::Sqrt(variance_e);
    auto coef0 = 1.0f / sqrt_variance_e;  // rsqrt(variance + epsilon)
    auto scaled_coef0 =
        scale / sqrt_variance_e;  // scale * rsqrt(variance + epsilon)

    auto scaled_coef0_bcast =
        dml::Reinterpret(scaled_coef0, input_sizes, broadcast_c_strides);

    // Unlike y_backprop we don't need to recompute the mean of x; it's provided
    // to us as an input. We do, however, need to broadcast it to cover the
    // entire tensor
    auto x_mean = dml::Reinterpret(mean, input_sizes, broadcast_c_strides);

    auto x_centered = x - x_mean;
    auto coef1 = y_backprop * x_centered;  // y_backprop * (x - mean(x))

    // Compute x_backprop:
    dml::Expression x_backprop;
    if (is_training) {
      // x_backprop = scale * rsqrt(variance + epsilon) *
      //              [y_backprop - mean(y_backprop) - (x - mean(x)) *
      //              mean(y_backprop * (x - mean(x))) / (variance + epsilon)]
      //
      // let coef0 = rsqrt(variance + epsilon)
      //     coef1 = y_backprop * (x - mean(x))
      //
      // => x_backprop = scale * coef0 * (y_backprop_centered - x_centered *
      //                                  mean(coef1) / (variance + epsilon))

      // Compute the mean of y_backprop for each C, and broadcast the mean
      // across the entire channel
      auto y_backprop_mean =
          dml::Reduce(y_backprop, DML_REDUCE_FUNCTION_AVERAGE, {0, 2, 3});
      y_backprop_mean =
          dml::Reinterpret(y_backprop_mean, input_sizes, broadcast_c_strides);

      auto variance_e_bcast =
          dml::Reinterpret(variance_e, input_sizes, broadcast_c_strides);

      auto y_backprop_centered = y_backprop - y_backprop_mean;

      auto coef1_mean =
          dml::Reduce(coef1, DML_REDUCE_FUNCTION_AVERAGE, {0, 2, 3});
      coef1_mean =
          dml::Reinterpret(coef1_mean, input_sizes, broadcast_c_strides);

      x_backprop =
          scaled_coef0_bcast *
          (y_backprop_centered - x_centered * coef1_mean / variance_e_bcast);
    } else {
      x_backprop = scaled_coef0_bcast * y_backprop;
    }

    // scale_backprop = sum(y_backprop *
    //                  (x - mean(x)) * rsqrt(variance + epsilon))
    //
    // let coef0 = rsqrt(variance + epsilon)
    //     coef1 = y_backprop * (x - mean(x))
    //
    // => scale_backprop = sum(coef1 * coef0) = sum(coef1) * coef0
    auto scale_backprop =
        dml::Reduce(coef1, DML_REDUCE_FUNCTION_SUM, {0, 2, 3}) * coef0;

    // offset_backprop = sum(y_backprop)
    auto offset_backprop =
        dml::Reduce(y_backprop, DML_REDUCE_FUNCTION_SUM, {0, 2, 3});

    // If necessary, cast outputs to their required types
    if (is_cast_required) {
      scale_backprop = dml::Cast(scale_backprop, DML_TENSOR_DATA_TYPE_FLOAT32);
      offset_backprop =
          dml::Cast(offset_backprop, DML_TENSOR_DATA_TYPE_FLOAT32);
    }

    auto outputs = {
        x_backprop,
        scale_backprop,
        offset_backprop,
    };

    auto compiled_op = scope.Compile(DML_EXECUTION_FLAG_NONE, outputs);
    Initialize(ctx, std::move(tensors), compiled_op.Get());
  }
};

#define DML_REGISTER_KERNEL(type)                                              \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("FusedBatchNormGrad").Device(DEVICE_DML).TypeConstraint<type>("T"), \
      DmlKernelWrapper<DmlFusedBatchNormGradKernel,                            \
                       BatchNormGradShapeHelper>);
// FusedBatchNormGrad only supports float32
TF_CALL_float(DML_REGISTER_KERNEL);
#undef DML_REGISTER_KERNEL

#define DML_REGISTER_KERNEL(type)                                       \
  REGISTER_KERNEL_BUILDER(Name("FusedBatchNormGradV2")                  \
                              .Device(DEVICE_DML)                       \
                              .TypeConstraint<type>("T")                \
                              .TypeConstraint<float>("U"),              \
                          DmlKernelWrapper<DmlFusedBatchNormGradKernel, \
                                           BatchNormGradShapeHelper>);
TF_CALL_DML_FLOAT_TYPES(DML_REGISTER_KERNEL);
#undef DML_REGISTER_KERNEL

#define DML_REGISTER_KERNEL(type)                                       \
  REGISTER_KERNEL_BUILDER(Name("FusedBatchNormGradV3")                  \
                              .Device(DEVICE_DML)                       \
                              .TypeConstraint<type>("T")                \
                              .TypeConstraint<float>("U"),              \
                          DmlKernelWrapper<DmlFusedBatchNormGradKernel, \
                                           BatchNormGradShapeHelper>);
TF_CALL_DML_FLOAT_TYPES(DML_REGISTER_KERNEL);
#undef DML_REGISTER_KERNEL

class DmlBatchGlobalNormGradKernel : public DmlKernel {
  enum InputIndex {
    kT,
    kM,
    kV,
    kGamma,
    kBackProp,
  };

  enum OutputIndex {
    kDX,
    kDM,
    kDV,
    kDB,
    kDG,
  };

 public:
  using InitHelper = BatchGlobalNormGradInitializationHelper;
  explicit DmlBatchGlobalNormGradKernel(DmlKernelConstruction* ctx,
                                        const InitHelper* init_helper) {
    CHECK(ctx->GetInputCount() == 5);
    CHECK(ctx->GetOutputCount() == 5);

    float epsilon = init_helper->GetVarianceEpsilon();
    bool scale_after_normalization = init_helper->ScaleAfterNormalization();

    DmlKernelParams params;

    if (scale_after_normalization) {
      params.kernel_input_indices = {kT, kM, kV, kGamma, kBackProp};
    } else {
      params.kernel_input_indices = {kT, kM, kV, kBackProp};
    }
    params.kernel_output_indices = {0, 1, 2, 3, 4};

    DmlKernelTensors tensors = GetTensorInfos(ctx, params);

    auto input_descs = GetDmlTensorDescs(tensors.inputs);
    auto output_descs = GetDmlTensorDescs(tensors.outputs);

    auto scope = dml::Graph(ctx->GetDmlDevice());

    const uint32_t back_prop_index =
        scale_after_normalization ? kBackProp : kBackProp - 1;
    auto input = dml::InputTensor(scope, kT, input_descs[kT]);
    auto mean = dml::InputTensor(scope, kM, input_descs[kM]);
    auto variance = dml::InputTensor(scope, kV, input_descs[kV]);
    auto back_prop =
        dml::InputTensor(scope, back_prop_index, input_descs[back_prop_index]);

    auto input_sizes = back_prop.GetOutputDesc().sizes;
    // The strides we need to set to broadcast C across an entire tensor
    dml::TensorDesc::Dimensions broadcast_c_strides = {/*N*/ 0,
                                                       /*H*/ 0,
                                                       /*W*/ 0,
                                                       /*C*/ 1};

    // Formulae copied from tensorflow\core\kernels\batch_norm_op.h:
    // db = out_backprop
    //
    // dg = out_backprop * ((x - m) * rsqrt(v + epsilon))
    //
    // dv = sum_over_rest(out_backprop * gamma * (x - m)) *
    //      (-1/2) * (v + epsilon) ^ (-3/2)
    //
    // dm = sum_over_rest(out_backprop * gamma) * (-1 / rsqrt(v + epsilon))
    //
    // dx = out_backprop * (gamma * rsqrt(v + epsilon))

    auto variance_e = variance + epsilon;
    auto sqrt_variance_e = dml::Sqrt(variance_e);
    auto mean_bcast = dml::Reinterpret(mean, input_sizes, broadcast_c_strides);

    // scratch1 = rsqrt(v + epsilon)
    auto scratch1 = 1.0f / sqrt_variance_e;
    auto scratch1_bcast =
        dml::Reinterpret(scratch1, input_sizes, broadcast_c_strides);

    // scratch2 = sum_over_rest(out_backprop * (x - m))
    auto scratch2_s = back_prop * (input - mean_bcast);
    auto scratch2 = dml::Reduce(scratch2_s, DML_REDUCE_FUNCTION_SUM, {0, 1, 2});

    // scratch3 = - 1/2 * (var + epsilon) ^ (-3/2)
    auto scratch3 = -1.0f / 2 * scratch1 / variance_e;

    auto db = dml::Reduce(back_prop, DML_REDUCE_FUNCTION_SUM, {0, 1, 2});
    dml::Expression dx, dv, dg, dm;

    if (scale_after_normalization) {
      auto gamma = dml::InputTensor(scope, kGamma, input_descs[kGamma]);
      auto scratch1_gamma_bcast =
          dml::Reinterpret(gamma * scratch1, input_sizes, broadcast_c_strides);
      dx = back_prop * scratch1_gamma_bcast;
      dm = -db * scratch1 * gamma;
      dv = scratch2 * scratch3 * gamma;
      dg = scratch2 * scratch1;
    } else {
      dx = back_prop * scratch1_bcast;
      dm = -db * scratch1;
      dv = scratch2 * scratch3;
      dg = dml::ScalarTensor(
          scope, 0.0f,
          variance.GetOutputDesc().sizes);  // Gamma is not learned.
    }

    auto outputs = {
        dx, dm, dv, db, dg,
    };

    auto compiled_op = scope.Compile(DML_EXECUTION_FLAG_NONE, outputs);
    Initialize(ctx, std::move(tensors), compiled_op.Get());
  }
};

#define DML_REGISTER_KERNEL(type)                  \
  REGISTER_KERNEL_BUILDER(                         \
      Name("BatchNormWithGlobalNormalizationGrad") \
          .Device(DEVICE_DML)                      \
          .TypeConstraint<type>("T"),              \
      DmlKernelWrapper<DmlBatchGlobalNormGradKernel, BatchNormShapeHelper>);
TF_CALL_DML_FLOAT_TYPES(DML_REGISTER_KERNEL);
#undef DML_REGISTER_KERNEL

}  // namespace tensorflow
