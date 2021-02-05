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
#include "tensorflow/core/kernels/dml_kernel_wrapper.h"
#include "tensorflow/core/kernels/dml_ops_common.h"

namespace tensorflow {

struct LRNAttributes {
  explicit LRNAttributes(OpKernelConstruction* ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("depth_radius", &depth_radius));
    OP_REQUIRES(ctx,
                FastBoundsCheck(depth_radius, std::numeric_limits<int>::max()),
                errors::InvalidArgument("depth_radius = ", depth_radius,
                                        " larger than int max"));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("bias", &bias));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("alpha", &alpha));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("beta", &beta));
  }

  int64 depth_radius;
  float bias;
  float alpha;
  float beta;
};

class LRNInitHelper : public InitializationHelper {
 public:
  using Attributes = LRNAttributes;

  LRNInitHelper(OpKernelContext* ctx, std::shared_ptr<const Attributes> attr)
      : attr_(attr) {
    const Tensor& in = ctx->input(0);
    OP_REQUIRES(ctx, in.dims() == 4,
                errors::InvalidArgument("in must be 4-dimensional"));
    OP_REQUIRES(
        ctx, FastBoundsCheck(in.NumElements(), std::numeric_limits<int>::max()),
        errors::InvalidArgument("argument to LRN too large"));

    const int depth = static_cast<int>(in.dim_size(3));
    OP_REQUIRES(
        ctx, (depth + attr_->depth_radius) <= std::numeric_limits<int>::max(),
        errors::InvalidArgument("depth ", depth, " + depth_radius ",
                                attr_->depth_radius, " exceeds int max."));
  }

  int64 GetDepthRadius() const { return attr_->depth_radius; }
  float GetBias() const { return attr_->bias; }
  float GetAlpha() const { return attr_->alpha; }
  float GetBeta() const { return attr_->beta; }

 private:
  const std::shared_ptr<const Attributes> attr_;
};

class DmlLRNKernel : public DmlKernel {
 public:
  using InitHelper = LRNInitHelper;

  explicit DmlLRNKernel(DmlKernelConstruction* ctx,
                        const InitHelper* init_helper) {
    DCHECK(ctx->GetInputCount() == 1);
    DCHECK(ctx->GetOutputCount() == 1);

    const TensorShape& tensor_shape = ctx->GetInputTensorShape(0);

    const auto tensor_layout =
        GetDmlTensorLayout(FORMAT_NHWC, tensor_shape.dims());

    DmlTensorInfo input;
    input.kernel_index = 0;
    input.desc = DmlTensorDesc::Create(ctx->GetInputDataType(0), tensor_shape,
                                       tensor_shape, tensor_layout);

    DmlTensorInfo output;
    output.kernel_index = 0;
    output.desc = DmlTensorDesc::Create(ctx->GetOutputDataType(0), tensor_shape,
                                        tensor_shape, tensor_layout);

    DmlKernelTensors tensors;
    tensors.inputs = {input};
    tensors.outputs = {output};

    auto inputs = GetDmlTensorDescs(tensors.inputs);
    auto outputs = GetDmlTensorDescs(tensors.outputs);

    uint32_t local_size = init_helper->GetDepthRadius() * 2 + 1;

    // DML divides Alpha by LocalSize, but tensorflow's Alpha already contains
    // that division
    float dml_alpha = init_helper->GetAlpha() * local_size;

    DML_LOCAL_RESPONSE_NORMALIZATION_OPERATOR_DESC lrn_desc = {};
    lrn_desc.InputTensor = &inputs[0];
    lrn_desc.OutputTensor = &outputs[0];
    lrn_desc.CrossChannel = true;
    lrn_desc.LocalSize = local_size;
    lrn_desc.Alpha = dml_alpha;
    lrn_desc.Beta = init_helper->GetBeta();
    lrn_desc.Bias = init_helper->GetBias();

    DML_OPERATOR_DESC op_desc = {DML_OPERATOR_LOCAL_RESPONSE_NORMALIZATION,
                                 &lrn_desc};
    Initialize(ctx, std::move(tensors), op_desc);
  }
};

#define REGISTER_KERNEL(type)                                   \
  REGISTER_KERNEL_BUILDER(                                      \
      Name("LRN").Device(DEVICE_DML).TypeConstraint<type>("T"), \
      DmlKernelWrapper<DmlLRNKernel, GetOutputShapeAsInputShapeHelper>)
TF_CALL_float(REGISTER_KERNEL);
TF_CALL_half(REGISTER_KERNEL);
#undef REGISTER_KERNEL

class LRNGradInitHelper : public InitializationHelper {
 public:
  using Attributes = LRNAttributes;

  LRNGradInitHelper(OpKernelContext* context,
                std::shared_ptr<const Attributes> attr)
      : attr_(attr) {
    const Tensor& in_grads = context->input(0);
    const Tensor& in_image = context->input(1);
    const Tensor& out_image = context->input(2);

    OP_REQUIRES(context, in_grads.dims() == 4 && in_image.dims() == 4,
                errors::InvalidArgument("inputs must be 4-dimensional"));
    const int64 batch = in_grads.dim_size(0);
    const int64 rows = in_grads.dim_size(1);
    const int64 cols = in_grads.dim_size(2);
    const int64 depth = in_grads.dim_size(3);
    OP_REQUIRES(
        context,
        in_image.dim_size(0) == batch && in_image.dim_size(1) == rows &&
            in_image.dim_size(2) == cols && in_image.dim_size(3) == depth &&
            out_image.dim_size(0) == batch && out_image.dim_size(1) == rows &&
            out_image.dim_size(2) == cols && out_image.dim_size(3) == depth,
        errors::InvalidArgument(
            "input_grads, input_image, and out_image should have the same "
            "shape"));
  }

  int64 GetDepthRadius() const { return attr_->depth_radius; }
  float GetBias() const { return attr_->bias; }
  float GetAlpha() const { return attr_->alpha; }
  float GetBeta() const { return attr_->beta; }

 private:
  const std::shared_ptr<const Attributes> attr_;
};

class DmlLRNGradKernel : public DmlKernel {
 public:
  using InitHelper = LRNGradInitHelper;

  explicit DmlLRNGradKernel(DmlKernelConstruction* ctx,
                            const InitHelper* init_helper) {
    DCHECK(ctx->GetInputCount() == 3);
    DCHECK(ctx->GetOutputCount() == 1);

    const TensorShape& tensor_shape = ctx->GetInputTensorShape(0);

    const auto tensor_layout =
        GetDmlTensorLayout(FORMAT_NHWC, tensor_shape.dims());

    DmlTensorInfo input;
    input.kernel_index = 1;
    input.desc = DmlTensorDesc::Create(ctx->GetInputDataType(0), tensor_shape,
                                       tensor_shape, tensor_layout);

    DmlTensorInfo inputGradient;
    inputGradient.kernel_index = 0;
    inputGradient.desc = DmlTensorDesc::Create(
        ctx->GetInputDataType(0), tensor_shape, tensor_shape, tensor_layout);

    DmlTensorInfo outputGradient;
    outputGradient.kernel_index = 0;
    outputGradient.desc = DmlTensorDesc::Create(
        ctx->GetOutputDataType(0), tensor_shape, tensor_shape, tensor_layout);

    DmlKernelTensors tensors;
    tensors.inputs = {input, inputGradient};
    tensors.outputs = {outputGradient};

    auto inputs = GetDmlTensorDescs(tensors.inputs);
    auto outputs = GetDmlTensorDescs(tensors.outputs);

    uint32_t local_size = init_helper->GetDepthRadius() * 2 + 1;

    // DML divides Alpha by LocalSize, but tensorflow's Alpha already contains
    // that division
    float dml_alpha = init_helper->GetAlpha() * local_size;

    DML_PREVIEW_LOCAL_RESPONSE_NORMALIZATION_GRAD_OPERATOR_DESC lrn_desc = {};
    lrn_desc.InputTensor = &inputs[0];
    lrn_desc.InputGradientTensor = &inputs[1];
    lrn_desc.OutputGradientTensor = &outputs[0];
    lrn_desc.CrossChannel = true;
    lrn_desc.LocalSize = local_size;
    lrn_desc.Alpha = dml_alpha;
    lrn_desc.Beta = init_helper->GetBeta();
    lrn_desc.Bias = init_helper->GetBias();

    DML_OPERATOR_DESC op_desc = {
        static_cast<DML_OPERATOR_TYPE>(DML_PREVIEW_OPERATOR_LOCAL_RESPONSE_NORMALIZATION_GRAD), &lrn_desc};
    Initialize(ctx, std::move(tensors), op_desc);
  }
};

#define REGISTER_KERNEL(type)                                       \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("LRNGrad").Device(DEVICE_DML).TypeConstraint<type>("T"), \
      DmlKernelWrapper<DmlLRNGradKernel, GetOutputShapeAsInputShapeHelper>)
TF_CALL_float(REGISTER_KERNEL);
TF_CALL_half(REGISTER_KERNEL);
#undef REGISTER_KERNEL

}  // namespace tensorflow