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
#include "tensorflow/core/util/bcast.h"

namespace tensorflow {

class LuGradInitHelper : public InitializationHelper {
 public:
  using Attributes = EmptyAttributes;

  LuGradInitHelper(OpKernelContext* ctx,
                   std::shared_ptr<const Attributes> attr) {
    BCast bcast_helper(ctx->input(1).shape().dim_sizes(),
                       ctx->input(0).shape().dim_sizes());
    feature_shape_ = TensorShape(bcast_helper.x_reshape());
    input_gradient_shape_ = TensorShape(bcast_helper.y_reshape());
    broadcasted_output_shape_ =
        BroadcastTensorShapes({feature_shape_, input_gradient_shape_});

    OP_REQUIRES(ctx, broadcasted_output_shape_.dims() <= kNcdhwDimensionCount,
                errors::InvalidArgument(
                    "DML doesn't support more than ", kNcdhwDimensionCount,
                    " dimensions for this operator, but ",
                    broadcasted_output_shape_.dims(), " were provided."));
  }

  const TensorShape& GetBroadcastedOutputShape() const {
    return broadcasted_output_shape_;
  }

  const TensorShape& GetFeatureShape() const { return feature_shape_; }

  const TensorShape& GetInputGradientShape() const {
    return input_gradient_shape_;
  }

 private:
  TensorShape feature_shape_;
  TensorShape input_gradient_shape_;
  TensorShape broadcasted_output_shape_;
};

class DmlReluKernel : public DmlKernel {
 public:
  using InitHelper = NoOpInitializationHelper;

  explicit DmlReluKernel(DmlKernelConstruction* ctx,
                         const InitHelper* init_helper) {
    CHECK(ctx->GetInputCount() == 1);
    CHECK(ctx->GetOutputCount() == 1);

    auto num_elements =
        static_cast<uint32_t>(ctx->GetInputTensorShape(0).num_elements());
    uint32_t tensor_sizes[4] = {1, 1, 1, num_elements};

    auto data_type = GetDmlDataTypeFromTfDataType(ctx->GetInputDataType(0));
    DmlTensorInfo tensor_info = {};
    tensor_info.kernel_index = 0;
    tensor_info.desc = DmlTensorDesc{data_type, tensor_sizes};

    DmlKernelTensors tensors = {};
    tensors.inputs = {tensor_info};
    tensors.outputs = {tensor_info};

    auto input_descs = GetDmlTensorDescs(tensors.inputs);
    auto output_descs = GetDmlTensorDescs(tensors.outputs);

    DML_ACTIVATION_RELU_OPERATOR_DESC relu_desc = {};
    relu_desc.InputTensor = &input_descs[0];
    relu_desc.OutputTensor = output_descs.data();

    DML_OPERATOR_DESC op_desc = {DML_OPERATOR_ACTIVATION_RELU, &relu_desc};
    Initialize(ctx, std::move(tensors), op_desc);
  }
};

#define DML_REGISTER_KERNEL(type)                                \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("Relu").Device(DEVICE_DML).TypeConstraint<type>("T"), \
      DmlKernelWrapper<DmlReluKernel, GetOutputShapeAsInputShapeHelper>);
TF_CALL_DML_FLOAT_TYPES(DML_REGISTER_KERNEL);
#undef DML_REGISTER_KERNEL

// Base CRTP class for linear unit (LU) grad ops: ReluGrad, SeluGrad, etc.
template <typename Impl>
class DmlLUGradKernel : public DmlKernel {
 public:
  using InitHelper = LuGradInitHelper;

  explicit DmlLUGradKernel(DmlKernelConstruction* ctx,
                           const InitHelper* init_helper) {
    CHECK(ctx->GetInputCount() == 2);
    CHECK(ctx->GetOutputCount() == 1);

    const TensorShape& feature_shape = init_helper->GetFeatureShape();
    const TensorShape& input_gradient_shape =
        init_helper->GetInputGradientShape();
    const TensorShape& output_shape = init_helper->GetBroadcastedOutputShape();

    DmlTensorInfo feature_tensor;
    feature_tensor.kernel_index = 1;
    feature_tensor.desc = DmlTensorDesc::Create(ctx->GetInputDataType(1),
                                                feature_shape, feature_shape);

    DmlTensorInfo input_gradient_tensor;
    input_gradient_tensor.kernel_index = 0;
    input_gradient_tensor.desc = DmlTensorDesc::Create(
        ctx->GetInputDataType(0), input_gradient_shape, input_gradient_shape);

    DmlTensorInfo output_tensor;
    output_tensor.kernel_index = 0;
    output_tensor.desc = DmlTensorDesc::Create(ctx->GetOutputDataType(0),
                                               output_shape, output_shape);

    DmlKernelTensors tensors = {};
    tensors.inputs = {feature_tensor, input_gradient_tensor};
    tensors.outputs = {output_tensor};

    auto input_descs = GetDmlTensorDescs(tensors.inputs);
    auto output_descs = GetDmlTensorDescs(tensors.outputs);

    static_cast<Impl*>(this)->Init(ctx, std::move(tensors), input_descs[0],
                                   input_descs[1], output_descs[0]);
  }
};

class DmlReluGradKernel : public DmlLUGradKernel<DmlReluGradKernel> {
 public:
  using InitHelper = LuGradInitHelper;

  explicit DmlReluGradKernel(DmlKernelConstruction* ctx,
                             const InitHelper* init_helper)
      : DmlLUGradKernel<DmlReluGradKernel>(ctx, init_helper) {}

  void Init(DmlKernelConstruction* ctx, DmlKernelTensors&& tensors,
            const DML_TENSOR_DESC& gradient_desc,
            const DML_TENSOR_DESC& feature_desc,
            const DML_TENSOR_DESC& output_desc) {
    DML_ACTIVATION_RELU_GRAD_OPERATOR_DESC relu_grad_desc = {};
    relu_grad_desc.InputTensor = &feature_desc;
    relu_grad_desc.InputGradientTensor = &gradient_desc;
    relu_grad_desc.OutputGradientTensor = &output_desc;

    DML_OPERATOR_DESC op_desc = {DML_OPERATOR_ACTIVATION_RELU_GRAD,
                                 &relu_grad_desc};

    Initialize(ctx, std::move(tensors), op_desc);
  }
};

#define DML_REGISTER_KERNEL(type)                                    \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("ReluGrad").Device(DEVICE_DML).TypeConstraint<type>("T"), \
      DmlKernelWrapper<DmlReluGradKernel, GetOutputShapeAsInputShapeHelper>);
TF_CALL_DML_FLOAT_TYPES(DML_REGISTER_KERNEL);
#undef DML_REGISTER_KERNEL

template <typename T>
class DmlRelu6GradKernel : public DmlLUGradKernel<DmlRelu6GradKernel<T>> {
 public:
  using InitHelper = LuGradInitHelper;

  explicit DmlRelu6GradKernel(DmlKernelConstruction* ctx,
                              const InitHelper* init_helper)
      : DmlLUGradKernel<DmlRelu6GradKernel>(ctx, init_helper) {}

  void Init(DmlKernelConstruction* ctx, DmlKernelTensors&& tensors,
            const DML_TENSOR_DESC& gradient_desc,
            const DML_TENSOR_DESC& feature_desc,
            const DML_TENSOR_DESC& output_desc) {
    auto scope = dml::Graph(ctx->GetDmlDevice());
    auto feature = dml::InputTensor(scope, 0, feature_desc);
    auto gradient = dml::InputTensor(scope, 1, gradient_desc);

    DML_TENSOR_DATA_TYPE feature_dtype = feature.GetOutputDesc().dataType;
    const auto& feature_sizes = feature.GetOutputDesc().sizes;

    auto zero = dml::ZeroTensor(scope, feature_dtype, feature_sizes);

    auto six_val = TfTensorTypeTraits<T>::FromFloat(6.0f);
    auto six = dml::ScalarTensor<T>(scope, six_val, feature_sizes);

    auto in_relu6 = feature > zero && feature < six;
    auto result = gradient * dml::Cast(in_relu6, feature_dtype);

    Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
        scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

    DmlKernel::Initialize(ctx, std::move(tensors), compiled_op.Get());
  }
};

#define DML_REGISTER_KERNEL(type)                                     \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("Relu6Grad").Device(DEVICE_DML).TypeConstraint<type>("T"), \
      DmlKernelWrapper<DmlRelu6GradKernel<type>,                      \
                       GetOutputShapeAsInputShapeHelper>);
TF_CALL_DML_FLOAT_TYPES(DML_REGISTER_KERNEL);
#undef DML_REGISTER_KERNEL

class DmlLeakyReluGradKernel : public DmlLUGradKernel<DmlLeakyReluGradKernel> {
 public:
  using InitHelper = LuGradInitHelper;

  explicit DmlLeakyReluGradKernel(DmlKernelConstruction* ctx,
                                  const InitHelper* init_helper)
      : DmlLUGradKernel<DmlLeakyReluGradKernel>(ctx, init_helper) {}

  void Init(DmlKernelConstruction* ctx, DmlKernelTensors&& tensors,
            const DML_TENSOR_DESC& gradient_desc,
            const DML_TENSOR_DESC& feature_desc,
            const DML_TENSOR_DESC& output_desc) {
    auto scope = dml::Graph(ctx->GetDmlDevice());
    auto feature = dml::InputTensor(scope, 0, feature_desc);
    auto gradient = dml::InputTensor(scope, 1, gradient_desc);

    DML_TENSOR_DATA_TYPE feature_dtype = feature.GetOutputDesc().dataType;
    const auto& feature_sizes = feature.GetOutputDesc().sizes;

    auto zero = dml::ZeroTensor(scope, feature_dtype, feature_sizes);

    float alpha = 0.2f;
    TF_CHECK_OK(ctx->GetAttr("alpha", &alpha));

    auto result = dml::If(feature > zero, gradient, gradient * alpha);

    Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
        scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

    Initialize(ctx, std::move(tensors), compiled_op.Get());
  }
};

#define DML_REGISTER_KERNEL(type)                                         \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("LeakyReluGrad").Device(DEVICE_DML).TypeConstraint<type>("T"), \
      DmlKernelWrapper<DmlLeakyReluGradKernel,                            \
                       GetOutputShapeAsInputShapeHelper>);
TF_CALL_DML_FLOAT_TYPES(DML_REGISTER_KERNEL);
#undef DML_REGISTER_KERNEL

class DmlEluGradKernel : public DmlLUGradKernel<DmlEluGradKernel> {
 public:
  using InitHelper = LuGradInitHelper;

  explicit DmlEluGradKernel(DmlKernelConstruction* ctx,
                            const InitHelper* init_helper)
      : DmlLUGradKernel<DmlEluGradKernel>(ctx, init_helper) {}

  void Init(DmlKernelConstruction* ctx, DmlKernelTensors&& tensors,
            const DML_TENSOR_DESC& gradient_desc,
            const DML_TENSOR_DESC& feature_desc,
            const DML_TENSOR_DESC& output_desc) {
    auto scope = dml::Graph(ctx->GetDmlDevice());
    auto feature = dml::InputTensor(scope, 0, feature_desc);
    auto gradient = dml::InputTensor(scope, 1, gradient_desc);

    DML_TENSOR_DATA_TYPE feature_dtype = feature.GetOutputDesc().dataType;
    const auto& feature_sizes = feature.GetOutputDesc().sizes;

    auto zero = dml::ZeroTensor(scope, feature_dtype, feature_sizes);

    auto result =
        dml::If(feature < zero, (feature + 1.0f) * gradient, gradient);

    Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
        scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

    Initialize(ctx, std::move(tensors), compiled_op.Get());
  }
};

#define DML_REGISTER_KERNEL(type)                                   \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("EluGrad").Device(DEVICE_DML).TypeConstraint<type>("T"), \
      DmlKernelWrapper<DmlEluGradKernel, GetOutputShapeAsInputShapeHelper>);
TF_CALL_DML_FLOAT_TYPES(DML_REGISTER_KERNEL);
#undef DML_REGISTER_KERNEL

class DmlSeluGradKernel : public DmlLUGradKernel<DmlSeluGradKernel> {
 public:
  using InitHelper = LuGradInitHelper;

  explicit DmlSeluGradKernel(DmlKernelConstruction* ctx,
                             const InitHelper* init_helper)
      : DmlLUGradKernel<DmlSeluGradKernel>(ctx, init_helper) {}

  void Init(DmlKernelConstruction* ctx, DmlKernelTensors&& tensors,
            const DML_TENSOR_DESC& gradient_desc,
            const DML_TENSOR_DESC& feature_desc,
            const DML_TENSOR_DESC& output_desc) {
    auto scope = dml::Graph(ctx->GetDmlDevice());
    auto feature = dml::InputTensor(scope, 0, feature_desc);
    auto gradient = dml::InputTensor(scope, 1, gradient_desc);

    DML_TENSOR_DATA_TYPE feature_dtype = feature.GetOutputDesc().dataType;
    const auto& feature_sizes = feature.GetOutputDesc().sizes;

    auto zero = dml::ZeroTensor(scope, feature_dtype, feature_sizes);

    constexpr float scale = 1.0507009873554804934193349852946f;
    constexpr float scale_alpha = 1.7580993408473768599402175208123f;

    auto result = dml::If(feature < zero, gradient * (feature + scale_alpha),
                          gradient * scale);

    Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
        scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

    Initialize(ctx, std::move(tensors), compiled_op.Get());
  }
};

#define DML_REGISTER_KERNEL(type)                                    \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("SeluGrad").Device(DEVICE_DML).TypeConstraint<type>("T"), \
      DmlKernelWrapper<DmlSeluGradKernel, GetOutputShapeAsInputShapeHelper>);
TF_CALL_DML_FLOAT_TYPES(DML_REGISTER_KERNEL);
#undef DML_REGISTER_KERNEL

}  // namespace tensorflow