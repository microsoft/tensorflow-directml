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

using Microsoft::WRL::ComPtr;

template <DML_OPERATOR_TYPE op_type>
static constexpr uint32_t GetMaxDimCount() {
  switch (op_type) {
    case DML_OPERATOR_ELEMENT_WISE_IDENTITY:
    case DML_OPERATOR_ELEMENT_WISE_ADD:
    case DML_OPERATOR_ELEMENT_WISE_MULTIPLY:
      return 5;
    default:
      return 4;
  }
}

static absl::InlinedVector<TensorShape, 2> GetCollapsedShapes(
    OpKernelContext* ctx) {
  if (ctx->num_inputs() == 1) {
    return {TensorShape({ctx->input(0).NumElements()})};
  }

  absl::InlinedVector<TensorShape, 2> shapes;

  // Shape collapsing for more than 2 inputs is not implemented
  if (ctx->num_inputs() > 2) {
    for (uint32_t i = 0; i < ctx->num_inputs(); ++i) {
      shapes.push_back(ctx->input(i).shape());
    }

    return shapes;
  }

  BCast bcast_helper(ctx->input(0).shape().dim_sizes(),
                     ctx->input(1).shape().dim_sizes());

  shapes.emplace_back(bcast_helper.x_reshape());
  shapes.emplace_back(bcast_helper.y_reshape());

  return shapes;
}

template <uint32_t max_dim_count>
class ElementWiseInitHelper
    : public GetBroadcastedOutputShapeHelper::InitHelper {
 public:
  struct Attributes
      : public GetBroadcastedOutputShapeHelper::InitHelper::Attributes {
    explicit Attributes(OpKernelConstruction* ctx)
        : GetBroadcastedOutputShapeHelper::InitHelper::Attributes(ctx) {}
  };

  ElementWiseInitHelper(OpKernelContext* ctx,
                        std::shared_ptr<const Attributes> attr)
      : GetBroadcastedOutputShapeHelper::InitHelper(ctx, attr) {
    collapsed_input_shapes_ = GetCollapsedShapes(ctx);
    collapsed_output_shape_ = BroadcastTensorShapes(collapsed_input_shapes_);

    OP_REQUIRES(ctx, collapsed_output_shape_.dims() <= max_dim_count,
                errors::InvalidArgument(
                    "DML doesn't support more than ", max_dim_count,
                    " dimensions for this operator, but ",
                    collapsed_output_shape_.dims(), " were provided."));
  }

  absl::Span<const TensorShape> GetCollapsedInputShapes() const {
    return collapsed_input_shapes_;
  }

  const TensorShape& GetCollapsedOutputShape() const {
    return collapsed_output_shape_;
  }

 private:
  absl::InlinedVector<TensorShape, 2> collapsed_input_shapes_;
  TensorShape collapsed_output_shape_;
};

static DmlKernelTensors CreateKernelTensors(
    DmlKernelConstruction* ctx, absl::Span<const TensorShape> input_shapes,
    const TensorShape& output_shape) {
  DmlKernelTensors tensors;

  for (uint32_t i = 0; i < ctx->GetInputCount(); ++i) {
    DmlTensorInfo input;
    input.kernel_index = i;
    input.desc = DmlTensorDesc::Create(ctx->GetInputDataType(i), output_shape,
                                       input_shapes[i]);

    tensors.inputs.push_back(std::move(input));
  }

  DmlTensorInfo output;
  output.kernel_index = 0;
  output.desc = DmlTensorDesc::Create(ctx->GetOutputDataType(0), output_shape,
                                      output_shape);

  tensors.outputs = {output};

  return tensors;
}

template <DML_OPERATOR_TYPE op_type, typename DML_OPERATOR_SPECIFIC_DESC>
class DmlBinaryKernel : public DmlKernel {
 public:
  using InitHelper = ElementWiseInitHelper<GetMaxDimCount<op_type>()>;

  explicit DmlBinaryKernel(DmlKernelConstruction* ctx,
                           const InitHelper* init_helper) {
    CHECK(ctx->GetInputCount() == 2);
    CHECK(ctx->GetOutputCount() == 1);

    auto input_shapes = init_helper->GetCollapsedInputShapes();
    const TensorShape& output_shape = init_helper->GetCollapsedOutputShape();

    DmlKernelTensors tensors =
        CreateKernelTensors(ctx, input_shapes, output_shape);
    auto inputs = GetDmlTensorDescs(tensors.inputs);
    auto outputs = GetDmlTensorDescs(tensors.outputs);

    DML_OPERATOR_SPECIFIC_DESC op_specific_desc = {
        &inputs[0],
        &inputs[1],
        outputs.data(),
    };

    DML_OPERATOR_DESC op_desc = {op_type, &op_specific_desc};
    Initialize(ctx, std::move(tensors), op_desc);
  }

  StatusOr<DmlGpuEvent> Compute(DmlKernelContext* ctx) const override {
    // Currently, 64-bit integers in DML are emulated using 32-bit integers
    // using striding to emulate a larger type. Because we can't guarantee that
    // our output tensor's memory is zero'd, we need to do so manually prior to
    // running running gather.
    Tensor* output = ctx->GetOutputTensor(0);

    if (Is64BitIntegerType(output->dtype())) {
      ctx->ZeroBuffer(ctx->CreateBufferForTensor(*output));
    }

    return DmlKernel::Compute(ctx);
  }
};

template <typename ExpressionFunctor, uint32_t max_dim_count>
class DmlCompositeBinaryKernel : public DmlKernel {
 public:
  using InitHelper = ElementWiseInitHelper<max_dim_count>;

  explicit DmlCompositeBinaryKernel(DmlKernelConstruction* ctx,
                                    const InitHelper* init_helper) {
    CHECK(ctx->GetInputCount() == 2);
    CHECK(ctx->GetOutputCount() == 1);

    auto input_shapes = init_helper->GetCollapsedInputShapes();
    const TensorShape& output_shape = init_helper->GetCollapsedOutputShape();

    DmlKernelTensors tensors =
        CreateKernelTensors(ctx, input_shapes, output_shape);
    auto inputs = GetDmlTensorDescs(tensors.inputs);
    auto outputs = GetDmlTensorDescs(tensors.outputs);

    auto scope = dml::Graph(ctx->GetDmlDevice());
    auto x = dml::InputTensor(scope, 0, inputs[0]);
    auto y = dml::InputTensor(scope, 1, inputs[1]);

    ExpressionFunctor expression;
    auto result = expression(x, y);

    ComPtr<IDMLCompiledOperator> compiled_op =
        scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

    Initialize(ctx, std::move(tensors), compiled_op.Get());
  }

  StatusOr<DmlGpuEvent> Compute(DmlKernelContext* ctx) const override {
    // Currently, 64-bit integers in DML are emulated using 32-bit integers
    // using striding to emulate a larger type. Because we can't guarantee that
    // our output tensor's memory is zero'd, we need to do so manually prior to
    // running running gather.
    Tensor* output = ctx->GetOutputTensor(0);

    if (Is64BitIntegerType(output->dtype())) {
      ctx->ZeroBuffer(ctx->CreateBufferForTensor(*output));
    }

    return DmlKernel::Compute(ctx);
  }
};

template <DML_OPERATOR_TYPE op_type, typename DML_OPERATOR_SPECIFIC_DESC,
          int... constants>
class DmlUnaryKernel : public DmlKernel {
 public:
  using InitHelper = ElementWiseInitHelper<UINT32_MAX>;

  explicit DmlUnaryKernel(DmlKernelConstruction* ctx,
                          const InitHelper* init_helper) {
    CHECK(ctx->GetInputCount() == 1);
    CHECK(ctx->GetOutputCount() == 1);

    TensorShape tensor_shape({ctx->GetOutputTensorShape(0).num_elements()});
    DmlKernelTensors tensors =
        CreateKernelTensors(ctx, {tensor_shape}, tensor_shape);
    auto inputs = GetDmlTensorDescs(tensors.inputs);
    auto outputs = GetDmlTensorDescs(tensors.outputs);

    DML_OPERATOR_SPECIFIC_DESC op_specific_desc = {&inputs[0], outputs.data(),
                                                   constants...};

    DML_OPERATOR_DESC op_desc = {op_type, &op_specific_desc};
    Initialize(ctx, std::move(tensors), op_desc);
  }

  StatusOr<DmlGpuEvent> Compute(DmlKernelContext* ctx) const override {
    // Currently, 64-bit integers in DML are emulated using 32-bit integers
    // using striding to emulate a larger type. Because we can't guarantee that
    // our output tensor's memory is zero'd, we need to do so manually prior to
    // running running gather.
    Tensor* output = ctx->GetOutputTensor(0);

    if (Is64BitIntegerType(output->dtype())) {
      ctx->ZeroBuffer(ctx->CreateBufferForTensor(*output));
    }

    return DmlKernel::Compute(ctx);
  }
};

template <DML_OPERATOR_TYPE op_type, typename DML_OPERATOR_SPECIFIC_DESC>
class DmlMaxActivationKernel : public DmlKernel {
 public:
  using InitHelper = ElementWiseInitHelper<UINT32_MAX>;

  explicit DmlMaxActivationKernel(DmlKernelConstruction* ctx,
                                  const InitHelper* init_helper) {
    CHECK(ctx->GetInputCount() == 1);
    CHECK(ctx->GetOutputCount() == 1);

    const TensorShape& input_shape = ctx->GetInputTensorShape(0);

    int batch_size = 1;
    int logits_size = input_shape.dim_size(input_shape.dims() - 1);

    // DML doesn't support tensors with rank > 2 for the max activation
    // functions, so collapse all the batch dimensions together
    for (int i = 0; i < input_shape.dims() - 1; ++i) {
      batch_size *= input_shape.dim_size(i);
    }

    TensorShape dml_tensor_shape;
    dml_tensor_shape.AddDim(batch_size);
    dml_tensor_shape.AddDim(logits_size);

    DmlTensorInfo input;
    input.kernel_index = 0;
    input.desc = DmlTensorDesc::Create(ctx->GetInputDataType(0),
                                       dml_tensor_shape, dml_tensor_shape);

    DmlTensorInfo output;
    output.kernel_index = 0;
    output.desc = DmlTensorDesc::Create(ctx->GetOutputDataType(0),
                                        dml_tensor_shape, dml_tensor_shape);

    DmlKernelTensors tensors;
    tensors.inputs = {input};
    tensors.outputs = {output};

    auto input_descs = GetDmlTensorDescs(tensors.inputs);
    auto output_descs = GetDmlTensorDescs(tensors.outputs);

    DML_OPERATOR_SPECIFIC_DESC op_specific_desc = {};
    op_specific_desc.InputTensor = input_descs.data();
    op_specific_desc.OutputTensor = output_descs.data();

    DML_OPERATOR_DESC op_desc = {op_type, &op_specific_desc};
    Initialize(ctx, std::move(tensors), op_desc);
  }
};

template <typename ExpressionFunctor, uint32_t max_dim_count>
class DmlCompositeUnaryKernel : public DmlKernel {
 public:
  using InitHelper = ElementWiseInitHelper<UINT32_MAX>;

  explicit DmlCompositeUnaryKernel(DmlKernelConstruction* ctx,
                                   const InitHelper* init_helper) {
    CHECK(ctx->GetInputCount() == 1);
    CHECK(ctx->GetOutputCount() == 1);

    TensorShape tensor_shape({ctx->GetOutputTensorShape(0).num_elements()});
    DmlKernelTensors tensors =
        CreateKernelTensors(ctx, {tensor_shape}, tensor_shape);

    auto inputs = GetDmlTensorDescs(tensors.inputs);
    auto outputs = GetDmlTensorDescs(tensors.outputs);

    auto scope = dml::Graph(ctx->GetDmlDevice());
    auto x = dml::InputTensor(scope, 0, inputs[0]);

    ExpressionFunctor expression;
    auto result = expression(x);

    ComPtr<IDMLCompiledOperator> compiled_op =
        scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

    Initialize(ctx, std::move(tensors), compiled_op.Get());
  }

  StatusOr<DmlGpuEvent> Compute(DmlKernelContext* ctx) const override {
    // Currently, 64-bit integers in DML are emulated using 32-bit integers
    // using striding to emulate a larger type. Because we can't guarantee that
    // our output tensor's memory is zero'd, we need to do so manually prior to
    // running running gather.
    Tensor* output = ctx->GetOutputTensor(0);

    if (Is64BitIntegerType(output->dtype())) {
      ctx->ZeroBuffer(ctx->CreateBufferForTensor(*output));
    }

    return DmlKernel::Compute(ctx);
  }
};

template <DML_OPERATOR_TYPE op_type, typename DML_OPERATOR_SPECIFIC_DESC,
          int scale = 1, int bias = 0, int... constants>
class DmlUnaryScaleBiasKernel : public DmlKernel {
 public:
  using InitHelper = ElementWiseInitHelper<UINT32_MAX>;

  explicit DmlUnaryScaleBiasKernel(DmlKernelConstruction* ctx,
                                   const InitHelper* init_helper) {
    CHECK(ctx->GetInputCount() == 1);
    CHECK(ctx->GetOutputCount() == 1);

    TensorShape tensor_shape({ctx->GetOutputTensorShape(0).num_elements()});
    DmlKernelTensors tensors =
        CreateKernelTensors(ctx, {tensor_shape}, tensor_shape);
    auto inputs = GetDmlTensorDescs(tensors.inputs);
    auto outputs = GetDmlTensorDescs(tensors.outputs);

    DML_SCALE_BIAS scale_bias = {scale, bias};
    DML_OPERATOR_SPECIFIC_DESC op_specific_desc = {
        &inputs[0],
        &outputs[0],
        &scale_bias,
        constants...,
    };

    DML_OPERATOR_DESC op_desc = {op_type, &op_specific_desc};
    Initialize(ctx, std::move(tensors), op_desc);
  }

  StatusOr<DmlGpuEvent> Compute(DmlKernelContext* ctx) const override {
    // Currently, 64-bit integers in DML are emulated using 32-bit integers
    // using striding to emulate a larger type. Because we can't guarantee that
    // our output tensor's memory is zero'd, we need to do so manually prior to
    // running running gather.
    Tensor* output = ctx->GetOutputTensor(0);

    if (Is64BitIntegerType(output->dtype())) {
      ctx->ZeroBuffer(ctx->CreateBufferForTensor(*output));
    }

    return DmlKernel::Compute(ctx);
  }
};

#define REGISTER_OP_KERNEL(opName, type)                          \
  REGISTER_KERNEL_BUILDER(                                        \
      Name(#opName).Device(DEVICE_DML).TypeConstraint<type>("T"), \
      DmlKernelWrapper<Dml##opName##Kernel, GetBroadcastedOutputShapeHelper>);

#define REGISTER_DML_FLOAT_OP_KERNEL(opName, kernelClassName, dmlOpType, \
                                     dmlOpDescType, ...)                 \
  using Dml##opName##Kernel =                                            \
      kernelClassName<dmlOpType, dmlOpDescType, ##__VA_ARGS__>;          \
  TF_CALL_DML_OP_FLOAT_TYPES(opName, REGISTER_OP_KERNEL);

#define REGISTER_DML_ALL_TYPES_OP_KERNEL(opName, kernelClassName, dmlOpType, \
                                         dmlOpDescType, ...)                 \
  using Dml##opName##Kernel =                                                \
      kernelClassName<dmlOpType, dmlOpDescType, ##__VA_ARGS__>;              \
  TF_CALL_DML_OP_ALL_TYPES(opName, REGISTER_OP_KERNEL);

#define REGISTER_DML_ALL_TYPES_EXCEPT_INT32_OP_KERNEL(          \
    opName, kernelClassName, dmlOpType, dmlOpDescType, ...)     \
  using Dml##opName##Kernel =                                   \
      kernelClassName<dmlOpType, dmlOpDescType, ##__VA_ARGS__>; \
  TF_CALL_DML_OP_ALL_TYPES_EXCEPT_INT32(opName, REGISTER_OP_KERNEL);

#define REGISTER_DML_FLOAT_INT32_INT64_TYPES_OP_KERNEL(         \
    opName, kernelClassName, dmlOpType, dmlOpDescType, ...)     \
  using Dml##opName##Kernel =                                   \
      kernelClassName<dmlOpType, dmlOpDescType, ##__VA_ARGS__>; \
  TF_CALL_DML_OP_FLOAT_INT32_INT64_TYPES(opName, REGISTER_OP_KERNEL);

#define REGISTER_DML_FLOAT_INT32_INT64_TYPES_EXCEPT_INT32_OP_KERNEL( \
    opName, kernelClassName, dmlOpType, dmlOpDescType, ...)          \
  using Dml##opName##Kernel =                                        \
      kernelClassName<dmlOpType, dmlOpDescType, ##__VA_ARGS__>;      \
  TF_CALL_DML_OP_FLOAT_INT32_INT64_TYPES_EXCEPT_INT32(opName,        \
                                                      REGISTER_OP_KERNEL);

#define REGISTER_DML_SIGNED_OP_KERNEL(opName, kernelClassName, dmlOpType, \
                                      dmlOpDescType, ...)                 \
  using Dml##opName##Kernel =                                             \
      kernelClassName<dmlOpType, dmlOpDescType, ##__VA_ARGS__>;           \
  TF_CALL_DML_OP_SIGNED_TYPES(opName, REGISTER_OP_KERNEL);

#define REGISTER_DML_SIGNED_EXCEPT_INT32_OP_KERNEL(             \
    opName, kernelClassName, dmlOpType, dmlOpDescType, ...)     \
  using Dml##opName##Kernel =                                   \
      kernelClassName<dmlOpType, dmlOpDescType, ##__VA_ARGS__>; \
  TF_CALL_DML_OP_SIGNED_TYPES_EXCEPT_INT32(opName, REGISTER_OP_KERNEL);

#define REGISTER_BOOL_OP_KERNEL(opName) \
  REGISTER_KERNEL_BUILDER(              \
      Name(#opName).Device(DEVICE_DML), \
      DmlKernelWrapper<Dml##opName##Kernel, GetBroadcastedOutputShapeHelper>);

#define REGISTER_DML_BOOL_OP_KERNEL(opName, kernelClassName, dmlOpType,  \
                                    dmlOpDescType)                       \
  using Dml##opName##Kernel = kernelClassName<dmlOpType, dmlOpDescType>; \
  REGISTER_BOOL_OP_KERNEL(opName);

#define REGISTER_DML_COMPOSITE_UNARY_FLOAT_KERNEL(opName, expression,      \
                                                  max_dim_count)           \
  struct Dml##opName##Functor {                                            \
    dml::Expression operator()(dml::Expression x) { return (expression); } \
  };                                                                       \
  using Dml##opName##Kernel =                                              \
      DmlCompositeUnaryKernel<Dml##opName##Functor, max_dim_count>;        \
  TF_CALL_DML_OP_FLOAT_TYPES(opName, REGISTER_OP_KERNEL);

#define REGISTER_DML_COMPOSITE_BINARY_FLOAT_KERNEL(opName, expression, \
                                                   max_dim_count)      \
  struct Dml##opName##Functor {                                        \
    dml::Expression operator()(dml::Expression x, dml::Expression y) { \
      return (expression);                                             \
    }                                                                  \
  };                                                                   \
  using Dml##opName##Kernel =                                          \
      DmlCompositeBinaryKernel<Dml##opName##Functor, max_dim_count>;   \
  TF_CALL_DML_OP_FLOAT_TYPES(opName, REGISTER_OP_KERNEL);

#define REGISTER_DML_COMPOSITE_BINARY_ALL_TYPES_KERNEL(opName, expression, \
                                                       max_dim_count)      \
  struct Dml##opName##Functor {                                            \
    dml::Expression operator()(dml::Expression x, dml::Expression y) {     \
      return (expression);                                                 \
    }                                                                      \
  };                                                                       \
  using Dml##opName##Kernel =                                              \
      DmlCompositeBinaryKernel<Dml##opName##Functor, max_dim_count>;       \
  TF_CALL_DML_OP_ALL_TYPES(opName, REGISTER_OP_KERNEL);

#define REGISTER_DML_COMPOSITE_BINARY_ALL_TYPES_EXCEPT_INT32_KERNEL(   \
    opName, expression, max_dim_count)                                 \
  struct Dml##opName##Functor {                                        \
    dml::Expression operator()(dml::Expression x, dml::Expression y) { \
      return (expression);                                             \
    }                                                                  \
  };                                                                   \
  using Dml##opName##Kernel =                                          \
      DmlCompositeBinaryKernel<Dml##opName##Functor, max_dim_count>;   \
  TF_CALL_DML_OP_ALL_TYPES_EXCEPT_INT32(opName, REGISTER_OP_KERNEL);

// TODO(b/25387198): A special kernel exists for int32 (see cwise_op_add1.cc).
REGISTER_DML_FLOAT_INT32_INT64_TYPES_EXCEPT_INT32_OP_KERNEL(
    Add, DmlBinaryKernel, DML_OPERATOR_ELEMENT_WISE_ADD,
    DML_ELEMENT_WISE_ADD_OPERATOR_DESC)
// TODO(b/25387198): A special kernel exists for int32 (see cwise_op_add1.cc).
REGISTER_DML_FLOAT_INT32_INT64_TYPES_EXCEPT_INT32_OP_KERNEL(
    AddV2, DmlBinaryKernel, DML_OPERATOR_ELEMENT_WISE_ADD,
    DML_ELEMENT_WISE_ADD_OPERATOR_DESC)
// TODO(b/25387198): A special kernel exists for int32 (see cwise_op_div.cc).
REGISTER_DML_FLOAT_INT32_INT64_TYPES_EXCEPT_INT32_OP_KERNEL(
    Div, DmlBinaryKernel, DML_OPERATOR_ELEMENT_WISE_DIVIDE,
    DML_ELEMENT_WISE_DIVIDE_OPERATOR_DESC)
REGISTER_DML_FLOAT_INT32_INT64_TYPES_OP_KERNEL(
    RealDiv, DmlBinaryKernel, DML_OPERATOR_ELEMENT_WISE_DIVIDE,
    DML_ELEMENT_WISE_DIVIDE_OPERATOR_DESC)
// TODO(b/25387198): A special kernel exists for int32 (see cwise_op_mul1.cc).
REGISTER_DML_FLOAT_INT32_INT64_TYPES_EXCEPT_INT32_OP_KERNEL(
    Mul, DmlBinaryKernel, DML_OPERATOR_ELEMENT_WISE_MULTIPLY,
    DML_ELEMENT_WISE_MULTIPLY_OPERATOR_DESC)
// TODO(b/25387198): A special kernel exists for int32 (see cwise_op_sub.cc).
REGISTER_DML_FLOAT_INT32_INT64_TYPES_EXCEPT_INT32_OP_KERNEL(
    Sub, DmlBinaryKernel, DML_OPERATOR_ELEMENT_WISE_SUBTRACT,
    DML_ELEMENT_WISE_SUBTRACT_OPERATOR_DESC)
REGISTER_DML_ALL_TYPES_OP_KERNEL(Relu6, DmlUnaryScaleBiasKernel,
                                 DML_OPERATOR_ELEMENT_WISE_CLIP,
                                 DML_ELEMENT_WISE_CLIP_OPERATOR_DESC, 1, 0, 0,
                                 6)
// TODO(b/25387198): A special kernel exists for int32 (see
// cwise_op_minimum.cc).
REGISTER_DML_ALL_TYPES_EXCEPT_INT32_OP_KERNEL(
    Minimum, DmlBinaryKernel, DML_OPERATOR_ELEMENT_WISE_MIN,
    DML_ELEMENT_WISE_MIN_OPERATOR_DESC)
// TODO(b/25387198): A special kernel exists for int32 (see
// cwise_op_maximum.cc).
REGISTER_DML_ALL_TYPES_EXCEPT_INT32_OP_KERNEL(
    Maximum, DmlBinaryKernel, DML_OPERATOR_ELEMENT_WISE_MAX,
    DML_ELEMENT_WISE_MAX_OPERATOR_DESC)
// TODO(b/25387198): A special kernel exists for int32 (see
// cwise_op_equal_to_1.cc).
REGISTER_DML_ALL_TYPES_EXCEPT_INT32_OP_KERNEL(
    Equal, DmlBinaryKernel, DML_OPERATOR_ELEMENT_WISE_LOGICAL_EQUALS,
    DML_ELEMENT_WISE_LOGICAL_EQUALS_OPERATOR_DESC)
// TODO(b/25387198): A special kernel exists for int32 (see
// cwise_op_greater.cc).
REGISTER_DML_ALL_TYPES_EXCEPT_INT32_OP_KERNEL(
    Greater, DmlBinaryKernel, DML_OPERATOR_ELEMENT_WISE_LOGICAL_GREATER_THAN,
    DML_ELEMENT_WISE_LOGICAL_GREATER_THAN_OPERATOR_DESC)
// TODO(b/25387198): A special kernel exists for int32 (see
// cwise_op_less.cc).
REGISTER_DML_ALL_TYPES_EXCEPT_INT32_OP_KERNEL(
    Less, DmlBinaryKernel, DML_OPERATOR_ELEMENT_WISE_LOGICAL_LESS_THAN,
    DML_ELEMENT_WISE_LOGICAL_LESS_THAN_OPERATOR_DESC)
// TODO(b/25387198): A special kernel exists for int32 (see cwise_op_mod.cc).
REGISTER_DML_ALL_TYPES_EXCEPT_INT32_OP_KERNEL(
    Mod, DmlBinaryKernel, DML_OPERATOR_ELEMENT_WISE_MODULUS_TRUNCATE,
    DML_ELEMENT_WISE_MODULUS_TRUNCATE_OPERATOR_DESC)
// TODO(b/25387198): A special kernel exists for int32 (see
// cwise_op_floor_mod.cc).
REGISTER_DML_ALL_TYPES_EXCEPT_INT32_OP_KERNEL(
    FloorMod, DmlBinaryKernel, DML_OPERATOR_ELEMENT_WISE_MODULUS_FLOOR,
    DML_ELEMENT_WISE_MODULUS_FLOOR_OPERATOR_DESC)
// TODO(b/25387198): A special kernel exists for int32 (see cwise_op_mod.cc).
REGISTER_DML_ALL_TYPES_EXCEPT_INT32_OP_KERNEL(
    TruncateMod, DmlBinaryKernel, DML_OPERATOR_ELEMENT_WISE_MODULUS_TRUNCATE,
    DML_ELEMENT_WISE_MODULUS_TRUNCATE_OPERATOR_DESC)
// TODO(b/25387198): A special kernel exists for int32 (see cwise_op_abs.cc).
REGISTER_DML_SIGNED_EXCEPT_INT32_OP_KERNEL(Abs, DmlUnaryScaleBiasKernel,
                                           DML_OPERATOR_ELEMENT_WISE_ABS,
                                           DML_ELEMENT_WISE_ABS_OPERATOR_DESC)
// TODO(b/25387198): A special kernel exists for int32 (see cwise_op_sign.cc).
REGISTER_DML_SIGNED_EXCEPT_INT32_OP_KERNEL(Sign, DmlUnaryKernel,
                                           DML_OPERATOR_ELEMENT_WISE_SIGN,
                                           DML_ELEMENT_WISE_SIGN_OPERATOR_DESC)
// TODO(b/25387198): A special kernel exists for int32 (see cwise_op_neg.cc).
REGISTER_DML_SIGNED_EXCEPT_INT32_OP_KERNEL(
    Neg, DmlUnaryScaleBiasKernel, DML_OPERATOR_ELEMENT_WISE_IDENTITY,
    DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC, -1, 0)
REGISTER_DML_FLOAT_OP_KERNEL(Acos, DmlUnaryScaleBiasKernel,
                             DML_OPERATOR_ELEMENT_WISE_ACOS,
                             DML_ELEMENT_WISE_ACOS_OPERATOR_DESC)
REGISTER_DML_FLOAT_OP_KERNEL(Acosh, DmlUnaryScaleBiasKernel,
                             DML_OPERATOR_ELEMENT_WISE_ACOSH,
                             DML_ELEMENT_WISE_ACOSH_OPERATOR_DESC)
REGISTER_DML_FLOAT_OP_KERNEL(Asin, DmlUnaryScaleBiasKernel,
                             DML_OPERATOR_ELEMENT_WISE_ASIN,
                             DML_ELEMENT_WISE_ASIN_OPERATOR_DESC)
REGISTER_DML_FLOAT_OP_KERNEL(Asinh, DmlUnaryScaleBiasKernel,
                             DML_OPERATOR_ELEMENT_WISE_ASINH,
                             DML_ELEMENT_WISE_ASINH_OPERATOR_DESC)
REGISTER_DML_FLOAT_OP_KERNEL(Atan, DmlUnaryScaleBiasKernel,
                             DML_OPERATOR_ELEMENT_WISE_ATAN,
                             DML_ELEMENT_WISE_ATAN_OPERATOR_DESC)
REGISTER_DML_FLOAT_OP_KERNEL(Atanh, DmlUnaryScaleBiasKernel,
                             DML_OPERATOR_ELEMENT_WISE_ATANH,
                             DML_ELEMENT_WISE_ATANH_OPERATOR_DESC)
REGISTER_DML_FLOAT_OP_KERNEL(Ceil, DmlUnaryScaleBiasKernel,
                             DML_OPERATOR_ELEMENT_WISE_CEIL,
                             DML_ELEMENT_WISE_CEIL_OPERATOR_DESC)
REGISTER_DML_FLOAT_OP_KERNEL(Cos, DmlUnaryScaleBiasKernel,
                             DML_OPERATOR_ELEMENT_WISE_COS,
                             DML_ELEMENT_WISE_COS_OPERATOR_DESC)
REGISTER_DML_FLOAT_OP_KERNEL(Cosh, DmlUnaryScaleBiasKernel,
                             DML_OPERATOR_ELEMENT_WISE_COSH,
                             DML_ELEMENT_WISE_COSH_OPERATOR_DESC)
REGISTER_DML_FLOAT_OP_KERNEL(Elu, DmlUnaryKernel, DML_OPERATOR_ACTIVATION_ELU,
                             DML_ACTIVATION_ELU_OPERATOR_DESC, 1)
REGISTER_DML_FLOAT_OP_KERNEL(Exp, DmlUnaryScaleBiasKernel,
                             DML_OPERATOR_ELEMENT_WISE_EXP,
                             DML_ELEMENT_WISE_EXP_OPERATOR_DESC)
REGISTER_DML_FLOAT_OP_KERNEL(Floor, DmlUnaryScaleBiasKernel,
                             DML_OPERATOR_ELEMENT_WISE_FLOOR,
                             DML_ELEMENT_WISE_FLOOR_OPERATOR_DESC)
REGISTER_DML_FLOAT_OP_KERNEL(Inv, DmlUnaryScaleBiasKernel,
                             DML_OPERATOR_ELEMENT_WISE_RECIP,
                             DML_ELEMENT_WISE_RECIP_OPERATOR_DESC)
REGISTER_DML_FLOAT_OP_KERNEL(IsInf, DmlUnaryKernel,
                             DML_OPERATOR_ELEMENT_WISE_IS_INFINITY,
                             DML_ELEMENT_WISE_IS_INFINITY_OPERATOR_DESC)
REGISTER_DML_FLOAT_OP_KERNEL(Log, DmlUnaryScaleBiasKernel,
                             DML_OPERATOR_ELEMENT_WISE_LOG,
                             DML_ELEMENT_WISE_LOG_OPERATOR_DESC)
REGISTER_DML_FLOAT_OP_KERNEL(Log1p, DmlUnaryScaleBiasKernel,
                             DML_OPERATOR_ELEMENT_WISE_LOG,
                             DML_ELEMENT_WISE_LOG_OPERATOR_DESC, 1, 1)
REGISTER_DML_BOOL_OP_KERNEL(LogicalAnd, DmlBinaryKernel,
                            DML_OPERATOR_ELEMENT_WISE_LOGICAL_AND,
                            DML_ELEMENT_WISE_LOGICAL_AND_OPERATOR_DESC)
REGISTER_DML_BOOL_OP_KERNEL(LogicalNot, DmlUnaryKernel,
                            DML_OPERATOR_ELEMENT_WISE_LOGICAL_NOT,
                            DML_ELEMENT_WISE_LOGICAL_NOT_OPERATOR_DESC)
REGISTER_DML_BOOL_OP_KERNEL(LogicalOr, DmlBinaryKernel,
                            DML_OPERATOR_ELEMENT_WISE_LOGICAL_OR,
                            DML_ELEMENT_WISE_LOGICAL_OR_OPERATOR_DESC)
REGISTER_DML_FLOAT_OP_KERNEL(Reciprocal, DmlUnaryScaleBiasKernel,
                             DML_OPERATOR_ELEMENT_WISE_RECIP,
                             DML_ELEMENT_WISE_RECIP_OPERATOR_DESC)
REGISTER_DML_FLOAT_OP_KERNEL(Sigmoid, DmlUnaryKernel,
                             DML_OPERATOR_ACTIVATION_SIGMOID,
                             DML_ACTIVATION_SIGMOID_OPERATOR_DESC)
REGISTER_DML_FLOAT_OP_KERNEL(Sin, DmlUnaryScaleBiasKernel,
                             DML_OPERATOR_ELEMENT_WISE_SIN,
                             DML_ELEMENT_WISE_SIN_OPERATOR_DESC)
REGISTER_DML_FLOAT_OP_KERNEL(Sinh, DmlUnaryScaleBiasKernel,
                             DML_OPERATOR_ELEMENT_WISE_SINH,
                             DML_ELEMENT_WISE_SINH_OPERATOR_DESC)
REGISTER_DML_FLOAT_OP_KERNEL(Softsign, DmlUnaryKernel,
                             DML_OPERATOR_ACTIVATION_SOFTSIGN,
                             DML_ACTIVATION_SOFTSIGN_OPERATOR_DESC)
REGISTER_DML_FLOAT_OP_KERNEL(Sqrt, DmlUnaryScaleBiasKernel,
                             DML_OPERATOR_ELEMENT_WISE_SQRT,
                             DML_ELEMENT_WISE_SQRT_OPERATOR_DESC)
REGISTER_DML_FLOAT_OP_KERNEL(Tan, DmlUnaryScaleBiasKernel,
                             DML_OPERATOR_ELEMENT_WISE_TAN,
                             DML_ELEMENT_WISE_TAN_OPERATOR_DESC)
REGISTER_DML_FLOAT_OP_KERNEL(Tanh, DmlUnaryKernel,
                             DML_OPERATOR_ELEMENT_WISE_TANH,
                             DML_ELEMENT_WISE_TANH_OPERATOR_DESC)
REGISTER_DML_FLOAT_OP_KERNEL(Erf, DmlUnaryScaleBiasKernel,
                             DML_OPERATOR_ELEMENT_WISE_ERF,
                             DML_ELEMENT_WISE_ERF_OPERATOR_DESC)
REGISTER_DML_FLOAT_OP_KERNEL(IsNan, DmlUnaryKernel,
                             DML_OPERATOR_ELEMENT_WISE_IS_NAN,
                             DML_ELEMENT_WISE_IS_NAN_OPERATOR_DESC)
// TODO(b/25387198): A special kernel exists for int32 (see cwise_op_square.cc).
REGISTER_DML_FLOAT_OP_KERNEL(Square, DmlUnaryScaleBiasKernel,
                             DML_OPERATOR_ELEMENT_WISE_CONSTANT_POW,
                             DML_ELEMENT_WISE_CONSTANT_POW_OPERATOR_DESC, 1, 0,
                             2)
REGISTER_DML_FLOAT_OP_KERNEL(Pow, DmlBinaryKernel,
                             DML_OPERATOR_ELEMENT_WISE_POW,
                             DML_ELEMENT_WISE_POW_OPERATOR_DESC)
REGISTER_DML_FLOAT_OP_KERNEL(Round, DmlUnaryKernel,
                             DML_OPERATOR_ELEMENT_WISE_ROUND,
                             DML_ELEMENT_WISE_ROUND_OPERATOR_DESC)
REGISTER_DML_FLOAT_OP_KERNEL(Softplus, DmlUnaryKernel,
                             DML_OPERATOR_ACTIVATION_SOFTPLUS,
                             DML_ACTIVATION_SOFTPLUS_OPERATOR_DESC, 1)
REGISTER_DML_FLOAT_OP_KERNEL(Softmax, DmlMaxActivationKernel,
                             DML_OPERATOR_ACTIVATION_SOFTMAX,
                             DML_ACTIVATION_SOFTMAX_OPERATOR_DESC)
REGISTER_DML_FLOAT_OP_KERNEL(LogSoftmax, DmlMaxActivationKernel,
                             DML_OPERATOR_ACTIVATION_LOG_SOFTMAX,
                             DML_ACTIVATION_LOG_SOFTMAX_OPERATOR_DESC)

REGISTER_DML_COMPOSITE_UNARY_FLOAT_KERNEL(Erfc, 1.0f - dml::Erf(x),
                                          kNchwDimensionCount)
REGISTER_DML_COMPOSITE_UNARY_FLOAT_KERNEL(Rsqrt, 1.0f / dml::Sqrt(x),
                                          kNchwDimensionCount)
REGISTER_DML_COMPOSITE_UNARY_FLOAT_KERNEL(Expm1, dml::Exp(x) - 1.0f,
                                          kNchwDimensionCount)
REGISTER_DML_COMPOSITE_UNARY_FLOAT_KERNEL(
    IsFinite, !(dml::IsNaN(x) || dml::IsInfinity(x)), kNchwDimensionCount)
// TODO(b/25387198): A special kernel exists for int32 (see
// cwise_op_not_equal_to_1.cc).
REGISTER_DML_COMPOSITE_BINARY_ALL_TYPES_EXCEPT_INT32_KERNEL(NotEqual, x != y,
                                                            kNchwDimensionCount)
// TODO(b/25387198): A special kernel exists for int32 (see
// cwise_op_less_equal.cc).
REGISTER_DML_COMPOSITE_BINARY_ALL_TYPES_EXCEPT_INT32_KERNEL(LessEqual, x <= y,
                                                            kNchwDimensionCount)
// TODO(b/25387198): A special kernel exists for int32 (see
// cwise_op_greater_equal.cc).
REGISTER_DML_COMPOSITE_BINARY_ALL_TYPES_EXCEPT_INT32_KERNEL(GreaterEqual,
                                                            x >= y,
                                                            kNchwDimensionCount)
// TODO: Register this operator for all types (except int32) when FloorDiv is
// added to DML. dml::Floor(x / y) works for float types, but it doesn't work
// for integer types since DML truncates towards zero.
// TFDML #25977645
// TODO(b/25387198): A special kernel exists for int32 (see
// cwise_op_floor_div.cc).
REGISTER_DML_COMPOSITE_BINARY_FLOAT_KERNEL(FloorDiv, dml::Floor(x / y),
                                           kNchwDimensionCount)
REGISTER_DML_COMPOSITE_BINARY_FLOAT_KERNEL(SigmoidGrad, y* x*(1 - x),
                                           kNchwDimensionCount)
REGISTER_DML_COMPOSITE_BINARY_FLOAT_KERNEL(TanhGrad, y*(1 - x * x),
                                           kNchwDimensionCount)
REGISTER_DML_COMPOSITE_BINARY_FLOAT_KERNEL(SqrtGrad, y * 0.5f / x,
                                           kNchwDimensionCount)
REGISTER_DML_COMPOSITE_BINARY_FLOAT_KERNEL(RsqrtGrad, y * (-0.5f * x) * (x * x),
                                           kNchwDimensionCount)
REGISTER_DML_COMPOSITE_BINARY_FLOAT_KERNEL(ReciprocalGrad, -y* x* x,
                                           kNchwDimensionCount)
REGISTER_DML_COMPOSITE_BINARY_FLOAT_KERNEL(SoftplusGrad, x / (dml::Exp(-y) + 1),
                                           kNchwDimensionCount)
// softsigngrad(gradients, features) = gradients / (1 + abs(features)) ** 2
REGISTER_DML_COMPOSITE_BINARY_FLOAT_KERNEL(SoftsignGrad,
                                           x / dml::Pow(1 + dml::Abs(y), 2),
                                           kNchwDimensionCount)
#undef REGISTER_DML_FLOAT_OP_KERNEL
#undef REGISTER_OP_KERNEL
#undef REGISTER_DML_BOOL_OP_KERNEL
#undef REGISTER_BOOL_OP_KERNEL

class DmlClipByValueKernel : public DmlKernel {
 public:
  using InitHelper = ElementWiseInitHelper<UINT32_MAX>;

  explicit DmlClipByValueKernel(DmlKernelConstruction* ctx,
                                const InitHelper* init_helper) {
    CHECK(ctx->GetInputCount() == 3);
    CHECK(ctx->GetOutputCount() == 1);

    DmlKernelParams params;

    // Broadcast inputs to match output shape
    params.input_shape = ctx->GetOutputTensorShape(0);

    // The DML operator takes fewer inputs than the TF kernel receives, so we
    // need to explicitly specify the kernel indices. In this case, the DML op
    // takes a single input which corresponds to the 0th input on the kernel.
    params.kernel_input_indices = {0};

    DmlKernelTensors tensors = GetTensorInfos(ctx, params);
    auto inputs = GetDmlTensorDescs(tensors.inputs);
    auto outputs = GetDmlTensorDescs(tensors.outputs);

    // Min/max are supplied as tensors for ClipByValue, which are required to be
    // constant CPU inputs
    const Tensor& min_tensor = ctx->GetConstantInputTensor(1);
    const Tensor& max_tensor = ctx->GetConstantInputTensor(2);

    DML_ELEMENT_WISE_CLIP_OPERATOR_DESC clip_desc = {};
    clip_desc.InputTensor = inputs.data();
    clip_desc.OutputTensor = outputs.data();
    clip_desc.Min = min_tensor.flat<float>()(0);
    clip_desc.Max = max_tensor.flat<float>()(0);

    DML_OPERATOR_DESC op_desc = {DML_OPERATOR_ELEMENT_WISE_CLIP, &clip_desc};
    Initialize(ctx, std::move(tensors), op_desc);
  }

  StatusOr<DmlGpuEvent> Compute(DmlKernelContext* ctx) const override {
    // Currently, 64-bit integers in DML are emulated using 32-bit integers
    // using striding to emulate a larger type. Because we can't guarantee that
    // our output tensor's memory is zero'd, we need to do so manually prior to
    // running running gather.
    Tensor* output = ctx->GetOutputTensor(0);

    if (Is64BitIntegerType(output->dtype())) {
      ctx->ZeroBuffer(ctx->CreateBufferForTensor(*output));
    }

    return DmlKernel::Compute(ctx);
  }
};

#define DML_REGISTER_KERNEL(type)                                \
  REGISTER_KERNEL_BUILDER(Name("ClipByValue")                    \
                              .Device(DEVICE_DML)                \
                              .TypeConstraint<type>("T")         \
                              .HostMemory("clip_value_min")      \
                              .HostMemory("clip_value_max"),     \
                          DmlKernelWrapper<DmlClipByValueKernel, \
                                           GetBroadcastedOutputShapeHelper>);
// TODO(b/25387198): A special kernel exists for int32 (see cwise_op_clip.cc).
TF_CALL_DML_ALL_TYPES_EXCEPT_INT32(DML_REGISTER_KERNEL);
#undef DML_REGISTER_KERNEL

template <typename Functor, uint32_t max_dim_count>
class DmlBinaryWithZeroKernel : public DmlKernel {
 public:
  using InitHelper = ElementWiseInitHelper<max_dim_count>;

  explicit DmlBinaryWithZeroKernel(DmlKernelConstruction* ctx,
                                   const InitHelper* init_helper) {
    CHECK(ctx->GetInputCount() == 2);
    CHECK(ctx->GetOutputCount() == 1);

    auto input_shapes = init_helper->GetCollapsedInputShapes();
    const TensorShape& output_shape = init_helper->GetCollapsedOutputShape();

    DmlKernelTensors tensors =
        CreateKernelTensors(ctx, input_shapes, output_shape);
    auto inputs = GetDmlTensorDescs(tensors.inputs);
    auto outputs = GetDmlTensorDescs(tensors.outputs);

    auto scope = dml::Graph(ctx->GetDmlDevice());
    auto x = dml::InputTensor(scope, 0, inputs[0]);
    auto y = dml::InputTensor(scope, 1, inputs[1]);
    auto zero = dml::ZeroTensor(scope, x.GetOutputDesc().dataType,
                                x.GetOutputDesc().sizes);

    Functor f;
    auto result = f(zero, x, y);

    ComPtr<IDMLCompiledOperator> compiled_op =
        scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

    Initialize(ctx, std::move(tensors), compiled_op.Get());
  }

  StatusOr<DmlGpuEvent> Compute(DmlKernelContext* ctx) const override {
    // Currently, 64-bit integers in DML are emulated using 32-bit integers
    // using striding to emulate a larger type. Because we can't guarantee that
    // our output tensor's memory is zero'd, we need to do so manually prior to
    // running running gather.
    Tensor* output = ctx->GetOutputTensor(0);

    if (Is64BitIntegerType(output->dtype())) {
      ctx->ZeroBuffer(ctx->CreateBufferForTensor(*output));
    }

    return DmlKernel::Compute(ctx);
  }
};

struct DmlDivNoNanFunctor {
  dml::Expression operator()(dml::Expression zero, dml::Expression x,
                             dml::Expression y) {
    return dml::If(y == zero, zero, x / y);
  }
};

#define DML_REGISTER_KERNEL(type)                                           \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("DivNoNan").Device(DEVICE_DML).TypeConstraint<type>("T"),        \
      DmlKernelWrapper<                                                     \
          DmlBinaryWithZeroKernel<DmlDivNoNanFunctor, kNchwDimensionCount>, \
          GetBroadcastedOutputShapeHelper>);
TF_CALL_DML_FLOAT_INT32_INT64_TYPES(DML_REGISTER_KERNEL);
#undef DML_REGISTER_KERNEL

struct DmlMulNoNanFunctor {
  dml::Expression operator()(dml::Expression zero, dml::Expression x,
                             dml::Expression y) {
    return dml::If(y == zero, zero, x * y);
  }
};

#define DML_REGISTER_KERNEL(type)                                           \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("MulNoNan").Device(DEVICE_DML).TypeConstraint<type>("T"),        \
      DmlKernelWrapper<                                                     \
          DmlBinaryWithZeroKernel<DmlMulNoNanFunctor, kNchwDimensionCount>, \
          GetBroadcastedOutputShapeHelper>);
TF_CALL_DML_FLOAT_TYPES(DML_REGISTER_KERNEL);
#undef DML_REGISTER_KERNEL

struct DmlXlogyFunctor {
  dml::Expression operator()(dml::Expression zero, dml::Expression x,
                             dml::Expression y) {
    return dml::If(x == zero, zero, x * dml::Log(y));
  }
};

#define DML_REGISTER_KERNEL(type)                                        \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("Xlogy").Device(DEVICE_DML).TypeConstraint<type>("T"),        \
      DmlKernelWrapper<                                                  \
          DmlBinaryWithZeroKernel<DmlXlogyFunctor, kNchwDimensionCount>, \
          GetBroadcastedOutputShapeHelper>);
TF_CALL_DML_FLOAT_TYPES(DML_REGISTER_KERNEL);
#undef DML_REGISTER_KERNEL

struct DmlXdivyFunctor {
  dml::Expression operator()(dml::Expression zero, dml::Expression x,
                             dml::Expression y) {
    return dml::If(x == zero, zero, x / y);
  }
};

#define DML_REGISTER_KERNEL(type)                                        \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("Xdivy").Device(DEVICE_DML).TypeConstraint<type>("T"),        \
      DmlKernelWrapper<                                                  \
          DmlBinaryWithZeroKernel<DmlXdivyFunctor, kNchwDimensionCount>, \
          GetBroadcastedOutputShapeHelper>);
TF_CALL_DML_FLOAT_TYPES(DML_REGISTER_KERNEL);
#undef DML_REGISTER_KERNEL

class DmlSquaredDifferenceKernel : public DmlKernel {
 public:
  using InitHelper = ElementWiseInitHelper<kNchwDimensionCount>;

  explicit DmlSquaredDifferenceKernel(DmlKernelConstruction* ctx,
                                      const InitHelper* init_helper) {
    CHECK(ctx->GetInputCount() == 2);
    CHECK(ctx->GetOutputCount() == 1);

    auto input_shapes = init_helper->GetCollapsedInputShapes();
    const TensorShape& output_shape = init_helper->GetCollapsedOutputShape();

    DmlKernelTensors tensors =
        CreateKernelTensors(ctx, input_shapes, output_shape);
    auto inputs = GetDmlTensorDescs(tensors.inputs);
    auto outputs = GetDmlTensorDescs(tensors.outputs);

    auto scope = dml::Graph(ctx->GetDmlDevice());
    auto x = dml::InputTensor(scope, 0, inputs[0]);
    auto y = dml::InputTensor(scope, 1, inputs[1]);
    auto diff = x - y;
    auto result = diff * diff;

    ComPtr<IDMLCompiledOperator> compiled_op =
        scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

    Initialize(ctx, std::move(tensors), compiled_op.Get());
  }

  StatusOr<DmlGpuEvent> Compute(DmlKernelContext* ctx) const override {
    // Currently, 64-bit integers in DML are emulated using 32-bit integers
    // using striding to emulate a larger type. Because we can't guarantee that
    // our output tensor's memory is zero'd, we need to do so manually prior to
    // running running gather.
    Tensor* output = ctx->GetOutputTensor(0);

    if (Is64BitIntegerType(output->dtype())) {
      ctx->ZeroBuffer(ctx->CreateBufferForTensor(*output));
    }

    return DmlKernel::Compute(ctx);
  }
};

#define DML_REGISTER_KERNEL(type)                                             \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("SquaredDifference").Device(DEVICE_DML).TypeConstraint<type>("T"), \
      DmlKernelWrapper<DmlSquaredDifferenceKernel,                            \
                       GetBroadcastedOutputShapeHelper>);
// TODO(b/25387198): A special kernel exists for int32 (see
// cwise_op_squared_difference.cc).
TF_CALL_DML_SIGNED_TYPES_EXCEPT_INT32(DML_REGISTER_KERNEL);
#undef DML_REGISTER_KERNEL

class DmlSeluKernel : public DmlKernel {
 public:
  using InitHelper = ElementWiseInitHelper<UINT32_MAX>;

  explicit DmlSeluKernel(DmlKernelConstruction* ctx,
                         const InitHelper* init_helper) {
    CHECK(ctx->GetInputCount() == 1);
    CHECK(ctx->GetOutputCount() == 1);

    DmlKernelParams params;

    // Broadcast inputs to match output shape
    params.input_shape = ctx->GetOutputTensorShape(0);

    DmlKernelTensors tensors = GetTensorInfos(ctx, params);
    auto inputs = GetDmlTensorDescs(tensors.inputs);
    auto outputs = GetDmlTensorDescs(tensors.outputs);

    DML_ACTIVATION_SCALED_ELU_OPERATOR_DESC selu_desc = {
        &inputs[0], outputs.data(),
        1.67326319217681884765625f,  // alpha
        1.05070102214813232421875f   // gamma
    };

    DML_OPERATOR_DESC op_desc = {DML_OPERATOR_ACTIVATION_SCALED_ELU,
                                 &selu_desc};
    Initialize(ctx, std::move(tensors), op_desc);
  }
};

#define DML_REGISTER_KERNEL(type)                                \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("Selu").Device(DEVICE_DML).TypeConstraint<type>("T"), \
      DmlKernelWrapper<DmlSeluKernel, GetBroadcastedOutputShapeHelper>);
TF_CALL_DML_FLOAT_TYPES(DML_REGISTER_KERNEL);
#undef DML_REGISTER_KERNEL

class DmlLeakyReluKernel : public DmlKernel {
 public:
  using InitHelper = ElementWiseInitHelper<UINT32_MAX>;

  explicit DmlLeakyReluKernel(DmlKernelConstruction* ctx,
                              const InitHelper* init_helper) {
    CHECK(ctx->GetInputCount() == 1);
    CHECK(ctx->GetOutputCount() == 1);

    DmlKernelParams params;

    // Broadcast inputs to match output shape
    params.input_shape = ctx->GetOutputTensorShape(0);

    DmlKernelTensors tensors = GetTensorInfos(ctx, params);
    auto inputs = GetDmlTensorDescs(tensors.inputs);
    auto outputs = GetDmlTensorDescs(tensors.outputs);

    float alpha;
    TF_CHECK_OK(ctx->GetAttr("alpha", &alpha));

    DML_ACTIVATION_LEAKY_RELU_OPERATOR_DESC leaky_relu_desc = {
        &inputs[0], outputs.data(), alpha};

    DML_OPERATOR_DESC op_desc = {DML_OPERATOR_ACTIVATION_LEAKY_RELU,
                                 &leaky_relu_desc};
    Initialize(ctx, std::move(tensors), op_desc);
  }
};

#define DML_REGISTER_KERNEL(type)                                     \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("LeakyRelu").Device(DEVICE_DML).TypeConstraint<type>("T"), \
      DmlKernelWrapper<DmlLeakyReluKernel, GetBroadcastedOutputShapeHelper>);
TF_CALL_DML_FLOAT_TYPES(DML_REGISTER_KERNEL);
#undef DML_REGISTER_KERNEL

template <typename T>
class DmlApproximateEqualKernel : public DmlKernel {
 public:
  using InitHelper = ElementWiseInitHelper<kNchwDimensionCount>;

  explicit DmlApproximateEqualKernel(DmlKernelConstruction* ctx,
                                     const InitHelper* init_helper) {
    CHECK(ctx->GetInputCount() == 2);
    CHECK(ctx->GetOutputCount() == 1);

    auto input_shapes = init_helper->GetCollapsedInputShapes();
    const TensorShape& output_shape = init_helper->GetCollapsedOutputShape();

    DmlKernelTensors tensors =
        CreateKernelTensors(ctx, input_shapes, output_shape);
    auto inputs = GetDmlTensorDescs(tensors.inputs);
    auto outputs = GetDmlTensorDescs(tensors.outputs);

    auto scope = dml::Graph(ctx->GetDmlDevice());
    auto x = dml::InputTensor(scope, 0, inputs[0]);
    auto y = dml::InputTensor(scope, 1, inputs[1]);

    float tolerance;
    TF_CHECK_OK(ctx->GetAttr("tolerance", &tolerance));
    auto tolerance_tensor =
        dml::ScalarTensor<T>(scope, TfTensorTypeTraits<T>::FromFloat(tolerance),
                             x.GetOutputDesc().sizes);

    auto result = dml::Abs(x - y) < tolerance_tensor;

    ComPtr<IDMLCompiledOperator> compiled_op =
        scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

    Initialize(ctx, std::move(tensors), compiled_op.Get());
  }
};

#define DML_REGISTER_KERNEL(type)                                            \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("ApproximateEqual").Device(DEVICE_DML).TypeConstraint<type>("T"), \
      DmlKernelWrapper<DmlApproximateEqualKernel<type>,                      \
                       GetBroadcastedOutputShapeHelper>);
TF_CALL_DML_FLOAT_TYPES(DML_REGISTER_KERNEL)
#undef DML_REGISTER_KERNEL

// ---------------------------------------------------
// BITWISE OPERATORS
// ---------------------------------------------------

class DmlBitwiseNotKernel : public DmlKernel {
 public:
  using InitHelper = tensorflow::NoOpInitializationHelper;

  explicit DmlBitwiseNotKernel(DmlKernelConstruction* ctx,
                               const InitHelper* init_helper) {
    CHECK(ctx->GetInputCount() == 1);
    CHECK(ctx->GetOutputCount() == 1);

    auto num_elements =
        static_cast<uint32_t>(ctx->GetInputTensorShape(0).num_elements());

    // DML doesn't support 64-bit integer types, but we can reinterpret
    // the tensor as twice as many 32-bit elements. Sign doesn't matter.
    DataType dtype = ctx->GetInputDataType(0);
    DCHECK(dtype == ctx->GetOutputDataType(0));
    if (Is64BitIntegerType(dtype)) {
      num_elements *= 2;
      dtype = DT_UINT32;
    }

    std::array<uint32_t, 4> sizes = {1, 1, 1, num_elements};

    DmlTensorInfo in;
    in.kernel_index = 0;
    in.desc = DmlTensorDesc::Create(dtype, sizes, sizes);
    in.desc.ForceUnsignedDataType();
    auto in_desc = in.desc.GetDmlDesc();

    DmlTensorInfo out;
    out.kernel_index = 0;
    out.desc = DmlTensorDesc::Create(dtype, sizes, sizes);
    out.desc.ForceUnsignedDataType();
    auto out_desc = out.desc.GetDmlDesc();

    DmlKernelTensors tensors;
    tensors.inputs = {in};
    tensors.outputs = {out};

    DML_ELEMENT_WISE_BIT_NOT_OPERATOR_DESC desc = {};
    desc.InputTensor = &in_desc;
    desc.OutputTensor = &out_desc;

    DML_OPERATOR_DESC op_desc = {DML_OPERATOR_ELEMENT_WISE_BIT_NOT, &desc};

    Initialize(ctx, std::move(tensors), op_desc);
  }
};

template <DML_OPERATOR_TYPE op_type, typename DML_OPERATOR_SPECIFIC_DESC>
class DmlBinaryBitwiseKernel : public DmlKernel {
 public:
  using InitHelper = ElementWiseInitHelper<GetMaxDimCount<op_type>()>;

  explicit DmlBinaryBitwiseKernel(DmlKernelConstruction* ctx,
                                  const InitHelper* init_helper) {
    CHECK(ctx->GetInputCount() == 2);
    CHECK(ctx->GetOutputCount() == 1);

    auto input_shapes = init_helper->GetCollapsedInputShapes();
    const TensorShape& output_shape = init_helper->GetCollapsedOutputShape();

    DmlKernelTensors tensors =
        CreateKernelTensors(ctx, input_shapes, output_shape);

    // DML only supports unsigned types, but sign doesn't matter for bitwise.
    tensors.inputs[0]->desc.ForceUnsignedDataType();
    tensors.inputs[1]->desc.ForceUnsignedDataType();
    tensors.outputs[0]->desc.ForceUnsignedDataType();

    auto inputs = GetDmlTensorDescs(tensors.inputs);
    auto outputs = GetDmlTensorDescs(tensors.outputs);

    DML_OPERATOR_SPECIFIC_DESC desc = {
        &inputs[0],
        &inputs[1],
        outputs.data(),
    };

    DML_OPERATOR_DESC op_desc = {op_type, &desc};

    Initialize(ctx, std::move(tensors), op_desc);
  }
};

class DmlBitCountKernel : public DmlKernel {
 public:
  using InitHelper = tensorflow::NoOpInitializationHelper;

  explicit DmlBitCountKernel(DmlKernelConstruction* ctx,
                             const InitHelper* init_helper) {
    CHECK(ctx->GetInputCount() == 1);
    CHECK(ctx->GetOutputCount() == 1);

    auto num_elements =
        static_cast<uint32_t>(ctx->GetInputTensorShape(0).num_elements());

    std::array<uint32_t, 4> sizes = {1, 1, 1, num_elements};

    DmlTensorInfo in;
    in.kernel_index = 0;
    in.desc = DmlTensorDesc::Create(ctx->GetInputDataType(0), sizes, sizes);
    in.desc.ForceUnsignedDataType();
    auto in_desc = in.desc.GetDmlDesc();

    DmlTensorInfo out;
    out.kernel_index = 0;
    out.desc = DmlTensorDesc::Create(ctx->GetOutputDataType(0), sizes, sizes);
    out.desc.ForceUnsignedDataType();
    auto out_desc = out.desc.GetDmlDesc();

    DmlKernelTensors tensors;
    tensors.inputs = {in};
    tensors.outputs = {out};

    if (Is64BitIntegerType(ctx->GetInputDataType(0))) {
      // DML doesn't support 64-bit integer types, but we can reinterpret
      // the input tensor as twice as many 32-bit elements. Sign doesn't matter.
      // This is followed by a sum of the two separate counts, so make the shape
      // 2D so that we can reduce each adjacent pair of counts.
      dml::TensorDesc::Dimensions double_sizes = {1, 1, num_elements, 2};

      auto scope = dml::Graph(ctx->GetDmlDevice());
      auto in_64_bit = dml::InputTensor(scope, 0, in_desc);
      auto in_32_bit = dml::Reinterpret(in_64_bit, DML_TENSOR_DATA_TYPE_UINT32,
                                        double_sizes, dml::NullOpt);

      // Reduce doesn't support UINT8, so output UINT32 bit counts and cast
      // down. This may be faster than doing the arithmetic in UINT8 anyway.
      auto bit_count = dml::BitCount(in_32_bit, DML_TENSOR_DATA_TYPE_UINT32);
      bit_count = dml::Reduce(bit_count, DML_REDUCE_FUNCTION_SUM, {3});
      bit_count = dml::Cast(bit_count, DML_TENSOR_DATA_TYPE_UINT8);

      ComPtr<IDMLCompiledOperator> compiled_op =
          scope.Compile(DML_EXECUTION_FLAG_NONE, {bit_count});

      Initialize(ctx, std::move(tensors), compiled_op.Get());

    } else {
      DML_ELEMENT_WISE_BIT_COUNT_OPERATOR_DESC desc = {};
      desc.InputTensor = &in_desc;
      desc.OutputTensor = &out_desc;

      DML_OPERATOR_DESC op_desc = {DML_OPERATOR_ELEMENT_WISE_BIT_COUNT, &desc};

      Initialize(ctx, std::move(tensors), op_desc);
    }
  }
};

#define DML_REGISTER_KERNEL(type)                                           \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("PopulationCount").Device(DEVICE_DML).TypeConstraint<type>("T"), \
      DmlKernelWrapper<DmlBitCountKernel, GetOutputShapeAsInputShapeHelper>);
TF_CALL_uint8(DML_REGISTER_KERNEL);
TF_CALL_uint16(DML_REGISTER_KERNEL);
TF_CALL_uint32(DML_REGISTER_KERNEL);
TF_CALL_uint64(DML_REGISTER_KERNEL);
TF_CALL_int8(DML_REGISTER_KERNEL);
TF_CALL_int16(DML_REGISTER_KERNEL);
TF_CALL_int32(DML_REGISTER_KERNEL);
TF_CALL_int64(DML_REGISTER_KERNEL);
#undef DML_REGISTER_KERNEL

#define DML_REGISTER_KERNEL(type)                                  \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("Invert").Device(DEVICE_DML).TypeConstraint<type>("T"), \
      DmlKernelWrapper<DmlBitwiseNotKernel,                        \
                       GetOutputShapeAsInputShapeHelper>);
TF_CALL_uint8(DML_REGISTER_KERNEL);
TF_CALL_uint16(DML_REGISTER_KERNEL);
TF_CALL_uint32(DML_REGISTER_KERNEL);
TF_CALL_uint64(DML_REGISTER_KERNEL);
TF_CALL_int8(DML_REGISTER_KERNEL);
TF_CALL_int16(DML_REGISTER_KERNEL);
TF_CALL_int32(DML_REGISTER_KERNEL);
TF_CALL_int64(DML_REGISTER_KERNEL);
#undef DML_REGISTER_KERNEL

#define DML_REGISTER_KERNEL(type)                                         \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("BitwiseAnd").Device(DEVICE_DML).TypeConstraint<type>("T"),    \
      DmlKernelWrapper<                                                   \
          DmlBinaryBitwiseKernel<DML_OPERATOR_ELEMENT_WISE_BIT_AND,       \
                                 DML_ELEMENT_WISE_BIT_AND_OPERATOR_DESC>, \
          GetBroadcastedOutputShapeHelper>);
TF_CALL_uint8(DML_REGISTER_KERNEL);
TF_CALL_uint16(DML_REGISTER_KERNEL);
TF_CALL_uint32(DML_REGISTER_KERNEL);
TF_CALL_int8(DML_REGISTER_KERNEL);
TF_CALL_int16(DML_REGISTER_KERNEL);
TF_CALL_int32(DML_REGISTER_KERNEL);
#undef DML_REGISTER_KERNEL

#define DML_REGISTER_KERNEL(type)                                        \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("BitwiseOr").Device(DEVICE_DML).TypeConstraint<type>("T"),    \
      DmlKernelWrapper<                                                  \
          DmlBinaryBitwiseKernel<DML_OPERATOR_ELEMENT_WISE_BIT_OR,       \
                                 DML_ELEMENT_WISE_BIT_OR_OPERATOR_DESC>, \
          GetBroadcastedOutputShapeHelper>);
TF_CALL_uint8(DML_REGISTER_KERNEL);
TF_CALL_uint16(DML_REGISTER_KERNEL);
TF_CALL_uint32(DML_REGISTER_KERNEL);
TF_CALL_int8(DML_REGISTER_KERNEL);
TF_CALL_int16(DML_REGISTER_KERNEL);
TF_CALL_int32(DML_REGISTER_KERNEL);
#undef DML_REGISTER_KERNEL

#define DML_REGISTER_KERNEL(type)                                         \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("BitwiseXor").Device(DEVICE_DML).TypeConstraint<type>("T"),    \
      DmlKernelWrapper<                                                   \
          DmlBinaryBitwiseKernel<DML_OPERATOR_ELEMENT_WISE_BIT_XOR,       \
                                 DML_ELEMENT_WISE_BIT_XOR_OPERATOR_DESC>, \
          GetBroadcastedOutputShapeHelper>);
TF_CALL_uint8(DML_REGISTER_KERNEL);
TF_CALL_uint16(DML_REGISTER_KERNEL);
TF_CALL_uint32(DML_REGISTER_KERNEL);
TF_CALL_int8(DML_REGISTER_KERNEL);
TF_CALL_int16(DML_REGISTER_KERNEL);
TF_CALL_int32(DML_REGISTER_KERNEL);
#undef DML_REGISTER_KERNEL

#define DML_REGISTER_KERNEL(type)                                           \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("RightShift").Device(DEVICE_DML).TypeConstraint<type>("T"),      \
      DmlKernelWrapper<DmlBinaryBitwiseKernel<                              \
                           DML_OPERATOR_ELEMENT_WISE_BIT_SHIFT_RIGHT,       \
                           DML_ELEMENT_WISE_BIT_SHIFT_RIGHT_OPERATOR_DESC>, \
                       GetBroadcastedOutputShapeHelper>);
// DML only supports logical shifts, not arithmetic shifts, so we can't support
// signed integers for RSHIFT.
TF_CALL_uint8(DML_REGISTER_KERNEL);
TF_CALL_uint16(DML_REGISTER_KERNEL);
TF_CALL_uint32(DML_REGISTER_KERNEL);
#undef DML_REGISTER_KERNEL

#define DML_REGISTER_KERNEL(type)                                          \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("LeftShift").Device(DEVICE_DML).TypeConstraint<type>("T"),      \
      DmlKernelWrapper<DmlBinaryBitwiseKernel<                             \
                           DML_OPERATOR_ELEMENT_WISE_BIT_SHIFT_LEFT,       \
                           DML_ELEMENT_WISE_BIT_SHIFT_LEFT_OPERATOR_DESC>, \
                       GetBroadcastedOutputShapeHelper>);
// DML only supports logical shifts, not arithmetic shifts, but for LSHIFT
// the two are identical. We reinterpret signed as unsigned.
TF_CALL_uint8(DML_REGISTER_KERNEL);
TF_CALL_uint16(DML_REGISTER_KERNEL);
TF_CALL_uint32(DML_REGISTER_KERNEL);
TF_CALL_int8(DML_REGISTER_KERNEL);
TF_CALL_int16(DML_REGISTER_KERNEL);
TF_CALL_int32(DML_REGISTER_KERNEL);
#undef DML_REGISTER_KERNEL

}  // namespace tensorflow