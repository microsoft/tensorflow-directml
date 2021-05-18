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

    // TFDML #24881131
    const dml::TensorPolicy out_policy =
        Is64BitUnsignedIntegerType(ctx->GetOutputDataType(0))
            ? GetEmulatedInt64TensorPolicy()
            : dml::TensorPolicy::Default();

    auto scope = dml::Graph(ctx->GetDmlDevice(), out_policy);
    auto x = dml::InputTensor(scope, 0, inputs[0]);
    auto y = dml::InputTensor(scope, 1, inputs[1]);

    ExpressionFunctor expression;
    auto result = expression(x, y);

    // TFDML #24881131
    if (Is64BitSignedIntegerType(ctx->GetOutputDataType(0))) {
      result = dml::ConvertInt32ToInt64(scope, result);
    }

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

    // TFDML #24881131
    if (Is64BitUnsignedIntegerType(output->dtype())) {
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

template <typename ExpressionFunctor>
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

    // TFDML #24881131
    const dml::TensorPolicy out_policy =
        Is64BitUnsignedIntegerType(ctx->GetOutputDataType(0))
            ? GetEmulatedInt64TensorPolicy()
            : dml::TensorPolicy::Default();

    auto scope = dml::Graph(ctx->GetDmlDevice(), out_policy);
    auto x = dml::InputTensor(scope, 0, inputs[0]);

    ExpressionFunctor expression;
    auto result = expression(x);

    // TFDML #24881131
    if (Is64BitSignedIntegerType(ctx->GetOutputDataType(0))) {
      result = dml::ConvertInt32ToInt64(scope, result);
    }

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

    // TFDML #24881131
    if (Is64BitUnsignedIntegerType(output->dtype())) {
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

#define REGISTER_BOOL_OP_KERNEL(opName) \
  REGISTER_KERNEL_BUILDER(              \
      Name(#opName).Device(DEVICE_DML), \
      DmlKernelWrapper<Dml##opName##Kernel, GetBroadcastedOutputShapeHelper>);

#define CONCAT_NAMESPACE_NAME_HELPER(opName, uniqueId) \
  Dml##opName##uniqueId##Namespace
#define CONCAT_NAMESPACE_NAME(opName, uniqueId) \
  CONCAT_NAMESPACE_NAME_HELPER(opName, uniqueId)

#define REGISTER_DML_COMPOSITE_UNARY_STRUCT(opName, expression)  \
  struct Dml##opName##Functor {                                            \
    dml::Expression operator()(dml::Expression x) { return (expression); } \
  };                                                                       \
  using Dml##opName##Kernel = DmlCompositeUnaryKernel<Dml##opName##Functor>;

#define REGISTER_DML_COMPOSITE_UNARY_BOOL_KERNEL(opName, expression)     \
  namespace CONCAT_NAMESPACE_NAME(opName, __COUNTER__) {                 \
    REGISTER_DML_COMPOSITE_UNARY_STRUCT(opName, expression) \
    REGISTER_BOOL_OP_KERNEL(opName);                                     \
  }

#define REGISTER_DML_COMPOSITE_BINARY_STRUCT(opName, expression,       \
                                             max_dim_count)            \
  struct Dml##opName##Functor {                                        \
    dml::Expression operator()(dml::Expression x, dml::Expression y) { \
      return (expression);                                             \
    }                                                                  \
  };                                                                   \
  using Dml##opName##Kernel =                                          \
      DmlCompositeBinaryKernel<Dml##opName##Functor, max_dim_count>;

#define REGISTER_DML_COMPOSITE_BINARY_BOOL_KERNEL(opName, expression,     \
                                                  max_dim_count)          \
  REGISTER_DML_COMPOSITE_BINARY_STRUCT(opName, expression, max_dim_count) \
  REGISTER_BOOL_OP_KERNEL(opName);

#define REGISTER_DML_COMPOSITE_UNARY_KERNEL_1(opName, expression, T1) \
  namespace CONCAT_NAMESPACE_NAME(opName, __COUNTER__) {              \
    REGISTER_DML_COMPOSITE_UNARY_STRUCT(opName, expression)           \
    REGISTER_OP_KERNEL(opName, T1);                                   \
  }

#define REGISTER_DML_COMPOSITE_UNARY_KERNEL_2(opName, expression, T1, T2) \
  namespace CONCAT_NAMESPACE_NAME(opName, __COUNTER__) {                  \
    REGISTER_DML_COMPOSITE_UNARY_STRUCT(opName, expression)               \
    REGISTER_OP_KERNEL(opName, T1);                                       \
    REGISTER_OP_KERNEL(opName, T2);                                       \
  }

#define REGISTER_DML_COMPOSITE_UNARY_KERNEL_3(opName, expression, T1, T2, T3) \
  namespace CONCAT_NAMESPACE_NAME(opName, __COUNTER__) {                      \
    REGISTER_DML_COMPOSITE_UNARY_STRUCT(opName, expression)                   \
    REGISTER_OP_KERNEL(opName, T1);                                           \
    REGISTER_OP_KERNEL(opName, T2);                                           \
    REGISTER_OP_KERNEL(opName, T3);                                           \
  }

#define REGISTER_DML_COMPOSITE_BINARY_KERNEL_1(opName, expression,          \
                                               max_dim_count, T1)           \
  namespace CONCAT_NAMESPACE_NAME(opName, __COUNTER__) {                    \
    REGISTER_DML_COMPOSITE_BINARY_STRUCT(opName, expression, max_dim_count) \
    REGISTER_OP_KERNEL(opName, T1);                                         \
  }

#define REGISTER_DML_COMPOSITE_BINARY_KERNEL_2(opName, expression,          \
                                               max_dim_count, T1, T2)       \
  namespace CONCAT_NAMESPACE_NAME(opName, __COUNTER__) {                    \
    REGISTER_DML_COMPOSITE_BINARY_STRUCT(opName, expression, max_dim_count) \
    REGISTER_OP_KERNEL(opName, T1);                                         \
    REGISTER_OP_KERNEL(opName, T2);                                         \
  }

#define REGISTER_DML_COMPOSITE_BINARY_KERNEL_3(opName, expression,          \
                                               max_dim_count, T1, T2, T3)   \
  namespace CONCAT_NAMESPACE_NAME(opName, __COUNTER__) {                    \
    REGISTER_DML_COMPOSITE_BINARY_STRUCT(opName, expression, max_dim_count) \
    REGISTER_OP_KERNEL(opName, T1);                                         \
    REGISTER_OP_KERNEL(opName, T2);                                         \
    REGISTER_OP_KERNEL(opName, T3);                                         \
  }

#define REGISTER_DML_COMPOSITE_BINARY_KERNEL_4(opName, expression,            \
                                               max_dim_count, T1, T2, T3, T4) \
  namespace CONCAT_NAMESPACE_NAME(opName, __COUNTER__) {                      \
    REGISTER_DML_COMPOSITE_BINARY_STRUCT(opName, expression, max_dim_count)   \
    REGISTER_OP_KERNEL(opName, T1);                                           \
    REGISTER_OP_KERNEL(opName, T2);                                           \
    REGISTER_OP_KERNEL(opName, T3);                                           \
    REGISTER_OP_KERNEL(opName, T4);                                           \
  }

#define REGISTER_DML_COMPOSITE_BINARY_KERNEL_5(                             \
    opName, expression, max_dim_count, T1, T2, T3, T4, T5)                  \
  namespace CONCAT_NAMESPACE_NAME(opName, __COUNTER__) {                    \
    REGISTER_DML_COMPOSITE_BINARY_STRUCT(opName, expression, max_dim_count) \
    REGISTER_OP_KERNEL(opName, T1);                                         \
    REGISTER_OP_KERNEL(opName, T2);                                         \
    REGISTER_OP_KERNEL(opName, T3);                                         \
    REGISTER_OP_KERNEL(opName, T4);                                         \
    REGISTER_OP_KERNEL(opName, T5);                                         \
  }

#define REGISTER_DML_COMPOSITE_BINARY_KERNEL_6(                             \
    opName, expression, max_dim_count, T1, T2, T3, T4, T5, T6)              \
  namespace CONCAT_NAMESPACE_NAME(opName, __COUNTER__) {                    \
    REGISTER_DML_COMPOSITE_BINARY_STRUCT(opName, expression, max_dim_count) \
    REGISTER_OP_KERNEL(opName, T1);                                         \
    REGISTER_OP_KERNEL(opName, T2);                                         \
    REGISTER_OP_KERNEL(opName, T3);                                         \
    REGISTER_OP_KERNEL(opName, T4);                                         \
    REGISTER_OP_KERNEL(opName, T5);                                         \
    REGISTER_OP_KERNEL(opName, T6);                                         \
  }

#define REGISTER_DML_COMPOSITE_BINARY_KERNEL_7(                             \
    opName, expression, max_dim_count, T1, T2, T3, T4, T5, T6, T7)          \
  namespace CONCAT_NAMESPACE_NAME(opName, __COUNTER__) {                    \
    REGISTER_DML_COMPOSITE_BINARY_STRUCT(opName, expression, max_dim_count) \
    REGISTER_OP_KERNEL(opName, T1);                                         \
    REGISTER_OP_KERNEL(opName, T2);                                         \
    REGISTER_OP_KERNEL(opName, T3);                                         \
    REGISTER_OP_KERNEL(opName, T4);                                         \
    REGISTER_OP_KERNEL(opName, T5);                                         \
    REGISTER_OP_KERNEL(opName, T6);                                         \
    REGISTER_OP_KERNEL(opName, T7);                                         \
  }

#define REGISTER_DML_COMPOSITE_BINARY_KERNEL_8(                             \
    opName, expression, max_dim_count, T1, T2, T3, T4, T5, T6, T7, T8)      \
  namespace CONCAT_NAMESPACE_NAME(opName, __COUNTER__) {                    \
    REGISTER_DML_COMPOSITE_BINARY_STRUCT(opName, expression, max_dim_count) \
    REGISTER_OP_KERNEL(opName, T1);                                         \
    REGISTER_OP_KERNEL(opName, T2);                                         \
    REGISTER_OP_KERNEL(opName, T3);                                         \
    REGISTER_OP_KERNEL(opName, T4);                                         \
    REGISTER_OP_KERNEL(opName, T5);                                         \
    REGISTER_OP_KERNEL(opName, T6);                                         \
    REGISTER_OP_KERNEL(opName, T7);                                         \
    REGISTER_OP_KERNEL(opName, T8);                                         \
  }

REGISTER_DML_COMPOSITE_BINARY_KERNEL_1(Atan2, dml::ATanYX(x, y), 8, float)
REGISTER_DML_COMPOSITE_BINARY_KERNEL_2(RealDiv, x / y, 8, Eigen::half, float)
// TODO: Register this operator for all types (except int32) when FloorDiv is
// added to DML. dml::Floor(x / y) works for float types, but it doesn't work
// for integer types since DML truncates towards zero.
// TFDML #25977645
// TODO(b/25387198): A special kernel exists for int32 (see
// cwise_op_floor_div.cc).
REGISTER_DML_COMPOSITE_BINARY_KERNEL_2(FloorDiv, dml::Floor(x / y), 8,
                                       Eigen::half, float)
REGISTER_DML_COMPOSITE_BINARY_KERNEL_2(SigmoidGrad, (y * x * (1 - x)), 8,
                                       Eigen::half, float)
REGISTER_DML_COMPOSITE_BINARY_KERNEL_2(TanhGrad, (y * (1 - x * x)), 8,
                                       Eigen::half, float)
REGISTER_DML_COMPOSITE_BINARY_KERNEL_2(SqrtGrad, (y * 0.5f / x), 8, Eigen::half,
                                       float)
REGISTER_DML_COMPOSITE_BINARY_KERNEL_2(RsqrtGrad, (y * (-0.5f * x) * (x * x)),
                                       8, Eigen::half, float)
REGISTER_DML_COMPOSITE_BINARY_KERNEL_2(ReciprocalGrad, (-y * x * x), 8,
                                       Eigen::half, float)
REGISTER_DML_COMPOSITE_BINARY_KERNEL_2(InvGrad, (-y * x * x), 8, Eigen::half,
                                       float)
REGISTER_DML_COMPOSITE_BINARY_KERNEL_2(SoftplusGrad, x / (dml::Exp(-y) + 1), 8,
                                       Eigen::half, float)
// softsigngrad(gradients, features) = gradients / (1 + abs(features)) ** 2
REGISTER_DML_COMPOSITE_BINARY_KERNEL_2(SoftsignGrad,
                                       x / dml::Pow(1 + dml::Abs(y), 2), 8,
                                       Eigen::half, float)
// TODO(b/25387198): A special kernel exists for int32 (see cwise_op_sub.cc).
REGISTER_DML_COMPOSITE_BINARY_KERNEL_3(Sub, x - y, 8, Eigen::half, float, int64)
// TODO(b/25387198): A special kernel exists for int32 (see
// cwise_op_minimum.cc).
REGISTER_DML_COMPOSITE_BINARY_KERNEL_3(Minimum, dml::Min(x, y), 8, Eigen::half,
                                       float, int64)
// TODO(b/25387198): A special kernel exists for int32 (see
// cwise_op_maximum.cc).
REGISTER_DML_COMPOSITE_BINARY_KERNEL_3(Maximum, dml::Max(x, y), 8, Eigen::half,
                                       float, int64)
REGISTER_DML_COMPOSITE_BINARY_KERNEL_3(SquaredDifference,
                                       dml::DifferenceSquare(x, y), 8,
                                       Eigen::half, float, int64)
REGISTER_DML_COMPOSITE_BINARY_KERNEL_3(Mul, (x * y), 8, Eigen::half, float,
                                       int64)
REGISTER_DML_COMPOSITE_BINARY_KERNEL_3(Pow, dml::Pow(x, y), 8, Eigen::half,
                                       float, int64)
// TODO(b/25387198): A special kernel exists for int32 (see cwise_op_add1.cc).
REGISTER_DML_COMPOSITE_BINARY_KERNEL_4(Add, x + y, 8, Eigen::half, float, uint8,
                                       int64)
// TODO(b/25387198): A special kernel exists for int32 (see cwise_op_add1.cc).
REGISTER_DML_COMPOSITE_BINARY_KERNEL_4(AddV2, x + y, 8, Eigen::half, float,
                                       uint8, int64)
REGISTER_DML_COMPOSITE_BINARY_KERNEL_4(TruncateDiv, x / y, 8, uint8, uint16,
                                       int16, int64)
// TODO(b/25387198): A special kernel exists for int32 (see cwise_op_div.cc).
REGISTER_DML_COMPOSITE_BINARY_KERNEL_6(Div, x / y, 8, Eigen::half, float, uint8,
                                       uint16, int16, int64)
// TODO(b/25387198): A special kernel exists for int32 (see
// cwise_op_greater.cc).
REGISTER_DML_COMPOSITE_BINARY_KERNEL_6(Greater, x > y, 8, Eigen::half, float,
                                       int64, uint8, int8, int16)
REGISTER_DML_COMPOSITE_BINARY_KERNEL_6(Less, x < y, 8, Eigen::half, float,
                                       int64, uint8, int8, int16)
// TODO(b/25387198): A special kernel exists for int32 (see
// cwise_op_less_equal.cc).
REGISTER_DML_COMPOSITE_BINARY_KERNEL_6(LessEqual, x <= y, 8, float, Eigen::half,
                                       int64, uint8, int8, int16)
// TODO(b/25387198): A special kernel exists for int32 (see
// cwise_op_greater_equal.cc).
REGISTER_DML_COMPOSITE_BINARY_KERNEL_6(GreaterEqual, x >= y, 8, float,
                                       Eigen::half, int64, uint8, int8, int16)
// TODO(b/25387198): A special kernel exists for int32 (see
// cwise_op_equal_to_1.cc).
REGISTER_DML_COMPOSITE_BINARY_KERNEL_7(Equal, x == y, 8, Eigen::half, float,
                                       uint8, int8, int16, int64, bool)
// TODO(b/25387198): A special kernel exists for int32 (see
// cwise_op_not_equal_to_1.cc).
REGISTER_DML_COMPOSITE_BINARY_KERNEL_7(NotEqual, x != y, 8, Eigen::half, float,
                                       uint8, int8, int16, int64, bool)
// TODO(b/25387198): A special kernel exists for int32 (see cwise_op_mul1.cc).
REGISTER_DML_COMPOSITE_BINARY_KERNEL_1(
    Mul,
    dml::Cast(dml::Cast(x, DML_TENSOR_DATA_TYPE_UINT32) *
                  dml::Cast(y, DML_TENSOR_DATA_TYPE_UINT32),
              DML_TENSOR_DATA_TYPE_UINT8),
    8, uint8)
REGISTER_DML_COMPOSITE_BINARY_KERNEL_1(
    Mul,
    dml::Cast(dml::Cast(x, DML_TENSOR_DATA_TYPE_INT32) *
                  dml::Cast(y, DML_TENSOR_DATA_TYPE_INT32),
              DML_TENSOR_DATA_TYPE_INT8),
    8, int8)
REGISTER_DML_COMPOSITE_BINARY_KERNEL_1(
    Mul,
    dml::Cast(dml::Cast(x, DML_TENSOR_DATA_TYPE_UINT32) *
                  dml::Cast(y, DML_TENSOR_DATA_TYPE_UINT32),
              DML_TENSOR_DATA_TYPE_UINT16),
    8, uint16)
REGISTER_DML_COMPOSITE_BINARY_KERNEL_1(
    Mul,
    dml::Cast(dml::Cast(x, DML_TENSOR_DATA_TYPE_INT32) *
                  dml::Cast(y, DML_TENSOR_DATA_TYPE_INT32),
              DML_TENSOR_DATA_TYPE_INT16),
    8, int16)
// TODO(b/25387198): A special kernel exists for int32 (see
// cwise_op_less.cc).
REGISTER_DML_COMPOSITE_UNARY_KERNEL_1(Acos, dml::ACos(x), float)
REGISTER_DML_COMPOSITE_UNARY_KERNEL_1(Acosh, dml::ACosh(x), float)
REGISTER_DML_COMPOSITE_UNARY_KERNEL_1(Asin, dml::ASin(x), float)
REGISTER_DML_COMPOSITE_UNARY_KERNEL_1(Asinh, dml::ASinh(x), float)
REGISTER_DML_COMPOSITE_UNARY_KERNEL_1(Atan, dml::ATan(x), float)
REGISTER_DML_COMPOSITE_UNARY_KERNEL_1(Atanh, dml::ATanh(x), float)
REGISTER_DML_COMPOSITE_UNARY_KERNEL_1(Cosh, dml::Cosh(x), float)
REGISTER_DML_COMPOSITE_UNARY_KERNEL_1(Sinh, dml::Sinh(x), float)
REGISTER_DML_COMPOSITE_UNARY_KERNEL_1(Rint, dml::Round(x), float)
// TFDML #24881131
REGISTER_DML_COMPOSITE_UNARY_KERNEL_1(
    Inv, dml::Recip(dml::Cast(x, DML_TENSOR_DATA_TYPE_FLOAT32)), int64)
// TFDML #24881131
REGISTER_DML_COMPOSITE_UNARY_KERNEL_1(
    Reciprocal, dml::Recip(dml::Cast(x, DML_TENSOR_DATA_TYPE_FLOAT32)), int64)
// TFDML #24881131
REGISTER_DML_COMPOSITE_UNARY_KERNEL_2(Relu6, dml::Clip(x, 0, 6), Eigen::half,
                                      float)
REGISTER_DML_COMPOSITE_UNARY_KERNEL_2(Ceil, dml::Ceil(x), Eigen::half, float)
REGISTER_DML_COMPOSITE_UNARY_KERNEL_2(Cos, dml::Cos(x), Eigen::half, float)
REGISTER_DML_COMPOSITE_UNARY_KERNEL_2(Elu, dml::ActivationElu(x), Eigen::half,
                                      float)
REGISTER_DML_COMPOSITE_UNARY_KERNEL_2(Exp, dml::Exp(x), Eigen::half, float)
REGISTER_DML_COMPOSITE_UNARY_KERNEL_2(Floor, dml::Floor(x), Eigen::half, float)
REGISTER_DML_COMPOSITE_UNARY_KERNEL_2(Inv, dml::Recip(x), Eigen::half, float)
REGISTER_DML_COMPOSITE_UNARY_KERNEL_2(Reciprocal, dml::Recip(x), Eigen::half,
                                      float)
REGISTER_DML_COMPOSITE_UNARY_KERNEL_2(IsInf, dml::IsInfinity(x), Eigen::half,
                                      float)
REGISTER_DML_COMPOSITE_UNARY_KERNEL_2(Log, dml::Log(x), Eigen::half, float)
REGISTER_DML_COMPOSITE_UNARY_KERNEL_2(Log1p, dml::Log(x, DML_SCALE_BIAS{1, 1}),
                                      Eigen::half, float)
REGISTER_DML_COMPOSITE_UNARY_KERNEL_2(Sigmoid, dml::ActivationSigmoid(x),
                                      Eigen::half, float)
REGISTER_DML_COMPOSITE_UNARY_KERNEL_2(Sin, dml::Sin(x), Eigen::half, float)
REGISTER_DML_COMPOSITE_UNARY_KERNEL_2(Softsign, dml::ActivationSoftsign(x),
                                      Eigen::half, float)
REGISTER_DML_COMPOSITE_UNARY_KERNEL_2(Sqrt, dml::Sqrt(x), Eigen::half, float)
REGISTER_DML_COMPOSITE_UNARY_KERNEL_2(Tan, dml::Tan(x), Eigen::half, float)
REGISTER_DML_COMPOSITE_UNARY_KERNEL_2(Tanh, dml::Tanh(x), Eigen::half, float)
REGISTER_DML_COMPOSITE_UNARY_KERNEL_2(Erf, dml::Erf(x), Eigen::half, float)
REGISTER_DML_COMPOSITE_UNARY_KERNEL_2(IsNan, dml::IsNaN(x), Eigen::half, float)
REGISTER_DML_COMPOSITE_UNARY_KERNEL_2(Round, dml::Round(x), Eigen::half, float)
REGISTER_DML_COMPOSITE_UNARY_KERNEL_2(Round, dml::Identity(x), int32, int64)
REGISTER_DML_COMPOSITE_UNARY_KERNEL_2(Softplus, dml::ActivationSoftplus(x),
                                      Eigen::half, float)
REGISTER_DML_COMPOSITE_UNARY_KERNEL_2(Erfc, 1.0f - dml::Erf(x), Eigen::half,
                                      float)
REGISTER_DML_COMPOSITE_UNARY_KERNEL_2(Rsqrt, 1.0f / dml::Sqrt(x), Eigen::half,
                                      float)
REGISTER_DML_COMPOSITE_UNARY_KERNEL_2(Expm1, dml::Exp(x) - 1.0f, Eigen::half,
                                      float)
REGISTER_DML_COMPOSITE_UNARY_KERNEL_2(IsFinite,
                                      !(dml::IsNaN(x) || dml::IsInfinity(x)),
                                      Eigen::half, float)
// TODO(b/25387198): A special kernel exists for int32 (see cwise_op_abs.cc).
REGISTER_DML_COMPOSITE_UNARY_KERNEL_3(Abs, dml::Abs(x), Eigen::half, float,
                                      int64)
// TODO(b/25387198): A special kernel exists for int32 (see cwise_op_sign.cc).
REGISTER_DML_COMPOSITE_UNARY_KERNEL_3(Sign, dml::Sign(x), Eigen::half, float,
                                      int64)
// TODO(b/25387198): A special kernel exists for int32 (see cwise_op_neg.cc).
REGISTER_DML_COMPOSITE_UNARY_KERNEL_3(Neg, -x, Eigen::half, float, int64)
// TODO(b/25387198): A special kernel exists for int32 (see cwise_op_square.cc).
REGISTER_DML_COMPOSITE_UNARY_KERNEL_3(Square, (x * x), Eigen::half, float,
                                      int64)
REGISTER_DML_COMPOSITE_UNARY_BOOL_KERNEL(LogicalNot, dml::LogicalNot(x))
REGISTER_DML_COMPOSITE_BINARY_BOOL_KERNEL(LogicalAnd, dml::LogicalAnd(x, y), 8)
REGISTER_DML_COMPOSITE_BINARY_BOOL_KERNEL(LogicalOr, dml::LogicalOr(x, y), 8)
REGISTER_DML_FLOAT_OP_KERNEL(Softmax, DmlMaxActivationKernel,
                             DML_OPERATOR_ACTIVATION_SOFTMAX,
                             DML_ACTIVATION_SOFTMAX_OPERATOR_DESC)
REGISTER_DML_FLOAT_OP_KERNEL(LogSoftmax, DmlMaxActivationKernel,
                             DML_OPERATOR_ACTIVATION_LOG_SOFTMAX,
                             DML_ACTIVATION_LOG_SOFTMAX_OPERATOR_DESC)

#undef REGISTER_OP_KERNEL
#undef REGISTER_DML_FLOAT_OP_KERNEL
#undef REGISTER_BOOL_OP_KERNEL
#undef REGISTER_DML_COMPOSITE_UNARY_STRUCT
#undef REGISTER_DML_COMPOSITE_UNARY_BOOL_KERNEL
#undef REGISTER_DML_COMPOSITE_BINARY_STRUCT
#undef REGISTER_DML_COMPOSITE_BINARY_BOOL_KERNEL
#undef REGISTER_DML_COMPOSITE_UNARY_KERNEL_1
#undef REGISTER_DML_COMPOSITE_UNARY_KERNEL_2
#undef REGISTER_DML_COMPOSITE_UNARY_KERNEL_3
#undef REGISTER_DML_COMPOSITE_BINARY_KERNEL_1
#undef REGISTER_DML_COMPOSITE_BINARY_KERNEL_2
#undef REGISTER_DML_COMPOSITE_BINARY_KERNEL_3
#undef REGISTER_DML_COMPOSITE_BINARY_KERNEL_4
#undef REGISTER_DML_COMPOSITE_BINARY_KERNEL_5
#undef REGISTER_DML_COMPOSITE_BINARY_KERNEL_6
#undef REGISTER_DML_COMPOSITE_BINARY_KERNEL_7
#undef REGISTER_DML_COMPOSITE_BINARY_KERNEL_8

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

    // Min/max are supplied as tensors for ClipByValue, which are required to be
    // constant CPU inputs
    const Tensor& min_tensor = ctx->GetConstantInputTensor(1);
    const Tensor& max_tensor = ctx->GetConstantInputTensor(2);

    auto scope = dml::Graph(ctx->GetDmlDevice());
    auto input = dml::InputTensor(scope, 0, inputs[0]);
    auto result = dml::Clip(input, min_tensor.flat<float>()(0),
                            max_tensor.flat<float>()(0));

    // TFDML #24881131
    if (Is64BitSignedIntegerType(ctx->GetOutputDataType(0))) {
      result = dml::ConvertInt32ToInt64(scope, result);
    }

    ComPtr<IDMLCompiledOperator> compiled_op =
        scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

    Initialize(ctx, std::move(tensors), compiled_op.Get());
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
TF_CALL_half(DML_REGISTER_KERNEL);
TF_CALL_float(DML_REGISTER_KERNEL);
TF_CALL_int8(DML_REGISTER_KERNEL);
TF_CALL_int16(DML_REGISTER_KERNEL);
TF_CALL_int64(DML_REGISTER_KERNEL);
TF_CALL_uint8(DML_REGISTER_KERNEL);
TF_CALL_uint16(DML_REGISTER_KERNEL);
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

    // TFDML #24881131
    const dml::TensorPolicy out_policy =
        Is64BitUnsignedIntegerType(ctx->GetOutputDataType(0))
            ? GetEmulatedInt64TensorPolicy()
            : dml::TensorPolicy::Default();

    auto scope = dml::Graph(ctx->GetDmlDevice(), out_policy);
    auto x = dml::InputTensor(scope, 0, inputs[0]);
    auto y = dml::InputTensor(scope, 1, inputs[1]);
    auto zero = dml::ZeroTensor(scope, x.GetOutputDesc().dataType,
                                x.GetOutputDesc().sizes);

    Functor f;
    auto result = f(zero, x, y);

    // TFDML #24881131
    if (Is64BitSignedIntegerType(ctx->GetOutputDataType(0))) {
      result = dml::ConvertInt32ToInt64(scope, result);
    }

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

    // TFDML #24881131
    if (Is64BitUnsignedIntegerType(output->dtype())) {
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
TF_CALL_DML_FLOAT_TYPES(DML_REGISTER_KERNEL);
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
TF_CALL_float(DML_REGISTER_KERNEL);
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
    // TFDML #24881131
    DataType dtype = ctx->GetInputDataType(0);
    DCHECK(dtype == ctx->GetOutputDataType(0));
    if (Is64BitIntegerType(dtype)) {
      num_elements *= 2;
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
  using InitHelper = ElementWiseInitHelper<8>;

  explicit DmlBinaryBitwiseKernel(DmlKernelConstruction* ctx,
                                  const InitHelper* init_helper) {
    CHECK(ctx->GetInputCount() == 2);
    CHECK(ctx->GetOutputCount() == 1);

    auto num_elements =
        static_cast<uint32_t>(ctx->GetInputTensorShape(0).num_elements());

    // DML doesn't support 64-bit integer types, but we can reinterpret
    // the tensor as twice as many 32-bit elements. Sign doesn't matter.
    // TFDML #24881131
    DataType dtype = ctx->GetInputDataType(0);
    DCHECK(dtype == ctx->GetOutputDataType(0));
    if (Is64BitIntegerType(dtype)) {
      num_elements *= 2;
    }

    std::array<uint32_t, 4> sizes = {1, 1, 1, num_elements};

    DmlTensorInfo in1;
    in1.kernel_index = 0;
    in1.desc = DmlTensorDesc::Create(dtype, sizes, sizes);
    in1.desc.ForceUnsignedDataType();
    auto in1_desc = in1.desc.GetDmlDesc();

    DmlTensorInfo in2;
    in2.kernel_index = 1;
    in2.desc = DmlTensorDesc::Create(dtype, sizes, sizes);
    in2.desc.ForceUnsignedDataType();
    auto in2_desc = in2.desc.GetDmlDesc();

    DmlTensorInfo out;
    out.kernel_index = 0;
    out.desc = DmlTensorDesc::Create(dtype, sizes, sizes);
    out.desc.ForceUnsignedDataType();
    auto out_desc = out.desc.GetDmlDesc();

    DmlKernelTensors tensors;
    tensors.inputs = {in1, in2};
    tensors.outputs = {out};

    auto input_shapes = init_helper->GetCollapsedInputShapes();
    const TensorShape& output_shape = init_helper->GetCollapsedOutputShape();

    DML_OPERATOR_SPECIFIC_DESC desc = {
        &in1_desc,
        &in2_desc,
        &out_desc,
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

    // TFDML #24881131
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
TF_CALL_int8(DML_REGISTER_KERNEL);
TF_CALL_uint16(DML_REGISTER_KERNEL);
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
TF_CALL_int8(DML_REGISTER_KERNEL);
TF_CALL_int16(DML_REGISTER_KERNEL);
TF_CALL_int32(DML_REGISTER_KERNEL);
TF_CALL_int64(DML_REGISTER_KERNEL);
TF_CALL_uint8(DML_REGISTER_KERNEL);
TF_CALL_uint16(DML_REGISTER_KERNEL);
TF_CALL_uint32(DML_REGISTER_KERNEL);
TF_CALL_uint64(DML_REGISTER_KERNEL);
#undef DML_REGISTER_KERNEL

#define DML_REGISTER_KERNEL(type)                                        \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("BitwiseOr").Device(DEVICE_DML).TypeConstraint<type>("T"),    \
      DmlKernelWrapper<                                                  \
          DmlBinaryBitwiseKernel<DML_OPERATOR_ELEMENT_WISE_BIT_OR,       \
                                 DML_ELEMENT_WISE_BIT_OR_OPERATOR_DESC>, \
          GetBroadcastedOutputShapeHelper>);
TF_CALL_int8(DML_REGISTER_KERNEL);
TF_CALL_int16(DML_REGISTER_KERNEL);
TF_CALL_int32(DML_REGISTER_KERNEL);
TF_CALL_int64(DML_REGISTER_KERNEL);
TF_CALL_uint8(DML_REGISTER_KERNEL);
TF_CALL_uint16(DML_REGISTER_KERNEL);
TF_CALL_uint32(DML_REGISTER_KERNEL);
TF_CALL_uint64(DML_REGISTER_KERNEL);
#undef DML_REGISTER_KERNEL

#define DML_REGISTER_KERNEL(type)                                         \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("BitwiseXor").Device(DEVICE_DML).TypeConstraint<type>("T"),    \
      DmlKernelWrapper<                                                   \
          DmlBinaryBitwiseKernel<DML_OPERATOR_ELEMENT_WISE_BIT_XOR,       \
                                 DML_ELEMENT_WISE_BIT_XOR_OPERATOR_DESC>, \
          GetBroadcastedOutputShapeHelper>);
TF_CALL_int8(DML_REGISTER_KERNEL);
TF_CALL_int16(DML_REGISTER_KERNEL);
TF_CALL_int32(DML_REGISTER_KERNEL);
TF_CALL_int64(DML_REGISTER_KERNEL);
TF_CALL_uint8(DML_REGISTER_KERNEL);
TF_CALL_uint16(DML_REGISTER_KERNEL);
TF_CALL_uint32(DML_REGISTER_KERNEL);
TF_CALL_uint64(DML_REGISTER_KERNEL);
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