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

namespace tensorflow {

using Microsoft::WRL::ComPtr;

class SparseXentInitializationHelper : public InitializationHelper {
 public:
  using Attributes = EmptyAttributes;

  bool IsNoOpKernel(
      OpKernelContext* ctx,
      absl::Span<const TensorShape> output_shapes) const override {
    if (!InitializationHelper::IsNoOpKernel(ctx, output_shapes)) {
      const Tensor& logits = ctx->input(0);
      return (logits.dim_size(0) == 0);
    }
    return true;
  }

  SparseXentInitializationHelper(OpKernelContext* ctx,
                                 std::shared_ptr<const Attributes> attr) {
    const Tensor& logits = ctx->input(0);
    const Tensor& labels = ctx->input(1);
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(logits.shape()),
                errors::InvalidArgument("logits must be 2-D, but got shape ",
                                        logits.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(labels.shape()),
                errors::InvalidArgument("labels must be 1-D, but got shape ",
                                        labels.shape().DebugString()));
    OP_REQUIRES(ctx, logits.dim_size(0) == labels.dim_size(0),
                errors::InvalidArgument(
                    "logits and labels must have the same first dimension, "
                    "got logits shape ",
                    logits.shape().DebugString(), " and labels shape ",
                    labels.shape().DebugString()));
    OP_REQUIRES(ctx, logits.dim_size(1) > 0,
                errors::InvalidArgument(
                    "Must have at least one class, but got logits shape ",
                    logits.shape().DebugString()));
  }
};

template <typename T>
struct SparseXentTraits;

template <>
struct SparseXentTraits<float> {
  static inline DML_SCALAR_UNION GetZeroValue() {
    return {};  // Zero
  }

  static inline DML_SCALAR_UNION GetOneValue() {
    return TfTensorTypeTraits<float>::ToDmlScalar(1.0f);
  }

  static inline float GetEpsilonValue() { return 1e-20f; }
};

template <>
struct SparseXentTraits<Eigen::half> {
  static inline DML_SCALAR_UNION GetZeroValue() {
    return {};  // Zero
  }

  static inline DML_SCALAR_UNION GetOneValue() {
    return TfTensorTypeTraits<Eigen::half>::ToDmlScalar(
        TfTensorTypeTraits<Eigen::half>::FromFloat(1.0f));
  }

  static inline Eigen::half GetEpsilonValue() {
    return TfTensorTypeTraits<Eigen::half>::FromFloat(1e-6f);
  }
};

template <typename T, typename TIndex>
class DmlSparseXentKernel : public DmlKernel {
 public:
  using InitHelper = SparseXentInitializationHelper;

  explicit DmlSparseXentKernel(DmlKernelConstruction* ctx,
                               const InitHelper* init_helper) {
    CHECK(ctx->GetInputCount() == 2);
    CHECK(ctx->GetOutputCount() == 2);

    const TensorShape& logits_shape = ctx->GetInputTensorShape(0);
    const TensorShape& labels_shape = ctx->GetInputTensorShape(1);

    const uint32_t batch_size = logits_shape.dim_size(/*Batch Index*/ 0);
    const uint32_t num_classes = logits_shape.dim_size(/*Class Index*/ 1);

    DmlKernelParams params;
    DmlKernelTensors tensors = GetTensorInfos(ctx, params);

    // DML OneHot requires shape [1,1,batch_size,1].
    std::array<uint32_t, 4> labels_sizes = {1, 1, batch_size, 1};
    tensors.inputs[1]->desc = DmlTensorDesc::Create(ctx->GetInputDataType(1),
                                                    labels_sizes, labels_sizes);

    // Here we use reinterpret to change sparse label's type. TIndex type can
    // be one of int32 or int64 and we need to convert it to uint32.
    tensors.inputs[1]->desc.ForceUnsignedDataType();

    auto input_descs = GetDmlTensorDescs(tensors.inputs);
    auto output_descs = GetDmlTensorDescs(tensors.outputs);

    auto scope = dml::Graph(ctx->GetDmlDevice());
    auto logits = dml::InputTensor(scope, 0, input_descs[0]);
    auto sparse_labels = dml::InputTensor(scope, 1, input_descs[1]);

    auto logits_size = tensors.inputs[0]->desc.GetSizes();
    dml::TensorDesc::Dimensions logits_dimensions(logits_size.begin(),
                                                  logits_size.end());

    // DML onehot operator requires values tensor containing high and low values
    auto values = dml::FillValueSequence(
        scope, dml::TensorDesc::Dimensions{1, 1, 1, 2},
        TfTensorTypeTraits<T>::dml_type, SparseXentTraits<T>::GetZeroValue(),
        SparseXentTraits<T>::GetOneValue());

    auto epsilon = dml::ScalarTensor(
        scope, SparseXentTraits<T>::GetEpsilonValue(), logits_dimensions);

    auto probs = dml::ActivationSoftmax(logits);

    const uint32_t axis = 3;
    auto labels =
        dml::OneHot(sparse_labels, values, logits_dimensions[axis], axis);

    dml::Expression bp;
    if (num_classes > 1) {
      bp = probs - labels;
    } else {
      bp = probs - probs;
    }
    auto intermediate = labels * dml::Log(probs + epsilon);
    auto loss =
        -dml::Reduce(intermediate, DML_REDUCE_FUNCTION_SUM, /*Axis*/ {3});

    Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
        scope.Compile(DML_EXECUTION_FLAG_NONE, {loss, bp});

    Initialize(ctx, std::move(tensors), compiled_op.Get());
  }
};

#define DML_REGISTER_KERNEL(type, index_type)                 \
  REGISTER_KERNEL_BUILDER(                                    \
      Name("SparseSoftmaxCrossEntropyWithLogits")             \
          .Device(DEVICE_DML)                                 \
          .TypeConstraint<type>("T")                          \
          .TypeConstraint<index_type>("Tlabels"),             \
      DmlKernelWrapper<DmlSparseXentKernel<type, index_type>, \
                       SparseXentShapeHelper>);
DML_REGISTER_KERNEL(float, int32)
DML_REGISTER_KERNEL(float, int64)
DML_REGISTER_KERNEL(Eigen::half, int32)
DML_REGISTER_KERNEL(Eigen::half, int64)
#undef DML_REGISTER_KERNEL

}  // namespace tensorflow