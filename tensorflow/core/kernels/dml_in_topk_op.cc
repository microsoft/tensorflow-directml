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

class DmlInTopKInitHelper : public InitializationHelper {
 public:
  using Attributes = EmptyAttributes;

  DmlInTopKInitHelper(OpKernelContext* ctx,
                      std::shared_ptr<const Attributes> attr) {
    const auto& predictions_in = ctx->input(0);
    const auto& targets_in = ctx->input(1);
    const auto& k_in = ctx->input(2);

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(k_in.shape()),
                errors::InvalidArgument("k must be 0-D, got shape ",
                                        k_in.shape().DebugString()));
    OP_REQUIRES(ctx, predictions_in.dims() == 2,
                errors::InvalidArgument("predictions must be 2-dimensional"));
    OP_REQUIRES(ctx, targets_in.dims() == 1,
                errors::InvalidArgument("targets must be 1-dimensional"));
    OP_REQUIRES(ctx, predictions_in.dim_size(0) == targets_in.dim_size(0),
                errors::InvalidArgument("First dimension of predictions ",
                                        predictions_in.dim_size(0),
                                        " must match length of targets ",
                                        targets_in.dim_size(0)));

    // TODO: Remove once K is moved into device memory
    if (k_in.dtype() == DT_INT32) {
      k_ = k_in.scalar<int32_t>()();
    } else {
      assert(k_in.dtype() == DT_INT64);
      k_ = k_in.scalar<int32_t>()();
    }
  }
  int GetK() const { return k_; }

 private:
  int64_t k_ = -1;
};

class DmlInTopKKernel : public DmlKernel {
 public:
  using InitHelper = DmlInTopKInitHelper;
  explicit DmlInTopKKernel(DmlKernelConstruction* ctx,
                           const InitHelper* init_helper) {
    DmlTensorInfo predictions_info;
    predictions_info.kernel_index = 0;
    predictions_info.desc = DmlTensorDesc::Create(ctx->GetInputDataType(0),
                                                  ctx->GetInputTensorShape(0),
                                                  ctx->GetInputTensorShape(0));

    DmlTensorInfo targets_info;
    targets_info.kernel_index = 1;
    targets_info.desc = DmlTensorDesc::Create(ctx->GetInputDataType(1),
                                              ctx->GetInputTensorShape(1),
                                              ctx->GetInputTensorShape(1));

    DmlTensorInfo output_info;
    output_info.kernel_index = 0;
    output_info.desc = DmlTensorDesc::Create(ctx->GetOutputDataType(0),
                                             ctx->GetOutputTensorShape(0),
                                             ctx->GetOutputTensorShape(0));
    output_info.desc.ForceUnsignedDataType();

    DmlKernelTensors tensors;
    tensors.inputs = {predictions_info, targets_info};
    tensors.outputs = {output_info};

    int64_t k = init_helper->GetK();

    // When K is smaller than 1, none of the targets are in the top K
    if (k < 1) {
      all_false_ = true;
      InitializeAsNoOp(ctx);
      return;
    }

    auto inputs = GetDmlTensorDescs(tensors.inputs);
    auto scope = dml::Graph(ctx->GetDmlDevice());
    auto predictions = dml::InputTensor(scope, 0, inputs[0]);
    auto targets = dml::InputTensor(scope, 1, inputs[1]);

    uint32_t axis = predictions_info.desc.GetDimensionCount() - 1;
    dml::TopKOutputs topk_result =
        dml::TopK(predictions, axis, k, DML_AXIS_DIRECTION_DECREASING);

    uint32_t num_classes = ctx->GetInputTensorShape(0).dim_size(1);
    dml::Expression classes;
    dml::Expression num_classes_tensor;
    dml::Expression zero_tensor;
    if (ctx->GetInputDataType(1) == DT_INT32) {
      classes = dml::Sequence<int32_t>(scope, 0, 1, {1, 1, 1, num_classes});
      num_classes_tensor = dml::ScalarTensor<int32_t>(
          scope, num_classes, targets.GetOutputDesc().sizes);
      zero_tensor =
          dml::ScalarTensor<int32_t>(scope, 0, targets.GetOutputDesc().sizes);
    } else {
      assert(ctx->GetInputDataType(1) == DT_INT64);
      classes = dml::Sequence<int64_t>(scope, 0, 1, {1, 1, 1, num_classes});
      num_classes_tensor = dml::ScalarTensor<int64_t>(
          scope, num_classes, targets.GetOutputDesc().sizes);
      zero_tensor =
          dml::ScalarTensor<int64_t>(scope, 0, targets.GetOutputDesc().sizes);
    }

    // Broadcast the classes list to the number of targets
    auto sparse_classes =
        dml::Reinterpret(classes, predictions.GetOutputDesc().sizes,
                         dml::TensorStrides({0, 0, 0, 1}));

    auto zero_float_tensor =
        dml::ScalarTensor<float>(scope, 0.0, predictions.GetOutputDesc().sizes);

    auto sparse_targets =
        dml::Reinterpret(targets, predictions.GetOutputDesc().sizes,
                         dml::TensorStrides({0, 0, 1, 0}));

    // Create a sparse predictions matrix where the value of the prediction
    // is set to 0 when the column doesn't match the target
    auto sparse_predictions = dml::If(sparse_targets == sparse_classes,
                                      predictions, zero_float_tensor);

    auto class_prediction_per_batch =
        dml::Reduce(sparse_predictions, DML_REDUCE_FUNCTION_SUM, {3});

    class_prediction_per_batch = dml::Reinterpret(
        class_prediction_per_batch, targets.GetOutputDesc().sizes,
        dml::TensorStrides({0, 0, 0, 1}));

    // To handle ties, we check to see if the class prediction is greater or
    // equal to the lowest prediction obtained from the TopK result
    auto lowest_topk_value_per_batch =
        k > 1 ? dml::Split(topk_result.value, 3,
                           {static_cast<uint32_t>(k) - 1u, 1u})[1]
              : topk_result.value;

    // Transpose lowest_topk_value_per_batch from a column to a row in order
    // to match the shape of class_prediction_per_batch
    lowest_topk_value_per_batch =
        dml::Reinterpret(lowest_topk_value_per_batch,
                         class_prediction_per_batch.GetOutputDesc().sizes,
                         dml::TensorStrides({0, 0, 0, 1}));

    // Out of bounds and "inf" targets always yield "false"
    auto result = targets >= zero_tensor && targets < num_classes_tensor &&
                  class_prediction_per_batch >= lowest_topk_value_per_batch &&
                  !dml::IsInfinity(class_prediction_per_batch);

    Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
        scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

    Initialize(ctx, std::move(tensors), compiled_op.Get());
  }

  StatusOr<DmlGpuEvent> Compute(DmlKernelContext* ctx) const override {
    if (all_false_) {
      Tensor& output = ctx->GetOutputTensor(0);
      return ctx->GetDmlDeviceContext()->ZeroBuffer(
          ctx->GetDmlDeviceContext()->GetBufferForTensor(output));
    }

    return DmlKernel::Compute(ctx);
  }

  bool all_false_ = false;
};

// TODO: Remove k pinning to host once we have an InTopKV2 DML kernel and K is
// moved into device memory
REGISTER_KERNEL_BUILDER(
    Name("InTopKV2")
        .Device(DEVICE_DML)
        .TypeConstraint<int32>("T")
        .HostMemory("k"),
    DmlKernelWrapper<DmlInTopKKernel, GetOutputShapeFromInputShapeHelper<1>>);
REGISTER_KERNEL_BUILDER(
    Name("InTopKV2")
        .Device(DEVICE_DML)
        .TypeConstraint<int64>("T")
        .HostMemory("k"),
    DmlKernelWrapper<DmlInTopKKernel, GetOutputShapeFromInputShapeHelper<1>>);

}  // namespace tensorflow