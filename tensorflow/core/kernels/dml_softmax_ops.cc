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

class SoftmaxXentWithLogitsShapeHelper : public ShapeHelper {
 public:
  std::vector<TensorShape> GetOutputShapes(
      OpKernelContext* ctx,
      const InitializationHelper* initialization_helper) const override {
    const Tensor& logits = ctx->input(0);
    const Tensor& labels = ctx->input(1);
    // logits must have the same shape as labels
    CHECK(logits.shape() == labels.shape());

    TensorShape outputShape({logits.dim_size(0), 1});

    return {outputShape, logits.shape()};
  }
};

class DmlSoftmaxXentWithLogitsInitHelper : public InitializationHelper {
 public:
  using Attributes = EmptyAttributes;

  DmlSoftmaxXentWithLogitsInitHelper(OpKernelContext* ctx,
                                     std::shared_ptr<const Attributes> attr) {
    const Tensor& logits_in = ctx->input(0);
    const Tensor& labels_in = ctx->input(1);
    TensorShape shape_in = logits_in.shape();

    OP_REQUIRES(ctx, logits_in.IsSameSize(labels_in),
                errors::InvalidArgument(
                    "logits and labels must be broadcastable: logits_size=",
                    logits_in.shape().DebugString(),
                    " labels_size=", labels_in.shape().DebugString()));

    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(shape_in),
                errors::InvalidArgument("logits and labels must be either "
                                        "2-dimensional, or broadcasted to be "
                                        "2-dimensional"));
  }
};

class DmlSoftmaxXentWithLogitsKernel : public DmlKernel {
 public:
  using InitHelper = DmlSoftmaxXentWithLogitsInitHelper;
  explicit DmlSoftmaxXentWithLogitsKernel(
      DmlKernelConstruction* ctx,
      const DmlSoftmaxXentWithLogitsInitHelper* init_helper) {
    CHECK(ctx->GetInputCount() == 2);
    CHECK(ctx->GetOutputCount() == 2);

    DmlKernelParams params;
    params.kernel_input_indices = {0, 1};
    params.kernel_output_indices = {0, 1};

    DmlKernelTensors tensors = GetTensorInfos(ctx, params);
    auto input_descs = GetDmlTensorDescs(tensors.inputs);

    auto scope = dml::Scope(ctx->GetDmlDevice());

    // logits: batch_size, num_classes.
    // labels: batch_size, num_classes.
    auto logits = dml::InputTensor(scope, 0, input_descs[0]);
    auto labels = dml::InputTensor(scope, 1, input_descs[1]);

    auto input_sizes = labels.GetOutputDesc().sizes;

    // The strides we need to set to broadcast class across an entire tensor
    dml::TensorDesc::Dimensions broadcast_c_strides = {/*NoSense*/ 0,
                                                       /*NoSense*/ 0,
                                                       /*kBatchDim*/ 1,
                                                       /*kClassDim*/ 0};

    // The strides we need to set to broadcast class across an entire tensor
    dml::TensorDesc::Dimensions reduce_strides = {0, 1, 3};

    // max_logits along classes.
    auto logits_max =
        dml::Reduce(logits, DML_REDUCE_FUNCTION_MAX, reduce_strides);
    auto logits_max_bcast =
        dml::Reinterpret(logits_max, input_sizes, broadcast_c_strides);

    // logits - max_logits.
    auto shifted_logits = logits - logits_max_bcast;

    // exp(logits - max_logits)
    auto exp_shifted_logits = dml::Exp(shifted_logits);

    // sum(exp(logits - max_logits)) along classes.
    auto sum_exp = dml::Reduce(exp_shifted_logits, DML_REDUCE_FUNCTION_SUM,
                               reduce_strides);

    // log(sum(exp(logits - max_logits)))
    auto log_sum_exp = dml::Log(sum_exp);
    auto log_sum_exp_bcast =
        dml::Reinterpret(log_sum_exp, input_sizes, broadcast_c_strides);

    // sum(-labels *
    //    ((logits - max_logits) - log(sum(exp(logits - max_logits)))))
    auto sub = log_sum_exp_bcast - shifted_logits;
    auto mul = labels * sub;
    auto loss = dml::Reduce(mul, DML_REDUCE_FUNCTION_SUM, reduce_strides);

    // backprop: prob - labels, where
    //   prob = exp(logits - max_logits) / sum(exp(logits - max_logits))
    //     (where the division broadcasts along the batch dimension)
    auto sum_exp_bcast =
        dml::Reinterpret(sum_exp, input_sizes, broadcast_c_strides);
    auto backprop = exp_shifted_logits / sum_exp_bcast - labels;

    // loss: output tensor for the loss, dims: batch_size.
    // backprop: output tensor for the backprop, dims: batch_size, num_classes.
    auto outputs = {loss, backprop};

    auto compiled_op = scope.Compile(DML_EXECUTION_FLAG_NONE, outputs);
    Initialize(ctx, std::move(tensors), compiled_op.Get());
  }
};

#define DML_REGISTER_KERNEL(type)                                          \
  REGISTER_KERNEL_BUILDER(Name("SoftmaxCrossEntropyWithLogits")            \
                              .Device(DEVICE_DML)                          \
                              .TypeConstraint<type>("T"),                  \
                          DmlKernelWrapper<DmlSoftmaxXentWithLogitsKernel, \
                                           SoftmaxXentWithLogitsShapeHelper>);
TF_CALL_DML_FLOAT_TYPES(DML_REGISTER_KERNEL);
#undef DML_REGISTER_KERNEL
}  // namespace tensorflow