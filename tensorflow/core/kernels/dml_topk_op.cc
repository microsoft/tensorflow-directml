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

class DmlTopKInitHelper : public InitializationHelper {
 public:
  struct Attributes {
    explicit Attributes(OpKernelConstruction* ctx) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("sorted", &sorted_));
      if (ctx->num_inputs() < 2) {  // k is an attr (TopK).
        OP_REQUIRES_OK(ctx, ctx->GetAttr("k", &k_));
      } else {  // k is an input (TopKV2), so we won't know it until Compute.
        k_ = -1;
      }
    }
    int k_;
    // sorted_ attr is not used because tensorflow has no specification for
    // sorted_=False
    bool sorted_;
  };

  DmlTopKInitHelper(OpKernelContext* ctx,
                    std::shared_ptr<const Attributes> attr)
      : attr_(std::move(attr)) {
    k = attr_->k_;
    if (ctx->num_inputs() >= 2) {
      const auto& k_in = ctx->input(1);
      OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(k_in.shape()),
                  errors::InvalidArgument("k must be scalar, got shape ",
                                          k_in.shape().DebugString()));
      k = k_in.scalar<int32>()();
    }
    OP_REQUIRES(ctx, k >= 0, errors::InvalidArgument("Need k >= 0, got ", k));
    const auto& input_in = ctx->input(0);
    OP_REQUIRES(ctx, input_in.dims() >= 1,
                errors::InvalidArgument("input must be >= 1-D, got shape ",
                                        input_in.shape().DebugString()));
    OP_REQUIRES(ctx, input_in.dim_size(input_in.dims() - 1) >= k,
                errors::InvalidArgument(
                    "input must have at least k columns. Had ",
                    input_in.dim_size(input_in.dims() - 1), ", needed ", k));
  }
  int GetK() const { return k; }

 private:
  int k = -1;
  const std::shared_ptr<const Attributes> attr_;
};

class TopKShapeHelper : public ShapeHelper {
 public:
  std::vector<TensorShape> GetOutputShapes(
      OpKernelContext* ctx,
      const InitializationHelper* initialization_helper) const override {
    auto init_helper =
        static_cast<const DmlTopKInitHelper*>(initialization_helper);
    int k = init_helper->GetK();
    const TensorShape& input_shape = ctx->input(0).shape();
    TensorShape output_shape;
    for (int i = 0; i < input_shape.dims() - 1; ++i) {
      output_shape.AddDim(input_shape.dim_size(i));
    }
    output_shape.AddDim(k);
    return {output_shape, output_shape};
  }
};

class DmlTopKKernel : public DmlKernel {
 public:
  using InitHelper = DmlTopKInitHelper;
  explicit DmlTopKKernel(DmlKernelConstruction* ctx,
                         const InitHelper* init_helper) {
    DCHECK(ctx->GetInputCount() == 1 || ctx->GetInputCount() == 2);
    DCHECK(ctx->GetOutputCount() == 2);

    const TensorShape& tensor_shape = ctx->GetInputTensorShape(0);
    const TensorShape& output_shape = ctx->GetOutputTensorShape(0);

    DmlKernelTensors tensors;
    DmlTensorInfo input;
    input.kernel_index = 0;
    input.desc = DmlTensorDesc::Create(ctx->GetInputDataType(0), tensor_shape,
                                       tensor_shape);

    tensors.inputs = {input};
    for (uint32_t i = 0; i < ctx->GetOutputCount(); ++i) {
      DmlTensorInfo output;
      output.kernel_index = i;
      output.desc = DmlTensorDesc::Create(ctx->GetOutputDataType(i),
                                          output_shape, output_shape);
      if (i == 1) {
        output.desc.ForceUnsignedDataType();
      }
      tensors.outputs.push_back(std::move(output));
    }

    auto inputs = GetDmlTensorDescs(tensors.inputs);
    auto outputs = GetDmlTensorDescs(tensors.outputs);

    uint32_t k = init_helper->GetK();
    uint32_t axis = input.desc.GetDimensionCount() - 1;
    DML_TOP_K_OPERATOR_DESC topk_desc = {};
    topk_desc.InputTensor = &inputs[0];
    topk_desc.OutputValueTensor = &outputs[0];
    topk_desc.OutputIndexTensor = &outputs[1];
    topk_desc.Axis = axis;
    topk_desc.K = k;

    DML_OPERATOR_DESC op_desc = {DML_OPERATOR_TOP_K, &topk_desc};
    Initialize(ctx, std::move(tensors), op_desc);
  }
};

#define DML_REGISTER_KERNEL(type)                                \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("TopK").Device(DEVICE_DML).TypeConstraint<type>("T"), \
      DmlKernelWrapper<DmlTopKKernel, TopKShapeHelper>);         \
  REGISTER_KERNEL_BUILDER(Name("TopKV2")                         \
                              .Device(DEVICE_DML)                \
                              .TypeConstraint<type>("T")         \
                              .HostMemory("k"),                  \
                          DmlKernelWrapper<DmlTopKKernel, TopKShapeHelper>);
TF_CALL_DML_FLOAT_TYPES(DML_REGISTER_KERNEL);
#undef DML_REGISTER_KERNEL
}  // namespace tensorflow