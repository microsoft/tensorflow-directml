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

struct SimplifiedReverseSequence {
  dml::TensorDesc::Dimensions input_output_sizes;
  dml::TensorDesc::Dimensions seq_lengths_sizes;
  dml::TensorDesc::Dimensions non_broadcast_seq_lengths_sizes;
  int32 seq_dim;
};

static SimplifiedReverseSequence SimplifyReverseSequence(
    const TensorShape& input_shape, const TensorShape& seq_lengths_shape,
    int32 batch_dim, int32 seq_dim) {
  // DML_REVERSE_SUBSEQUENCES_OPERATOR_DESC only supports up to 5D, but since
  // tensorflow has a single batch dim and sequence dim, there's always a way
  // to reduce all dimensions to 5D.
  SimplifiedReverseSequence desc = {};

  int32 lower_index = std::min(batch_dim, seq_dim);
  int32 upper_index = std::max(batch_dim, seq_dim);

  uint32_t dim_before = 1;
  uint32_t dim_between = 1;
  uint32_t dim_after = 1;

  for (int i = 0; i < lower_index; ++i) {
    dim_before *= input_shape.dim_size(i);
  }

  for (int i = lower_index + 1; i < upper_index; ++i) {
    dim_between *= input_shape.dim_size(i);
  }

  for (int i = upper_index + 1; i < input_shape.dims(); ++i) {
    dim_after *= input_shape.dim_size(i);
  }

  desc.seq_dim = seq_dim < batch_dim ? 1 : 3;

  desc.input_output_sizes = {
      dim_before,  static_cast<uint32_t>(input_shape.dim_size(lower_index)),
      dim_between, static_cast<uint32_t>(input_shape.dim_size(upper_index)),
      dim_after,
  };

  desc.non_broadcast_seq_lengths_sizes = {1u, 1u, 1u, 1u, 1u};
  desc.seq_lengths_sizes = {dim_before, 1u, dim_between, 1u, dim_after};

  uint32 new_batch_dim = batch_dim < seq_dim ? 1 : 3;
  desc.non_broadcast_seq_lengths_sizes[new_batch_dim] =
      input_shape.dim_size(batch_dim);
  desc.seq_lengths_sizes[new_batch_dim] = input_shape.dim_size(batch_dim);

  return desc;
}

class ReverseSequenceInitHelper : public InitializationHelper {
 public:
  struct Attributes {
    explicit Attributes(OpKernelConstruction* ctx) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("batch_dim", &batch_dim));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("seq_dim", &seq_dim));
    }

    int32 batch_dim;
    int32 seq_dim;
  };

  ReverseSequenceInitHelper(OpKernelContext* ctx,
                            std::shared_ptr<const Attributes> attr) {
    const Tensor& input = ctx->input(0);
    const Tensor& seq_lens = ctx->input(1);

    // Preliminary validation of sizes.
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(seq_lens.shape()),
                errors::InvalidArgument("seq_lens input must be 1-dim, not ",
                                        seq_lens.dims()));

    OP_REQUIRES(
        ctx, attr->batch_dim != attr->seq_dim,
        errors::InvalidArgument("batch_dim == seq_dim == ", attr->seq_dim));

    OP_REQUIRES(
        ctx, attr->seq_dim < input.dims(),
        errors::InvalidArgument("seq_dim must be < input.dims()", "( ",
                                attr->seq_dim, " vs. ", input.dims(), ")"));
    OP_REQUIRES(
        ctx, attr->batch_dim < input.dims(),
        errors::InvalidArgument("batch_dim must be < input.dims()", "( ",
                                attr->batch_dim, " vs. ", input.dims(), ")"));

    OP_REQUIRES(
        ctx, seq_lens.NumElements() == input.dim_size(attr->batch_dim),
        errors::InvalidArgument("len(seq_lens) != input.dims(", attr->batch_dim,
                                "), ", "(", seq_lens.NumElements(), " vs. ",
                                input.dim_size(attr->batch_dim), ")"));

    // Collapse the non-batch and non-axis dimensions together
    simple_reverse_sequence_ = SimplifyReverseSequence(
        input.shape(), seq_lens.shape(), attr->batch_dim, attr->seq_dim);
  }

  const SimplifiedReverseSequence& GetSimplifiedReverseSequence() const {
    return simple_reverse_sequence_;
  }

 private:
  SimplifiedReverseSequence simple_reverse_sequence_;
};

class DmlReverseSequenceKernel : public DmlKernel {
 public:
  using InitHelper = ReverseSequenceInitHelper;

  explicit DmlReverseSequenceKernel(DmlKernelConstruction* ctx,
                                    const InitHelper* init_helper) {
    auto simple_reverse_sequence = init_helper->GetSimplifiedReverseSequence();

    DmlTensorInfo input;
    input.kernel_index = 0;
    input.desc = DmlTensorDesc::Create(
        ctx->GetInputDataType(0), simple_reverse_sequence.input_output_sizes,
        simple_reverse_sequence.input_output_sizes);

    DmlTensorInfo seq_lengths;
    seq_lengths.kernel_index = 1;
    seq_lengths.desc = DmlTensorDesc::Create(
        ctx->GetInputDataType(1), simple_reverse_sequence.seq_lengths_sizes,
        simple_reverse_sequence.non_broadcast_seq_lengths_sizes);
    seq_lengths.desc.ForceUnsignedDataType();

    DmlTensorInfo output;
    output.kernel_index = 0;
    output.desc = DmlTensorDesc::Create(
        ctx->GetOutputDataType(0), simple_reverse_sequence.input_output_sizes,
        simple_reverse_sequence.input_output_sizes);

    DmlKernelTensors tensors;
    tensors.inputs = {input, seq_lengths};
    tensors.outputs = {output};

    auto inputs = GetDmlTensorDescs(tensors.inputs);
    auto outputs = GetDmlTensorDescs(tensors.outputs);

    DML_REVERSE_SUBSEQUENCES_OPERATOR_DESC reverse_desc = {};
    reverse_desc.InputTensor = &inputs[0];
    reverse_desc.SequenceLengthsTensor = &inputs[1];
    reverse_desc.OutputTensor = &outputs[0];
    reverse_desc.Axis = simple_reverse_sequence.seq_dim;

    DML_OPERATOR_DESC op_desc = {DML_OPERATOR_REVERSE_SUBSEQUENCES,
                                 &reverse_desc};
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

#define REGISTER_KERNELS(type)                                                \
  REGISTER_KERNEL_BUILDER(Name("ReverseSequence")                             \
                              .Device(DEVICE_DML)                             \
                              .TypeConstraint<type>("T")                      \
                              .TypeConstraint<int32>("Tlen"),                 \
                          DmlKernelWrapper<DmlReverseSequenceKernel,          \
                                           GetOutputShapeAsInputShapeHelper>) \
  REGISTER_KERNEL_BUILDER(Name("ReverseSequence")                             \
                              .Device(DEVICE_DML)                             \
                              .TypeConstraint<type>("T")                      \
                              .TypeConstraint<int64>("Tlen"),                 \
                          DmlKernelWrapper<DmlReverseSequenceKernel,          \
                                           GetOutputShapeAsInputShapeHelper>)

TF_CALL_DML_ALL_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

}  // namespace tensorflow