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

class SplitInitHelper : public InitializationHelper {
 public:
  using Attributes = EmptyAttributes;

  // Split is only a no-op if all output tensors are empty
  bool IsNoOpKernel(
      OpKernelContext* ctx,
      absl::Span<const TensorShape> output_shapes) const override {
    for (int i = 0; i < output_shapes.size(); ++i) {
      if (output_shapes[i].num_elements() != 0) {
        return false;
      }
    }

    return true;
  }

  SplitInitHelper(OpKernelContext* context,
                  std::shared_ptr<const Attributes> attr) {
    // SplitV has 3 inputs and Split has 2 inputs
    CHECK(context->num_inputs() == 2 || context->num_inputs() == 3);
    bool is_split_v = context->num_inputs() == 3;
    input_tensor_index_ = is_split_v ? 0 : 1;

    const TensorShape& input_shape =
        context->input(input_tensor_index_).shape();
    const Tensor& split_dim_tensor =
        is_split_v ? context->input(2) : context->input(0);

    OP_REQUIRES(
        context, split_dim_tensor.shape().dims() == 0,
        errors::InvalidArgument("split_dim must be a scalar but has rank ",
                                split_dim_tensor.shape().dims()));
    const int32 split_dim_orig = split_dim_tensor.flat<int32>()(0);
    split_dim_ = split_dim_orig < 0 ? split_dim_orig + input_shape.dims()
                                    : split_dim_orig;
    int32 num_split = context->num_outputs();

    OP_REQUIRES(context, 0 <= split_dim_ && split_dim_ < input_shape.dims(),
                errors::InvalidArgument("-input rank(-", input_shape.dims(),
                                        ") <= split_dim < input rank (",
                                        input_shape.dims(), "), but got ",
                                        split_dim_orig));

    OP_REQUIRES(
        context, num_split > 0,
        errors::InvalidArgument(
            "Number of ways to split should be > 0, but got ", num_split));

    int input_size_split_dim = input_shape.dim_size(split_dim_);

    if (is_split_v) {
      const Tensor& split_tensor = context->input(1);

      OP_REQUIRES(
          context,
          split_tensor.dims() == 1 && split_tensor.NumElements() == num_split,
          errors::InvalidArgument(
              "size of the split_tensor must be 1-D and have "
              "the same elements as outputs got ",
              split_tensor.dims(), " -D and ", split_tensor.NumElements(),
              " elements"));

      split_sizes_ = IntTensorToVec<int64>(split_tensor);

      // Determine sizes of output, in case of a -1 input value
      int neg_one_dim = -1;
      int64 determined_size = 0;
      for (int d = 0; d < split_sizes_.size(); ++d) {
        int64 size = split_sizes_[d];

        if (size == -1) {
          OP_REQUIRES(context, neg_one_dim == -1,
                      errors::InvalidArgument("There can only be one -1 in the "
                                              "input."));
          neg_one_dim = d;
        } else {
          determined_size += size;
        }
      }

      OP_REQUIRES(
          context,
          (neg_one_dim == -1 && determined_size == input_size_split_dim) ||
              (neg_one_dim >= 0 && determined_size <= input_size_split_dim),
          errors::InvalidArgument(
              "Determined shape must either match input shape along split_dim "
              "exactly if fully specified, or be less than the size of the "
              "input along split_dim if not fully specified.  Got: ",
              determined_size));

      if (neg_one_dim >= 0) {
        split_sizes_[neg_one_dim] = input_size_split_dim - determined_size;
      }
    } else {
      OP_REQUIRES(context, input_size_split_dim % num_split == 0,
                  errors::InvalidArgument(
                      "Number of ways to split should evenly divide the split "
                      "dimension, but got split_dim ",
                      split_dim_, " (size = ", input_size_split_dim, ") ",
                      "and num_split ", num_split));

      split_sizes_.assign(num_split, input_size_split_dim / num_split);
    }
  }

  int32 GetSplitDim() const { return split_dim_; }
  absl::Span<const int64> GetSplitSizes() const { return split_sizes_; }
  int GetInputTensorIndex() const { return input_tensor_index_; }

 private:
  absl::InlinedVector<int64, 5> split_sizes_;
  int32 split_dim_;
  int input_tensor_index_;
};

using InitHelper = SplitInitHelper;

class SplitShapeHelper : public ShapeHelper {
 public:
  std::vector<TensorShape> GetOutputShapes(
      OpKernelContext* ctx,
      const InitializationHelper* initialization_helper) const override {
    auto init_helper = static_cast<const InitHelper*>(initialization_helper);
    int32 split_dim = init_helper->GetSplitDim();
    absl::Span<const int64> split_sizes = init_helper->GetSplitSizes();

    int input_tensor_index = init_helper->GetInputTensorIndex();
    const TensorShape& input_shape = ctx->input(input_tensor_index).shape();
    std::vector<TensorShape> output_shapes;
    output_shapes.reserve(split_sizes.size());

    for (int64 split_size : split_sizes) {
      TensorShape output_shape = input_shape;
      output_shape.set_dim(split_dim, split_size);
      output_shapes.push_back(std::move(output_shape));
    }

    return output_shapes;
  }
};

class DmlSplitKernel : public DmlKernel {
 public:
  using InitHelper = tensorflow::InitHelper;

  explicit DmlSplitKernel(DmlKernelConstruction* ctx,
                          const InitHelper* init_helper) {
    CHECK(ctx->GetInputCount() == 2 || ctx->GetInputCount() == 3);
    CHECK(ctx->GetOutputCount() > 0);

    int input_tensor_index = init_helper->GetInputTensorIndex();
    int64 split_dim = init_helper->GetSplitDim();

    // We can collapse all dimensions to the left together and all dimensions
    // to the right together. This allows us to send tensors with an "unlimited"
    // number of dimensions to DirectML
    int left_dim_size = 1;
    int right_dim_size = 1;
    TensorShape input_shape = ctx->GetInputTensorShape(input_tensor_index);

    for (int i = 0; i < split_dim; ++i) {
      left_dim_size *= input_shape.dim_size(i);
    }

    for (int i = split_dim + 1; i < input_shape.dims(); ++i) {
      right_dim_size *= input_shape.dim_size(i);
    }

    int split_dim_size = input_shape.dim_size(split_dim);
    input_shape = TensorShape({left_dim_size, split_dim_size, right_dim_size});

    DmlKernelTensors tensors;

    DmlTensorInfo input;
    input.kernel_index = input_tensor_index;
    input.desc = DmlTensorDesc::Create(
        ctx->GetInputDataType(input_tensor_index), input_shape, input_shape);
    tensors.inputs = {input};

    // DML doesn't support empty tensors, so filter them out when generating the
    // kernel output indices (which is what determines the mapping between
    // kernel outputs and DML op outputs)
    for (uint32_t i = 0; i < ctx->GetOutputCount(); ++i) {
      if (ctx->GetOutputTensorShape(i).num_elements() == 0) {
        // Empty tensor; ignore this output
        continue;
      }

      int axis_dim_size = ctx->GetOutputTensorShape(i).dim_size(split_dim);
      TensorShape output_shape({left_dim_size, axis_dim_size, right_dim_size});

      DmlTensorInfo output;
      output.kernel_index = i;
      output.desc = DmlTensorDesc::Create(ctx->GetOutputDataType(i),
                                          output_shape, output_shape);
      tensors.outputs.push_back(std::move(output));
    }

    auto inputs = GetDmlTensorDescs(tensors.inputs);
    auto outputs = GetDmlTensorDescs(tensors.outputs);

    DML_SPLIT_OPERATOR_DESC split_desc = {};
    split_desc.InputTensor = inputs.data();
    split_desc.OutputCount = outputs.size();
    split_desc.OutputTensors = outputs.data();
    split_desc.Axis = kNchwDimensionCount - 2;

    DML_OPERATOR_DESC op_desc = {DML_OPERATOR_SPLIT, &split_desc};
    Initialize(ctx, std::move(tensors), op_desc);
  }

  StatusOr<DmlGpuEvent> Compute(DmlKernelContext* ctx) const override {
    // Currently, 64-bit integers in DML are emulated using 32-bit integers
    // using striding to emulate a larger type. Because we can't guarantee that
    // our output tensor's memory is zero'd, we need to do so manually prior to
    // running running gather.
    if (Is64BitIntegerType(ctx->GetOutputTensor(0)->dtype())) {
      for (uint32_t i = 0; i < ctx->GetOutputCount(); ++i) {
        if (ctx->GetOutputTensor(i)->NumElements() != 0) {
          Tensor* output = ctx->GetOutputTensor(i);
          ctx->ZeroBuffer(ctx->CreateBufferForTensor(*output));
        }
      }
    }

    return DmlKernel::Compute(ctx);
  }
};

#define REGISTER_KERNEL(type)                                                 \
  REGISTER_KERNEL_BUILDER(Name("Split")                                       \
                              .Device(DEVICE_DML)                             \
                              .TypeConstraint<type>("T")                      \
                              .HostMemory("split_dim"),                       \
                          DmlKernelWrapper<DmlSplitKernel, SplitShapeHelper>) \
  REGISTER_KERNEL_BUILDER(Name("SplitV")                                      \
                              .Device(DEVICE_DML)                             \
                              .TypeConstraint<int32>("Tlen")                  \
                              .TypeConstraint<type>("T")                      \
                              .HostMemory("size_splits")                      \
                              .HostMemory("split_dim"),                       \
                          DmlKernelWrapper<DmlSplitKernel, SplitShapeHelper>) \
  REGISTER_KERNEL_BUILDER(Name("SplitV")                                      \
                              .Device(DEVICE_DML)                             \
                              .TypeConstraint<int64>("Tlen")                  \
                              .TypeConstraint<type>("T")                      \
                              .HostMemory("size_splits")                      \
                              .HostMemory("split_dim"),                       \
                          DmlKernelWrapper<DmlSplitKernel, SplitShapeHelper>)

TF_CALL_DML_ALL_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL

}  // namespace tensorflow