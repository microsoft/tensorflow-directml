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

template <DML_OPERATOR_TYPE operator_type>
class SpaceDepthInitHelper : public InitializationHelper {
 public:
  struct Attributes {
    explicit Attributes(OpKernelConstruction* ctx) {
      std::string data_format_attr;
      OP_REQUIRES_OK(ctx, ctx->GetAttr("data_format", &data_format_attr));
      OP_REQUIRES(ctx, FormatFromString(data_format_attr, &data_format),
                  errors::InvalidArgument("Invalid data format"));

      OP_REQUIRES(ctx, data_format == FORMAT_NHWC || data_format == FORMAT_NCHW,
                  errors::InvalidArgument(
                      "DML only supports NHWC and NCHW for the "
                      "SpaceToDepth and DepthToSpace operators, but received ",
                      data_format_attr));

      OP_REQUIRES_OK(ctx, ctx->GetAttr("block_size", &block_size));
      OP_REQUIRES(ctx, block_size > 1,
                  errors::InvalidArgument("Block size should be > 1, but was: ",
                                          block_size));
    }

    int block_size;
    TensorFormat data_format;
  };

  SpaceDepthInitHelper(OpKernelContext* ctx,
                       std::shared_ptr<const Attributes> attr)
      : attr_(attr) {
    const TensorShape& input_shape = ctx->input(0).shape();
    OP_REQUIRES(ctx, input_shape.dims() == 4,
                errors::InvalidArgument("Input rank should be 4 instead of ",
                                        input_shape.dims()));

    batch_size_ = input_shape.dim_size(
        GetTensorDimIndex<kNchwSpatialDimensionCount>(attr->data_format, 'N'));
    input_height_ = input_shape.dim_size(
        GetTensorDimIndex<kNchwSpatialDimensionCount>(attr->data_format, 'H'));
    input_width_ = input_shape.dim_size(
        GetTensorDimIndex<kNchwSpatialDimensionCount>(attr->data_format, 'W'));
    input_depth_ = input_shape.dim_size(
        GetTensorDimIndex<kNchwSpatialDimensionCount>(attr->data_format, 'C'));

    if (operator_type == DML_OPERATOR_DEPTH_TO_SPACE) {
      const int block_size_sq = attr->block_size * attr->block_size;

      // The depth must be divisible by block_size * block_size
      OP_REQUIRES(
          ctx, input_depth_ % block_size_sq == 0,
          errors::InvalidArgument("Input depth dimension ", input_depth_,
                                  " should be divisible by: ", block_size_sq));
    } else {
      DCHECK(operator_type == DML_OPERATOR_SPACE_TO_DEPTH);

      // Both width and height must be divisible by block_size.
      OP_REQUIRES(
          ctx,
          (input_width_ % attr->block_size) == 0 &&
              (input_height_ % attr->block_size) == 0,
          errors::InvalidArgument(
              "Image width ", input_width_, " and height ", input_height_,
              " should be divisible by block_size: ", attr->block_size));
    }
  }

  TensorFormat GetDataFormat() const { return attr_->data_format; }
  int GetBlockSize() const { return attr_->block_size; }
  int GetBatchSize() const { return batch_size_; }
  int GetInputHeight() const { return input_height_; }
  int GetInputWidth() const { return input_width_; }
  int GetInputDepth() const { return input_depth_; }

 private:
  const std::shared_ptr<const Attributes> attr_;
  int batch_size_;
  int input_height_;
  int input_width_;
  int input_depth_;
};

template <DML_OPERATOR_TYPE operator_type>
using InitHelper = SpaceDepthInitHelper<operator_type>;

template <DML_OPERATOR_TYPE operator_type>
class SpaceDepthShapeHelper : public ShapeHelper {
 public:
  std::vector<TensorShape> GetOutputShapes(
      OpKernelContext* ctx,
      const InitializationHelper* initialization_helper) const override {
    auto init_helper =
        static_cast<const InitHelper<operator_type>*>(initialization_helper);

    TensorFormat data_format = init_helper->GetDataFormat();
    int batch_size = init_helper->GetBatchSize();
    int input_height = init_helper->GetInputHeight();
    int input_width = init_helper->GetInputWidth();
    int input_depth = init_helper->GetInputDepth();
    int block_size = init_helper->GetBlockSize();

    if (operator_type == DML_OPERATOR_DEPTH_TO_SPACE) {
      const int block_size_sq = block_size * block_size;
      const int output_depth = input_depth / block_size_sq;
      const int output_width = input_width * block_size;
      const int output_height = input_height * block_size;

      return {ShapeFromFormat(data_format, batch_size, output_height,
                              output_width, output_depth)};
    } else {
      DCHECK(operator_type == DML_OPERATOR_SPACE_TO_DEPTH);
      // The 'spatial' block of size block_size X block_size will be moved to
      // depth.
      const int output_depth = input_depth * block_size * block_size;
      const int output_width = input_width / block_size;
      const int output_height = input_height / block_size;

      return {ShapeFromFormat(data_format, batch_size, output_height,
                              output_width, output_depth)};
    }
  }
};

template <typename OPERATOR_DESC, DML_OPERATOR_TYPE operator_type>
class DmlSpaceDepthKernel : public DmlKernel {
 public:
  using InitHelper = InitHelper<operator_type>;

  explicit DmlSpaceDepthKernel(DmlKernelConstruction* ctx,
                               const InitHelper* init_helper) {
    CHECK(ctx->GetInputCount() == 1);
    CHECK(ctx->GetOutputCount() == 1);

    const TensorShape& input_shape = ctx->GetInputTensorShape(0);
    const TensorShape& output_shape = ctx->GetOutputTensorShape(1);

    auto layout =
        GetDmlTensorLayout(init_helper->GetDataFormat(), input_shape.dims());

    DmlTensorInfo input;
    input.kernel_index = 0;
    input.desc = CreateTensorDescFromInput(ctx, 0, layout);

    DmlTensorInfo output;
    output.kernel_index = 0;
    output.desc = CreateTensorDescFromOutput(ctx, 0, layout);

    DmlKernelTensors tensors;
    tensors.inputs = {input};
    tensors.outputs = {output};

    auto input_descs = GetDmlTensorDescs(tensors.inputs);
    auto output_descs = GetDmlTensorDescs(tensors.outputs);

    OPERATOR_DESC specific_op_desc = {};
    specific_op_desc.InputTensor = &input_descs[0];
    specific_op_desc.OutputTensor = &output_descs[0];
    specific_op_desc.BlockSize = init_helper->GetBlockSize();

    DML_OPERATOR_DESC op_desc = {operator_type, &specific_op_desc};
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

template <typename OP_DESC, DML_OPERATOR_TYPE operator_type>
using DmlSpaceDepthKernelWrapper =
    DmlKernelWrapper<DmlSpaceDepthKernel<OP_DESC, operator_type>,
                     SpaceDepthShapeHelper<operator_type>>;

#define DML_REGISTER_KERNELS(type)                                       \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("SpaceToDepth").Device(DEVICE_DML).TypeConstraint<type>("T"), \
      DmlSpaceDepthKernelWrapper<DML_SPACE_TO_DEPTH_OPERATOR_DESC,       \
                                 DML_OPERATOR_SPACE_TO_DEPTH>)           \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("DepthToSpace").Device(DEVICE_DML).TypeConstraint<type>("T"), \
      DmlSpaceDepthKernelWrapper<DML_DEPTH_TO_SPACE_OPERATOR_DESC,       \
                                 DML_OPERATOR_DEPTH_TO_SPACE>)

TF_CALL_DML_ALL_TYPES(DML_REGISTER_KERNELS);
#undef DML_REGISTER_KERNELS

}  // namespace tensorflow