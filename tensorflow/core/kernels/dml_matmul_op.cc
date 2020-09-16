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
#include "tensorflow/core/util/matmul_bcast.h"

namespace tensorflow {

class MatMulInitHelper : public InitializationHelper {
 public:
  struct Attributes {
    explicit Attributes(OpKernelConstruction* ctx) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_a", &transpose_a));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_b", &transpose_b));
    }

    bool transpose_a;
    bool transpose_b;
  };

  MatMulInitHelper(OpKernelContext* ctx, std::shared_ptr<const Attributes> attr)
      : attr_(attr) {
    const TensorShape& a_shape = ctx->input(0).shape();
    const TensorShape& b_shape = ctx->input(1).shape();

    OP_REQUIRES(
        ctx, TensorShapeUtils::IsMatrix(a_shape),
        errors::InvalidArgument("In[0] is not a matrix. Instead it has shape ",
                                a_shape.DebugString()));
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsMatrix(b_shape),
        errors::InvalidArgument("In[1] is not a matrix. Instead it has shape ",
                                b_shape.DebugString()));

    int in0_k_index = attr->transpose_a ? 0 : 1;
    int in1_k_index = attr->transpose_b ? 1 : 0;

    OP_REQUIRES(ctx,
                a_shape.dim_size(in0_k_index) == b_shape.dim_size(in1_k_index),
                errors::InvalidArgument(
                    "Matrix size-incompatible: In[0]: ", a_shape.DebugString(),
                    ", In[1]: ", b_shape.DebugString()));
  }

  bool TransposeA() const { return attr_->transpose_a; }
  bool TransposeB() const { return attr_->transpose_b; }

 private:
  const std::shared_ptr<const Attributes> attr_;
};

class BaseBatchMatMulInitHelper : public InitializationHelper {
 public:
  struct Attributes {
    explicit Attributes(OpKernelConstruction* ctx) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("adj_x", &adj_x));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("adj_y", &adj_y));
    }

    bool adj_x;
    bool adj_y;
  };

  BaseBatchMatMulInitHelper(OpKernelContext* ctx,
                            std::shared_ptr<const Attributes> attr)
      : attr_(attr) {
    const Tensor& in0 = ctx->input(0);
    const Tensor& in1 = ctx->input(1);

    MatMulBCast bcast(in0.shape().dim_sizes(), in1.shape().dim_sizes());
    OP_REQUIRES(
        ctx, bcast.IsValid(),
        errors::InvalidArgument(
            "In[0] and In[1] must have compatible batch dimensions: ",
            in0.shape().DebugString(), " vs. ", in1.shape().DebugString()));

    auto batch_size = bcast.output_batch_size();
    auto d0 = in0.dim_size(in0.dims() - 2);
    auto d1 = in0.dim_size(in0.dims() - 1);
    Tensor in0_reshaped;
    OP_REQUIRES(
        ctx,
        in0_reshaped.CopyFrom(in0, TensorShape({bcast.x_batch_size(), d0, d1})),
        errors::Internal("Failed to reshape In[0] from ",
                         in0.shape().DebugString()));
    auto d2 = in1.dim_size(in1.dims() - 2);
    auto d3 = in1.dim_size(in1.dims() - 1);
    Tensor in1_reshaped;
    OP_REQUIRES(
        ctx,
        in1_reshaped.CopyFrom(in1, TensorShape({bcast.y_batch_size(), d2, d3})),
        errors::Internal("Failed to reshape In[1] from ",
                         in1.shape().DebugString()));
    if (attr->adj_x) std::swap(d0, d1);
    if (attr->adj_y) std::swap(d2, d3);
    OP_REQUIRES(ctx, d1 == d2,
                errors::InvalidArgument(
                    "In[0] mismatch In[1] shape: ", d1, " vs. ", d2, ": ",
                    in0.shape().DebugString(), " ", in1.shape().DebugString(),
                    " ", attr->adj_x, " ", attr->adj_y));

    output_batch_shape_ = bcast.output_batch_shape();
  }

  const TensorShape& GetOutputBatchShape() const { return output_batch_shape_; }
  bool AdjX() const { return attr_->adj_x; }
  bool AdjY() const { return attr_->adj_y; }

 protected:
  virtual void ValidateInputTensors(OpKernelContext* ctx, const Tensor& in0,
                                    const Tensor& in1) = 0;

 private:
  TensorShape output_batch_shape_;
  const std::shared_ptr<const Attributes> attr_;
};

class BatchMatMulInitHelper : public BaseBatchMatMulInitHelper {
 public:
  explicit BatchMatMulInitHelper(
      OpKernelContext* ctx,
      std::shared_ptr<const BaseBatchMatMulInitHelper::Attributes> attr)
      : BaseBatchMatMulInitHelper(ctx, attr) {}

 private:
  void ValidateInputTensors(OpKernelContext* ctx, const Tensor& in0,
                            const Tensor& in1) override {
    // Disallow broadcasting support. Ensure that all batch dimensions of the
    // input tensors match.
    OP_REQUIRES(ctx, in0.dims() == in1.dims(),
                errors::InvalidArgument("In[0] and In[1] has different ndims: ",
                                        in0.shape().DebugString(), " vs. ",
                                        in1.shape().DebugString()));
    const int ndims = in0.dims();
    OP_REQUIRES(
        ctx, ndims >= 2,
        errors::InvalidArgument("In[0] and In[1] ndims must be >= 2: ", ndims));
    for (int i = 0; i < ndims - 2; ++i) {
      OP_REQUIRES(ctx, in0.dim_size(i) == in1.dim_size(i),
                  errors::InvalidArgument(
                      "In[0].dim(", i, ") and In[1].dim(", i,
                      ") must be the same: ", in0.shape().DebugString(), " vs ",
                      in1.shape().DebugString()));
    }
  }
};

class BatchMatMulV2InitHelper : public BaseBatchMatMulInitHelper {
 public:
  explicit BatchMatMulV2InitHelper(
      OpKernelContext* ctx,
      std::shared_ptr<const BaseBatchMatMulInitHelper::Attributes> attr)
      : BaseBatchMatMulInitHelper(ctx, attr) {}

 private:
  void ValidateInputTensors(OpKernelContext* ctx, const Tensor& in0,
                            const Tensor& in1) override {
    // Enable broadcasting support. Validity of broadcasting is checked in
    // BaseBatchMatMulInitHelper.
    OP_REQUIRES(
        ctx, in0.dims() >= 2,
        errors::InvalidArgument("In[0] ndims must be >= 2: ", in0.dims()));
    OP_REQUIRES(
        ctx, in1.dims() >= 2,
        errors::InvalidArgument("In[1] ndims must be >= 2: ", in1.dims()));
  }
};

class MatMulShapeHelper : public ShapeHelper {
 public:
  std::vector<TensorShape> GetOutputShapes(
      OpKernelContext* ctx,
      const InitializationHelper* initialization_helper) const override {
    const TensorShape& a_shape = ctx->input(0).shape();
    const TensorShape& b_shape = ctx->input(1).shape();

    auto init_helper =
        static_cast<const MatMulInitHelper*>(initialization_helper);

    TensorShape out_shape = {
        !init_helper->TransposeA() ? a_shape.dim_size(0) : a_shape.dim_size(1),
        !init_helper->TransposeB() ? b_shape.dim_size(1) : b_shape.dim_size(0)};

    return {out_shape};
  }
};

static std::vector<TensorShape> GetBatchMatMulOutputShapes(
    OpKernelContext* ctx, const BaseBatchMatMulInitHelper* init_helper) {
  const TensorShape& in0 = ctx->input(0).shape();
  const TensorShape& in1 = ctx->input(1).shape();

  int64 in0_rows = in0.dim_size(in0.dims() - 2);
  int64 in0_cols = in0.dim_size(in0.dims() - 1);

  int64 in1_rows = in1.dim_size(in1.dims() - 2);
  int64 in1_cols = in1.dim_size(in1.dims() - 1);

  if (init_helper->AdjX()) {
    std::swap(in0_rows, in0_cols);
  }

  if (init_helper->AdjY()) {
    std::swap(in1_rows, in1_cols);
  }

  // The matrices must have matching shapes; this should have been validated
  // earlier
  assert(in0_cols == in1_rows);

  TensorShape output_shape = init_helper->GetOutputBatchShape();
  output_shape.AddDim(in0_rows);
  output_shape.AddDim(in1_cols);

  return {output_shape};
}

template <typename TInitHelper>
class BatchMatMulShapeHelper : public ShapeHelper {
 public:
  std::vector<TensorShape> GetOutputShapes(
      OpKernelContext* ctx,
      const InitializationHelper* initialization_helper) const override {
    auto init_helper = static_cast<const TInitHelper*>(initialization_helper);

    return GetBatchMatMulOutputShapes(ctx, init_helper);
  }
};

class DmlMatMulKernel : public DmlKernel {
 public:
  using InitHelper = MatMulInitHelper;

  explicit DmlMatMulKernel(DmlKernelConstruction* ctx,
                           const InitHelper* init_helper) {
    CHECK(ctx->GetInputCount() == 2);
    CHECK(ctx->GetOutputCount() == 1);

    DmlKernelParams params;
    params.kernel_input_indices = {
        0, 1,
        absl::nullopt  // We don't use the GEMM's 'C' tensor
    };

    DmlKernelTensors tensors = GetTensorInfos(ctx, params);

    auto input_descs = GetDmlTensorDescs(tensors.inputs);
    auto output_descs = GetDmlTensorDescs(tensors.outputs);

    DML_GEMM_OPERATOR_DESC gemm_desc = {};
    gemm_desc.ATensor = &input_descs[0];
    gemm_desc.BTensor = &input_descs[1];
    gemm_desc.CTensor = nullptr;
    gemm_desc.OutputTensor = &output_descs[0];
    gemm_desc.TransA = init_helper->TransposeA()
                           ? DML_MATRIX_TRANSFORM_TRANSPOSE
                           : DML_MATRIX_TRANSFORM_NONE;
    gemm_desc.TransB = init_helper->TransposeB()
                           ? DML_MATRIX_TRANSFORM_TRANSPOSE
                           : DML_MATRIX_TRANSFORM_NONE;
    gemm_desc.Alpha = 1.0f;
    gemm_desc.Beta = 0.0f;
    gemm_desc.FusedActivation = nullptr;

    DML_OPERATOR_DESC op_desc = {DML_OPERATOR_GEMM, &gemm_desc};
    Initialize(ctx, std::move(tensors), op_desc);
  }
};

#define DML_REGISTER_KERNEL(type)                                  \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("MatMul").Device(DEVICE_DML).TypeConstraint<type>("T"), \
      DmlKernelWrapper<DmlMatMulKernel, MatMulShapeHelper>);
TF_CALL_DML_FLOAT_TYPES(DML_REGISTER_KERNEL);
#undef DML_REGISTER_KERNEL

template <typename TInitHelper>
class DmlBatchMatMulKernel : public DmlKernel {
 public:
  using InitHelper = TInitHelper;

  explicit DmlBatchMatMulKernel(DmlKernelConstruction* ctx,
                                const InitHelper* init_helper) {
    CHECK(ctx->GetInputCount() == 2);
    CHECK(ctx->GetOutputCount() == 1);

    // Dimensions of input tensors, prior to broadcasting
    auto in0_physical_shape = ctx->GetInputTensorShape(0);
    auto in1_physical_shape = ctx->GetInputTensorShape(1);

    // All dimensions but the last two are batch dimensions.
    const TensorShape& output_shape = ctx->GetOutputTensorShape(0);
    const int batch_dimension_count = output_shape.dims() - 2;

    // Broadcast the batch dimensions of the input shapes if necessary by
    // setting the batch dimensions equal to the output's
    TensorShape in0_shape;
    TensorShape in1_shape;
    for (int i = 0; i < batch_dimension_count; ++i) {
      in0_shape.AddDim(output_shape.dim_size(i));
      in1_shape.AddDim(output_shape.dim_size(i));
    }

    // Add spatial dimensions for input shapes. Spatial dimensions are the
    // last two of the input shapes.
    in0_shape.AddDim(
        in0_physical_shape.dim_size(in0_physical_shape.dims() - 2));
    in0_shape.AddDim(
        in0_physical_shape.dim_size(in0_physical_shape.dims() - 1));
    in1_shape.AddDim(
        in1_physical_shape.dim_size(in1_physical_shape.dims() - 2));
    in1_shape.AddDim(
        in1_physical_shape.dim_size(in1_physical_shape.dims() - 1));

    DmlKernelParams params;
    params.kernel_input_indices = {
        0, 1,
        absl::nullopt  // We don't use GEMM's 'C' tensor
    };

    DmlKernelTensors tensors = GetTensorInfos(ctx, params);
    tensors.inputs[0]->desc = CreateTensorDescFromInput(ctx, 0, in0_shape);
    tensors.inputs[1]->desc = CreateTensorDescFromInput(ctx, 1, in1_shape);

    auto input_descs = GetDmlTensorDescs(tensors.inputs);
    auto output_descs = GetDmlTensorDescs(tensors.outputs);

    // This kernel doesn't support complex types, so adjointed matrices are
    // equivalent to their transpose.
    DML_MATRIX_TRANSFORM trans_a = init_helper->AdjX()
                                       ? DML_MATRIX_TRANSFORM_TRANSPOSE
                                       : DML_MATRIX_TRANSFORM_NONE;
    DML_MATRIX_TRANSFORM trans_b = init_helper->AdjY()
                                       ? DML_MATRIX_TRANSFORM_TRANSPOSE
                                       : DML_MATRIX_TRANSFORM_NONE;

    DML_GEMM_OPERATOR_DESC gemm_desc = {};
    gemm_desc.ATensor = &input_descs[0];
    gemm_desc.BTensor = &input_descs[1];
    gemm_desc.CTensor = nullptr;
    gemm_desc.OutputTensor = &output_descs[0];
    gemm_desc.TransA = trans_a;
    gemm_desc.TransB = trans_b;
    gemm_desc.Alpha = 1.0f;
    gemm_desc.Beta = 0.0f;
    gemm_desc.FusedActivation = nullptr;

    DML_OPERATOR_DESC op_desc = {DML_OPERATOR_GEMM, &gemm_desc};
    Initialize(ctx, std::move(tensors), op_desc);
  }
};

#define DML_REGISTER_KERNEL(type)                                       \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("BatchMatMul").Device(DEVICE_DML).TypeConstraint<type>("T"), \
      DmlKernelWrapper<DmlBatchMatMulKernel<BatchMatMulInitHelper>,     \
                       BatchMatMulShapeHelper<BatchMatMulInitHelper>>);
TF_CALL_DML_FLOAT_TYPES(DML_REGISTER_KERNEL);
#undef DML_REGISTER_KERNEL

#define DML_REGISTER_KERNEL(type)                                         \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("BatchMatMulV2").Device(DEVICE_DML).TypeConstraint<type>("T"), \
      DmlKernelWrapper<DmlBatchMatMulKernel<BatchMatMulV2InitHelper>,     \
                       BatchMatMulShapeHelper<BatchMatMulV2InitHelper>>);
TF_CALL_DML_FLOAT_TYPES(DML_REGISTER_KERNEL);
#undef DML_REGISTER_KERNEL

}  // namespace tensorflow
