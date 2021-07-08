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
#include "tensorflow/core/kernels/fused_eigen_output_kernels.h"
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

  BaseBatchMatMulInitHelper(
      OpKernelContext* ctx, std::shared_ptr<const Attributes> attr,
      std::function<void(OpKernelContext*, const TensorShape&,
                         const TensorShape&)>
          ValidateInputTensors)
      : attr_(attr) {
    const TensorShape& in0_shape = ctx->input(0).shape();
    const TensorShape& in1_shape = ctx->input(1).shape();
    ValidateInputTensors(ctx, in0_shape, in1_shape);

    TensorShape in0_batches_shape;
    for (int i = 0; i < in0_shape.dims() - 2; ++i) {
      in0_batches_shape.AddDim(in0_shape.dim_size(i));
    }

    TensorShape in1_batches_shape;
    for (int i = 0; i < in1_shape.dims() - 2; ++i) {
      in1_batches_shape.AddDim(in1_shape.dim_size(i));
    }

    BCast batches_bcast(BCast::FromShape(in0_batches_shape),
                        BCast::FromShape(in1_batches_shape));

    OP_REQUIRES(ctx, batches_bcast.IsValid(),
                errors::InvalidArgument(
                    "In[0] and In[1] must have compatible batch dimensions: ",
                    in0_shape.DebugString(), " vs. ", in1_shape.DebugString()));

    auto in0_rows = in0_shape.dim_size(in0_shape.dims() - 2);
    auto in0_cols = in0_shape.dim_size(in0_shape.dims() - 1);
    auto in1_rows = in1_shape.dim_size(in1_shape.dims() - 2);
    auto in1_cols = in1_shape.dim_size(in1_shape.dims() - 1);

    collapsed_in0_shape_ = BCast::ToShape(batches_bcast.x_reshape());
    collapsed_in0_shape_.AddDim(in0_rows);
    collapsed_in0_shape_.AddDim(in0_cols);

    collapsed_in1_shape_ = BCast::ToShape(batches_bcast.y_reshape());
    collapsed_in1_shape_.AddDim(in1_rows);
    collapsed_in1_shape_.AddDim(in1_cols);

    if (attr->adj_x) std::swap(in0_rows, in0_cols);
    if (attr->adj_y) std::swap(in1_rows, in1_cols);

    collapsed_output_shape_ = BCast::ToShape(batches_bcast.output_shape());
    collapsed_output_shape_.AddDim(in0_rows);
    collapsed_output_shape_.AddDim(in1_cols);

    OP_REQUIRES(ctx, collapsed_output_shape_.dims() <= 8,
                errors::InvalidArgument(
                    "DML doesn't support more than 8D for BroadcastTo after "
                    "collapsing dimensions together, but the output has ",
                    collapsed_output_shape_.dims(), " dimensions."));

    OP_REQUIRES(ctx, in0_cols == in1_rows,
                errors::InvalidArgument(
                    "In[0] mismatch In[1] shape: ", in0_cols, " vs. ", in1_rows,
                    ": ", in0_shape.DebugString(), " ", in1_shape.DebugString(),
                    " ", attr->adj_x, " ", attr->adj_y));
  }

  const TensorShape& GetCollapsedIn0Shape() const {
    return collapsed_in0_shape_;
  }

  const TensorShape& GetCollapsedIn1Shape() const {
    return collapsed_in1_shape_;
  }

  const TensorShape& GetCollapsedOutputShape() const {
    return collapsed_output_shape_;
  }

  bool AdjX() const { return attr_->adj_x; }
  bool AdjY() const { return attr_->adj_y; }

 private:
  TensorShape collapsed_in0_shape_;
  TensorShape collapsed_in1_shape_;
  TensorShape collapsed_output_shape_;
  const std::shared_ptr<const Attributes> attr_;
};

class BatchMatMulInitHelper : public BaseBatchMatMulInitHelper {
 public:
  explicit BatchMatMulInitHelper(
      OpKernelContext* ctx,
      std::shared_ptr<const BaseBatchMatMulInitHelper::Attributes> attr)
      : BaseBatchMatMulInitHelper(ctx, attr, ValidateInputTensors) {}

 private:
  static void ValidateInputTensors(OpKernelContext* ctx,
                                   const TensorShape& in0_shape,
                                   const TensorShape& in1_shape) {
    // Disallow broadcasting support. Ensure that all batch dimensions of the
    // input tensors match.
    OP_REQUIRES(ctx, in0_shape.dims() == in1_shape.dims(),
                errors::InvalidArgument("In[0] and In[1] has different ndims: ",
                                        in0_shape.DebugString(), " vs. ",
                                        in1_shape.DebugString()));
    const int ndims = in0_shape.dims();
    OP_REQUIRES(
        ctx, ndims >= 2,
        errors::InvalidArgument("In[0] and In[1] ndims must be >= 2: ", ndims));
    for (int i = 0; i < ndims - 2; ++i) {
      OP_REQUIRES(ctx, in0_shape.dim_size(i) == in1_shape.dim_size(i),
                  errors::InvalidArgument(
                      "In[0].dim(", i, ") and In[1].dim(", i,
                      ") must be the same: ", in0_shape.DebugString(), " vs ",
                      in1_shape.DebugString()));
    }
  }
};

class BatchMatMulV2InitHelper : public BaseBatchMatMulInitHelper {
 public:
  explicit BatchMatMulV2InitHelper(
      OpKernelContext* ctx,
      std::shared_ptr<const BaseBatchMatMulInitHelper::Attributes> attr)
      : BaseBatchMatMulInitHelper(ctx, attr, ValidateInputTensors) {}

 private:
  static void ValidateInputTensors(OpKernelContext* ctx,
                                   const TensorShape& in0_shape,
                                   const TensorShape& in1_shape) {
    // Enable broadcasting support. Validity of broadcasting is checked in
    // BaseBatchMatMulInitHelper.
    OP_REQUIRES(ctx, in0_shape.dims() >= 2,
                errors::InvalidArgument("In[0] ndims must be >= 2: ",
                                        in0_shape.dims()));
    OP_REQUIRES(ctx, in1_shape.dims() >= 2,
                errors::InvalidArgument("In[1] ndims must be >= 2: ",
                                        in1_shape.dims()));
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

template <typename TInitHelper>
class BatchMatMulShapeHelper : public ShapeHelper {
 public:
  std::vector<TensorShape> GetOutputShapes(
      OpKernelContext* ctx,
      const InitializationHelper* initialization_helper) const override {
    auto init_helper = static_cast<const TInitHelper*>(initialization_helper);

    const Tensor& in0 = ctx->input(0);
    const Tensor& in1 = ctx->input(1);
    auto in0_rows = in0.dim_size(in0.dims() - 2);
    auto in0_cols = in0.dim_size(in0.dims() - 1);
    auto in1_rows = in1.dim_size(in1.dims() - 2);
    auto in1_cols = in1.dim_size(in1.dims() - 1);

    if (init_helper->AdjX()) {
      std::swap(in0_rows, in0_cols);
    }

    if (init_helper->AdjY()) {
      std::swap(in1_rows, in1_cols);
    }

    MatMulBCast bcast(in0.shape().dim_sizes(), in1.shape().dim_sizes());
    TensorShape output_shape = bcast.output_batch_shape();
    output_shape.AddDim(in0_rows);
    output_shape.AddDim(in1_cols);

    return {output_shape};
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
    TensorShape batches_shape;
    for (int i = 0; i < batch_dimension_count; ++i) {
      batches_shape.AddDim(output_shape.dim_size(i));
    }

    TensorShape in0_shape = batches_shape;
    TensorShape in1_shape = batches_shape;

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
    params.kernel_input_indices = {0, 1};

    DmlKernelTensors tensors = GetTensorInfos(ctx, params);
    tensors.inputs[0]->desc = CreateTensorDescFromInput(ctx, 0, in0_shape);
    tensors.inputs[1]->desc = CreateTensorDescFromInput(ctx, 1, in1_shape);

    auto input_descs = GetDmlTensorDescs(tensors.inputs);
    auto scope = dml::Graph(ctx->GetDmlDevice());
    auto a_tensor = dml::InputTensor(scope, 0, input_descs[0]);
    auto b_tensor = dml::InputTensor(scope, 1, input_descs[1]);

    auto a_tensor_sizes = a_tensor.GetOutputDesc().sizes;
    auto b_tensor_sizes = b_tensor.GetOutputDesc().sizes;

    // DML doesn't support more than 4 dimensions for GEMM, so do the
    // broadcasting manually by using identity beforehand
    if (a_tensor_sizes.size() > kNchwDimensionCount) {
      a_tensor = dml::Identity(a_tensor);
      dml::TensorDimensions collapsed_sizes = {
          static_cast<uint32_t>(batches_shape.num_elements()),
          1,
          a_tensor_sizes[a_tensor_sizes.size() - 2],
          a_tensor_sizes[a_tensor_sizes.size() - 1],
      };

      a_tensor = dml::Reinterpret(a_tensor, collapsed_sizes, absl::nullopt);
    }

    if (b_tensor_sizes.size() > kNchwDimensionCount) {
      b_tensor = dml::Identity(b_tensor);

      dml::TensorDimensions collapsed_sizes = {
          static_cast<uint32_t>(batches_shape.num_elements()),
          1,
          b_tensor_sizes[b_tensor_sizes.size() - 2],
          b_tensor_sizes[b_tensor_sizes.size() - 1],
      };

      b_tensor = dml::Reinterpret(b_tensor, collapsed_sizes, absl::nullopt);
    }

    // This kernel doesn't support complex types, so adjointed matrices are
    // equivalent to their transpose.
    DML_MATRIX_TRANSFORM trans_a = init_helper->AdjX()
                                       ? DML_MATRIX_TRANSFORM_TRANSPOSE
                                       : DML_MATRIX_TRANSFORM_NONE;
    DML_MATRIX_TRANSFORM trans_b = init_helper->AdjY()
                                       ? DML_MATRIX_TRANSFORM_TRANSPOSE
                                       : DML_MATRIX_TRANSFORM_NONE;

    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;
    auto result = dml::Gemm(a_tensor, b_tensor, absl::nullopt, trans_a, trans_b,
                            alpha, beta);

    Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
        scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

    Initialize(ctx, std::move(tensors), compiled_op.Get());
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

class FusedMatMulInitHelper : public MatMulInitHelper {
 public:
  struct Attributes : public MatMulInitHelper::Attributes {
    explicit Attributes(OpKernelConstruction* ctx)
        : MatMulInitHelper::Attributes(ctx) {
      std::vector<FusedComputationPattern> patterns = {
          {FusedComputationType::kBiasAdd, {"BiasAdd"}},
          {FusedComputationType::kBiasAddWithRelu, {"BiasAdd", "Relu"}},
          {FusedComputationType::kBiasAddWithElu, {"BiasAdd", "Elu"}},
      };

      // Only used for FusedBatchNorm
      FusedComputationArgs fused_computation_args;

      OP_REQUIRES_OK(ctx,
                     InitializeFusedComputation(ctx, "DmlFusedMatMul", patterns,
                                                &fused_computation_type,
                                                &fused_computation_args));
    }

    FusedComputationType fused_computation_type;
  };

  FusedMatMulInitHelper(OpKernelContext* ctx,
                        std::shared_ptr<const Attributes> attr)
      : MatMulInitHelper(ctx, attr), attr_(attr) {}

  FusedComputationType GetFusedComputationType() const {
    return attr_->fused_computation_type;
  }

 private:
  const std::shared_ptr<const Attributes> attr_;
};

class DmlFusedMatMulKernel : public DmlKernel {
 public:
  using InitHelper = FusedMatMulInitHelper;

  explicit DmlFusedMatMulKernel(DmlKernelConstruction* ctx,
                                const InitHelper* init_helper) {
    CHECK(ctx->GetInputCount() == 3);
    CHECK(ctx->GetOutputCount() == 1);

    DML_OPERATOR_DESC fused_op_desc = {};
    DML_ACTIVATION_RELU_OPERATOR_DESC relu_desc = {};
    DML_ACTIVATION_ELU_OPERATOR_DESC elu_desc = {};

    const auto fused_computation_type = init_helper->GetFusedComputationType();
    switch (fused_computation_type) {
      case FusedComputationType::kBiasAdd:
        break;
      case FusedComputationType::kBiasAddWithRelu:
        fused_op_desc.Type = DML_OPERATOR_ACTIVATION_RELU;
        fused_op_desc.Desc = &relu_desc;
        break;
      case FusedComputationType::kBiasAddWithElu:
        fused_op_desc.Type = DML_OPERATOR_ACTIVATION_ELU;
        elu_desc.Alpha = 1.0f;
        fused_op_desc.Desc = &elu_desc;
        break;
    }

    DmlTensorInfo a;
    a.kernel_index = 0;
    a.desc = DmlTensorDesc::Create(ctx->GetInputDataType(0),
                                   ctx->GetInputTensorShape(0),
                                   ctx->GetInputTensorShape(0));

    DmlTensorInfo b;
    b.kernel_index = 1;
    b.desc = DmlTensorDesc::Create(ctx->GetInputDataType(1),
                                   ctx->GetInputTensorShape(1),
                                   ctx->GetInputTensorShape(1));

    DmlTensorInfo c;
    c.kernel_index = 2;
    c.desc = DmlTensorDesc::Create(ctx->GetInputDataType(2),
                                   ctx->GetOutputTensorShape(0),
                                   ctx->GetInputTensorShape(2));

    DmlTensorInfo output;
    output.kernel_index = 0;
    output.desc = DmlTensorDesc::Create(ctx->GetOutputDataType(0),
                                        ctx->GetOutputTensorShape(0),
                                        ctx->GetOutputTensorShape(0));

    DmlKernelTensors tensors;
    tensors.inputs = {a, b, c};
    tensors.outputs = {output};

    auto input_descs = GetDmlTensorDescs(tensors.inputs);
    auto output_descs = GetDmlTensorDescs(tensors.outputs);

    DML_GEMM_OPERATOR_DESC gemm_desc = {};
    gemm_desc.ATensor = &input_descs[0];
    gemm_desc.BTensor = &input_descs[1];
    gemm_desc.CTensor = &input_descs[2];
    gemm_desc.OutputTensor = &output_descs[0];
    gemm_desc.TransA = init_helper->TransposeA()
                           ? DML_MATRIX_TRANSFORM_TRANSPOSE
                           : DML_MATRIX_TRANSFORM_NONE;
    gemm_desc.TransB = init_helper->TransposeB()
                           ? DML_MATRIX_TRANSFORM_TRANSPOSE
                           : DML_MATRIX_TRANSFORM_NONE;
    gemm_desc.Alpha = 1.0f;
    gemm_desc.Beta = 1.0f;
    gemm_desc.FusedActivation = fused_op_desc.Desc ? &fused_op_desc : nullptr;

    DML_OPERATOR_DESC op_desc = {DML_OPERATOR_GEMM, &gemm_desc};
    Initialize(ctx, std::move(tensors), op_desc);
  }
};

#define DML_REGISTER_KERNEL(type)                                        \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("_FusedMatMul").Device(DEVICE_DML).TypeConstraint<type>("T"), \
      DmlKernelWrapper<DmlFusedMatMulKernel, MatMulShapeHelper>);
// _FusedMatMul only supports float32
TF_CALL_float(DML_REGISTER_KERNEL);
#undef DML_REGISTER_KERNEL

}  // namespace tensorflow
