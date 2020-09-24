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
#include "tensorflow/core/kernels/rnn/gru_ops.h"

namespace tensorflow {

class GruInitHelper : public InitializationHelper {
 public:
  using Attributes = EmptyAttributes;

  GruInitHelper(OpKernelContext* ctx, std::shared_ptr<const Attributes> attr) {
    const Tensor* x_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("x", &x_tensor));

    const Tensor* h_prev_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("h_prev", &h_prev_tensor));

    const Tensor* w_ru_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("w_ru", &w_ru_tensor));

    const Tensor* w_c_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("w_c", &w_c_tensor));

    const Tensor* b_ru_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("b_ru", &b_ru_tensor));

    const Tensor* b_c_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("b_c", &b_c_tensor));

    batch_size_ = x_tensor->dim_size(0);
    input_size_ = x_tensor->dim_size(1);
    cell_size_ = h_prev_tensor->dim_size(1);

    // Sanity checks for input shapes.

    // Shape of 'h' must be [batch_size, cell_size]
    OP_REQUIRES(ctx, h_prev_tensor->dim_size(0) == batch_size_,
                errors::InvalidArgument("h_prev.dims(0) != batch_size: ",
                                        h_prev_tensor->dim_size(0), " vs. ",
                                        batch_size_));
    OP_REQUIRES(ctx, h_prev_tensor->dim_size(1) == cell_size_,
                errors::InvalidArgument(
                    "h_prev.dims(1) != cell_size: ", h_prev_tensor->dim_size(1),
                    " vs. ", cell_size_));

    // Shape of 'w_ru' must be [input_size+cell_size, 2*cell_size]
    OP_REQUIRES(
        ctx, w_ru_tensor->dim_size(0) == input_size_ + cell_size_,
        errors::InvalidArgument("w_ru.dim_size(0) != input_size + cell_size: ",
                                w_ru_tensor->dim_size(0), " vs. ",
                                input_size_ + cell_size_));
    OP_REQUIRES(ctx, w_ru_tensor->dim_size(1) == cell_size_ * 2,
                errors::InvalidArgument("w_ru.dim_size(1) != cell_size * 2: ",
                                        w_ru_tensor->dim_size(1), " vs. ",
                                        cell_size_ * 2));

    // Shape of 'w_c' must be [input_size+cell_size, cell_size]
    OP_REQUIRES(
        ctx, w_c_tensor->dim_size(0) == input_size_ + cell_size_,
        errors::InvalidArgument("w_c.dim_size(0) != input_size + cell_size: ",
                                w_c_tensor->dim_size(0), " vs. ",
                                input_size_ + cell_size_));
    OP_REQUIRES(ctx, w_c_tensor->dim_size(1) == cell_size_,
                errors::InvalidArgument(
                    "w_c.dim_size(1) != cell_size: ", w_c_tensor->dim_size(1),
                    " vs. ", cell_size_));

    // Shape of 'b_ru' must be [2*cell_size]
    OP_REQUIRES(ctx, b_ru_tensor->dim_size(0) == cell_size_ * 2,
                errors::InvalidArgument("b_ru.dim_size(0) != cell_size * 2: ",
                                        b_ru_tensor->dim_size(0), " vs. ",
                                        cell_size_ * 2));
    OP_REQUIRES(ctx, b_ru_tensor->dims() == 1,
                errors::InvalidArgument("Rank of b_ru must be 1",
                                        b_ru_tensor->dims(), " vs. 1", 1));

    // Shape of 'b_c' must be [cell_size]
    OP_REQUIRES(ctx, b_c_tensor->dim_size(0) == cell_size_,
                errors::InvalidArgument(
                    "b_c.dim_size(0) != cell_size: ", b_c_tensor->dim_size(0),
                    " vs. ", cell_size_));
    OP_REQUIRES(ctx, b_c_tensor->dims() == 1,
                errors::InvalidArgument("Rank of b_c must be 1",
                                        b_c_tensor->dims(), " vs. 1"));
  }

  int64 GetBatchSize() const { return batch_size_; }
  int64 GetInputSize() const { return input_size_; }
  int64 GetCellSize() const { return cell_size_; }

 private:
  int64 batch_size_ = 0;
  int64 input_size_ = 0;
  int64 cell_size_ = 0;
};

class GruShapeHelper : public ShapeHelper {
 public:
  using InitHelper = tensorflow::GruInitHelper;
  std::vector<TensorShape> GetOutputShapes(
      OpKernelContext* ctx,
      const InitializationHelper* initialization_helper) const override {
    auto init_helper = static_cast<const InitHelper*>(initialization_helper);
    std::vector<TensorShape> outputShapes;
    outputShapes.reserve(4);

    auto batch_size = init_helper->GetBatchSize();
    auto cell_size = init_helper->GetCellSize();
    // r_tensor shape
    outputShapes.push_back(TensorShape({batch_size, cell_size}));

    // u_tensor shape
    outputShapes.push_back(TensorShape({batch_size, cell_size}));

    // c_tensor shape
    outputShapes.push_back(TensorShape({batch_size, cell_size}));

    // h_tensor shape
    outputShapes.push_back(TensorShape({batch_size, cell_size}));

    return outputShapes;
  }
};

class DmlGruBlockCellOp : public DmlKernel {
 public:
  using InitHelper = tensorflow::GruInitHelper;
  explicit DmlGruBlockCellOp(DmlKernelConstruction* ctx,
                             const InitHelper* init_helper) {
    CHECK(ctx->GetInputCount() == 6);
    CHECK(ctx->GetOutputCount() == 4);

    const uint32_t batch_size = init_helper->GetBatchSize();
    const uint32_t input_size = init_helper->GetInputSize();
    const uint32_t cell_size = init_helper->GetCellSize();

    DmlKernelParams params;
    auto tensors = GetTensorInfos(ctx, params);

    // Reshape b_ru and b_c for future operations
    DmlTensorInfo b_ru_info;
    b_ru_info.desc = DmlTensorDesc::Create(
        ctx->GetInputDataType(4), TensorShape({batch_size, cell_size * 2}),
        TensorShape({1, cell_size * 2}));
    b_ru_info.kernel_index = 4;
    tensors.inputs[4] = b_ru_info;

    DmlTensorInfo b_c_info;
    b_c_info.desc = DmlTensorDesc::Create(ctx->GetInputDataType(5),
                                          TensorShape({batch_size, cell_size}),
                                          TensorShape({1, cell_size}));
    b_c_info.kernel_index = 5;
    tensors.inputs[5] = b_c_info;

    auto input_descs = GetDmlTensorDescs(tensors.inputs);
    auto output_descs = GetDmlTensorDescs(tensors.outputs);

    dml::TensorDesc::Dimensions x_offset = {0, 0, 0, 0};
    dml::TensorDesc::Dimensions x_extent = {1, 1, batch_size, input_size};
    dml::TensorDesc::Dimensions h_offset = {0, 0, 0, input_size};
    dml::TensorDesc::Dimensions h_extent = {1, 1, batch_size, cell_size};
    dml::TensorDesc::Dimensions ru_r_offsets = {0, 0, 0, 0};
    dml::TensorDesc::Dimensions ru_u_offsets = {0, 0, 0, cell_size};
    dml::TensorDesc::Dimensions cell_extents = {1, 1, batch_size, cell_size};
    int32_t slice_strides[] = {1, 1, 1, 1};

    auto scope = dml::Graph(ctx->GetDmlDevice());
    auto x = dml::InputTensor(scope, 0, input_descs[0]);
    auto h_prev = dml::InputTensor(scope, 1, input_descs[1]);
    auto w_ru = dml::InputTensor(scope, 2, input_descs[2]);
    auto w_c = dml::InputTensor(scope, 3, input_descs[3]);
    auto b_ru = dml::InputTensor(scope, 4, input_descs[4]);
    auto b_c = dml::InputTensor(scope, 5, input_descs[5]);

    // Concat x_h_prev = [x, h_prev].
    auto x_h_prev = dml::Join({x, h_prev}, 3);

    // r_u_bar = x_h_prev * w_ru + b_ru
    auto r_u_bar_gemm = dml::Gemm(x_h_prev, w_ru);
    dml::Expression r_u_bar = r_u_bar_gemm;
    r_u_bar += b_ru;

    // Slice r_u_bar into r, u and apply the sigmoid.
    auto r = dml::Slice(r_u_bar, ru_r_offsets, cell_extents, slice_strides);
    r = dml::ActivationSigmoid(r);

    auto u = dml::Slice(r_u_bar, ru_u_offsets, cell_extents, slice_strides);
    u = dml::ActivationSigmoid(u);

    // Concat x_h_prevr = [x,h_prev*r]
    auto h_prevr = h_prev * r;
    auto x_h_prevr = dml::Join({x, h_prevr}, 3);

    // c = tanh(x_h_prevr*w_c+b_c), Note b_c is broadcasted before adding.
    auto c_gemm = dml::Gemm(x_h_prevr, w_c);
    dml::Expression c = c_gemm;
    c += b_c;
    c = dml::ActivationTanh(c);

    // h= u*h_prev + (1-u)*c
    auto h = u * (h_prev - c) + c;

    std::vector<dml::Expression> outputs = {r, u, c, h};
    Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
        scope.Compile(DML_EXECUTION_FLAG_NONE, outputs);

    Initialize(ctx, std::move(tensors), compiled_op.Get());
  }

  StatusOr<DmlGpuEvent> Compute(DmlKernelContext* ctx) const override {
    for (int i = 0; i < ctx->GetOutputCount(); ++i) {
      ctx->ZeroBuffer(ctx->CreateBufferForTensor(*ctx->GetOutputTensor(i)));
    }
    return DmlKernel::Compute(ctx);
  }
};

#define DML_REGISTER_KERNEL(type)                                        \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("GRUBlockCell").Device(DEVICE_DML).TypeConstraint<type>("T"), \
      DmlKernelWrapper<DmlGruBlockCellOp, GruShapeHelper>);
DML_REGISTER_KERNEL(float);
#undef DML_REGISTER_KERNEL

class GruGradInitHelper : public InitializationHelper {
 public:
  using Attributes = EmptyAttributes;

  GruGradInitHelper(OpKernelContext* ctx,
                    std::shared_ptr<const Attributes> attr) {
    // Grab the input tensors.
    const Tensor* x_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("x", &x_tensor));

    const Tensor* h_prev_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("h_prev", &h_prev_tensor));

    const Tensor* w_ru_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("w_ru", &w_ru_tensor));

    const Tensor* w_c_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("w_c", &w_c_tensor));

    const Tensor* b_ru_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("b_ru", &b_ru_tensor));

    const Tensor* b_c_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("b_c", &b_c_tensor));

    const Tensor* r_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("r", &r_tensor));

    const Tensor* u_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("u", &u_tensor));

    const Tensor* c_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("c", &c_tensor));

    const Tensor* d_h_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("d_h", &d_h_tensor));

    batch_size_ = x_tensor->dim_size(0);
    input_size_ = x_tensor->dim_size(1);
    cell_size_ = h_prev_tensor->dim_size(1);

    // Sanity checks for input shapes.

    // Shape of 'h_prev' must be [batch_size, cell_size]
    OP_REQUIRES(ctx, h_prev_tensor->dim_size(0) == batch_size_,
                errors::InvalidArgument("h_prev.dims(0) != batch_size: ",
                                        h_prev_tensor->dim_size(0), " vs. ",
                                        batch_size_));
    OP_REQUIRES(ctx, h_prev_tensor->dim_size(1) == cell_size_,
                errors::InvalidArgument(
                    "h_prev.dims(1) != cell_size: ", h_prev_tensor->dim_size(1),
                    " vs. ", cell_size_));

    // Shape of 'w_ru' must be [input_size+cell_size, 2*cell_size]
    OP_REQUIRES(
        ctx, w_ru_tensor->dim_size(0) == input_size_ + cell_size_,
        errors::InvalidArgument("w_ru.dim_size(0) != input_size + cell_size: ",
                                w_ru_tensor->dim_size(0), " vs. ",
                                input_size_ + cell_size_));

    OP_REQUIRES(ctx, w_ru_tensor->dim_size(1) == cell_size_ * 2,
                errors::InvalidArgument("w_ru.dim_size(1) != cell_size * 2: ",
                                        w_ru_tensor->dim_size(1), " vs. ",
                                        cell_size_ * 2));

    // Shape of 'w_c' must be [input_size+cell_size, cell_size]
    OP_REQUIRES(
        ctx, w_c_tensor->dim_size(0) == input_size_ + cell_size_,
        errors::InvalidArgument("w_c.dim_size(0) != input_size + cell_size: ",
                                w_c_tensor->dim_size(0), " vs. ",
                                input_size_ + cell_size_));

    OP_REQUIRES(ctx, w_c_tensor->dim_size(1) == cell_size_,
                errors::InvalidArgument(
                    "w_c.dim_size(1) != cell_size: ", w_c_tensor->dim_size(1),
                    " vs. ", cell_size_));

    // Shape of 'b_ru' must be [2*cell_size]
    OP_REQUIRES(ctx, b_ru_tensor->dim_size(0) == cell_size_ * 2,
                errors::InvalidArgument("b_ru.dim_size(0) != cell_size * 2: ",
                                        b_ru_tensor->dim_size(0), " vs. ",
                                        cell_size_ * 2));

    OP_REQUIRES(ctx, b_ru_tensor->dims() == 1,
                errors::InvalidArgument("Rank of b_ru must be 1",
                                        b_ru_tensor->dims(), " vs. 1"));

    // Shape of 'b_c' must be [cell_size]
    OP_REQUIRES(ctx, b_c_tensor->dim_size(0) == cell_size_,
                errors::InvalidArgument(
                    "b_c.dim_size(0) != cell_size: ", b_c_tensor->dim_size(0),
                    " vs. ", cell_size_));

    OP_REQUIRES(ctx, b_c_tensor->dims() == 1,
                errors::InvalidArgument("Rank of b_c must be 1 ",
                                        b_c_tensor->dims(), " vs. 1"));

    // Shape of 'r' must be [batch_size, cell_size]
    OP_REQUIRES(ctx, r_tensor->dim_size(0) == batch_size_,
                errors::InvalidArgument(
                    "r.dims(0) != batch_size: ", r_tensor->dim_size(0), " vs. ",
                    batch_size_));
    OP_REQUIRES(ctx, r_tensor->dim_size(1) == cell_size_,
                errors::InvalidArgument(
                    "r.dims(1) != cell_size: ", r_tensor->dim_size(1), " vs. ",
                    cell_size_));

    // Shape of 'u' must be [batch_size, cell_size]
    OP_REQUIRES(ctx, u_tensor->dim_size(0) == batch_size_,
                errors::InvalidArgument(
                    "u.dims(0) != batch_size: ", u_tensor->dim_size(0), " vs. ",
                    batch_size_));
    OP_REQUIRES(ctx, u_tensor->dim_size(1) == cell_size_,
                errors::InvalidArgument(
                    "u.dims(1) != cell_size: ", u_tensor->dim_size(1), " vs. ",
                    cell_size_));

    // Shape of 'c' must be [batch_size, cell_size]
    OP_REQUIRES(ctx, c_tensor->dim_size(0) == batch_size_,
                errors::InvalidArgument(
                    "c.dims(0) != batch_size: ", c_tensor->dim_size(0), " vs. ",
                    batch_size_));
    OP_REQUIRES(ctx, c_tensor->dim_size(1) == cell_size_,
                errors::InvalidArgument(
                    "c.dims(1) != cell_size: ", c_tensor->dim_size(1), " vs. ",
                    cell_size_));

    // Shape of 'd_h' must be [batch_size, cell_size]
    OP_REQUIRES(ctx, d_h_tensor->dim_size(0) == batch_size_,
                errors::InvalidArgument(
                    "d_h.dims(0) != batch_size: ", d_h_tensor->dim_size(0),
                    " vs. ", batch_size_));
    OP_REQUIRES(ctx, d_h_tensor->dim_size(1) == cell_size_,
                errors::InvalidArgument(
                    "d_h.dims(1) != cell_size: ", d_h_tensor->dim_size(1),
                    " vs. ", cell_size_));
  }

  int64 GetBatchSize() const { return batch_size_; }
  int64 GetInputSize() const { return input_size_; }
  int64 GetCellSize() const { return cell_size_; }

 private:
  int64 batch_size_ = 0;
  int64 input_size_ = 0;
  int64 cell_size_ = 0;
};

class GruGradShapeHelper : public ShapeHelper {
 public:
  using InitHelper = tensorflow::GruGradInitHelper;
  std::vector<TensorShape> GetOutputShapes(
      OpKernelContext* ctx,
      const InitializationHelper* initialization_helper) const override {
    auto init_helper = static_cast<const InitHelper*>(initialization_helper);

    std::vector<TensorShape> outputShapes;
    outputShapes.reserve(4);

    auto batch_size = init_helper->GetBatchSize();
    auto cell_size = init_helper->GetCellSize();
    auto input_size = init_helper->GetInputSize();

    // d_x_tensor shape
    outputShapes.push_back(TensorShape({batch_size, input_size}));

    // d_h_prevtensor shape
    outputShapes.push_back(TensorShape({batch_size, cell_size}));

    // d_c_bar_tensor shape
    outputShapes.push_back(TensorShape({batch_size, cell_size}));

    // d_r_bar_u_bar_tensor shape
    outputShapes.push_back(TensorShape({batch_size, 2 * cell_size}));

    return outputShapes;
  }
};

class DmlGruCellBlockGradOp : public DmlKernel {
 public:
  using InitHelper = tensorflow::GruGradInitHelper;
  explicit DmlGruCellBlockGradOp(DmlKernelConstruction* ctx,
                                 const InitHelper* init_helper) {
    CHECK(ctx->GetInputCount() == 10);
    CHECK(ctx->GetOutputCount() == 4);

    const uint32_t batch_size = init_helper->GetBatchSize();
    const uint32_t input_size = init_helper->GetInputSize();
    const uint32_t cell_size = init_helper->GetCellSize();

    DmlKernelParams params;
    params.kernel_input_indices =
        absl::InlinedVector<absl::optional<uint32_t>, 8>({1, 2, 3, 6, 7, 8, 9});
    auto tensors = GetTensorInfos(ctx, params);

    auto input_descs = GetDmlTensorDescs(tensors.inputs);
    auto output_descs = GetDmlTensorDescs(tensors.outputs);

    auto scope = dml::Graph(ctx->GetDmlDevice());
    auto h_prev = dml::InputTensor(scope, 0, input_descs[0]);
    auto w_ru = dml::InputTensor(scope, 1, input_descs[1]);
    auto w_c = dml::InputTensor(scope, 2, input_descs[2]);
    auto r = dml::InputTensor(scope, 3, input_descs[3]);
    auto u = dml::InputTensor(scope, 4, input_descs[4]);
    auto c = dml::InputTensor(scope, 5, input_descs[5]);
    auto d_h = dml::InputTensor(scope, 6, input_descs[6]);

    dml::TensorDesc::Dimensions x_offset = {0, 0, 0, 0};
    dml::TensorDesc::Dimensions x_extent = {1, 1, batch_size, input_size};
    dml::TensorDesc::Dimensions h_offset = {0, 0, 0, input_size};
    dml::TensorDesc::Dimensions h_extent = {1, 1, batch_size, cell_size};
    int32_t slice_stride[] = {1, 1, 1, 1};

    // d_c_bar = d_h*(1-u)*(1-(c*c))
    auto d_c_bar = (d_h * (1 - u)) * (1 - dml::Pow(c, 2.0f));
    // d_u_bar = d_h*(h-c)*(u*(1-u))
    auto d_u_bar = d_h * (h_prev - c) * u * (1 - u);

    // [2nd_component_of_d_x d_h_prevr] = d_c_bar X w_c^T
    auto d_x_comp2_and_h_prevr_gemm =
        dml::Gemm(d_c_bar, w_c, dml::NullOpt, DML_MATRIX_TRANSFORM_NONE,
                  DML_MATRIX_TRANSFORM_TRANSPOSE);
    dml::Expression d_x_comp2_and_h_prevr = d_x_comp2_and_h_prevr_gemm;

    auto d_hr =
        dml::Slice(d_x_comp2_and_h_prevr, h_offset, h_extent, slice_stride);
    auto d_r_bar = (d_hr * h_prev * r) * (1 - r);

    // d_r_bar_u_bar = concatenate(d_r_bar, d_u_bar) along axis = 3.
    auto d_r_bar_u_bar = dml::Join({d_r_bar, d_u_bar}, 3);

    // [1st_component_of_d_x 1st_component_of_d_h_prev] = [d_r_bar d_u_bar] X
    // w_ru^T
    auto d_x_comp1_and_h_prev_comp1_gemm =
        dml::Gemm(d_r_bar_u_bar, w_ru, dml::NullOpt, DML_MATRIX_TRANSFORM_NONE,
                  DML_MATRIX_TRANSFORM_TRANSPOSE);
    dml::Expression d_x_comp1_and_h_prev_comp1 =
        d_x_comp1_and_h_prev_comp1_gemm;
    auto comp1_plus_comp2 = d_x_comp1_and_h_prev_comp1 + d_x_comp2_and_h_prevr;

    // d_x = d_x_comp1 + d_x_comp2
    auto d_x = dml::Slice(comp1_plus_comp2, x_offset, x_extent, slice_stride);
    // d_h_prev = d_h_comp1 + d_hr*r + d_h*u
    auto d_h_prev = dml::Slice(d_x_comp1_and_h_prev_comp1, h_offset, h_extent,
                               slice_stride);
    d_h_prev += (d_hr * r);
    d_h_prev += (d_h * u);

    std::vector<dml::Expression> outputs = {d_x, d_h_prev, d_c_bar,
                                            d_r_bar_u_bar};
    Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
        scope.Compile(DML_EXECUTION_FLAG_NONE, outputs);

    Initialize(ctx, std::move(tensors), compiled_op.Get());
  }
};

#define DML_REGISTER_KERNEL(type)                                            \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("GRUBlockCellGrad").Device(DEVICE_DML).TypeConstraint<type>("T"), \
      DmlKernelWrapper<DmlGruCellBlockGradOp, GruGradShapeHelper>);
DML_REGISTER_KERNEL(float);
#undef DML_REGISTER_KERNEL

}  // namespace tensorflow