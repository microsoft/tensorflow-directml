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
#include "tensorflow/core/kernels/rnn/lstm_ops.h"

namespace tensorflow {

inline dml::TensorDesc::Dimensions DimensionFromOffset(
    const Eigen::array<Eigen::DenseIndex, 2>& offset) {
  return dml::TensorDesc::Dimensions{0, 0, static_cast<uint32_t>(offset[0]),
                                     static_cast<uint32_t>(offset[1])};
};

inline dml::TensorDesc::Dimensions DimensionFromExtent(
    const Eigen::array<Eigen::DenseIndex, 2>& extent) {
  return dml::TensorDesc::Dimensions{1, 1, static_cast<uint32_t>(extent[0]),
                                     static_cast<uint32_t>(extent[1])};
};

class LstmInitHelper : public InitializationHelper {
 public:
  struct Attributes {
    explicit Attributes(OpKernelConstruction* ctx) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("forget_bias", &forget_bias));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("cell_clip", &cell_clip));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("use_peephole", &use_peephole));
    }
    float forget_bias;
    float cell_clip;
    bool use_peephole;
  };

  LstmInitHelper(OpKernelContext* ctx, std::shared_ptr<const Attributes> attr)
      : attr_(std::move(attr)) {
    const Tensor* x_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("x", &x_tensor));

    const Tensor* cs_prev_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("cs_prev", &cs_prev_tensor));

    const Tensor* h_prev_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("h_prev", &h_prev_tensor));

    const Tensor* w_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("w", &w_tensor));

    const Tensor* wci_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("wci", &wci_tensor));

    const Tensor* wcf_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("wcf", &wcf_tensor));

    const Tensor* wco_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("wco", &wco_tensor));

    const Tensor* b_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("b", &b_tensor));

    batch_size_ = x_tensor->dim_size(0);
    input_size_ = x_tensor->dim_size(1);
    cell_size_ = cs_prev_tensor->dim_size(1);

    // Sanity checks for our input shapes.
    OP_REQUIRES(ctx, cs_prev_tensor->dim_size(0) == batch_size_,
                errors::InvalidArgument("cs_prev.dims(0) != batch_size: ",
                                        cs_prev_tensor->dim_size(0), " vs. ",
                                        batch_size_));
    OP_REQUIRES(ctx, cs_prev_tensor->dim_size(1) == cell_size_,
                errors::InvalidArgument("cs_prev.dims(1) != cell_size: ",
                                        cs_prev_tensor->dim_size(1), " vs. ",
                                        cell_size_));

    OP_REQUIRES(ctx, h_prev_tensor->dim_size(0) == batch_size_,
                errors::InvalidArgument("h_prev.dims(0) != batch_size: ",
                                        h_prev_tensor->dim_size(0), " vs. ",
                                        batch_size_));
    OP_REQUIRES(ctx, h_prev_tensor->dim_size(1) == cell_size_,
                errors::InvalidArgument(
                    "h_prev.dims(1) != cell_size: ", h_prev_tensor->dim_size(1),
                    " vs. ", cell_size_));

    OP_REQUIRES(ctx, w_tensor->dim_size(0) == input_size_ + cell_size_,
                errors::InvalidArgument(
                    "w.dim_size(0) != input_size + cell_size: ",
                    w_tensor->dim_size(0), " vs. ", input_size_ + cell_size_));

    OP_REQUIRES(ctx, w_tensor->dim_size(1) == cell_size_ * 4,
                errors::InvalidArgument(
                    "w.dim_size(1) != cell_size * 4: ", w_tensor->dim_size(1),
                    " vs. ", cell_size_ * 4));

    OP_REQUIRES(ctx, b_tensor->dim_size(0) == cell_size_ * 4,
                errors::InvalidArgument(
                    "b.dim_size(0) != cell_size * 4: ", b_tensor->dim_size(0),
                    " vs. ", cell_size_ * 4));
  }

  int64 GetBatchSize() const { return batch_size_; }
  int64 GetInputSize() const { return input_size_; }
  int64 GetCellSize() const { return cell_size_; }
  float GetForgetBias() const { return attr_->forget_bias; }
  float GetCellClip() const { return attr_->cell_clip; }
  bool GetUsePeepHole() const { return attr_->use_peephole; }

 private:
  std::shared_ptr<const Attributes> attr_;
  int64 batch_size_ = 0;
  int64 input_size_ = 0;
  int64 cell_size_ = 0;
};

class LstmShapeHelper : public ShapeHelper {
 public:
  using InitHelper = tensorflow::LstmInitHelper;
  std::vector<TensorShape> GetOutputShapes(
      OpKernelContext* ctx,
      const InitializationHelper* initialization_helper) const override {
    auto init_helper = static_cast<const InitHelper*>(initialization_helper);
    std::vector<TensorShape> outputShapes;
    outputShapes.reserve(7);

    auto batch_size = init_helper->GetBatchSize();
    auto cell_size = init_helper->GetCellSize();

    // i tensor shape
    outputShapes.push_back(TensorShape({batch_size, cell_size}));

    // cs tensor shape
    outputShapes.push_back(TensorShape({batch_size, cell_size}));

    // f tensor shape
    outputShapes.push_back(TensorShape({batch_size, cell_size}));

    // o tensor shape
    outputShapes.push_back(TensorShape({batch_size, cell_size}));

    // ci tensor shape
    outputShapes.push_back(TensorShape({batch_size, cell_size}));

    // co tensor shape
    outputShapes.push_back(TensorShape({batch_size, cell_size}));

    // h tensor shape
    outputShapes.push_back(TensorShape({batch_size, cell_size}));

    return outputShapes;
  }
};

template <typename T, GateLayout gate_layout>
class DmlLstmBlockCellOp : public DmlKernel {
 public:
  using InitHelper = tensorflow::LstmInitHelper;

  explicit DmlLstmBlockCellOp(DmlKernelConstruction* ctx,
                              const InitHelper* init_helper) {
    CHECK(ctx->GetInputCount() == 8);
    CHECK(ctx->GetOutputCount() == 7);

    const uint32_t batch_size = init_helper->GetBatchSize();
    const uint32_t input_size = init_helper->GetInputSize();
    const uint32_t cell_size = init_helper->GetCellSize();
    const float forget_bias = init_helper->GetForgetBias();
    const float cell_clip = init_helper->GetCellClip();
    const bool use_peephole = init_helper->GetUsePeepHole();

    DmlKernelParams params;
    params.kernel_input_indices = {0, 1, 2, 3};
    auto tensors = GetTensorInfos(ctx, params);

    if (use_peephole) {
      TensorShape inshape({cell_size});
      TensorShape outshape({batch_size, cell_size});
      DmlTensorInfo wci_info;
      wci_info.kernel_index = 4;
      wci_info.desc =
          DmlTensorDesc::Create(ctx->GetInputDataType(4), outshape, inshape);
      tensors.inputs.push_back(wci_info);

      DmlTensorInfo wcf_info;
      wcf_info.kernel_index = 5;
      wcf_info.desc =
          DmlTensorDesc::Create(ctx->GetInputDataType(5), outshape, inshape);
      tensors.inputs.push_back(wcf_info);

      DmlTensorInfo wco_info;
      wco_info.kernel_index = 6;
      wco_info.desc =
          DmlTensorDesc::Create(ctx->GetInputDataType(6), outshape, inshape);
      tensors.inputs.push_back(wco_info);
    }

    DmlTensorInfo b_info;
    b_info.desc = DmlTensorDesc::Create(
        ctx->GetInputDataType(7), TensorShape({batch_size, cell_size * 4}),
        TensorShape({cell_size * 4}));
    b_info.kernel_index = 7;
    tensors.inputs.push_back(b_info);

    auto input_descs = GetDmlTensorDescs(tensors.inputs);
    auto output_descs = GetDmlTensorDescs(tensors.outputs);

    functor::LSTMBlockCell cell(batch_size, input_size, cell_size);

    dml::TensorDesc::Dimensions i_offset =
        DimensionFromOffset(cell.gates_i_offsets());
    dml::TensorDesc::Dimensions c_offset =
        DimensionFromOffset(cell.gates_c_offsets(gate_layout));
    dml::TensorDesc::Dimensions f_offset =
        DimensionFromOffset(cell.gates_f_offsets(gate_layout));
    dml::TensorDesc::Dimensions o_offset =
        DimensionFromOffset(cell.gates_o_offsets());
    dml::TensorDesc::Dimensions cell_extent =
        DimensionFromExtent(cell.cell_extents());
    int32_t slice_stride[] = {1, 1, 1, 1};

    auto scope = dml::Graph(ctx->GetDmlDevice());
    auto x = dml::InputTensor(scope, 0, input_descs[0]);
    auto cs_prev = dml::InputTensor(scope, 1, input_descs[1]);
    auto h_prev = dml::InputTensor(scope, 2, input_descs[2]);
    auto w = dml::InputTensor(scope, 3, input_descs[3]);

    dml::Expression wci;
    dml::Expression wcf;
    dml::Expression wco;
    dml::Expression b;

    if (use_peephole) {
      wci = dml::InputTensor(scope, 4, input_descs[4]);
      wcf = dml::InputTensor(scope, 5, input_descs[5]);
      wco = dml::InputTensor(scope, 6, input_descs[6]);
      b = dml::InputTensor(scope, 7, input_descs[7]);
    } else {
      b = dml::InputTensor(scope, 4, input_descs[4]);
    }

    // Concat xh = [x, h].
    auto xh = dml::Join({x, h_prev}, 3);

    // states1 = xh * w + b
    auto gates_gemm = dml::Gemm(xh, w);
    dml::Expression gates = gates_gemm;
    gates += b;

    // Input gate.
    auto i = dml::Slice(gates, i_offset, cell_extent, slice_stride);
    if (use_peephole) {
      auto i_peep = cs_prev * wci;
      i = dml::ActivationSigmoid(i + i_peep);
    } else {
      i = dml::ActivationSigmoid(i);
    };

    // Cell input.
    auto ci = dml::Slice(gates, c_offset, cell_extent, slice_stride);
    ci = dml::ActivationTanh(ci);

    // Forget gate (w/ bias).
    auto f = dml::Slice(gates, f_offset, cell_extent, slice_stride);
    auto forget_bias_tensor =
        dml::ScalarTensor(scope, TfTensorTypeTraits<T>::FromFloat(forget_bias),
                          f.GetOutputDesc().sizes);
    if (use_peephole) {
      auto f_peep = cs_prev * wcf;
      f = dml::ActivationSigmoid(f + forget_bias_tensor + f_peep);
    } else {
      f = dml::ActivationSigmoid(f + forget_bias_tensor);
    }

    // cs = ci .* i + f .* cs_prev
    auto cs = i * ci + f * cs_prev;

    if (cell_clip > 0) {
      cs = dml::Clip(cs, 0, cell_clip);
    }

    // co = tanh(cs)
    auto co = dml::ActivationTanh(cs);

    // Output gate.
    auto o = dml::Slice(gates, o_offset, cell_extent, slice_stride);
    if (use_peephole) {
      auto o_peep = cs * wco;
      o = dml::ActivationSigmoid(o + o_peep);
    } else {
      o = dml::ActivationSigmoid(o);
    }

    // h = o * co
    auto h = o * co;

    std::vector<dml::Expression> outputs = {i, cs, f, o, ci, co, h};
    Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
        scope.Compile(DML_EXECUTION_FLAG_NONE, outputs);

    Initialize(ctx, std::move(tensors), compiled_op.Get());
  }
};

#define DML_REGISTER_KERNEL(type)                                         \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("LSTMBlockCell").Device(DEVICE_DML).TypeConstraint<type>("T"), \
      DmlKernelWrapper<DmlLstmBlockCellOp<type, ICFO>, LstmShapeHelper>);
TF_CALL_DML_FLOAT_TYPES(DML_REGISTER_KERNEL);
#undef DML_REGISTER_KERNEL

class LstmGradInitHelper : public InitializationHelper {
 public:
  struct Attributes {
    explicit Attributes(OpKernelConstruction* ctx) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("use_peephole", &use_peephole));
    }
    bool use_peephole;
  };

  LstmGradInitHelper(OpKernelContext* ctx,
                     std::shared_ptr<const Attributes> attr)
      : attr_(std::move(attr)) {
    const Tensor* x_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("x", &x_tensor));

    const Tensor* cs_prev_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("cs_prev", &cs_prev_tensor));

    const Tensor* h_prev_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("h_prev", &h_prev_tensor));

    const Tensor* w_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("w", &w_tensor));

    const Tensor* wci_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("wci", &wci_tensor));

    const Tensor* wcf_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("wcf", &wcf_tensor));

    const Tensor* wco_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("wco", &wco_tensor));

    const Tensor* b_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("b", &b_tensor));

    const Tensor* i_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("i", &i_tensor));

    const Tensor* cs_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("cs", &cs_tensor));

    const Tensor* f_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("f", &f_tensor));

    const Tensor* o_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("o", &o_tensor));

    const Tensor* ci_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("ci", &ci_tensor));

    const Tensor* co_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("co", &co_tensor));

    const Tensor* cs_grad_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("cs_grad", &cs_grad_tensor));

    const Tensor* h_grad_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("h_grad", &h_grad_tensor));

    batch_size_ = x_tensor->dim_size(0);
    input_size_ = x_tensor->dim_size(1);
    cell_size_ = cs_prev_tensor->dim_size(1);
    wci_shape_ = wci_tensor->shape();
    wcf_shape_ = wcf_tensor->shape();
    wco_shape_ = wco_tensor->shape();

    // Sanity checks for our input shapes.
    OP_REQUIRES(ctx, cs_prev_tensor->dim_size(0) == batch_size_,
                errors::InvalidArgument("cs_prev.dims(0) != batch_size: ",
                                        cs_prev_tensor->dim_size(0), " vs. ",
                                        batch_size_));
    OP_REQUIRES(ctx, cs_prev_tensor->dim_size(1) == cell_size_,
                errors::InvalidArgument("cs_prev.dims(1) != cell_size: ",
                                        cs_prev_tensor->dim_size(1), " vs. ",
                                        cell_size_));

    OP_REQUIRES(ctx, h_prev_tensor->dim_size(0) == batch_size_,
                errors::InvalidArgument("h_prev.dims(0) != batch_size: ",
                                        h_prev_tensor->dim_size(0), " vs. ",
                                        batch_size_));
    OP_REQUIRES(ctx, h_prev_tensor->dim_size(1) == cell_size_,
                errors::InvalidArgument(
                    "h_prev.dims(1) != cell_size: ", h_prev_tensor->dim_size(1),
                    " vs. ", cell_size_));

    OP_REQUIRES(ctx, w_tensor->dim_size(0) == input_size_ + cell_size_,
                errors::InvalidArgument(
                    "w.dim_size(0) != input_size + cell_size: ",
                    w_tensor->dim_size(0), " vs. ", input_size_ + cell_size_));
    OP_REQUIRES(ctx, w_tensor->dim_size(1) == cell_size_ * 4,
                errors::InvalidArgument(
                    "w.dim_size(1) != cell_size * 4: ", w_tensor->dim_size(1),
                    " vs. ", cell_size_ * 4));

    OP_REQUIRES(ctx, b_tensor->dim_size(0) == cell_size_ * 4,
                errors::InvalidArgument(
                    "b.dim_size(0) != cell_size * 4: ", b_tensor->dim_size(0),
                    " vs. ", cell_size_ * 4));

    OP_REQUIRES(ctx, i_tensor->dim_size(0) == batch_size_,
                errors::InvalidArgument(
                    "i.dim_size(0) != batch_size: ", i_tensor->dim_size(0),
                    " vs. ", batch_size_));
    OP_REQUIRES(ctx, i_tensor->dim_size(1) == cell_size_,
                errors::InvalidArgument(
                    "i.dim_size(1) != cell_size: ", i_tensor->dim_size(1),
                    " vs. ", cell_size_));

    OP_REQUIRES(ctx, cs_tensor->dim_size(0) == batch_size_,
                errors::InvalidArgument(
                    "cs.dim_size(0) != batch_size: ", cs_tensor->dim_size(0),
                    " vs. ", batch_size_));
    OP_REQUIRES(ctx, cs_tensor->dim_size(1) == cell_size_,
                errors::InvalidArgument(
                    "cs.dim_size(1) != cell_size: ", cs_tensor->dim_size(1),
                    " vs. ", cell_size_));

    OP_REQUIRES(ctx, f_tensor->dim_size(0) == batch_size_,
                errors::InvalidArgument(
                    "f.dim_size(0) != batch_size: ", f_tensor->dim_size(0),
                    " vs. ", batch_size_));
    OP_REQUIRES(ctx, f_tensor->dim_size(1) == cell_size_,
                errors::InvalidArgument(
                    "i.dim_size(1) != cell_size: ", f_tensor->dim_size(1),
                    " vs. ", cell_size_));

    OP_REQUIRES(ctx, o_tensor->dim_size(0) == batch_size_,
                errors::InvalidArgument(
                    "o.dim_size(0) != batch_size: ", o_tensor->dim_size(0),
                    " vs. ", batch_size_));
    OP_REQUIRES(ctx, o_tensor->dim_size(1) == cell_size_,
                errors::InvalidArgument(
                    "o.dim_size(1) != cell_size: ", o_tensor->dim_size(1),
                    " vs. ", cell_size_));

    OP_REQUIRES(ctx, ci_tensor->dim_size(0) == batch_size_,
                errors::InvalidArgument(
                    "ci.dim_size(0) != batch_size: ", ci_tensor->dim_size(0),
                    " vs. ", batch_size_));
    OP_REQUIRES(ctx, ci_tensor->dim_size(1) == cell_size_,
                errors::InvalidArgument(
                    "ci.dim_size(1) != cell_size: ", ci_tensor->dim_size(1),
                    " vs. ", cell_size_));

    OP_REQUIRES(ctx, co_tensor->dim_size(0) == batch_size_,
                errors::InvalidArgument(
                    "co.dim_size(0) != batch_size: ", co_tensor->dim_size(0),
                    " vs. ", batch_size_));
    OP_REQUIRES(ctx, co_tensor->dim_size(1) == cell_size_,
                errors::InvalidArgument(
                    "co.dim_size(1) != cell_size: ", co_tensor->dim_size(1),
                    " vs. ", cell_size_));

    OP_REQUIRES(ctx, cs_grad_tensor->dim_size(0) == batch_size_,
                errors::InvalidArgument(
                    "cs_grad_tensor.dims(0) != batch_size: ",
                    cs_grad_tensor->dim_size(0), " vs. ", batch_size_));
    OP_REQUIRES(ctx, cs_grad_tensor->dim_size(1) == cell_size_,
                errors::InvalidArgument("cs_grad_tensor.dims(1) != cell_size: ",
                                        cs_grad_tensor->dim_size(1), " vs. ",
                                        cell_size_));

    OP_REQUIRES(ctx, h_grad_tensor->dim_size(0) == batch_size_,
                errors::InvalidArgument("h_grad_tensor.dims(0) != batch_size: ",
                                        h_grad_tensor->dim_size(0), " vs. ",
                                        batch_size_));
    OP_REQUIRES(ctx, h_grad_tensor->dim_size(1) == cell_size_,
                errors::InvalidArgument("h_grad_tensor.dims(1) != cell_size: ",
                                        h_grad_tensor->dim_size(1), " vs. ",
                                        cell_size_));
  }

  bool IsNoOpKernel(
      OpKernelContext* ctx,
      absl::Span<const TensorShape> output_shapes) const override {
    for (int i = 0; i < ctx->num_inputs(); ++i) {
      const Tensor& input = ctx->input(i);

      if (input.NumElements() == 0) {
        return true;
      }
    }

    // If peephole is not used, last 3 inputs will be empty, so we don't want to
    // return true in those cases
    int count = 0;
    for (const auto& output_shape : output_shapes) {
      count++;
      if (attr_->use_peephole && count > 2) {
        continue;
      }
      if (output_shape.num_elements() == 0) {
        return true;
      }
    }

    return false;
  }

  int64 GetBatchSize() const { return batch_size_; }
  int64 GetInputSize() const { return input_size_; }
  int64 GetCellSize() const { return cell_size_; }
  bool GetUsePeepHole() const { return attr_->use_peephole; }
  TensorShape GetWciShape() const { return wci_shape_; }
  TensorShape GetWcfShape() const { return wcf_shape_; }
  TensorShape GetWcoShape() const { return wco_shape_; }

 private:
  std::shared_ptr<const Attributes> attr_;
  int64 batch_size_ = 0;
  int64 input_size_ = 0;
  int64 cell_size_ = 0;
  TensorShape wci_shape_;
  TensorShape wcf_shape_;
  TensorShape wco_shape_;
};

class LstmGradShapeHelper : public ShapeHelper {
 public:
  using InitHelper = tensorflow::LstmGradInitHelper;
  std::vector<TensorShape> GetOutputShapes(
      OpKernelContext* ctx,
      const InitializationHelper* initialization_helper) const override {
    auto init_helper = static_cast<const InitHelper*>(initialization_helper);
    std::vector<TensorShape> outputShapes;
    outputShapes.reserve(5);

    auto batch_size = init_helper->GetBatchSize();
    auto cell_size = init_helper->GetCellSize();

    const Tensor* wci_tensor = nullptr;
    ctx->input("wci", &wci_tensor);

    const Tensor* wcf_tensor = nullptr;
    ctx->input("wcf", &wcf_tensor);

    const Tensor* wco_tensor = nullptr;
    ctx->input("wco", &wco_tensor);

    // cs_prev_grad tensor shape
    outputShapes.push_back(TensorShape({batch_size, cell_size}));

    // dgates tensor shape
    outputShapes.push_back(TensorShape({batch_size, cell_size * 4}));

    // wci_grad tensor shape
    outputShapes.push_back(wci_tensor->shape());

    // wcf_grad tensor shape
    outputShapes.push_back(wcf_tensor->shape());

    // wco_grad tensor shape
    outputShapes.push_back(wco_tensor->shape());

    return outputShapes;
  }
};

class DmlLstmCellBlockGradOp : public DmlKernel {
 public:
  using InitHelper = tensorflow::LstmGradInitHelper;
  explicit DmlLstmCellBlockGradOp(DmlKernelConstruction* ctx,
                                  const InitHelper* init_helper) {
    CHECK(ctx->GetInputCount() == 16);
    CHECK(ctx->GetOutputCount() == 5);

    const uint32_t batch_size = init_helper->GetBatchSize();
    const uint32_t input_size = init_helper->GetInputSize();
    const uint32_t cell_size = init_helper->GetCellSize();
    const bool use_peephole = init_helper->GetUsePeepHole();

    constexpr int cs_prev_ind = 1;
    constexpr int wci_ind = 4;
    constexpr int wcf_ind = 5;
    constexpr int wco_ind = 6;
    constexpr int i_ind = 8;
    constexpr int cs_ind = 9;
    constexpr int f_ind = 10;
    constexpr int o_ind = 11;
    constexpr int ci_ind = 12;
    constexpr int co_ind = 13;
    constexpr int cs_grad_ind = 14;
    constexpr int h_grad_ind = 15;

    DmlKernelParams params;
    if (!use_peephole) {
      params.kernel_input_indices = {cs_prev_ind, i_ind,     f_ind,
                                     o_ind,       ci_ind,    co_ind,
                                     cs_grad_ind, h_grad_ind};
    } else {
      params.kernel_input_indices = {
          cs_prev_ind, wci_ind, wcf_ind, wco_ind, i_ind,       cs_ind,
          f_ind,       o_ind,   ci_ind,  co_ind,  cs_grad_ind, h_grad_ind};
    }
    auto tensors = GetTensorInfos(ctx, params);

    if (use_peephole) {
      TensorShape inshape({cell_size});
      TensorShape outshape({batch_size, cell_size});
      DmlTensorInfo wci_info;
      wci_info.kernel_index = wci_ind;
      wci_info.desc = DmlTensorDesc::Create(ctx->GetInputDataType(wci_ind),
                                            outshape, inshape);
      tensors.inputs[1] = wci_info;

      DmlTensorInfo wcf_info;
      wcf_info.kernel_index = wcf_ind;
      wcf_info.desc = DmlTensorDesc::Create(ctx->GetInputDataType(wcf_ind),
                                            outshape, inshape);
      tensors.inputs[2] = wcf_info;

      DmlTensorInfo wco_info;
      wco_info.kernel_index = wco_ind;
      wco_info.desc = DmlTensorDesc::Create(ctx->GetInputDataType(wco_ind),
                                            outshape, inshape);
      tensors.inputs[3] = wco_info;
    }

    auto input_descs = GetDmlTensorDescs(tensors.inputs);
    auto output_descs = GetDmlTensorDescs(tensors.outputs);

    auto scope = dml::Graph(ctx->GetDmlDevice());

    auto cs_prev = dml::InputTensor(scope, 0, input_descs[0]);

    dml::Expression wci, wcf, wco, b, i, cs, f, o, ci, co, cs_grad, h_grad;

    if (use_peephole) {
      wci = dml::InputTensor(scope, 1, input_descs[1]);  // peephole only
      wcf = dml::InputTensor(scope, 2, input_descs[2]);  // peephole only
      wco = dml::InputTensor(scope, 3, input_descs[3]);  // peephole only
      i = dml::InputTensor(scope, 4, input_descs[4]);
      cs = dml::InputTensor(scope, 5, input_descs[5]);  // peephole only
      f = dml::InputTensor(scope, 6, input_descs[6]);
      o = dml::InputTensor(scope, 7, input_descs[7]);
      ci = dml::InputTensor(scope, 8, input_descs[8]);
      co = dml::InputTensor(scope, 9, input_descs[9]);
      cs_grad = dml::InputTensor(scope, 10, input_descs[10]);
      h_grad = dml::InputTensor(scope, 11, input_descs[11]);
    } else {
      i = dml::InputTensor(scope, 1, input_descs[1]);
      f = dml::InputTensor(scope, 2, input_descs[2]);
      o = dml::InputTensor(scope, 3, input_descs[3]);
      ci = dml::InputTensor(scope, 4, input_descs[4]);
      co = dml::InputTensor(scope, 5, input_descs[5]);
      cs_grad = dml::InputTensor(scope, 6, input_descs[6]);
      h_grad = dml::InputTensor(scope, 7, input_descs[7]);
    }

    // do[t] = sigm'(o[t]) .* dh[t] .* co[t]
    auto do_tensor = o * (1 - o) * h_grad * co;

    // dcs[t] += tanh'(cs[t]) .* dh[t] .* o[t] + dcs[t + 1] .* f[t + 1]
    auto dcs = (1 - dml::Pow(co, 2.0f)) * h_grad * o + cs_grad;

    if (use_peephole) {
      dcs = dcs + do_tensor * wco;
    }

    // dci[t] = tanh'(ci[t]) dcs[t] i[t]
    auto dci = (1 - dml::Pow(ci, 2.0f)) * dcs * i;

    // df[t] = sigm'(f[t]) dcs[t] cs[t - 1]
    auto df = f * (1 - f) * dcs * cs_prev;

    // di[t] = sigm'(i[t]) dcs[t] ci[t]
    auto di = i * (1 - i) * dcs * ci;

    // Use ICFO gate layout as in CPU
    auto dgates = dml::Join({di, dci, df, do_tensor}, 3);

    auto cs_prev_grad = dcs * f;

    dml::Expression wci_grad;
    dml::Expression wcf_grad;
    dml::Expression wco_grad;

    absl::InlinedVector<uint32_t, 4> wci_shape =
        NarrowTensorShape(init_helper->GetWciShape());
    absl::InlinedVector<uint32_t, 4> wcf_shape =
        NarrowTensorShape(init_helper->GetWcfShape());
    absl::InlinedVector<uint32_t, 4> wco_shape =
        NarrowTensorShape(init_helper->GetWcoShape());

    dml::TensorDesc::Dimensions wci_dims(wci_shape.begin(), wci_shape.end());
    dml::TensorDesc::Dimensions wcf_dims(wcf_shape.begin(), wcf_shape.end());
    dml::TensorDesc::Dimensions wco_dims(wco_shape.begin(), wco_shape.end());

    if (use_peephole) {
      cs_prev_grad = cs_prev_grad + (di * wci) + (df * wcf);
      wci_grad = dml::Reduce(di * cs_prev, DML_REDUCE_FUNCTION_SUM, {0});
      wcf_grad = dml::Reduce(df * cs_prev, DML_REDUCE_FUNCTION_SUM, {0});
      wco_grad = dml::Reduce(do_tensor * cs, DML_REDUCE_FUNCTION_SUM, {0});
    } else {
      wci_grad = dml::ZeroTensor(scope, DML_TENSOR_DATA_TYPE_FLOAT32, wci_dims);
      wcf_grad = dml::ZeroTensor(scope, DML_TENSOR_DATA_TYPE_FLOAT32, wcf_dims);
      wco_grad = dml::ZeroTensor(scope, DML_TENSOR_DATA_TYPE_FLOAT32, wco_dims);
    }

    std::vector<dml::Expression> outputs = {cs_prev_grad, dgates, wci_grad,
                                            wcf_grad, wco_grad};
    Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
        scope.Compile(DML_EXECUTION_FLAG_NONE, outputs);

    Initialize(ctx, std::move(tensors), compiled_op.Get());
  }
};

#define DML_REGISTER_KERNEL(type)                                             \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("LSTMBlockCellGrad").Device(DEVICE_DML).TypeConstraint<type>("T"), \
      DmlKernelWrapper<DmlLstmCellBlockGradOp, LstmGradShapeHelper>);
TF_CALL_DML_FLOAT_TYPES(DML_REGISTER_KERNEL);
#undef DML_REGISTER_KERNEL

class BlockLstmInitHelper : public InitializationHelper {
 public:
  struct Attributes {
    explicit Attributes(OpKernelConstruction* ctx) {
      if (ctx->HasAttr("forget_bias")) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("forget_bias", &forget_bias));
      } else {
        // V2 version does not have "forget_bias" attribute.
        forget_bias = 0.0;
      }
      OP_REQUIRES_OK(ctx, ctx->GetAttr("cell_clip", &cell_clip));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("use_peephole", &use_peephole));
    }
    float forget_bias;
    float cell_clip;
    bool use_peephole;
  };

  BlockLstmInitHelper(OpKernelContext* ctx,
                      std::shared_ptr<const Attributes> attr)
      : attr_(std::move(attr)) {
    const Tensor* seq_len_max_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("seq_len_max", &seq_len_max_tensor));

    const Tensor* x = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("x", &x));

    const Tensor* cs_prev_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("cs_prev", &cs_prev_tensor));

    const Tensor* h_prev_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("h_prev", &h_prev_tensor));

    const Tensor* w_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("w", &w_tensor));

    const Tensor* wci_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("wci", &wci_tensor));

    const Tensor* wcf_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("wcf", &wcf_tensor));

    const Tensor* wco_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("wco", &wco_tensor));

    const Tensor* b_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("b", &b_tensor));

    timelen_ = x->dim_size(0);
    batch_size_ = x->dim_size(1);
    input_size_ = x->dim_size(2);
    cell_size_ = cs_prev_tensor->dim_size(1);

    if (batch_size_ * input_size_ % 2 == 1) {
      LOG(WARNING) << "BlockLSTMOp is inefficient when both batch_size and "
                   << "input_size are odd. You are using: batch_size="
                   << batch_size_ << ", input_size=" << input_size_;
    }
    if (batch_size_ * cell_size_ % 2 == 1) {
      LOG(WARNING) << "BlockLSTMOp is inefficient when both batch_size and "
                   << "cell_size are odd. You are using: batch_size="
                   << batch_size_ << ", cell_size=" << cell_size_;
    }

    // Sanity checks for our input shapes.
    OP_REQUIRES(ctx, x->dims() == 3, errors::InvalidArgument("x must be 3D"));

    OP_REQUIRES(ctx, cs_prev_tensor->dims() == 2,
                errors::InvalidArgument("cs_prev must be 2D"));

    OP_REQUIRES(ctx, cs_prev_tensor->dim_size(0) == batch_size_,
                errors::InvalidArgument("cs_prev.dims(0) != batch_size: ",
                                        cs_prev_tensor->dim_size(0), " vs. ",
                                        batch_size_));

    OP_REQUIRES(ctx, h_prev_tensor->dims() == 2,
                errors::InvalidArgument("h_prev must be 2D"));
    OP_REQUIRES(ctx, h_prev_tensor->dim_size(0) == batch_size_,
                errors::InvalidArgument("h_prev.dims(0) != batch_size: ",
                                        h_prev_tensor->dim_size(0), " vs. ",
                                        batch_size_));
    OP_REQUIRES(ctx, h_prev_tensor->dim_size(1) == cell_size_,
                errors::InvalidArgument(
                    "h_prev.dims(1) != cell_size: ", h_prev_tensor->dim_size(1),
                    " vs. ", cell_size_));

    OP_REQUIRES(ctx, w_tensor->dims() == 2,
                errors::InvalidArgument("w must be 2D"));
    OP_REQUIRES(ctx, w_tensor->dim_size(0) == input_size_ + cell_size_,
                errors::InvalidArgument(
                    "w.dim_size(0) != input_size + cell_size: ",
                    w_tensor->dim_size(0), " vs. ", input_size_ + cell_size_));
    OP_REQUIRES(ctx, w_tensor->dim_size(1) == cell_size_ * 4,
                errors::InvalidArgument(
                    "w.dim_size(1) != cell_size * 4: ", w_tensor->dim_size(1),
                    " vs. ", cell_size_ * 4));

    OP_REQUIRES(ctx, wci_tensor->dims() == 1,
                errors::InvalidArgument("wci must be 1D"));
    OP_REQUIRES(ctx, wci_tensor->dim_size(0) == cell_size_,
                errors::InvalidArgument(
                    "wci.dim_size(0) != cell_size: ", wci_tensor->dim_size(0),
                    " vs. ", cell_size_));

    OP_REQUIRES(ctx, wcf_tensor->dims() == 1,
                errors::InvalidArgument("wcf must be 1D"));
    OP_REQUIRES(ctx, wcf_tensor->dim_size(0) == cell_size_,
                errors::InvalidArgument(
                    "wcf.dim_size(0) != cell_size: ", wcf_tensor->dim_size(0),
                    " vs. ", cell_size_));

    OP_REQUIRES(ctx, wco_tensor->dims() == 1,
                errors::InvalidArgument("wco must be 1D"));
    OP_REQUIRES(ctx, wco_tensor->dim_size(0) == cell_size_,
                errors::InvalidArgument(
                    "wco.dim_size(0) != cell_size: ", wco_tensor->dim_size(0),
                    " vs. ", cell_size_));

    OP_REQUIRES(ctx, b_tensor->dims() == 1,
                errors::InvalidArgument("b must be 1D"));
    OP_REQUIRES(ctx, b_tensor->dim_size(0) == cell_size_ * 4,
                errors::InvalidArgument(
                    "b.dim_size(0) != cell_size * 4: ", b_tensor->dim_size(0),
                    " vs. ", cell_size_ * 4));
  }

  int64 GetBatchSize() const { return batch_size_; }
  int64 GetInputSize() const { return input_size_; }
  int64 GetCellSize() const { return cell_size_; }
  int64 GetTimeLength() const { return timelen_; }
  float GetForgetBias() const { return attr_->forget_bias; }
  float GetCellClip() const { return attr_->cell_clip; }
  bool GetUsePeepHole() const { return attr_->use_peephole; }

 private:
  std::shared_ptr<const Attributes> attr_;
  int64 timelen_ = 0;
  int64 batch_size_ = 0;
  int64 input_size_ = 0;
  int64 cell_size_ = 0;
};

class BlockLstmShapeHelper : public ShapeHelper {
 public:
  using InitHelper = tensorflow::BlockLstmInitHelper;
  std::vector<TensorShape> GetOutputShapes(
      OpKernelContext* ctx,
      const InitializationHelper* initialization_helper) const override {
    auto init_helper = static_cast<const InitHelper*>(initialization_helper);
    std::vector<TensorShape> outputShapes;
    outputShapes.reserve(7);

    auto timelen = init_helper->GetTimeLength();
    auto batch_size = init_helper->GetBatchSize();
    auto cell_size = init_helper->GetCellSize();

    CHECK(ctx->input_memory_type(0) == HOST_MEMORY);

    TensorShape batch_cell_shape({timelen, batch_size, cell_size});

    // i tensor shape
    outputShapes.push_back(batch_cell_shape);

    // cs tensor shape
    outputShapes.push_back(batch_cell_shape);

    // f tensor shape
    outputShapes.push_back(batch_cell_shape);

    // o tensor shape
    outputShapes.push_back(batch_cell_shape);

    // ci tensor shape
    outputShapes.push_back(batch_cell_shape);

    // co tensor shape
    outputShapes.push_back(batch_cell_shape);

    // h tensor shape
    outputShapes.push_back(batch_cell_shape);

    return outputShapes;
  }
};

template <typename T, GateLayout gate_layout>
class DmlBlockLstmOp : public DmlKernel {
 public:
  using InitHelper = tensorflow::BlockLstmInitHelper;

  explicit DmlBlockLstmOp(DmlKernelConstruction* ctx,
                          const InitHelper* init_helper) {
    CHECK(ctx->GetInputCount() == 9);
    CHECK(ctx->GetOutputCount() == 7);

    const Tensor& seq_len_max = ctx->GetConstantInputTensor(0);
    const int32 seq_len_max_int =
        static_cast<int32>(seq_len_max.scalar<int64>()());

    skip_ = seq_len_max_int == 0 ? true : false;

    if (skip_) {
      DmlKernel::InitializeAsNoOp(ctx);
      return;
    }

    const uint32_t timelen = init_helper->GetTimeLength();
    const uint32_t batch_size = init_helper->GetBatchSize();
    const uint32_t input_size = init_helper->GetInputSize();
    const uint32_t cell_size = init_helper->GetCellSize();
    const float forget_bias = init_helper->GetForgetBias();
    const float cell_clip = init_helper->GetCellClip();
    const bool use_peephole = init_helper->GetUsePeepHole();

    DmlKernelParams params;
    params.kernel_input_indices = {1, 2, 3, 4};
    auto tensors = GetTensorInfos(ctx, params);

    if (use_peephole) {
      TensorShape inshape({cell_size});
      TensorShape outshape({batch_size, cell_size});
      DmlTensorInfo wci_info;
      wci_info.kernel_index = 5;
      wci_info.desc =
          DmlTensorDesc::Create(ctx->GetInputDataType(5), outshape, inshape);
      tensors.inputs.push_back(wci_info);

      DmlTensorInfo wcf_info;
      wcf_info.kernel_index = 6;
      wcf_info.desc =
          DmlTensorDesc::Create(ctx->GetInputDataType(6), outshape, inshape);
      tensors.inputs.push_back(wcf_info);

      DmlTensorInfo wco_info;
      wco_info.kernel_index = 7;
      wco_info.desc =
          DmlTensorDesc::Create(ctx->GetInputDataType(7), outshape, inshape);
      tensors.inputs.push_back(wco_info);
    }

    DmlTensorInfo b_info;
    b_info.desc = DmlTensorDesc::Create(
        ctx->GetInputDataType(8), TensorShape({batch_size, cell_size * 4}),
        TensorShape({cell_size * 4}));
    b_info.kernel_index = 8;
    tensors.inputs.push_back(b_info);

    auto input_descs = GetDmlTensorDescs(tensors.inputs);

    functor::LSTMBlockCell cell(batch_size, input_size, cell_size);

    dml::TensorDesc::Dimensions i_offset =
        DimensionFromOffset(cell.gates_i_offsets());
    dml::TensorDesc::Dimensions c_offset =
        DimensionFromOffset(cell.gates_c_offsets(gate_layout));
    dml::TensorDesc::Dimensions f_offset =
        DimensionFromOffset(cell.gates_f_offsets(gate_layout));
    dml::TensorDesc::Dimensions o_offset =
        DimensionFromOffset(cell.gates_o_offsets());
    dml::TensorDesc::Dimensions cell_extent =
        DimensionFromExtent(cell.cell_extents());
    int32_t slice_stride[] = {1, 1, 1, 1};

    auto scope = dml::Graph(ctx->GetDmlDevice());

    dml::Expression wci, wcf, wco, b;

    auto x = dml::InputTensor(scope, 0, input_descs[0]);
    auto cs_prev = dml::InputTensor(scope, 1, input_descs[1]);
    auto h_prev = dml::InputTensor(scope, 2, input_descs[2]);
    auto w = dml::InputTensor(scope, 3, input_descs[3]);

    if (use_peephole) {
      wci = dml::InputTensor(scope, 4, input_descs[4]);
      wcf = dml::InputTensor(scope, 5, input_descs[5]);
      wco = dml::InputTensor(scope, 6, input_descs[6]);
      b = dml::InputTensor(scope, 7, input_descs[7]);
    } else {
      b = dml::InputTensor(scope, 4, input_descs[4]);
    }

    dml::TensorDesc::Dimensions output_dims{1, timelen, batch_size, cell_size};
    dml::TensorDesc::Dimensions x_extent{1, 1, batch_size, input_size};
    dml::TensorDesc::Dimensions output_extent{1, 1, batch_size, cell_size};

    std::vector<dml::Expression> i_tensors;
    std::vector<dml::Expression> cs_tensors;
    std::vector<dml::Expression> f_tensors;
    std::vector<dml::Expression> o_tensors;
    std::vector<dml::Expression> ci_tensors;
    std::vector<dml::Expression> co_tensors;
    std::vector<dml::Expression> h_tensors;

    i_tensors.reserve(seq_len_max_int);
    cs_tensors.reserve(seq_len_max_int);
    f_tensors.reserve(seq_len_max_int);
    o_tensors.reserve(seq_len_max_int);
    ci_tensors.reserve(seq_len_max_int);
    co_tensors.reserve(seq_len_max_int);
    h_tensors.reserve(seq_len_max_int);

    for (uint32_t t = 0; t < seq_len_max_int; ++t) {
      dml::TensorDesc::Dimensions tensor_offset{0, t, 0, 0};
      dml::TensorDesc::Dimensions prev_offset{0, t - 1, 0, 0};

      auto x_tensor = dml::Slice(x, tensor_offset, x_extent, slice_stride);

      auto cs_prev_tensor = t == 0 ? cs_prev : cs_tensors.at(t - 1);
      auto h_prev_tensor = t == 0 ? h_prev : h_tensors.at(t - 1);

      // Concat xh = [x, h].
      auto xh = dml::Join({x_tensor, h_prev_tensor}, 3);

      // states1 = xh * w + b
      auto gates_gemm = dml::Gemm(xh, w);
      dml::Expression gates = gates_gemm;
      gates += b;

      // Input gate.
      auto i = dml::Slice(gates, i_offset, cell_extent, slice_stride);
      if (use_peephole) {
        auto i_peep = cs_prev_tensor * wci;
        i = dml::ActivationSigmoid(i + i_peep);
      } else {
        i = dml::ActivationSigmoid(i);
      };

      // Cell input.
      auto ci = dml::Slice(gates, c_offset, cell_extent, slice_stride);
      ci = dml::ActivationTanh(ci);

      // Forget gate (w/ bias).
      auto f = dml::Slice(gates, f_offset, cell_extent, slice_stride);
      auto forget_bias_tensor = dml::ScalarTensor(
          scope, TfTensorTypeTraits<T>::FromFloat(forget_bias),
          f.GetOutputDesc().sizes);
      if (use_peephole) {
        auto f_peep = cs_prev_tensor * wcf;
        f = dml::ActivationSigmoid(f + forget_bias_tensor + f_peep);
      } else {
        f = dml::ActivationSigmoid(f + forget_bias_tensor);
      }

      // cs = ci .* i + f .* cs_prev
      auto cs = i * ci + f * cs_prev_tensor;

      if (cell_clip > 0) {
        cs = dml::Clip(cs, -1.0, cell_clip);
      }

      // co = tanh(cs)
      auto co = dml::ActivationTanh(cs);

      // Output gate.
      auto o = dml::Slice(gates, o_offset, cell_extent, slice_stride);
      if (use_peephole) {
        auto o_peep = cs * wco;
        o = dml::ActivationSigmoid(o + o_peep);
      } else {
        o = dml::ActivationSigmoid(o);
      }

      // h = o * co
      auto h = o * co;

      // add to vectors of tensors
      i_tensors.push_back(i);
      cs_tensors.push_back(cs);
      f_tensors.push_back(f);
      o_tensors.push_back(o);
      ci_tensors.push_back(ci);
      co_tensors.push_back(co);
      h_tensors.push_back(h);
    }

    auto i = dml::Join(i_tensors, 1);
    auto cs = dml::Join(cs_tensors, 1);
    auto f = dml::Join(f_tensors, 1);
    auto o = dml::Join(o_tensors, 1);
    auto ci = dml::Join(ci_tensors, 1);
    auto co = dml::Join(co_tensors, 1);
    auto h = dml::Join(h_tensors, 1);

    std::vector<dml::Expression> outputs = {i, cs, f, o, ci, co, h};
    Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
        scope.Compile(DML_EXECUTION_FLAG_NONE, outputs);

    Initialize(ctx, std::move(tensors), compiled_op.Get());
  }

  StatusOr<DmlGpuEvent> Compute(DmlKernelContext* ctx) const override {
    if (skip_) {
      uint32_t num_out = ctx->GetOutputCount();
      for (uint32_t i = 0; i < num_out; ++i) {
        Tensor* output = ctx->GetOutputTensor(i);
        ctx->ZeroBuffer(ctx->CreateBufferForTensor(*output));
      }
      return ctx->GetCurrentCompletionEvent();
    }
    return DmlKernel::Compute(ctx);
  }

 private:
  bool skip_ = false;
};

#define DML_REGISTER_KERNEL(type)                                          \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("BlockLSTM")                                                    \
          .Device(DEVICE_DML)                                              \
          .HostMemory("seq_len_max")                                       \
          .TypeConstraint<type>("T"),                                      \
      DmlKernelWrapper<DmlBlockLstmOp<type, ICFO>, BlockLstmShapeHelper>); \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("BlockLSTMV2")                                                  \
          .Device(DEVICE_DML)                                              \
          .HostMemory("seq_len_max")                                       \
          .TypeConstraint<type>("T"),                                      \
      DmlKernelWrapper<DmlBlockLstmOp<type, IFCO>, BlockLstmShapeHelper>);
TF_CALL_DML_FLOAT_TYPES(DML_REGISTER_KERNEL);
#undef DML_REGISTER_KERNEL

class BlockLstmGradInitHelper : public InitializationHelper {
 public:
  struct Attributes {
    explicit Attributes(OpKernelConstruction* ctx) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("use_peephole", &use_peephole));
    }
    bool use_peephole;
  };

  BlockLstmGradInitHelper(OpKernelContext* ctx,
                          std::shared_ptr<const Attributes> attr)
      : attr_(std::move(attr)) {
    const Tensor* seq_len_max_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("seq_len_max", &seq_len_max_tensor));

    const Tensor* x = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("x", &x));

    const Tensor* cs_prev_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("cs_prev", &cs_prev_tensor));

    const Tensor* h_prev_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("h_prev", &h_prev_tensor));

    const Tensor* w_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("w", &w_tensor));

    const Tensor* wci_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("wci", &wci_tensor));

    const Tensor* wcf_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("wcf", &wcf_tensor));

    const Tensor* wco_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("wco", &wco_tensor));

    const Tensor* b_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("b", &b_tensor));

    const Tensor* i_out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("i", &i_out));

    const Tensor* cs_out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("cs", &cs_out));

    const Tensor* f_out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("f", &f_out));

    const Tensor* o_out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("o", &o_out));

    const Tensor* ci_out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("ci", &ci_out));

    const Tensor* co_out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("co", &co_out));

    const Tensor* h_out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("h", &h_out));

    const Tensor* cs_grad = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("cs_grad", &cs_grad));

    const Tensor* h_grad = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("h_grad", &h_grad));

    timelen_ = x->dim_size(0);
    batch_size_ = x->dim_size(1);
    input_size_ = x->dim_size(2);
    cell_size_ = w_tensor->dim_size(1) / 4;

    // Sanity checks for our input shapes.
    OP_REQUIRES(ctx, x->dims() == 3, errors::InvalidArgument("x must be 3D"));

    OP_REQUIRES(ctx, cs_prev_tensor->dims() == 2,
                errors::InvalidArgument("cs_prev must be 2D"));
    OP_REQUIRES(ctx, cs_prev_tensor->dim_size(0) == batch_size_,
                errors::InvalidArgument("cs_prev.dims(0) != batch_size: ",
                                        cs_prev_tensor->dim_size(0), " vs. ",
                                        batch_size_));
    OP_REQUIRES(ctx, cs_prev_tensor->dim_size(1) == cell_size_,
                errors::InvalidArgument("cs_prev.dims(1) != cell_size: ",
                                        cs_prev_tensor->dim_size(1), " vs. ",
                                        cell_size_));

    OP_REQUIRES(ctx, h_prev_tensor->dims() == 2,
                errors::InvalidArgument("h_prev must be 2D"));
    OP_REQUIRES(ctx, h_prev_tensor->dim_size(0) == batch_size_,
                errors::InvalidArgument("h_prev.dims(0) != batch_size: ",
                                        h_prev_tensor->dim_size(0), " vs. ",
                                        batch_size_));
    OP_REQUIRES(ctx, h_prev_tensor->dim_size(1) == cell_size_,
                errors::InvalidArgument(
                    "h_prev.dims(1) != cell_size: ", h_prev_tensor->dim_size(1),
                    " vs. ", cell_size_));

    OP_REQUIRES(ctx, w_tensor->dims() == 2,
                errors::InvalidArgument("w must be 2D"));
    OP_REQUIRES(ctx, input_size_ + cell_size_ == w_tensor->dim_size(0),
                errors::InvalidArgument(
                    "w matrix rows don't match: ", input_size_ + cell_size_,
                    " vs. ", w_tensor->dim_size(0)));

    OP_REQUIRES(ctx, wci_tensor->dims() == 1,
                errors::InvalidArgument("wci must be 1D"));
    OP_REQUIRES(ctx, wci_tensor->dim_size(0) == cell_size_,
                errors::InvalidArgument(
                    "wci.dim_size(0) != cell_size: ", wci_tensor->dim_size(0),
                    " vs. ", cell_size_));

    OP_REQUIRES(ctx, wcf_tensor->dims() == 1,
                errors::InvalidArgument("wcf must be 1D"));
    OP_REQUIRES(ctx, wcf_tensor->dim_size(0) == cell_size_,
                errors::InvalidArgument(
                    "wcf.dim_size(0) != cell_size: ", wcf_tensor->dim_size(0),
                    " vs. ", cell_size_));

    OP_REQUIRES(ctx, wco_tensor->dims() == 1,
                errors::InvalidArgument("wco must be 1D"));
    OP_REQUIRES(ctx, wco_tensor->dim_size(0) == cell_size_,
                errors::InvalidArgument(
                    "wco.dim_size(0) != cell_size: ", wco_tensor->dim_size(0),
                    " vs. ", cell_size_));

    OP_REQUIRES(ctx, b_tensor->dims() == 1,
                errors::InvalidArgument("b must be 1D"));
    OP_REQUIRES(
        ctx, cell_size_ == b_tensor->dim_size(0) / 4,
        errors::InvalidArgument("w and b cell_size don't match: ", cell_size_,
                                " vs. ", b_tensor->dim_size(0)));
  }

  bool IsNoOpKernel(
      OpKernelContext* ctx,
      absl::Span<const TensorShape> output_shapes) const override {
    for (int i = 0; i < ctx->num_inputs(); ++i) {
      const Tensor& input = ctx->input(i);

      if (input.NumElements() == 0) {
        return true;
      }
    }

    // If peephole is not used, wci/wcf/wco will be empty, so we don't
    // want to return true in those cases
    int count = 0;
    for (const auto& output_shape : output_shapes) {
      count++;
      if (attr_->use_peephole && count > 4 && count < 8) {
        continue;
      }
      if (output_shape.num_elements() == 0) {
        return true;
      }
    }

    return false;
  }

  int64 GetBatchSize() const { return batch_size_; }
  int64 GetInputSize() const { return input_size_; }
  int64 GetCellSize() const { return cell_size_; }
  int64 GetTimeLength() const { return timelen_; }
  bool GetUsePeepHole() const { return attr_->use_peephole; }

 private:
  std::shared_ptr<const Attributes> attr_;
  int64 timelen_ = 0;
  int64 batch_size_ = 0;
  int64 input_size_ = 0;
  int64 cell_size_ = 0;
};

class BlockLstmGradShapeHelper : public ShapeHelper {
 public:
  using InitHelper = tensorflow::BlockLstmGradInitHelper;
  std::vector<TensorShape> GetOutputShapes(
      OpKernelContext* ctx,
      const InitializationHelper* initialization_helper) const override {
    auto init_helper = static_cast<const InitHelper*>(initialization_helper);
    std::vector<TensorShape> outputShapes;
    outputShapes.reserve(8);

    auto timelen = init_helper->GetTimeLength();
    auto batch_size = init_helper->GetBatchSize();
    auto input_size = init_helper->GetInputSize();

    CHECK(ctx->input_memory_type(0) == HOST_MEMORY);

    TensorShape batch_input_shape({timelen, batch_size, input_size});

    const Tensor* cs_prev_tensor = nullptr;
    ctx->input("cs_prev", &cs_prev_tensor);

    const Tensor* h_prev_tensor = nullptr;
    ctx->input("h_prev", &h_prev_tensor);

    const Tensor* w_tensor = nullptr;
    ctx->input("w", &w_tensor);

    const Tensor* wci_tensor = nullptr;
    ctx->input("wci", &wci_tensor);

    const Tensor* wcf_tensor = nullptr;
    ctx->input("wcf", &wcf_tensor);

    const Tensor* wco_tensor = nullptr;
    ctx->input("wco", &wco_tensor);

    const Tensor* b_tensor = nullptr;
    ctx->input("b", &b_tensor);

    // x_grad tensor shape
    outputShapes.push_back(batch_input_shape);

    // cs_prev_grad tensor shape
    outputShapes.push_back(cs_prev_tensor->shape());

    // h_prev_grad tensor shape
    outputShapes.push_back(h_prev_tensor->shape());

    // w_grad tensor shape
    outputShapes.push_back(w_tensor->shape());

    // wci_grad tensor shape
    outputShapes.push_back(wci_tensor->shape());

    // wcf_grad tensor shape
    outputShapes.push_back(wcf_tensor->shape());

    // wco_grad tensor shape
    outputShapes.push_back(wco_tensor->shape());

    // b_grad tensor shape
    outputShapes.push_back(b_tensor->shape());

    return outputShapes;
  }
};

template <typename T, GateLayout gate_layout>
class DmlBlockLstmGradOp : public DmlKernel {
 public:
  using InitHelper = tensorflow::BlockLstmGradInitHelper;

  explicit DmlBlockLstmGradOp(DmlKernelConstruction* ctx,
                              const InitHelper* init_helper) {
    CHECK(ctx->GetInputCount() == 18);
    CHECK(ctx->GetOutputCount() == 8);

    const Tensor& seq_len_max = ctx->GetConstantInputTensor(0);
    const int32 seq_len_max_int =
        static_cast<int32>(seq_len_max.scalar<int64>()());

    skip_ = seq_len_max_int == 0 ? true : false;

    if (skip_) {
      DmlKernel::InitializeAsNoOp(ctx);
      return;
    }

    const uint32_t timelen = init_helper->GetTimeLength();
    const uint32_t batch_size = init_helper->GetBatchSize();
    const uint32_t input_size = init_helper->GetInputSize();
    const uint32_t cell_size = init_helper->GetCellSize();
    const bool use_peephole = init_helper->GetUsePeepHole();

    constexpr int x_ind = 1;
    constexpr int cs_prev_ind = 2;
    constexpr int h_prev_ind = 3;
    constexpr int w_ind = 4;
    constexpr int wci_ind = 5;
    constexpr int wcf_ind = 6;
    constexpr int wco_ind = 7;
    constexpr int i_ind = 9;
    constexpr int cs_ind = 10;
    constexpr int f_ind = 11;
    constexpr int o_ind = 12;
    constexpr int ci_ind = 13;
    constexpr int co_ind = 14;
    constexpr int h_ind = 15;
    constexpr int cs_grad_ind = 16;
    constexpr int h_grad_ind = 17;

    DmlKernelParams params;

    if (use_peephole) {
      if (seq_len_max_int == 1) {
        // no h
        params.kernel_input_indices = {x_ind,   cs_prev_ind, h_prev_ind, w_ind,
                                       wci_ind, wcf_ind,     wco_ind,    i_ind,
                                       cs_ind,  f_ind,       o_ind,      ci_ind,
                                       co_ind,  cs_grad_ind, h_grad_ind};
      } else {
        params.kernel_input_indices = {
            x_ind,   cs_prev_ind, h_prev_ind,  w_ind,     wci_ind, wcf_ind,
            wco_ind, i_ind,       cs_ind,      f_ind,     o_ind,   ci_ind,
            co_ind,  h_ind,       cs_grad_ind, h_grad_ind};
      }
    } else {
      if (seq_len_max_int == 1) {
        // no cs, h
        params.kernel_input_indices = {x_ind,  cs_prev_ind, h_prev_ind, w_ind,
                                       i_ind,  f_ind,       o_ind,      ci_ind,
                                       co_ind, cs_grad_ind, h_grad_ind};
      } else {
        params.kernel_input_indices = {
            x_ind,  cs_prev_ind, h_prev_ind, w_ind,  i_ind,
            cs_ind, f_ind,       o_ind,      ci_ind, co_ind,
            h_ind,  cs_grad_ind, h_grad_ind};
      }
    }
    auto tensors = GetTensorInfos(ctx, params);

    if (use_peephole) {
      TensorShape inshape({cell_size});
      TensorShape outshape({batch_size, cell_size});
      DmlTensorInfo wci_info;
      wci_info.kernel_index = wci_ind;
      wci_info.desc = DmlTensorDesc::Create(ctx->GetInputDataType(wci_ind),
                                            outshape, inshape);
      tensors.inputs[4] = wci_info;

      DmlTensorInfo wcf_info;
      wcf_info.kernel_index = wcf_ind;
      wcf_info.desc = DmlTensorDesc::Create(ctx->GetInputDataType(wcf_ind),
                                            outshape, inshape);
      tensors.inputs[5] = wcf_info;

      DmlTensorInfo wco_info;
      wco_info.kernel_index = wco_ind;
      wco_info.desc = DmlTensorDesc::Create(ctx->GetInputDataType(wco_ind),
                                            outshape, inshape);
      tensors.inputs[6] = wco_info;
    }

    auto input_descs = GetDmlTensorDescs(tensors.inputs);

    auto scope = dml::Graph(ctx->GetDmlDevice());

    auto x = dml::InputTensor(scope, 0, input_descs[0]);
    auto cs_prev = dml::InputTensor(scope, 1, input_descs[1]);
    auto h_prev = dml::InputTensor(scope, 2, input_descs[2]);
    auto w = dml::InputTensor(scope, 3, input_descs[3]);

    dml::Expression wci, wcf, wco, i, cs, f, o, ci, co, h, cs_grad, h_grad;

    if (use_peephole) {
      wci = dml::InputTensor(scope, 4, input_descs[4]);
      wcf = dml::InputTensor(scope, 5, input_descs[5]);
      wco = dml::InputTensor(scope, 6, input_descs[6]);
      i = dml::InputTensor(scope, 7, input_descs[7]);
      cs = dml::InputTensor(scope, 8, input_descs[8]);
      f = dml::InputTensor(scope, 9, input_descs[9]);
      o = dml::InputTensor(scope, 10, input_descs[10]);
      ci = dml::InputTensor(scope, 11, input_descs[11]);
      co = dml::InputTensor(scope, 12, input_descs[12]);
      if (seq_len_max_int == 1) {
        // no h
        cs_grad = dml::InputTensor(scope, 13, input_descs[13]);
        h_grad = dml::InputTensor(scope, 14, input_descs[14]);
      } else {
        h = dml::InputTensor(scope, 13, input_descs[13]);
        cs_grad = dml::InputTensor(scope, 14, input_descs[14]);
        h_grad = dml::InputTensor(scope, 15, input_descs[15]);
      }
    } else {
      i = dml::InputTensor(scope, 4, input_descs[4]);
      if (seq_len_max_int == 1) {
        // no cs or h
        f = dml::InputTensor(scope, 5, input_descs[5]);
        o = dml::InputTensor(scope, 6, input_descs[6]);
        ci = dml::InputTensor(scope, 7, input_descs[7]);
        co = dml::InputTensor(scope, 8, input_descs[8]);
        cs_grad = dml::InputTensor(scope, 9, input_descs[9]);
        h_grad = dml::InputTensor(scope, 10, input_descs[10]);
      } else {
        cs = dml::InputTensor(scope, 5, input_descs[5]);
        f = dml::InputTensor(scope, 6, input_descs[6]);
        o = dml::InputTensor(scope, 7, input_descs[7]);
        ci = dml::InputTensor(scope, 8, input_descs[8]);
        co = dml::InputTensor(scope, 9, input_descs[9]);
        h = dml::InputTensor(scope, 10, input_descs[10]);
        cs_grad = dml::InputTensor(scope, 11, input_descs[11]);
        h_grad = dml::InputTensor(scope, 12, input_descs[12]);
      }
    }

    dml::TensorDesc::Dimensions b_dims{1, 1, 1, cell_size * 4};
    dml::TensorDesc::Dimensions peep_dims{1, 1, 1, cell_size};
    dml::TensorDesc::Dimensions w_dims{1, 1, input_size + cell_size,
                                       cell_size * 4};

    dml::TensorDesc::Dimensions x_extent{1, 1, batch_size, input_size};
    dml::TensorDesc::Dimensions output_extent{1, 1, batch_size, cell_size};
    dml::TensorDesc::Dimensions xh_x_offset{0, 0, 0, 0};
    dml::TensorDesc::Dimensions xh_h_offset{0, 0, 0, input_size};
    int32_t slice_stride[] = {1, 1, 1, 1};

    std::vector<dml::Expression> x_grad_tensors;

    DML_TENSOR_DATA_TYPE dtype =
        GetDmlDataTypeFromTfDataType(ctx->GetInputDataType(1));

    auto b_grad = dml::ZeroTensor(scope, dtype, b_dims);
    auto cs_prev_grad = dml::ZeroTensor(scope, dtype, output_extent);
    auto h_prev_grad = dml::ZeroTensor(scope, dtype, output_extent);
    auto w_grad = dml::ZeroTensor(scope, dtype, w_dims);

    auto wci_grad = dml::ZeroTensor(scope, dtype, peep_dims);
    auto wcf_grad = dml::ZeroTensor(scope, dtype, peep_dims);
    auto wco_grad = dml::ZeroTensor(scope, dtype, peep_dims);

    for (int32 t = seq_len_max_int - 1; t >= 0; --t) {
      uint32_t t_ind = static_cast<uint32_t>(t);
      dml::TensorDesc::Dimensions tensor_offset{0, t_ind, 0, 0};
      dml::TensorDesc::Dimensions prev_offset{0, t_ind - 1, 0, 0};

      auto x_tensor = dml::Slice(x, tensor_offset, x_extent, slice_stride);

      auto cs_prev_tensor =
          t == 0 ? cs_prev
                 : dml::Slice(cs, prev_offset, output_extent, slice_stride);
      auto h_prev_tensor =
          t == 0 ? h_prev
                 : dml::Slice(h, prev_offset, output_extent, slice_stride);

      auto i_tensor = dml::Slice(i, tensor_offset, output_extent, slice_stride);

      dml::Expression cs_tensor;
      if (use_peephole) {
        cs_tensor = dml::Slice(cs, tensor_offset, output_extent, slice_stride);
      }

      auto f_tensor = dml::Slice(f, tensor_offset, output_extent, slice_stride);
      auto o_tensor = dml::Slice(o, tensor_offset, output_extent, slice_stride);
      auto ci_tensor =
          dml::Slice(ci, tensor_offset, output_extent, slice_stride);
      auto co_tensor =
          dml::Slice(co, tensor_offset, output_extent, slice_stride);

      auto cs_grad_tensor =
          dml::Slice(cs_grad, tensor_offset, output_extent, slice_stride);

      cs_grad_tensor += cs_prev_grad;

      auto h_grad_tensor =
          dml::Slice(h_grad, tensor_offset, output_extent, slice_stride);

      h_grad_tensor += h_prev_grad;

      // do[t] = sigm'(o[t]) .* dh[t] .* co[t]
      auto do_tensor = o_tensor * (1 - o_tensor) * h_grad_tensor * co_tensor;

      // dcs[t] += tanh'(cs[t]) .* dh[t] .* o[t] + dcs[t + 1] .* f[t + 1]
      auto dcs = (1 - dml::Pow(co_tensor, 2.0f)) * h_grad_tensor * o_tensor +
                 cs_grad_tensor;

      if (use_peephole) {
        dcs = dcs + do_tensor * wco;
      }

      // dci[t] = tanh'(ci[t]) dcs[t] i[t]
      auto dci = (1 - dml::Pow(ci_tensor, 2.0f)) * dcs * i_tensor;

      // df[t] = sigm'(f[t]) dcs[t] cs[t - 1]
      auto df = f_tensor * (1 - f_tensor) * dcs * cs_prev_tensor;

      // di[t] = sigm'(i[t]) dcs[t] ci[t]
      auto di = i_tensor * (1 - i_tensor) * dcs * ci_tensor;

      dml::Expression dgates;
      if (gate_layout == ICFO) {
        dgates = dml::Join({di, dci, df, do_tensor}, 3);
      } else {
        dgates = dml::Join({di, df, dci, do_tensor}, 3);
      }

      cs_prev_grad = dcs * f_tensor;

      if (use_peephole) {
        cs_prev_grad = cs_prev_grad + (di * wci) + (df * wcf);
      }

      auto xh_grad_gemm =
          dml::Gemm(dgates, w, dml::NullOpt, DML_MATRIX_TRANSFORM_NONE,
                    DML_MATRIX_TRANSFORM_TRANSPOSE);
      dml::Expression xh_grad = xh_grad_gemm;

      auto xh = dml::Join({x_tensor, h_prev_tensor}, 3);

      auto x_grad_tensor =
          dml::Slice(xh_grad, xh_x_offset, x_extent, slice_stride);
      h_prev_grad =
          dml::Slice(xh_grad, xh_h_offset, output_extent, slice_stride);

      auto w_grad_gemm =
          dml::Gemm(xh, dgates, dml::NullOpt, DML_MATRIX_TRANSFORM_TRANSPOSE,
                    DML_MATRIX_TRANSFORM_NONE);
      w_grad += w_grad_gemm;

      b_grad += dml::Reduce(dgates, DML_REDUCE_FUNCTION_SUM, {2});

      if (use_peephole) {
        wci_grad +=
            dml::Reduce(di * cs_prev_tensor, DML_REDUCE_FUNCTION_SUM, {2});
        wcf_grad +=
            dml::Reduce(df * cs_prev_tensor, DML_REDUCE_FUNCTION_SUM, {2});
        wco_grad +=
            dml::Reduce(do_tensor * cs_tensor, DML_REDUCE_FUNCTION_SUM, {2});
      }

      // add to vector of x_grad tensors
      x_grad_tensors.insert(x_grad_tensors.begin(), x_grad_tensor);
    }

    auto x_grad = dml::Join(x_grad_tensors, 1);

    std::vector<dml::Expression> outputs = {x_grad,   cs_prev_grad, h_prev_grad,
                                            w_grad,   wci_grad,     wcf_grad,
                                            wco_grad, b_grad};
    Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
        scope.Compile(DML_EXECUTION_FLAG_NONE, outputs);

    Initialize(ctx, std::move(tensors), compiled_op.Get());
  }

  StatusOr<DmlGpuEvent> Compute(DmlKernelContext* ctx) const override {
    for (int i = 0; i < ctx->GetOutputCount(); ++i) {
      ctx->ZeroBuffer(ctx->CreateBufferForTensor(*ctx->GetOutputTensor(i)));
    }
    if (skip_) {
      return ctx->GetCurrentCompletionEvent();
    }
    return DmlKernel::Compute(ctx);
  }

 private:
  bool skip_ = false;
};

#define DML_REGISTER_KERNEL(type)                                          \
  REGISTER_KERNEL_BUILDER(Name("BlockLSTMGrad")                            \
                              .Device(DEVICE_DML)                          \
                              .HostMemory("seq_len_max")                   \
                              .TypeConstraint<type>("T"),                  \
                          DmlKernelWrapper<DmlBlockLstmGradOp<type, ICFO>, \
                                           BlockLstmGradShapeHelper>);     \
  REGISTER_KERNEL_BUILDER(Name("BlockLSTMGradV2")                          \
                              .Device(DEVICE_DML)                          \
                              .HostMemory("seq_len_max")                   \
                              .TypeConstraint<type>("T"),                  \
                          DmlKernelWrapper<DmlBlockLstmGradOp<type, IFCO>, \
                                           BlockLstmGradShapeHelper>);
TF_CALL_DML_FLOAT_TYPES(DML_REGISTER_KERNEL);
#undef DML_REGISTER_KERNEL

}  // namespace tensorflow