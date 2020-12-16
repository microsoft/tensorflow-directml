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
#include "tensorflow/core/util/overflow.h"

namespace tensorflow {

class OneHotInitHelper : public InitializationHelper {
 public:
  struct Attributes {
    explicit Attributes(OpKernelConstruction* ctx) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("axis", &axis));
    }

    int axis;
  };

  OneHotInitHelper(OpKernelContext* ctx, std::shared_ptr<const Attributes> attr)
      : attr_(attr) {
    const Tensor& indices = ctx->input(0);
    const Tensor& depth_tensor = ctx->input(1);
    const Tensor& on_value = ctx->input(2);
    const Tensor& off_value = ctx->input(3);
    const TensorShape& indices_shape = indices.shape();
    const int indices_dims = indices_shape.dims();
    const int output_dims = indices_dims + 1;

    // Preliminary validation of sizes.
    OP_REQUIRES(
        ctx,
        attr_->axis == -1 || (attr_->axis >= 0 && attr_->axis < output_dims),
        errors::InvalidArgument("Expected axis to be -1 or between [0, ",
                                output_dims,
                                ").  But received: ", attr_->axis));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(depth_tensor.shape()),
                errors::InvalidArgument("depth must be a scalar, but got: ",
                                        depth_tensor.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(on_value.shape()),
                errors::InvalidArgument("on_value must be a scalar, but got: ",
                                        on_value.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(off_value.shape()),
                errors::InvalidArgument("off_value must be a scalar, but got: ",
                                        off_value.shape().DebugString()));

    // The one-hot dimension.
    depth_ = depth_tensor.scalar<int32>()();
    OP_REQUIRES(
        ctx, depth_ >= 0,
        errors::InvalidArgument("depth must be non-negative, got: ", depth_));
    OP_REQUIRES(
        ctx, MultiplyWithoutOverflow(indices_shape.num_elements(), depth_) >= 0,
        errors::InvalidArgument("OneHot result would have shape ",
                                indices_shape.DebugString(), " + [", depth_,
                                "], which exceeds 2**63 - 1 elements"));

    positive_axis_ = attr_->axis == -1 ? indices_shape.dims() : attr_->axis;
  }

  int GetAxis() const { return positive_axis_; }
  int GetDepth() const { return depth_; }

 private:
  const std::shared_ptr<const Attributes> attr_;
  int positive_axis_;
  int depth_;
};

using InitHelper = OneHotInitHelper;

class OneHotShapeHelper : public ShapeHelper {
 public:
  std::vector<TensorShape> GetOutputShapes(
      OpKernelContext* ctx,
      const InitializationHelper* initialization_helper) const override {
    auto init_helper = static_cast<const InitHelper*>(initialization_helper);
    const int axis = init_helper->GetAxis();
    const int depth = init_helper->GetDepth();

    TensorShape output_shape = ctx->input(0).shape();
    output_shape.InsertDim(axis, depth);

    return {std::move(output_shape)};
  };
};

class DmlOneHotKernel : public DmlKernel {
 public:
  using InitHelper = tensorflow::InitHelper;

  explicit DmlOneHotKernel(DmlKernelConstruction* ctx,
                           const InitHelper* init_helper) {
    CHECK(ctx->GetInputCount() == 4);
    CHECK(ctx->GetOutputCount() == 1);

    TensorShape indices_shape = ctx->GetInputTensorShape(0);
    int axis = init_helper->GetAxis();

    // OneHot generates a tensor with rank N+1, where N is the rank of the
    // indices tensor. The +1 comes from inserting one-hot values along a new
    // dimension, with size 'Depth', at a specified axis. For example, consider
    // an indices tensor with shape [2,2,4,3,2]. If axis = 2, then the output
    // shape will be [2,2,Depth,4,3,2]. DML can only handle 4D tensors, but it
    // can support the higher-order shapes by collapsing indices to the left and
    // right of the inserted one-hot dimension. In the above example, the output
    // shape can be collapsed into [4,Depth,24]: 2*2 elements left of axis and
    // 4*3*2 elements right of axis. More generally, where L = elements left of
    // axis and R = elements right of axis:
    // - indices shape = [1,L,1,R]
    // - output shape  = [1,L,Depth,R]
    // Note that L and R will be a minimum of 1, since a dimension cannot have a
    // size of 0. These shapes effectively fix the axis in DML such that it is
    // always 2.

    uint32_t left = 1;
    for (int i = 0; i < axis; i++) {
      left *= indices_shape.dim_size(i);
    }
    uint32_t right = indices_shape.num_elements() / left;

    const uint32_t depth =
        static_cast<uint32_t>(ctx->GetConstantInputTensor(1).scalar<int32>()());

    const uint32_t axis_dml = 2;
    const uint32_t indices_shape_dml[4] = {1, left, 1, right};
    const uint32_t output_shape_dml[4] = {1, left, depth, right};

    DML_TENSOR_DATA_TYPE value_type =
        GetDmlDataTypeFromTfDataType(ctx->GetInputDataType(2));

    // Use DmlTensorDesc::Create helper on indices to apply int64 magic.
    const DataType indices_type = ctx->GetInputDataType(0);
    DmlTensorInfo indices_info = {};
    indices_info.kernel_index = 0;
    indices_info.desc = DmlTensorDesc::Create(indices_type, indices_shape_dml,
                                              indices_shape_dml);

    DmlTensorInfo on_value_info = {};
    on_value_info.kernel_index = 2;
    on_value_info.desc = DmlTensorDesc{value_type, {1, 1, 1, 1}};

    DmlTensorInfo off_value_info = {};
    off_value_info.kernel_index = 3;
    off_value_info.desc = DmlTensorDesc{value_type, {1, 1, 1, 1}};

    DmlTensorInfo output_info = {};
    output_info.kernel_index = 0;
    output_info.desc = DmlTensorDesc{value_type, output_shape_dml};

    DmlKernelTensors tensors = {};
    tensors.inputs = {indices_info, on_value_info, off_value_info};
    tensors.outputs = {output_info};

    auto inputs = GetDmlTensorDescs(tensors.inputs);
    auto outputs = GetDmlTensorDescs(tensors.outputs);

    auto scope = dml::Graph(ctx->GetDmlDevice());
    auto indices = dml::InputTensor(scope, 0, inputs[0]);
    auto on_value = dml::InputTensor(scope, 1, inputs[1]);
    auto off_value = dml::InputTensor(scope, 2, inputs[2]);

    // DML's OneHot only supports uint32 for the indices tensor, but TF models
    // use either int64 or int32. int64 already gets converted to uint32 with
    // the strides hack
    // (TFDML #24881131), so we
    // only need to reinterpret the int32 data to uint32 here.
    if (indices.GetOutputDesc().dataType == DML_TENSOR_DATA_TYPE_INT32) {
      indices = dml::Reinterpret(indices, DML_TENSOR_DATA_TYPE_UINT32);
    }

    // TF provides the on/off values as separate tensors, but DML expects a
    // single two-element tensor with values [off, on].
    auto values = dml::Join({off_value, on_value}, 3);
    auto one_hot = dml::OneHot(indices, values, depth, axis_dml);

    Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
        scope.Compile(DML_EXECUTION_FLAG_NONE, {one_hot});

    Initialize(ctx, std::move(tensors), compiled_op.Get());
  }
};

#define DML_REGISTER_KERNELS(type)                          \
  REGISTER_KERNEL_BUILDER(                                  \
      Name("OneHot")                                        \
          .Device(DEVICE_DML)                               \
          .TypeConstraint<type>("T")                        \
          .TypeConstraint<int32>("TI")                      \
          .HostMemory("depth"),                             \
      DmlKernelWrapper<DmlOneHotKernel, OneHotShapeHelper>) \
  REGISTER_KERNEL_BUILDER(                                  \
      Name("OneHot")                                        \
          .Device(DEVICE_DML)                               \
          .TypeConstraint<type>("T")                        \
          .TypeConstraint<int64>("TI")                      \
          .HostMemory("depth"),                             \
      DmlKernelWrapper<DmlOneHotKernel, OneHotShapeHelper>)

// Composite operators can't easily support int64
TF_CALL_float(DML_REGISTER_KERNELS);
TF_CALL_half(DML_REGISTER_KERNELS);
TF_CALL_uint32(DML_REGISTER_KERNELS);
TF_CALL_uint16(DML_REGISTER_KERNELS);
TF_CALL_uint8(DML_REGISTER_KERNELS);
TF_CALL_int32(DML_REGISTER_KERNELS);
TF_CALL_int16(DML_REGISTER_KERNELS);
TF_CALL_int8(DML_REGISTER_KERNELS);
TF_CALL_bool(DML_REGISTER_KERNELS);
#undef DML_REGISTER_KERNELS

}  // namespace tensorflow
