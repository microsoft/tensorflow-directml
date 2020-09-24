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

template <typename Tshift, typename Taxis>
class RollInitHelper : public InitializationHelper {
 public:
  using Attributes = EmptyAttributes;

  RollInitHelper(OpKernelContext* ctx, std::shared_ptr<const Attributes> attr) {
    // Grab the input tensor
    const Tensor& input = ctx->input(0);
    const Tensor& shift = ctx->input(1);
    const Tensor& axis = ctx->input(2);

    auto shift_flat = shift.flat<Tshift>();
    auto axis_flat = axis.flat<Taxis>();

    OP_REQUIRES(ctx, TensorShapeUtils::IsVectorOrHigher(input.shape()),
                errors::InvalidArgument("input must be 1-D or higher"));
    OP_REQUIRES(ctx, shift.shape().dims() <= 1,
                errors::InvalidArgument(
                    "shift must be a scalar or a 1-D vector. Found: ",
                    shift.shape().DebugString()));
    OP_REQUIRES(ctx, axis.shape().dims() <= 1,
                errors::InvalidArgument(
                    "axis must be a scalar or a 1-D vector. Found: ",
                    axis.shape().DebugString()));
    OP_REQUIRES(
        ctx, shift.shape() == axis.shape(),
        errors::InvalidArgument("shift and axis must have the same size"));
    const int num_shifts = static_cast<int>(shift_flat.size());
    const int num_dims = input.dims();

    // if there are any duplicate axes, shift_mod_sum_ will have the
    // total modulo sum of shifts for each dimension
    shift_mod_sum_.resize(num_dims);
    for (int i = 0; i < num_shifts; i++) {
      int axis = axis_flat(i);
      if (axis < 0) {
        axis += num_dims;
      }
      OP_REQUIRES(ctx, FastBoundsCheck(axis, num_dims),
                  errors::InvalidArgument("axis ", axis, " is out of range"));
      const int ds = std::max<int>(static_cast<int>(input.dim_size(axis)), 1);
      const int sum = shift_mod_sum_[axis] + static_cast<int>(shift_flat(i));
      // modulo that works with negatives: ((x % y) + y) % y
      shift_mod_sum_[axis] = (sum % ds + ds) % ds;
    }
  }

  absl::Span<const int32> GetShiftModSum() const { return shift_mod_sum_; }

 private:
  gtl::InlinedVector<int32, 4> shift_mod_sum_;
};

template <typename Tshift, typename Taxis>
class DmlRollKernel : public DmlKernel {
 public:
  using InitHelper = RollInitHelper<Tshift, Taxis>;

  DmlRollKernel(DmlKernelConstruction* ctx, const InitHelper* init_helper) {
    // Flatten the tensor shape since we will reinterpret the sizes in the
    // gather loop, and we want to support more than 5D
    TensorShape tensor_shape({ctx->GetOutputTensorShape(0).num_elements()});

    DmlTensorInfo input;
    input.kernel_index = 0;
    input.desc = DmlTensorDesc::Create(ctx->GetInputDataType(0), tensor_shape,
                                       tensor_shape);

    DmlTensorInfo output;
    output.kernel_index = 0;
    output.desc = DmlTensorDesc::Create(ctx->GetOutputDataType(0), tensor_shape,
                                        tensor_shape);

    DmlKernelTensors tensors;
    tensors.inputs = {input};
    tensors.outputs = {output};

    auto inputs = GetDmlTensorDescs(tensors.inputs);

    absl::Span<const int32> shift_mod_sum = init_helper->GetShiftModSum();
    bool is_identity = std::all_of(shift_mod_sum.begin(), shift_mod_sum.end(),
                                   [](int32 axis) { return axis == 0; });

    if (is_identity) {
      auto outputs = GetDmlTensorDescs(tensors.outputs);

      DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC identity_desc = {};
      identity_desc.InputTensor = inputs.data();
      identity_desc.OutputTensor = outputs.data();

      DML_OPERATOR_DESC op_desc = {DML_OPERATOR_ELEMENT_WISE_IDENTITY,
                                   &identity_desc};
      Initialize(ctx, std::move(tensors), op_desc);
      return;
    }

    auto scope = dml::Graph(ctx->GetDmlDevice());
    auto result = dml::InputTensor(scope, 0, inputs[0]);

    DML_SCALAR_UNION indices_start;
    indices_start.UInt32 = 0;

    DML_SCALAR_UNION indices_delta;
    indices_delta.UInt32 = 1;

    const TensorShape& input_shape = ctx->GetInputTensorShape(0);

    // Iteratively roll each axis of the tensor by using Gather
    for (int axis = 0; axis < shift_mod_sum.size(); ++axis) {
      if (shift_mod_sum[axis] == 0) {
        // No shift was requested for this dimension, so leave it alone
        continue;
      }

      is_identity = false;

      // Coalesce the non-axis dimensions to the left and to the right together
      // to allow N dimensional support
      uint32_t before_dim_size = 1;
      for (int i = 0; i < axis; ++i) {
        before_dim_size *= input_shape.dim_size(i);
      }

      uint32_t axis_dim_size = input_shape.dim_size(axis);

      uint32_t after_dim_size = 1;
      for (int i = axis + 1; i < input_shape.dims(); ++i) {
        after_dim_size *= input_shape.dim_size(i);
      }

      dml::TensorDesc::Dimensions new_sizes = {1, before_dim_size,
                                               axis_dim_size, after_dim_size};

      result = dml::Reinterpret(result, new_sizes, {});

      auto indices = dml::FillValueSequence(scope, {1, 1, 1, axis_dim_size},
                                            DML_TENSOR_DATA_TYPE_UINT32,
                                            indices_start, indices_delta);

      DML_SCALAR_UNION axis_constant;
      axis_constant.UInt32 = axis_dim_size;

      auto axis_tensor = dml::FillValueConstant(
          scope, {1, 1, 1, 1}, DML_TENSOR_DATA_TYPE_UINT32, axis_constant);

      axis_tensor = dml::Reinterpret(axis_tensor, {1, 1, 1, axis_dim_size},
                                     dml::TensorDesc::Dimensions{0, 0, 0, 0});

      // For Gather's indices, the shift needs to be reversed
      int shift = axis_dim_size - shift_mod_sum[axis];

      // Shift the indices
      indices = (indices + shift) % axis_tensor;

      // After reinterpreting the sizes, the gather axis is always 2
      constexpr uint32_t dml_axis = 2;
      constexpr uint32_t index_dimensions = 1;
      result = dml::Gather(result, indices, dml_axis, index_dimensions);
    }

    Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
        scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

    Initialize(ctx, std::move(tensors), compiled_op.Get());
  }
};

#define REGISTER_DML_KERNEL(type)                                             \
  REGISTER_KERNEL_BUILDER(Name("Roll")                                        \
                              .Device(DEVICE_DML)                             \
                              .TypeConstraint<type>("T")                      \
                              .TypeConstraint<int32>("Tshift")                \
                              .TypeConstraint<int32>("Taxis")                 \
                              .HostMemory("shift")                            \
                              .HostMemory("axis"),                            \
                          DmlKernelWrapper<DmlRollKernel<int32, int32>,       \
                                           GetOutputShapeAsInputShapeHelper>) \
  REGISTER_KERNEL_BUILDER(Name("Roll")                                        \
                              .Device(DEVICE_DML)                             \
                              .TypeConstraint<type>("T")                      \
                              .TypeConstraint<int64>("Tshift")                \
                              .TypeConstraint<int32>("Taxis")                 \
                              .HostMemory("shift")                            \
                              .HostMemory("axis"),                            \
                          DmlKernelWrapper<DmlRollKernel<int64, int32>,       \
                                           GetOutputShapeAsInputShapeHelper>) \
  REGISTER_KERNEL_BUILDER(Name("Roll")                                        \
                              .Device(DEVICE_DML)                             \
                              .TypeConstraint<type>("T")                      \
                              .TypeConstraint<int32>("Tshift")                \
                              .TypeConstraint<int64>("Taxis")                 \
                              .HostMemory("shift")                            \
                              .HostMemory("axis"),                            \
                          DmlKernelWrapper<DmlRollKernel<int32, int64>,       \
                                           GetOutputShapeAsInputShapeHelper>) \
  REGISTER_KERNEL_BUILDER(Name("Roll")                                        \
                              .Device(DEVICE_DML)                             \
                              .TypeConstraint<type>("T")                      \
                              .TypeConstraint<int64>("Tshift")                \
                              .TypeConstraint<int64>("Taxis")                 \
                              .HostMemory("shift")                            \
                              .HostMemory("axis"),                            \
                          DmlKernelWrapper<DmlRollKernel<int64, int64>,       \
                                           GetOutputShapeAsInputShapeHelper>)

TF_CALL_int32(REGISTER_DML_KERNEL);
TF_CALL_float(REGISTER_DML_KERNEL);
TF_CALL_half(REGISTER_DML_KERNEL);

}  // namespace tensorflow