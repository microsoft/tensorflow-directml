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
#include "tensorflow/core/kernels/reduction_ops_common.h"

namespace tensorflow {

template <DML_REDUCE_FUNCTION reduce_function>
class ReduceInitializationHelper : public InitializationHelper {
 public:
  struct Attributes {
    explicit Attributes(OpKernelConstruction* ctx) {
      if (!ctx->GetAttr("keep_dims", &keep_dims).ok()) {
        keep_dims = false;
      }
    }

    bool keep_dims;
  };

  bool IsNoOpKernel(
      OpKernelContext* ctx,
      absl::Span<const TensorShape> output_shapes) const override {
    const Tensor& data_tensor = ctx->input(0);

    if (output_shapes[0].num_elements() == 0) {
      return true;
    }

    // TF's Prod and All operators are different to the other reductions in that
    // reduction of an empty tensor is defined to return 1, not 0. Because of
    // this, reduction of empty tensors needs to be handled by the kernel to
    // explicitly output 1. Min and Max are also different because they need to
    // return inf and -inf, respectively.
    if (reduce_function == DML_REDUCE_FUNCTION_MULTIPLY ||
        reduce_function == DML_REDUCE_FUNCTION_MIN ||
        reduce_function == DML_REDUCE_FUNCTION_MAX) {
      return false;
    }

    // Ignore the axes tensor when deciding whether to no-op. This is because
    // it's legal to have an empty axes tensor, which turns this reduction into
    // an identity.
    if (data_tensor.NumElements() == 0) {
      return true;
    }

    return false;
  }

  ReduceInitializationHelper(OpKernelContext* ctx,
                             std::shared_ptr<const Attributes> attr) {
    // We delegate most of the work to TF's existing ReductionHelper
    const Tensor& data_tensor = ctx->input(0);
    const Tensor& axes_tensor = ctx->input(1);
    OP_REQUIRES_OK(ctx, reduction_helper_.Simplify(data_tensor, axes_tensor,
                                                   attr->keep_dims));
  }

  const ReductionHelper& GetReductionHelper() const {
    return reduction_helper_;
  }

 private:
  ReductionHelper reduction_helper_;
};

template <DML_REDUCE_FUNCTION reduce_function>
using InitHelper = ReduceInitializationHelper<reduce_function>;

template <DML_REDUCE_FUNCTION reduce_function>
class ReduceOutputShapeHelper : public ShapeHelper {
 public:
  std::vector<TensorShape> GetOutputShapes(
      OpKernelContext* ctx,
      const InitializationHelper* initialization_helper) const override {
    auto init_helper =
        static_cast<const InitHelper<reduce_function>*>(initialization_helper);
    return {init_helper->GetReductionHelper().out_shape()};
  }
};

template <typename T>
static T EmptyKernelReturnValue(DML_REDUCE_FUNCTION reduce_function) {
  switch (reduce_function) {
    case DML_REDUCE_FUNCTION_MULTIPLY:
      return static_cast<T>(1);
    case DML_REDUCE_FUNCTION_MIN:
      return std::numeric_limits<T>::has_infinity
                 ? std::numeric_limits<T>::infinity()
                 : std::numeric_limits<T>::max();
    case DML_REDUCE_FUNCTION_MAX:
      return std::numeric_limits<T>::has_infinity
                 ? -std::numeric_limits<T>::infinity()
                 : std::numeric_limits<T>::min();
    default:
      LOG(FATAL) << "Invalid reduce function type.";
  }
}

template <>
static bool EmptyKernelReturnValue(DML_REDUCE_FUNCTION reduce_function) {
  switch (reduce_function) {
    case DML_REDUCE_FUNCTION_MIN:
      return true;
    case DML_REDUCE_FUNCTION_MAX:
      return false;
    default:
      LOG(FATAL) << "Invalid reduce function type.";
  }
}

template <DML_REDUCE_FUNCTION reduce_function, typename TAxis>
class DmlReduceKernel : public DmlKernel {
 public:
  using InitHelper = tensorflow::InitHelper<reduce_function>;

  explicit DmlReduceKernel(DmlKernelConstruction* ctx,
                           const InitHelper* init_helper) {
    CHECK(ctx->GetInputCount() == 2);
    CHECK(ctx->GetOutputCount() == 1);

    const Tensor& axes_tensor = ctx->GetConstantInputTensor(1);

    // Use TF's existing ReductionHelper to help compute axes for reduction. The
    // ReductionHelper does a couple of useful things.
    //
    // One, it collapses adjacent reduction axes together. For example if you
    // have a 5D (e.g. NCDHW) tensor and want to reduce along the 2nd axis (e.g.
    // 'D'), you can collapse the NC dimensions together and the HW dimensions
    // together, then perform a 3D reduction along the 1st axis. This allows us
    // to perform reductions across higher-dimensional tensors than we would
    // usually support, since DML only supports 4D reductions.
    //
    // Second, if the number of axes that need reduction exceed what's
    // supported, the ReductionHelper also provides shapes for transposing the
    // dimensions of the input tensor such that all the reduction axes are
    // shuffled to the end. This allows an N-dimensional reduction to be
    // performed as a 2D reduction, albeit at the cost of a tensor copy. We
    // don't use this facility in DML, though. If the dimensionality of the
    // reduction (after collapsing adjacent axes) exceeds what DML supports,
    // this kernel returns an error.
    const ReductionHelper& reduce_helper = init_helper->GetReductionHelper();

    constexpr bool is_special_empty_kernel =
        reduce_function == DML_REDUCE_FUNCTION_MULTIPLY ||
        reduce_function == DML_REDUCE_FUNCTION_MIN ||
        reduce_function == DML_REDUCE_FUNCTION_MAX;

    // Special-case for Prod operator: when reducing an empty tensor, we
    // explicitly need to return a value of 1.0 (not zero, which is the
    // default for a no-op'd operator)
    if (is_special_empty_kernel &&
        ctx->GetInputTensorShape(0).num_elements() == 0) {
      DataType output_type = ctx->GetOutputDataType(0);

      DmlKernelTensors tensors;
      tensors.outputs.resize(1);

      constexpr uint32_t dml_output_shape[] = {1u};

      tensors.outputs[0].emplace();
      tensors.outputs[0]->desc = DmlTensorDesc::Create(
          output_type, dml_output_shape, dml_output_shape);
      tensors.outputs[0]->kernel_index = 0;

      auto output_descs = GetDmlTensorDescs(tensors.outputs);

      DML_FILL_VALUE_CONSTANT_OPERATOR_DESC fill_desc = {};
      fill_desc.OutputTensor = &output_descs[0];
      fill_desc.ValueDataType = GetDmlDataTypeFromTfDataType(output_type);

      switch (fill_desc.ValueDataType) {
        case DML_TENSOR_DATA_TYPE_FLOAT32: {
          fill_desc.Value.Float32 =
              EmptyKernelReturnValue<float>(reduce_function);
        } break;

        case DML_TENSOR_DATA_TYPE_FLOAT16: {
          // Copy the bits as a UINT16
          fill_desc.Value.UInt16 =
              EmptyKernelReturnValue<Eigen::half>(reduce_function).x;
        } break;

        case DML_TENSOR_DATA_TYPE_UINT32: {
          fill_desc.Value.UInt32 =
              EmptyKernelReturnValue<uint32>(reduce_function);
        } break;

        case DML_TENSOR_DATA_TYPE_INT32: {
          fill_desc.Value.Int32 =
              EmptyKernelReturnValue<int32>(reduce_function);
        } break;

        case DML_TENSOR_DATA_TYPE_UINT8: {
          fill_desc.Value.UInt8 = EmptyKernelReturnValue<bool>(reduce_function);
        } break;

        default:
          assert(false);
          LOG(FATAL) << "Unsupported datatype";
      }

      DML_OPERATOR_DESC op_desc = {DML_OPERATOR_FILL_VALUE_CONSTANT,
                                   &fill_desc};
      Initialize(ctx, std::move(tensors), op_desc);

      return;
    }

    // ARGMIN and ARGMAX are special reduce functions that can never be replaced
    // by identity
    constexpr bool is_arg_function =
        reduce_function == DML_REDUCE_FUNCTION_ARGMIN ||
        reduce_function == DML_REDUCE_FUNCTION_ARGMAX;

    // This logic copied from the CPU implementation:
    // tensorflow/core/kernels/reduction_ops_common.h(155)
    bool is_identity =
        !is_arg_function &&
        (reduce_helper.ndims() == 0 ||
         (reduce_helper.ndims() == 1 && !reduce_helper.reduce_first_axis()));

    if (is_identity) {
      // Since the reduce helper may have removed dimensions of size 1, we can't
      // just take its shape as-is. We know that this is an identity scenario,
      // so we can just collapse all dimensions into one and use the same tensor
      // desc for both the input and the output.
      TensorShape in_out_shape(
          {1, 1, 1, ctx->GetInputTensorShape(0).num_elements()});

      DmlTensorInfo in_out_tensor;
      in_out_tensor.kernel_index = 0;
      in_out_tensor.desc = DmlTensorDesc::Create(ctx->GetInputDataType(0),
                                                 in_out_shape, in_out_shape);

      DmlKernelTensors tensors;
      tensors.inputs = {in_out_tensor};
      tensors.outputs = {in_out_tensor};

      auto input_descs = GetDmlTensorDescs(tensors.inputs);
      auto output_descs = GetDmlTensorDescs(tensors.outputs);

      DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC identity_desc = {};
      identity_desc.InputTensor = &input_descs[0];
      identity_desc.OutputTensor = &output_descs[0];

      DML_OPERATOR_DESC op_desc = {DML_OPERATOR_ELEMENT_WISE_IDENTITY,
                                   &identity_desc};
      Initialize(ctx, std::move(tensors), op_desc);

      return;
    }

    // The input shape after adjacent reduction axes have been collapsed.
    const TensorShape& input_shape = reduce_helper.data_reshape();

    // DML's reduction operator only supports up to 4 dimensions
    CHECK(input_shape.dims() <= kNchwDimensionCount);

    uint32_t reduce_axis_offset = kNchwDimensionCount - input_shape.dims();

    // Compute the DML reduce axes based on the input shape. If
    // reduce_first_axis() is true we reduce over axes 0 and 2, otherwise we
    // reduce over axes 1 and 3 (up to the dimension count of the input.)
    uint32_t first_reduce_axis = reduce_helper.reduce_first_axis() ? 0 : 1;
    absl::InlinedVector<uint32_t, 4> reduce_axes;
    for (uint32_t axis = first_reduce_axis; axis < input_shape.dims();
         axis += 2) {
      reduce_axes.push_back(axis + reduce_axis_offset);
    }

    // Use the axes and input shape to compute the output shape as required by
    // DML. We can't use the TF-style output shape as computed by the
    // ReductionHelper, because DML requires a very specific output tensor shape
    // for reduction (namely, that reduced axes must have dimension 1 in the
    // output tensor)
    TensorShape output_shape;
    for (int i = 0; i < input_shape.dims(); ++i) {
      uint32_t axis = i + reduce_axis_offset;
      const bool is_reduce_axis =
          std::count(reduce_axes.begin(), reduce_axes.end(), axis) > 0;

      if (is_reduce_axis) {
        output_shape.AddDim(1);
      } else {
        output_shape.AddDim(input_shape.dim_size(i));
      }
    }

    // Currently, 64-bit integers in DML are emulated using 32-bit integers
    // using striding to emulate a larger type. Because we can't guarantee
    // that our output tensor's memory is zero'd, we need to do so manually
    // prior to running the reduction.
    if (Is64BitIntegerType(ctx->GetOutputDataType(0))) {
      zero_outputs_ = true;
    }

    assert(input_shape.dims() == output_shape.dims());

    DmlTensorInfo input;
    input.kernel_index = 0;
    input.desc = DmlTensorDesc::Create(ctx->GetInputDataType(0), input_shape,
                                       input_shape);

    DmlTensorInfo output;
    output.kernel_index = 0;
    output.desc = DmlTensorDesc::Create(ctx->GetOutputDataType(0), output_shape,
                                        output_shape);

    // Coerce the output datatype to unsigned, for argmin/argmax
    if (DataTypeIsInteger(output.desc.GetTfDataType())) {
      output.desc.ForceUnsignedDataType();
    }

    DmlKernelTensors tensors;
    tensors.inputs = {input};
    tensors.outputs = {output};

    DML_TENSOR_DATA_TYPE dml_input_data_type =
        GetDmlDataTypeFromTfDataType(ctx->GetInputDataType(0));

    // For logical operators like Any and All, we need to cast from uint8 to
    // float
    if (dml_input_data_type != DML_TENSOR_DATA_TYPE_FLOAT32 &&
        dml_input_data_type != DML_TENSOR_DATA_TYPE_FLOAT16) {
      auto input_descs = GetDmlTensorDescs(tensors.inputs);

      auto scope = dml::Graph(ctx->GetDmlDevice());
      auto result = dml::InputTensor(scope, 0, input_descs[0]);
      result = dml::Cast(result, DML_TENSOR_DATA_TYPE_FLOAT32);
      result = dml::Reduce(result, reduce_function, reduce_axes);
      result = dml::Cast(result, dml_input_data_type);

      Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
          scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

      Initialize(ctx, std::move(tensors), compiled_op.Get());
    } else {
      auto input_descs = GetDmlTensorDescs(tensors.inputs);
      auto output_descs = GetDmlTensorDescs(tensors.outputs);

      DML_REDUCE_OPERATOR_DESC reduce_desc = {};
      reduce_desc.InputTensor = &input_descs[0];
      reduce_desc.OutputTensor = &output_descs[0];
      reduce_desc.Function = reduce_function;
      reduce_desc.AxisCount = static_cast<UINT>(reduce_axes.size());
      reduce_desc.Axes = reduce_axes.data();

      DML_OPERATOR_DESC op_desc = {DML_OPERATOR_REDUCE, &reduce_desc};
      Initialize(ctx, std::move(tensors), op_desc);
    }
  }

  StatusOr<DmlGpuEvent> Compute(DmlKernelContext* ctx) const override {
    if (zero_outputs_) {
      Tensor* output = ctx->GetOutputTensor(0);
      ctx->ZeroBuffer(ctx->CreateBufferForTensor(*output));
    }

    return DmlKernel::Compute(ctx);
  }

 private:
  bool zero_outputs_ = false;
};

template <DML_REDUCE_FUNCTION reduce_function, typename TAxis>
using DmlReduceWrapper =
    DmlKernelWrapper<DmlReduceKernel<reduce_function, TAxis>,
                     ReduceOutputShapeHelper<reduce_function>>;

#define DML_REGISTER_KERNELS(type)                                           \
  REGISTER_KERNEL_BUILDER(Name("Sum")                                        \
                              .Device(DEVICE_DML)                            \
                              .TypeConstraint<type>("T")                     \
                              .TypeConstraint<int32>("Tidx")                 \
                              .HostMemory("reduction_indices"),              \
                          DmlReduceWrapper<DML_REDUCE_FUNCTION_SUM, int32>); \
  REGISTER_KERNEL_BUILDER(Name("Sum")                                        \
                              .Device(DEVICE_DML)                            \
                              .TypeConstraint<type>("T")                     \
                              .TypeConstraint<int64>("Tidx")                 \
                              .HostMemory("reduction_indices"),              \
                          DmlReduceWrapper<DML_REDUCE_FUNCTION_SUM, int64>);
// TODO(b/25387198): A special kernel exists for int32 (see
// reduction_ops_sum.cc).
TF_CALL_float(DML_REGISTER_KERNELS);
TF_CALL_half(DML_REGISTER_KERNELS);
TF_CALL_uint32(DML_REGISTER_KERNELS);
#undef DML_REGISTER_KERNELS

#define DML_REGISTER_KERNELS(type)                           \
  REGISTER_KERNEL_BUILDER(                                   \
      Name("Mean")                                           \
          .Device(DEVICE_DML)                                \
          .TypeConstraint<type>("T")                         \
          .TypeConstraint<int32>("Tidx")                     \
          .HostMemory("reduction_indices"),                  \
      DmlReduceWrapper<DML_REDUCE_FUNCTION_AVERAGE, int32>); \
  REGISTER_KERNEL_BUILDER(                                   \
      Name("Mean")                                           \
          .Device(DEVICE_DML)                                \
          .TypeConstraint<type>("T")                         \
          .TypeConstraint<int64>("Tidx")                     \
          .HostMemory("reduction_indices"),                  \
      DmlReduceWrapper<DML_REDUCE_FUNCTION_AVERAGE, int64>);
TF_CALL_DML_FLOAT_TYPES(DML_REGISTER_KERNELS);
#undef DML_REGISTER_KERNELS

#define DML_REGISTER_KERNELS(type)                            \
  REGISTER_KERNEL_BUILDER(                                    \
      Name("Prod")                                            \
          .Device(DEVICE_DML)                                 \
          .TypeConstraint<type>("T")                          \
          .TypeConstraint<int32>("Tidx")                      \
          .HostMemory("reduction_indices"),                   \
      DmlReduceWrapper<DML_REDUCE_FUNCTION_MULTIPLY, int32>); \
  REGISTER_KERNEL_BUILDER(                                    \
      Name("Prod")                                            \
          .Device(DEVICE_DML)                                 \
          .TypeConstraint<type>("T")                          \
          .TypeConstraint<int64>("Tidx")                      \
          .HostMemory("reduction_indices"),                   \
      DmlReduceWrapper<DML_REDUCE_FUNCTION_MULTIPLY, int64>);
TF_CALL_float(DML_REGISTER_KERNELS);
TF_CALL_half(DML_REGISTER_KERNELS);
TF_CALL_int32(DML_REGISTER_KERNELS);
TF_CALL_uint32(DML_REGISTER_KERNELS);
#undef DML_REGISTER_KERNELS

#define DML_REGISTER_KERNELS(type)                                           \
  REGISTER_KERNEL_BUILDER(Name("Min")                                        \
                              .Device(DEVICE_DML)                            \
                              .TypeConstraint<type>("T")                     \
                              .TypeConstraint<int32>("Tidx")                 \
                              .HostMemory("reduction_indices"),              \
                          DmlReduceWrapper<DML_REDUCE_FUNCTION_MIN, int32>); \
  REGISTER_KERNEL_BUILDER(Name("Min")                                        \
                              .Device(DEVICE_DML)                            \
                              .TypeConstraint<type>("T")                     \
                              .TypeConstraint<int64>("Tidx")                 \
                              .HostMemory("reduction_indices"),              \
                          DmlReduceWrapper<DML_REDUCE_FUNCTION_MIN, int64>);
// TODO(b/25387198): A special kernel exists for int32 (see
// reduction_ops_min.cc).
TF_CALL_float(DML_REGISTER_KERNELS);
TF_CALL_half(DML_REGISTER_KERNELS);
TF_CALL_uint32(DML_REGISTER_KERNELS);
#undef DML_REGISTER_KERNELS

#define DML_REGISTER_KERNELS(type)                                           \
  REGISTER_KERNEL_BUILDER(Name("Max")                                        \
                              .Device(DEVICE_DML)                            \
                              .TypeConstraint<type>("T")                     \
                              .TypeConstraint<int32>("Tidx")                 \
                              .HostMemory("reduction_indices"),              \
                          DmlReduceWrapper<DML_REDUCE_FUNCTION_MAX, int32>); \
  REGISTER_KERNEL_BUILDER(Name("Max")                                        \
                              .Device(DEVICE_DML)                            \
                              .TypeConstraint<type>("T")                     \
                              .TypeConstraint<int64>("Tidx")                 \
                              .HostMemory("reduction_indices"),              \
                          DmlReduceWrapper<DML_REDUCE_FUNCTION_MAX, int64>);
// TODO(b/25387198): A special kernel exists for int32 (see
// reduction_ops_max.cc).
TF_CALL_float(DML_REGISTER_KERNELS);
TF_CALL_half(DML_REGISTER_KERNELS);
TF_CALL_uint32(DML_REGISTER_KERNELS);
#undef DML_REGISTER_KERNELS

#define DML_REGISTER_KERNELS(type)                          \
  REGISTER_KERNEL_BUILDER(                                  \
      Name("ArgMin")                                        \
          .Device(DEVICE_DML)                               \
          .TypeConstraint<type>("T")                        \
          .TypeConstraint<int32>("Tidx")                    \
          .HostMemory("dimension"),                         \
      DmlReduceWrapper<DML_REDUCE_FUNCTION_ARGMIN, int32>); \
  REGISTER_KERNEL_BUILDER(                                  \
      Name("ArgMin")                                        \
          .Device(DEVICE_DML)                               \
          .TypeConstraint<type>("T")                        \
          .TypeConstraint<int64>("Tidx")                    \
          .HostMemory("dimension"),                         \
      DmlReduceWrapper<DML_REDUCE_FUNCTION_ARGMIN, int64>);

TF_CALL_DML_FLOAT_TYPES(DML_REGISTER_KERNELS);
#undef DML_REGISTER_KERNELS

#define DML_REGISTER_KERNELS(type)                          \
  REGISTER_KERNEL_BUILDER(                                  \
      Name("ArgMax")                                        \
          .Device(DEVICE_DML)                               \
          .TypeConstraint<type>("T")                        \
          .TypeConstraint<int32>("Tidx")                    \
          .HostMemory("dimension"),                         \
      DmlReduceWrapper<DML_REDUCE_FUNCTION_ARGMAX, int32>); \
  REGISTER_KERNEL_BUILDER(                                  \
      Name("ArgMax")                                        \
          .Device(DEVICE_DML)                               \
          .TypeConstraint<type>("T")                        \
          .TypeConstraint<int64>("Tidx")                    \
          .HostMemory("dimension"),                         \
      DmlReduceWrapper<DML_REDUCE_FUNCTION_ARGMAX, int64>);

TF_CALL_DML_FLOAT_TYPES(DML_REGISTER_KERNELS);
#undef DML_REGISTER_KERNELS

REGISTER_KERNEL_BUILDER(Name("Any")
                            .Device(DEVICE_DML)
                            .TypeConstraint<int32>("Tidx")
                            .HostMemory("reduction_indices"),
                        DmlReduceWrapper<DML_REDUCE_FUNCTION_MAX, int32>);
REGISTER_KERNEL_BUILDER(Name("Any")
                            .Device(DEVICE_DML)
                            .TypeConstraint<int64>("Tidx")
                            .HostMemory("reduction_indices"),
                        DmlReduceWrapper<DML_REDUCE_FUNCTION_MAX, int64>);

REGISTER_KERNEL_BUILDER(Name("All")
                            .Device(DEVICE_DML)
                            .TypeConstraint<int32>("Tidx")
                            .HostMemory("reduction_indices"),
                        DmlReduceWrapper<DML_REDUCE_FUNCTION_MIN, int32>);
REGISTER_KERNEL_BUILDER(Name("All")
                            .Device(DEVICE_DML)
                            .TypeConstraint<int64>("Tidx")
                            .HostMemory("reduction_indices"),
                        DmlReduceWrapper<DML_REDUCE_FUNCTION_MIN, int64>);

}  // namespace tensorflow