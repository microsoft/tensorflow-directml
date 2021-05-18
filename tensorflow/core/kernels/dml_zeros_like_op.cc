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
#include "tensorflow/core/kernels/data/optional_ops.h"
#include "tensorflow/core/kernels/dml_kernel_wrapper.h"
#include "tensorflow/core/kernels/dml_ops_common.h"
#include "tensorflow/core/kernels/list_kernels.h"

namespace tensorflow {

static Status DmlUnaryOpVariant(OpKernelContext* ctx, VariantUnaryOp op,
                                const Variant& v, Variant* v_out);

template <typename T>
static Status DmlVariantZerosLike(OpKernelContext* ctx, const T& x, T* y);

static void SetTensorToZero(OpKernelContext* ctx, const Tensor& tensor) {
  DmlDevice* device = static_cast<DmlDevice*>(ctx->device());

  D3D12BufferRegion output_buffer =
      dml_util::CreateBufferForTensor(device, tensor);

  uint8_t pattern[] = {0};

  device->GetExecutionContext()->FillBufferWithPattern(
      output_buffer.Resource(), output_buffer.Offset(),
      output_buffer.SizeInBytes(), pattern);
}

static Status DmlZerosLikeTensor(OpKernelContext* ctx, const Tensor& x,
                                 Tensor* out) {
  AllocatorAttributes attr;
  if (x.dtype() == DT_VARIANT) {
    attr.set_on_host(true);
  }
  TF_RETURN_IF_ERROR(ctx->allocate_temp(x.dtype(), x.shape(), out, attr));

  switch (out->dtype()) {
#define DTYPE_CASE(dtype)            \
  case DataTypeToEnum<dtype>::value: \
    SetTensorToZero(ctx, *out);      \
    break;

    TF_CALL_POD_TYPES(DTYPE_CASE)
#undef DTYPE_CASE

    case DT_INVALID: {
      *out = Tensor(DT_INVALID);
      break;
    }
    case DataTypeToEnum<Variant>::value: {
      Variant* out_variant = out->scalar<Variant>().data();
      TF_RETURN_IF_ERROR(DmlUnaryOpVariant(ctx, ZEROS_LIKE_VARIANT_UNARY_OP,
                                           x.scalar<Variant>()(), out_variant));
      break;
    }
    default:
      return errors::InvalidArgument(
          "Trying to compute zeros_like for unsupported dtype ",
          DataTypeString(out->dtype()));
  }
  return Status::OK();
}

template <>
static Status DmlVariantZerosLike<data::OptionalVariant>(
    OpKernelContext* ctx, const data::OptionalVariant& x,
    data::OptionalVariant* y) {
  if (!x.has_value()) {
    *y = x;
    return Status::OK();
  }
  std::vector<Tensor> zero_tensors;
  for (const Tensor& tensor : x.get_values()) {
    Tensor zero_t;
    TF_RETURN_IF_ERROR(DmlZerosLikeTensor(ctx, tensor, &zero_t));
    zero_tensors.push_back(std::move(zero_t));
  }
  *y = data::OptionalVariant(zero_tensors);
  return Status::OK();
}

template <>
static Status DmlVariantZerosLike<TensorList>(OpKernelContext* c,
                                              const TensorList& x,
                                              TensorList* y) {
  y->element_dtype = x.element_dtype;
  y->element_shape = x.element_shape;
  y->tensors().reserve(x.tensors().size());
  for (const Tensor& t : x.tensors()) {
    Tensor out_tensor;
    TF_RETURN_IF_ERROR(DmlZerosLikeTensor(c, t, &out_tensor));
    y->tensors().emplace_back(out_tensor);
  }
  return Status::OK();
}

static Status DmlUnaryOpVariant(OpKernelContext* ctx, VariantUnaryOp op,
                                const Variant& v, Variant* v_out) {
  UnaryVariantOpRegistry::VariantUnaryOpFn* unary_op_fn =
      UnaryVariantOpRegistry::Global()->GetUnaryOpFn(op, DEVICE_DML,
                                                     v.TypeId());
  if (unary_op_fn == nullptr) {
    return errors::Internal(
        "No unary variant unary_op function found for unary variant op enum: ",
        op, " Variant type_name: ", v.TypeName(),
        " for device type: ", DEVICE_DML);
  }
  return (*unary_op_fn)(ctx, v, v_out);
}

class DmlZerosLikeKernel : public OpKernel {
 public:
  explicit DmlZerosLikeKernel(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, ctx->input(0).shape(), &output_tensor));

    SetTensorToZero(ctx, *output_tensor);
  }
};

class DmlZerosLikeKernelVariant : public OpKernel {
 public:
  explicit DmlZerosLikeKernelVariant(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);

    OP_REQUIRES(ctx, input.dims() == 0,
                errors::InvalidArgument("ZerosLike non-scalar Tensor with "
                                        "dtype=DT_VARIANT is not supported."));
    const Variant& v = input.scalar<Variant>()();
    // DT_VARIANT tensors must be allocated on CPU since they wrap C++
    // objects which can not be efficiently represented in GPU memory.
    int numa_node = DeviceNumaNode(ctx->device());
    Tensor out(cpu_allocator(numa_node), DT_VARIANT, TensorShape({}));
    Variant* out_v = &(out.scalar<Variant>()());
    OP_REQUIRES_OK(
        ctx, DmlUnaryOpVariant(ctx, ZEROS_LIKE_VARIANT_UNARY_OP, v, out_v));
    ctx->set_output(0, out);
  }
};

#define REGISTER_DML_KERNEL(TYPE)                                     \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("ZerosLike").Device(DEVICE_DML).TypeConstraint<TYPE>("T"), \
      DmlZerosLikeKernel);

// TODO(b/25387198): A special kernel exists for int32 (see constant_op.cc).
TF_CALL_bool(REGISTER_DML_KERNEL)
TF_CALL_half(REGISTER_DML_KERNEL)
TF_CALL_float(REGISTER_DML_KERNEL)
TF_CALL_int64(REGISTER_DML_KERNEL)
#undef REGISTER_DML_KERNEL

REGISTER_KERNEL_BUILDER(
    Name("ZerosLike").Device(DEVICE_DML).TypeConstraint<Variant>("T"),
    DmlZerosLikeKernelVariant);

#define REGISTER_VARIANT_DML_KERNEL(TYPE)                               \
  REGISTER_UNARY_VARIANT_UNARY_OP_FUNCTION(ZEROS_LIKE_VARIANT_UNARY_OP, \
                                           DEVICE_DML, TYPE,            \
                                           DmlVariantZerosLike<TYPE>);

REGISTER_VARIANT_DML_KERNEL(data::OptionalVariant)
REGISTER_VARIANT_DML_KERNEL(TensorList)
#undef REGISTER_VARIANT_DML_KERNEL

}  // namespace tensorflow
