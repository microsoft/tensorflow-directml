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

namespace tensorflow {

static Status DmlUnaryOpVariant(OpKernelContext* ctx, VariantUnaryOp op,
                                const Variant& v, Variant* v_out);

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

static Status DmlOptionalZerosLike(OpKernelContext* ctx,
                                   const data::OptionalVariant& x,
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

static Status DmlUnaryOpVariant(OpKernelContext* ctx, VariantUnaryOp op,
                                const Variant& v, Variant* v_out) {
  DCHECK_NE(v_out, nullptr);
  *v_out = data::OptionalVariant();
  if (v.get<data::OptionalVariant>() == nullptr) {
    return errors::Internal(
        "VariantUnaryOpFn: Could not access object, type_index: ",
        port::MaybeAbiDemangle(MakeTypeIndex<data::OptionalVariant>().name()));
  }
  const data::OptionalVariant& t = *v.get<data::OptionalVariant>();
  data::OptionalVariant* t_out = v_out->get<data::OptionalVariant>();
  return DmlOptionalZerosLike(ctx, t, t_out);
}

class DmlZerosLikeKernel : public OpKernel {
 public:
  explicit DmlZerosLikeKernel(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);

    if (input.dtype() == DT_VARIANT) {
      OP_REQUIRES(
          ctx, input.dims() == 0,
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
    } else {
      Tensor* output_tensor = nullptr;
      OP_REQUIRES_OK(ctx,
                     ctx->allocate_output(0, input.shape(), &output_tensor));

      SetTensorToZero(ctx, *output_tensor);
    }
  }
};

#define REGISTER_DML_KERNEL(TYPE)                                     \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("ZerosLike").Device(DEVICE_DML).TypeConstraint<TYPE>("T"), \
      DmlZerosLikeKernel);

// TODO(b/25387198): A special kernel exists for int32 (see constant_op.cc).
TF_CALL_DML_ALL_TYPES_EXCEPT_INT32(REGISTER_DML_KERNEL)
TF_CALL_variant(REGISTER_DML_KERNEL)
#undef REGISTER_DML_KERNEL

}  // namespace tensorflow
