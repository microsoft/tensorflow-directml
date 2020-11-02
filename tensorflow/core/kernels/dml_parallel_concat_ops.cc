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

class DmlParallelConcatStartKernel : public OpKernel {
 public:
  explicit DmlParallelConcatStartKernel(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shape", &shape_));
  }

  void Compute(OpKernelContext* ctx) override {
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shape_, &out));
  }

 private:
  TensorShape shape_;
};

class DmlParallelConcatUpdateKernel : public OpKernel {
 public:
  explicit DmlParallelConcatUpdateKernel(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("loc", &loc_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& value_tensor = ctx->input(0);
    const Tensor& update_tensor = ctx->input(1);

    OP_REQUIRES(
        ctx, value_tensor.dims() == update_tensor.dims(),
        errors::InvalidArgument("value and update shape doesn't match: ",
                                value_tensor.shape().DebugString(), " vs. ",
                                update_tensor.shape().DebugString()));
    for (int i = 1; i < value_tensor.dims(); ++i) {
      OP_REQUIRES(
          ctx, value_tensor.dim_size(i) == update_tensor.dim_size(i),
          errors::InvalidArgument("value and update shape doesn't match ",
                                  value_tensor.shape().DebugString(), " vs. ",
                                  update_tensor.shape().DebugString()));
    }
    OP_REQUIRES(ctx, 1 == update_tensor.dim_size(0),
                errors::InvalidArgument("update shape doesn't match: ",
                                        update_tensor.shape().DebugString()));

    DmlDevice* device = static_cast<DmlDevice*>(ctx->device());

    // This creates an alias intentionally
    Tensor output_tensor = value_tensor;

    D3D12BufferRegion update_buffer =
        dml_util::CreateBufferForTensor(device, update_tensor);
    D3D12BufferRegion output_buffer =
        dml_util::CreateBufferForTensor(device, output_tensor);

    const int64 nrows = output_tensor.dim_size(0);
    const int dtype_size_in_bytes = DataTypeSize(output_tensor.dtype());
    const uint64_t stride =
        dtype_size_in_bytes * output_tensor.NumElements() / nrows;

    // Guard the row index range
    const int64 row_index = (loc_ % nrows + nrows) % nrows;
    const uint64_t dst_offset = output_buffer.Offset() + row_index * stride;
    const uint64_t src_offset = update_buffer.Offset();

    device->GetExecutionContext()->CopyBufferRegion(
        output_buffer.Resource(), dst_offset, D3D12_RESOURCE_STATE_COPY_DEST,
        update_buffer.Resource(), src_offset, D3D12_RESOURCE_STATE_COPY_SOURCE,
        stride);

    ctx->set_output(0, output_tensor);
  }

 private:
  int32 loc_;
};

#define DML_REGISTER_KERNELS(type)                            \
  REGISTER_KERNEL_BUILDER(Name("_ParallelConcatStart")        \
                              .Device(DEVICE_DML)             \
                              .TypeConstraint<type>("dtype"), \
                          DmlParallelConcatStartKernel);      \
  REGISTER_KERNEL_BUILDER(Name("_ParallelConcatUpdate")       \
                              .Device(DEVICE_DML)             \
                              .TypeConstraint<type>("T"),     \
                          DmlParallelConcatUpdateKernel);

DML_REGISTER_KERNELS(float);
DML_REGISTER_KERNELS(Eigen::half);
#undef DML_REGISTER_KERNELS

}  // namespace tensorflow
