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

class DmlDeepCopyKernel : public OpKernel {
 public:
  explicit DmlDeepCopyKernel(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);
    const TensorShape& input_shape = input.shape();

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input_shape, &output));

    DmlDevice* device = static_cast<DmlDevice*>(ctx->device());
    const auto& execution_context = device->GetExecutionContext();

    D3D12BufferRegion input_buffer =
        dml_util::CreateBufferForTensor(device, input);

    D3D12BufferRegion output_buffer =
        dml_util::CreateBufferForTensor(device, *output);

    uint64_t copy_size = std::min(output_buffer.SizeInBytes(), input_buffer.SizeInBytes());

    execution_context->CopyBufferRegion(
        output_buffer, input_buffer.Subregion(0, copy_size));
  }
};

#define DML_REGISTER_KERNEL(TYPE)                                    \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("DeepCopy").Device(DEVICE_DML).TypeConstraint<TYPE>("T"), \
      DmlDeepCopyKernel);

TF_CALL_half(DML_REGISTER_KERNEL);
TF_CALL_float(DML_REGISTER_KERNEL);
TF_CALL_int64(DML_REGISTER_KERNEL);
#undef DML_REGISTER_KERNEL

}  // namespace tensorflow
