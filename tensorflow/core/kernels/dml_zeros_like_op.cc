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

class DmlZerosLikeKernel : public OpKernel {
 public:
  explicit DmlZerosLikeKernel(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, ctx->input(0).shape(), &output_tensor));

    DmlDevice* device = static_cast<DmlDevice*>(ctx->device());

    D3D12BufferRegion output_buffer =
        dml_util::CreateBufferForTensor(device, *output_tensor);

    uint8_t pattern[] = {0};

    device->GetExecutionContext()->FillBufferWithPattern(
        output_buffer.Resource(), output_buffer.Offset(),
        output_buffer.SizeInBytes(), pattern);
  }
};

#define REGISTER_DML_KERNEL(TYPE)                                     \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("ZerosLike").Device(DEVICE_DML).TypeConstraint<TYPE>("T"), \
      DmlZerosLikeKernel);

// TODO(b/25387198): A special kernel exists for int32 (see constant_op.cc).
TF_CALL_DML_ALL_TYPES_EXCEPT_INT32(REGISTER_DML_KERNEL)
#undef REGISTER_DML_KERNEL

}  // namespace tensorflow
