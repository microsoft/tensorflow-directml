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

class DmlZerosLikeKernel : public DmlKernel {
 public:
  using InitHelper = NoOpInitializationHelper;

  explicit DmlZerosLikeKernel(DmlKernelConstruction* ctx,
                              const InitHelper* init_helper) {}

  DmlGpuEvent Compute(DmlKernelContext* ctx) const override {
    Tensor* output = ctx->GetOutputTensor(0);
    return ctx->ZeroBuffer(ctx->CreateBufferForTensor(*output));
  }
};

#define REGISTER_DML_KERNEL(TYPE)                                     \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("ZerosLike").Device(DEVICE_DML).TypeConstraint<TYPE>("T"), \
      DmlKernelWrapper<DmlZerosLikeKernel, GetOutputShapeAsInputShapeHelper>);

// TODO(b/25387198): A special kernel exists for int32 (see constant_op.cc).
TF_CALL_DML_ALL_TYPES_EXCEPT_INT32(REGISTER_DML_KERNEL)
#undef REGISTER_DML_KERNEL

}  // namespace tensorflow
