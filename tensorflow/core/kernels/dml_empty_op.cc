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

namespace tensorflow {

class DmlEmptyKernel : public OpKernel {
 public:
  explicit DmlEmptyKernel(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("init", &init_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& shape = ctx->input(0);
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsVector(shape.shape()),
        errors::InvalidArgument("shape must be a vector of int32, got shape ",
                                shape.shape().DebugString()));
    auto dims = shape.flat<int32>();
    TensorShape out_shape;
    OP_REQUIRES_OK(ctx, TensorShapeUtils::MakeShape(
                            reinterpret_cast<const int32*>(dims.data()),
                            dims.size(), &out_shape));

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &output_tensor));

    if (init_ && out_shape.num_elements() > 0) {
      DmlDevice* device = static_cast<DmlDevice*>(ctx->device());

      D3D12BufferRegion output_buffer =
          dml_util::CreateBufferForTensor(device, *output_tensor);

      uint8_t pattern[] = {0};

      device->GetExecutionContext()->FillBufferWithPattern(
          output_buffer.Resource(), output_buffer.Offset(),
          output_buffer.SizeInBytes(), pattern);
    }
  }

 private:
  bool init_;
};

#define DML_REGISTER_KERNEL(type)                             \
  REGISTER_KERNEL_BUILDER(Name("Empty")                       \
                              .Device(DEVICE_DML)             \
                              .HostMemory("shape")            \
                              .TypeConstraint<type>("dtype"), \
                          DmlEmptyKernel)

DML_REGISTER_KERNEL(float);
DML_REGISTER_KERNEL(Eigen::half);
DML_REGISTER_KERNEL(int64);
DML_REGISTER_KERNEL(int32);
#undef DML_REGISTER_KERNEL

}  // namespace tensorflow
