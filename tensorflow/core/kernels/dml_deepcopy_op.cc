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

    ID3D12Resource* input_buffer = dml_util::GetBufferFromTensor(device, input);

    ID3D12Resource* output_buffer =
        dml_util::GetBufferFromTensor(device, *output);

    std::array<D3D12_RESOURCE_BARRIER, 2> barriers = {
        CD3DX12_RESOURCE_BARRIER::Transition(
            input_buffer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE),
        CD3DX12_RESOURCE_BARRIER::Transition(
            output_buffer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_DEST),
    };

    execution_context->ResourceBarrier(barriers);

    constexpr uint64_t dst_offset = 0;
    constexpr uint64_t src_offset = 0;

    execution_context->CopyBufferRegion(
            output_buffer, dst_offset, D3D12_RESOURCE_STATE_COPY_DEST,
            input_buffer, src_offset, D3D12_RESOURCE_STATE_COPY_SOURCE,
            input.TotalBytes());

    for (auto& barrier : barriers) {
      std::swap(barrier.Transition.StateBefore, barrier.Transition.StateAfter);
    }

    execution_context->ResourceBarrier(barriers);
  }
};

#define DML_REGISTER_KERNEL(TYPE)                                    \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("DeepCopy").Device(DEVICE_DML).TypeConstraint<TYPE>("T"), \
      DmlDeepCopyKernel);

TF_CALL_DML_ALL_TYPES(DML_REGISTER_KERNEL);
#undef DML_REGISTER_KERNEL

}  // namespace tensorflow
