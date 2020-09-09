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
#include "tensorflow/core/kernels/dml_ops_common.h"

namespace tensorflow {

class DmlDataFormatVecPermuteKernel : public OpKernel {
 public:
  explicit DmlDataFormatVecPermuteKernel(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    std::string src_format;
    std::string dst_format;

    OP_REQUIRES_OK(ctx, ctx->GetAttr("src_format", &src_format));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dst_format", &dst_format));

    // The CPU and CUDA implementions don't check the size of src_format or
    // dst_format and just leave garbage data for the dimensions that are not
    // included. We do the same thing here.
    size_t src_format_length = std::min<size_t>(src_format.length(), 4u);
    size_t dst_format_length = std::min<size_t>(dst_format.length(), 4u);

    for (size_t dst_index = 0; dst_index < dst_format_length; ++dst_index) {
      for (size_t src_index = 0; src_index < src_format_length; ++src_index) {
        if (dst_format[dst_index] == src_format[src_index]) {
          permutations_.push_back(src_index);
          break;
        }
      }
    }
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);
    const TensorShape& input_shape = input.shape();

    OP_REQUIRES(ctx, input_shape.dims() == 1 || input_shape.dims() == 2,
                errors::InvalidArgument(
                    "input must be a vector or 2D tensor, but got shape ",
                    input_shape.DebugString()));

    if (input_shape.dims() == 1) {
      OP_REQUIRES(
          ctx, input_shape.dim_size(0) == 4,
          errors::InvalidArgument("1D input must be of size 4, but got shape ",
                                  input_shape.DebugString()));
    } else if (input_shape.dims() == 2) {
      OP_REQUIRES(
          ctx, input_shape.dim_size(0) == 4,
          errors::InvalidArgument(
              "First dimension of 2D input must be of size 4, but got shape ",
              input_shape.DebugString()));
      OP_REQUIRES(
          ctx, input_shape.dim_size(1) == 2,
          errors::InvalidArgument(
              "Second dimension of 2D input must be of size 2, but got shape ",
              input_shape.DebugString()));
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input_shape, &output));

    DmlDevice* device = static_cast<DmlDevice*>(ctx->device());
    const auto& execution_context = device->GetExecutionContext();

    D3D12BufferRegion input_buffer =
        dml_util::CreateBufferForTensor(device, input);

    D3D12BufferRegion output_buffer =
        dml_util::CreateBufferForTensor(device, *output);

    std::array<D3D12_RESOURCE_BARRIER, 2> barriers = {
        CD3DX12_RESOURCE_BARRIER::Transition(
            input_buffer.Resource(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE),
        CD3DX12_RESOURCE_BARRIER::Transition(
            output_buffer.Resource(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_DEST),
    };

    execution_context->ResourceBarrier(barriers);

    // Currently, 64-bit integers in DML are emulated using 32-bit integers
    // using striding to emulate a larger type. Because we can't guarantee that
    // our output tensor's memory is zero'd, we need to do so manually prior to
    // running running gather.
    int perm_stride = DataTypeSize(input.dtype()) * input_shape.dims();
    bool is_64_bit_int = Is64BitIntegerType(output->dtype());
    uint32_t permutation_size = is_64_bit_int ? perm_stride / 2 : perm_stride;

    // For int64 data types, we copy only half the data since the other half is
    // most likely garbage data
    if (is_64_bit_int) {
      const uint8_t pattern[] = {0};
      execution_context->FillBufferWithPattern(
          output_buffer.Resource(), output_buffer.Offset(),
          output_buffer.SizeInBytes(), pattern);
    }

    for (uint32_t i = 0; i < permutations_.size(); ++i) {
      uint64_t dst_offset = output_buffer.Offset() + i * perm_stride;
      uint64_t src_offset =
          input_buffer.Offset() + permutations_[i] * perm_stride;

      // For int64 data types, we need to do 2 to separate copies for 2D tensors
      // in order to skip the garbage data between elements
      if (is_64_bit_int && input.dims() == 2) {
        execution_context->CopyBufferRegion(
            output_buffer.Resource(), dst_offset,
            D3D12_RESOURCE_STATE_COPY_DEST, input_buffer.Resource(), src_offset,
            D3D12_RESOURCE_STATE_COPY_SOURCE, permutation_size / 2);

        execution_context->CopyBufferRegion(
            output_buffer.Resource(), dst_offset + permutation_size,
            D3D12_RESOURCE_STATE_COPY_DEST, input_buffer.Resource(),
            src_offset + permutation_size, D3D12_RESOURCE_STATE_COPY_SOURCE,
            permutation_size / 2);
      } else {
        execution_context->CopyBufferRegion(
            output_buffer.Resource(), dst_offset,
            D3D12_RESOURCE_STATE_COPY_DEST, input_buffer.Resource(), src_offset,
            D3D12_RESOURCE_STATE_COPY_SOURCE, permutation_size);
      }
    }

    for (auto& barrier : barriers) {
      std::swap(barrier.Transition.StateBefore, barrier.Transition.StateAfter);
    }

    execution_context->ResourceBarrier(barriers);
  }

 private:
  absl::InlinedVector<uint32_t, 4> permutations_;
};

#define REGISTER_KERNEL(type)                             \
  REGISTER_KERNEL_BUILDER(Name("DataFormatVecPermute")    \
                              .Device(DEVICE_DML)         \
                              .TypeConstraint<type>("T"), \
                          DmlDataFormatVecPermuteKernel);
TF_CALL_int32(REGISTER_KERNEL);
TF_CALL_int64(REGISTER_KERNEL);
#undef REGISTER_KERNEL

}  // namespace tensorflow