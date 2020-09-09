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

static bool SameExtraShape(const Tensor& data0, const Tensor& indices0,
                           const Tensor& data1, const Tensor& indices1) {
  const int extra0 = data0.dims() - indices0.dims();
  const int extra1 = data1.dims() - indices1.dims();
  if (extra0 != extra1) return false;
  for (int i = 0; i < extra0; i++) {
    if (data0.dim_size(indices0.dims() + i) !=
        data1.dim_size(indices1.dims() + i)) {
      return false;
    }
  }
  return true;
}

class DmlDynamicStitchKernel : public OpKernel {
 public:
  explicit DmlDynamicStitchKernel(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    // Find maximum index in the indices vectors
    OpInputList indices_inputs;
    OP_REQUIRES_OK(ctx, ctx->input_list("indices", &indices_inputs));

    int32 max_index = -1;

    for (const Tensor& indices : indices_inputs) {
      if (indices.NumElements() > 0) {
        Eigen::Tensor<int32, 0, Eigen::RowMajor> m =
            indices.flat<int32>().maximum();
        max_index = std::max(m(), max_index);
      }
    }

    int first_dim_size = max_index + 1;

    // Validate that data[i].shape = indices[i].shape + constant
    OpInputList data_inputs;
    OP_REQUIRES_OK(ctx, ctx->input_list("data", &data_inputs));
    const Tensor& data0 = (data_inputs)[0];
    const Tensor& indices0 = (indices_inputs)[0];
    for (int input_num = 0; input_num < indices_inputs.size(); input_num++) {
      const Tensor& indices = (indices_inputs)[input_num];
      const Tensor& data = (data_inputs)[input_num];
      OP_REQUIRES(
          ctx, TensorShapeUtils::StartsWith(data.shape(), indices.shape()),
          errors::InvalidArgument("data[", input_num,
                                  "].shape = ", data.shape().DebugString(),
                                  " does not start with indices[", input_num,
                                  "].shape = ", indices.shape().DebugString()));
      OP_REQUIRES(
          ctx, input_num == 0 || SameExtraShape(data0, indices0, data, indices),
          errors::InvalidArgument(
              "Need data[0].shape[", indices0.dims(), ":] = data[", input_num,
              "].shape[", indices.dims(),
              ":], got data[0].shape = ", data0.shape().DebugString(),
              ", data[", input_num, "].shape = ", data.shape().DebugString(),
              ", indices[0].shape = ", indices0.shape().DebugString(),
              ", indices[", input_num,
              "].shape = ", indices.shape().DebugString()));
    }

    TensorShape output_shape({first_dim_size});
    for (int d = indices0.dims(); d < data0.dims(); d++) {
      output_shape.AddDim(data0.dim_size(d));
    }

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output_tensor));

    if (output_shape.num_elements() == 0) {
      return;
    }

    DmlDevice* device = static_cast<DmlDevice*>(ctx->device());

    const uint64_t data_type_size = DataTypeSize(ctx->expected_output_dtype(0));

    const uint64_t byte_stride =
        output_shape.num_elements() / output_shape.dim_size(0) * data_type_size;

    std::vector<D3D12BufferRegion> input_buffers;
    input_buffers.reserve(data_inputs.size());

    std::vector<D3D12_RESOURCE_BARRIER> barriers;
    barriers.reserve(data_inputs.size() + 1);

    for (const Tensor& data_tensor : data_inputs) {
      if (data_tensor.NumElements() == 0) {
        input_buffers.push_back({});
        continue;
      }

      D3D12BufferRegion input_buffer =
          dml_util::CreateBufferForTensor(device, data_tensor);

      barriers.push_back(CD3DX12_RESOURCE_BARRIER::Transition(
          input_buffer.Resource(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
          D3D12_RESOURCE_STATE_COPY_SOURCE));

      input_buffers.push_back(std::move(input_buffer));
    }

    D3D12BufferRegion output_buffer =
        dml_util::CreateBufferForTensor(device, *output_tensor);

    barriers.push_back(CD3DX12_RESOURCE_BARRIER::Transition(
        output_buffer.Resource(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_COPY_DEST));

    device->GetExecutionContext()->ResourceBarrier(barriers);

    DCHECK(indices_inputs.size() == data_inputs.size());
    for (int tensor_idx = 0; tensor_idx < indices_inputs.size(); ++tensor_idx) {
      const Tensor& indices_tensor = indices_inputs[tensor_idx];
      const Tensor& data_tensor = data_inputs[tensor_idx];

      const D3D12BufferRegion& input_buffer = input_buffers[tensor_idx];

      if (!input_buffer) {
        DCHECK(indices_tensor.NumElements() == 0);
        continue;
      }

      const auto& indices = indices_tensor.flat<int32>();
      for (int i = 0; i < indices_tensor.NumElements(); ++i) {
        int32 output_idx = indices(i);

        const uint64_t src_offset = input_buffer.Offset() + byte_stride * i;
        const uint64_t dst_offset =
            output_buffer.Offset() + byte_stride * output_idx;

        device->GetExecutionContext()->CopyBufferRegion(
            output_buffer.Resource(), dst_offset,
            D3D12_RESOURCE_STATE_COPY_DEST, input_buffer.Resource(), src_offset,
            D3D12_RESOURCE_STATE_COPY_SOURCE, byte_stride);
      }
    }

    for (auto& barrier : barriers) {
      std::swap(barrier.Transition.StateBefore, barrier.Transition.StateAfter);
    }

    device->GetExecutionContext()->ResourceBarrier(barriers);
  }
};

#define DML_REGISTER_KERNELS(type)                        \
  REGISTER_KERNEL_BUILDER(Name("DynamicStitch")           \
                              .Device(DEVICE_DML)         \
                              .HostMemory("indices")      \
                              .TypeConstraint<type>("T"), \
                          DmlDynamicStitchKernel)         \
  REGISTER_KERNEL_BUILDER(Name("ParallelDynamicStitch")   \
                              .Device(DEVICE_DML)         \
                              .HostMemory("indices")      \
                              .TypeConstraint<type>("T"), \
                          DmlDynamicStitchKernel)

TF_CALL_DML_ALL_TYPES(DML_REGISTER_KERNELS);
#undef DML_REGISTER_KERNELS

}  // namespace tensorflow