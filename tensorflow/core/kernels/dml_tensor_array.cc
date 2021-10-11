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

#include "dml_tensor_array.h"

#include "tensorflow/core/common_runtime/dml/dml_common.h"
#include "tensorflow/core/common_runtime/dml/dml_device.h"
#include "tensorflow/core/common_runtime/dml/dml_util.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {
namespace tensor_array {

Status DmlAddToTensor(OpKernelContext* ctx, Tensor* sum, const Tensor* current,
                      const Tensor* add) {
  NodeDef def;
  def.set_op("AddV2");
  def.add_input("x");
  def.add_input("y");
  SetAttrValue(sum->dtype(), &(*def.mutable_attr())["T"]);

  // Delegate to DML's kernel for Add
  Status s;
  auto op_kernel = CreateOpKernel(DEVICE_DML, ctx->device(),
                                  ctx->get_allocator(AllocatorAttributes()),
                                  def, TF_GRAPH_DEF_VERSION, &s);
  TF_RETURN_IF_ERROR(s);

  absl::InlinedVector<TensorValue, 4> inputs = {
      TensorValue(const_cast<Tensor*>(current)),
      TensorValue(const_cast<Tensor*>(add))};

  AllocatorAttributes output_attrs[] = {AllocatorAttributes()};

  OpKernelContext::Params op_ctx_params;
  op_ctx_params.op_kernel = op_kernel.get();
  op_ctx_params.device = ctx->device();
  op_ctx_params.inputs = &inputs;
  op_ctx_params.output_attr_array = output_attrs;

  OpKernelContext op_ctx(&op_ctx_params, 1);
  op_kernel->Compute(&op_ctx);

  TensorValue out = op_ctx.release_output(0);
  ctx->device()->CopyTensorInSameDevice(
      out.tensor, sum, ctx->op_device_context(), [ctx, out](const Status& s) {
        OP_REQUIRES_OK(ctx, s);
        delete out.tensor;
      });

  return op_ctx.status();
}

void DmlTensorSetZero(OpKernelContext* ctx, Tensor* value) {
  auto device_context =
      static_cast<DMLDeviceContext*>(ctx->op_device_context());

  D3D12BufferRegion dst = device_context->GetBufferForTensor(*value);
  device_context->ZeroBuffer(dst);
}

void DmlConcatTensors(OpKernelContext* ctx, Tensor* output_tensor,
                      absl::Span<PersistentTensor> values) {
  auto device_context =
      static_cast<DMLDeviceContext*>(ctx->op_device_context());

  D3D12BufferRegion dst = device_context->GetBufferForTensor(*output_tensor);
  uint64_t dst_offset = 0;

  for (PersistentTensor& value : values) {
    const Tensor& input_tensor = *value.AccessTensor(ctx);

    // These should have been validated earlier
    assert(input_tensor.dtype() == output_tensor->dtype());
    assert(input_tensor.NumElements() <= output_tensor->NumElements());

    uint64_t bytes_to_copy = input_tensor.TotalBytes();
    D3D12BufferRegion src = device_context->GetBufferForTensor(input_tensor);

    device_context->CopyBufferToBuffer(dst.Subregion(dst_offset),
                                       src.Subregion(0, bytes_to_copy));

    dst_offset += bytes_to_copy;
    CHECK(dst_offset <= dst.ResourceInCopyDstState()->GetDesc().Width);
  }
}

void DmlSplitTensor(OpKernelContext* ctx, Tensor* output_tensor,
                    const Tensor& input_tensor, int64 start_element,
                    int64 element_count) {
  // These should have been validated earlier
  assert(input_tensor.dtype() == output_tensor->dtype());
  assert(output_tensor->NumElements() == element_count);
  assert(start_element + element_count <= input_tensor.NumElements());

  uint64_t element_byte_size =
      output_tensor->TotalBytes() / output_tensor->NumElements();
  uint64_t bytes_to_copy = element_count * element_byte_size;
  uint64_t src_offset = start_element * element_byte_size;

  auto device_context =
      static_cast<DMLDeviceContext*>(ctx->op_device_context());
  D3D12BufferRegion dst = device_context->GetBufferForTensor(*output_tensor);
  D3D12BufferRegion src = device_context->GetBufferForTensor(input_tensor);

  device_context->CopyBufferToBuffer(dst,
                                     src.Subregion(src_offset, bytes_to_copy));
}

Status DmlTensorCopy(OpKernelContext* ctx, Tensor* src, Tensor* dst) {
  TF_RETURN_IF_ERROR(ctx->allocate_temp(src->dtype(), src->shape(), dst));
  ctx->device()->CopyTensorInSameDevice(
      src, dst, ctx->op_device_context(),
      [ctx](const Status& s) { OP_REQUIRES_OK(ctx, s); });
  return Status::OK();
}

}  // namespace tensor_array
}  // namespace tensorflow