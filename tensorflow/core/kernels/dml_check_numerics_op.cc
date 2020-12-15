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

class CheckNumericsInitHelper : public InitializationHelper {
 public:
  struct Attributes {
    explicit Attributes(OpKernelConstruction* ctx) {
      // message is used as the prefix for the assertion error message. For
      // instance, this can be the name of the input op that produced the
      // tensor.
      OP_REQUIRES_OK(ctx, ctx->GetAttr("message", &message));
    }

    std::string message;
  };

  CheckNumericsInitHelper(OpKernelContext* ctx,
                          std::shared_ptr<const Attributes> attr)
      : attr_(attr) {}

  const std::string& GetMessage() const { return attr_->message; }

 private:
  const std::shared_ptr<const Attributes> attr_;
};

class DmlCheckNumericsKernel : public DmlKernel {
 public:
  using InitHelper = CheckNumericsInitHelper;

  explicit DmlCheckNumericsKernel(DmlKernelConstruction* ctx,
                                  const InitHelper* init_helper) {
    DCHECK(ctx->GetInputCount() == 1);
    DCHECK(ctx->GetOutputCount() == 1);

    message_ = init_helper->GetMessage();

    const TensorShape& input_shape = ctx->GetInputTensorShape(0);

    DmlTensorInfo input;
    input.kernel_index = 0;
    input.desc = DmlTensorDesc::Create(ctx->GetInputDataType(0), input_shape,
                                       input_shape);

    DmlTensorInfo output;
    output.kernel_index = 0;
    output.desc =
        DmlTensorDesc::Create(DT_INT32, TensorShape({}), TensorShape({}));

    DmlKernelTensors tensors;
    tensors.inputs = {input};
    tensors.outputs = {output};

    auto inputs = GetDmlTensorDescs(tensors.inputs);
    auto scope = dml::Graph(ctx->GetDmlDevice());
    auto input_tensor = dml::InputTensor(scope, 0, inputs[0]);

    // Reduce doesn't support less than 32bit integer datatypes, so we need to
    // cast to uint32 beforehand
    auto has_nan = dml::Reduce(
        dml::Cast(dml::IsNaN(input_tensor), DML_TENSOR_DATA_TYPE_UINT32),
        DML_REDUCE_FUNCTION_MAX);
    auto has_inf = dml::Reduce(
        dml::Cast(dml::IsInfinity(input_tensor), DML_TENSOR_DATA_TYPE_UINT32),
        DML_REDUCE_FUNCTION_MAX);

    // We pack the NaN and Inf bits into 1 byte, where NaN is the bit at 2^1 and
    // Inf is the bit at 2^0
    auto result = dml::Cast(has_nan * 2 + has_inf, DML_TENSOR_DATA_TYPE_UINT8);

    Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
        scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

    Initialize(ctx, std::move(tensors), compiled_op.Get());
  }

  StatusOr<DmlGpuEvent> Compute(DmlKernelContext* ctx) const override {
    DmlKernel::Compute(ctx);

    OpKernelContext* op_ctx = ctx->GetOpKernelContext();

    // Copy the result to the CPU
    AllocatorAttributes attr;
    Tensor is_error_tensor;
    attr.set_on_host(true);
    TF_RETURN_IF_ERROR(op_ctx->allocate_temp(
        op_ctx->input(0).dtype(), TensorShape({}), &is_error_tensor, attr));

    auto device = static_cast<DmlDevice*>(op_ctx->device());
    Tensor* output_tensor = ctx->GetOutputTensor(0);

    // Copy the Inf and NaN bits from the GPU to the CPU
    Notification note;
    Status status;
    op_ctx->op_device_context()->CopyDeviceTensorToCPU(
        output_tensor, "", device, &is_error_tensor,
        [&note, &status](const Status& copy_status) {
          note.Notify();
          status = copy_status;
        });

    note.WaitForNotification();
    TF_RETURN_IF_ERROR(status);

    uint8_t nan_inf_bits = is_error_tensor.scalar<uint8_t>()();

    // The NaN bit is 2^1 and the Inf bit is 2^0
    if (nan_inf_bits) {
      bool is_nan = nan_inf_bits & 2;
      bool is_inf = nan_inf_bits & 1;
      std::string status;

      if (is_nan && is_inf) {
        status = "Inf and NaN";
      } else if (is_nan) {
        status = "NaN";
      } else if (is_inf) {
        status = "Inf";
      }

      return errors::InvalidArgument(message_, " : Tensor had ", status,
                                     " values");
    } else {
      // If everything is fine, we simply copy the input to the output
      D3D12BufferRegion input_buffer =
          dml_util::CreateBufferForTensor(device, ctx->GetInputTensor(0));

      D3D12BufferRegion output_buffer =
          dml_util::CreateBufferForTensor(device, *output_tensor);

      ctx->CopyBufferToBuffer(output_buffer.Resource(), output_buffer.Offset(),
                              input_buffer.Resource(), input_buffer.Offset(),
                              output_tensor->TotalBytes());
    }

    return ctx->GetCurrentCompletionEvent();
  }

 private:
  std::string message_;
};

#define REGISTER_KERNEL(type)                                             \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("CheckNumerics").Device(DEVICE_DML).TypeConstraint<type>("T"), \
      DmlKernelWrapper<DmlCheckNumericsKernel,                            \
                       GetOutputShapeAsInputShapeHelper>)
TF_CALL_float(REGISTER_KERNEL);
TF_CALL_half(REGISTER_KERNEL);
#undef REGISTER_KERNEL

}  // namespace tensorflow