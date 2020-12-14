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

#include "tensorflow/core/kernels/dml_ops_common.h"

#include "tensorflow/core/common_runtime/dml/dml_util.h"
#include "tensorflow/core/lib/strings/numbers.h"

#if _WIN32
#include "tensorflow/core/platform/windows/wide_char.h"
#endif

using Microsoft::WRL::ComPtr;

namespace tensorflow {

// Retrieves the number of input tensors for the DML operator corresponding to
// the given kernel parameters. Recall that the DML operator may have a
// different number of inputs to the TF kernel.
static uint32_t GetDmlInputTensorCount(DmlKernelConstruction* ctx,
                                       const DmlKernelParams& params) {
  // The kernel_input_indices is optional; if it's not supplied, then it means
  // the DML operator and TF kernel have the same number of inputs (and in the
  // same order).
  if (params.kernel_input_indices.empty()) {
    return ctx->GetInputCount();
  }

  return params.kernel_input_indices.size();
}

// Same as GetDmlInputTensorCount, but for outputs
static uint32_t GetDmlOutputTensorCount(DmlKernelConstruction* ctx,
                                        const DmlKernelParams& params) {
  if (params.kernel_output_indices.empty()) {
    return ctx->GetOutputCount();
  }

  return params.kernel_output_indices.size();
}

/*static*/ DmlTensorDesc DmlKernel::CreateTensorDescFromInput(
    DmlKernelConstruction* ctx, uint32_t kernel_index,
    absl::Span<const DmlTensorAxis> tensor_layout,
    const absl::optional<TensorShape>& tensor_shape) {
  CHECK(!tensor_layout.empty());

  DataType data_type = ctx->GetInputDataType(kernel_index);
  TensorShape actual_tensor_shape = ctx->GetInputTensorShape(kernel_index);
  TensorShape desired_shape =
      tensor_shape ? *tensor_shape : actual_tensor_shape;

  return DmlTensorDesc::Create(data_type, desired_shape, actual_tensor_shape,
                               tensor_layout);
}

/*static*/ DmlTensorDesc DmlKernel::CreateTensorDescFromOutput(
    DmlKernelConstruction* ctx, uint32_t kernel_index,
    absl::Span<const DmlTensorAxis> tensor_layout,
    const absl::optional<TensorShape>& tensor_shape) {
  CHECK(!tensor_layout.empty());

  DataType data_type = ctx->GetOutputDataType(kernel_index);
  TensorShape actual_tensor_shape = ctx->GetOutputTensorShape(kernel_index);
  TensorShape desired_shape =
      tensor_shape ? *tensor_shape : actual_tensor_shape;

  return DmlTensorDesc::Create(data_type, desired_shape, actual_tensor_shape,
                               tensor_layout);
}

/*static*/ DmlTensorDesc DmlKernel::CreateTensorDescFromInput(
    DmlKernelConstruction* ctx, uint32_t kernel_index,
    const absl::optional<TensorShape>& tensor_shape) {
  DataType data_type = ctx->GetInputDataType(kernel_index);
  TensorShape actual_tensor_shape = ctx->GetInputTensorShape(kernel_index);
  TensorShape desired_shape =
      tensor_shape ? *tensor_shape : actual_tensor_shape;

  return DmlTensorDesc::Create(data_type, desired_shape, actual_tensor_shape);
}

/*static*/ DmlTensorDesc DmlKernel::CreateTensorDescFromOutput(
    DmlKernelConstruction* ctx, uint32_t kernel_index,
    const absl::optional<TensorShape>& tensor_shape) {
  DataType data_type = ctx->GetOutputDataType(kernel_index);
  TensorShape actual_tensor_shape = ctx->GetOutputTensorShape(kernel_index);
  TensorShape desired_shape =
      tensor_shape ? *tensor_shape : actual_tensor_shape;

  return DmlTensorDesc::Create(data_type, desired_shape, actual_tensor_shape);
}

/*static*/ DmlKernelTensors DmlKernel::GetTensorInfos(
    DmlKernelConstruction* ctx, const DmlKernelParams& params) {
  DmlKernelTensors tensor_descs;
  tensor_descs.output_refs_forwarding = params.output_refs_forwarding;

  // The number of inputs/outputs the DML operator takes
  uint32_t dml_input_count = GetDmlInputTensorCount(ctx, params);
  uint32_t dml_output_count = GetDmlOutputTensorCount(ctx, params);

  // Get input DML tensor descs
  for (uint32_t i = 0; i < dml_input_count; ++i) {
    // Retrieve the TF kernel tensor index correponding to this DML input
    // tensor
    absl::optional<uint32_t> kernel_index;
    if (!params.kernel_input_indices.empty()) {
      kernel_index = params.kernel_input_indices[i];
    } else {
      kernel_index = i;
    }

    // This is a null-valued optional input tensor
    if (!kernel_index) {
      tensor_descs.inputs.push_back(absl::nullopt);
      continue;
    }

    DmlTensorInfo tensor_info = {};
    tensor_info.desc =
        CreateTensorDescFromInput(ctx, *kernel_index, params.input_shape);
    tensor_info.kernel_index = *kernel_index;

    tensor_descs.inputs.push_back(std::move(tensor_info));
  }

  // Get output DML tensor descs
  for (uint32_t i = 0; i < dml_output_count; ++i) {
    // Retrieve the TF kernel tensor index corresponding to this DML output
    // tensor
    absl::optional<uint32_t> kernel_index;
    if (!params.kernel_output_indices.empty()) {
      kernel_index = params.kernel_output_indices[i];
    } else {
      kernel_index = i;
    }

    // This is a null-valued optional input tensor
    if (!kernel_index) {
      tensor_descs.outputs.push_back(absl::nullopt);
      continue;
    }

    DmlTensorInfo tensor_info = {};
    tensor_info.desc =
        CreateTensorDescFromOutput(ctx, *kernel_index, params.output_shape);
    tensor_info.kernel_index = *kernel_index;

    tensor_descs.outputs.push_back(std::move(tensor_info));
  }

  return tensor_descs;
}

void DmlKernel::Initialize(DmlKernelConstruction* ctx,
                           DmlKernelTensors&& tensor_descs,
                           const DML_OPERATOR_DESC& op_desc) {
  IDMLDevice* dml_device = ctx->GetDmlDevice();

  ComPtr<IDMLOperator> op;
  DML_CHECK_SUCCEEDED(dml_device->CreateOperator(&op_desc, IID_PPV_ARGS(&op)));

  // For now, we don't set any flags
  DML_EXECUTION_FLAGS execution_flags = DML_EXECUTION_FLAG_NONE;

  // Compile the operator
  ComPtr<IDMLCompiledOperator> compiled_op;
  DML_CHECK_SUCCEEDED(dml_device->CompileOperator(op.Get(), execution_flags,
                                                  IID_PPV_ARGS(&compiled_op)));

  // Defer to the other overload of Initialize(), which does the actual work
  Initialize(ctx, std::move(tensor_descs), compiled_op.Get());
}

void DmlKernel::Initialize(DmlKernelConstruction* ctx,
                           DmlKernelTensors&& tensor_descs,
                           IDMLCompiledOperator* compiled_op) {
  assert(!compiled_op_);  // Initialize must only be called once

#if _WIN32
  // Set the name of this compiled op, for debugging purposes. We use the name
  // of the op (e.g. "Conv2D") rather than the name of the node because this
  // kernel may be shared across many nodes.
  std::wstring op_type =
      Utf8ToWideChar(ctx->GetOpKernelContext()->op_kernel().type_string());
  DML_CHECK_SUCCEEDED(compiled_op->SetName(op_type.c_str()));
#endif

  compiled_op_ = compiled_op;
  input_descs_ = std::move(tensor_descs.inputs);
  output_descs_ = std::move(tensor_descs.outputs);
  output_refs_forwarding_ = std::move(tensor_descs.output_refs_forwarding);
  init_helper_ = ctx->GetInitializationHelper();

  // Create the persistent resource, if necessary

  DML_BINDING_PROPERTIES binding_props = compiled_op_->GetBindingProperties();

  if (binding_props.PersistentResourceSize != 0) {
    VLOG(2) << "Allocating"
            << strings::HumanReadableNumBytes(
                   binding_props.PersistentResourceSize)
            << " persistent resource for kernel "
            << ctx->GetOpKernelContext()->op_kernel().type_string();

    persistent_resource_ =
        ctx->AllocateDefaultBuffer(binding_props.PersistentResourceSize);

    OP_REQUIRES(ctx->GetOpKernelContext(), persistent_resource_,
                errors::ResourceExhausted("OOM when allocating a buffer of ",
                                          binding_props.PersistentResourceSize,
                                          " bytes"));

    persistent_resource_binding_ = persistent_resource_.GetBufferBinding();
  }

  // Initialize the operator

  // We don't supply any input bindings, because we never set OWNED_BY_DML
  absl::Span<const DML_BUFFER_BINDING> input_init_bindings = {};

  ctx->InitializeOperator(compiled_op_.Get(), GetPersistentResourceBinding(),
                          input_init_bindings);
}

StatusOr<DmlGpuEvent> DmlKernel::Compute(DmlKernelContext* ctx) const {
  auto input_buffers = CreateInputBuffers(ctx);
  auto output_buffers = CreateOutputBuffers(ctx);

  // Set up input, output, and persistent resource binding descs
  auto input_bindings = dml_util::GetBufferBindings(input_buffers);
  auto output_bindings = dml_util::GetBufferBindings(output_buffers);

  return ctx->ExecuteOperator(compiled_op_.Get(),
                              GetPersistentResourceBinding(), input_bindings,
                              output_bindings);
}

absl::InlinedVector<D3D12BufferRegion, 8> DmlKernel::CreateInputBuffers(
    DmlKernelContext* ctx) const {
  absl::InlinedVector<D3D12BufferRegion, 8> input_buffers(input_descs_.size());

  for (uint32_t i = 0; i < input_descs_.size(); ++i) {
    if (input_descs_[i]) {
      uint32_t kernel_index = input_descs_[i]->kernel_index;

      const Tensor& input_tensor = ctx->GetInputTensor(kernel_index);
      input_buffers[i] = ctx->CreateBufferForTensor(input_tensor);
    }
  }

  return input_buffers;
}

absl::InlinedVector<D3D12BufferRegion, 4> DmlKernel::CreateOutputBuffers(
    DmlKernelContext* ctx) const {
  absl::InlinedVector<D3D12BufferRegion, 4> output_buffers(
      output_descs_.size());

  for (uint32_t i = 0; i < output_descs_.size(); ++i) {
    if (output_descs_[i]) {
      uint32_t kernel_index = output_descs_[i]->kernel_index;

      Tensor* output_tensor = ctx->GetOutputTensor(kernel_index);
      output_buffers[i] = ctx->CreateBufferForTensor(*output_tensor);
    }
  }

  return output_buffers;
}

const DML_BUFFER_BINDING* DmlKernel::GetPersistentResourceBinding() const {
  return persistent_resource_binding_ ? &*persistent_resource_binding_
                                      : nullptr;
}

/*static*/ absl::InlinedVector<DML_TENSOR_DESC, 8> DmlKernel::GetDmlTensorDescs(
    absl::Span<absl::optional<DmlTensorInfo>> tensor_infos) {
  absl::InlinedVector<DML_TENSOR_DESC, 8> descs;

  for (auto& info : tensor_infos) {
    DML_TENSOR_DESC desc = {};
    if (info) {
      desc = info->desc.GetDmlDesc();
    }
    descs.push_back(desc);
  }

  return descs;
}

}  // namespace tensorflow

namespace dml {
DML_SCALAR_UNION ScalarUnion(double value, DML_TENSOR_DATA_TYPE data_type) {
  DML_SCALAR_UNION scalar{};

  switch (data_type) {
    case DML_TENSOR_DATA_TYPE_INT8:
      scalar.Int8 = static_cast<int8_t>(value);
      break;

    case DML_TENSOR_DATA_TYPE_UINT8:
      scalar.UInt8 = static_cast<uint8_t>(value);
      break;

    case DML_TENSOR_DATA_TYPE_INT16:
      scalar.Int16 = static_cast<int16_t>(value);
      break;

    case DML_TENSOR_DATA_TYPE_UINT16:
      scalar.UInt16 = static_cast<uint16_t>(value);
      break;

    case DML_TENSOR_DATA_TYPE_INT32:
      scalar.Int32 = static_cast<int32_t>(value);
      break;

    case DML_TENSOR_DATA_TYPE_UINT32:
      scalar.UInt32 = static_cast<uint32_t>(value);
      break;

    case DML_TENSOR_DATA_TYPE_INT64:
      scalar.Int64 = static_cast<int64_t>(value);
      break;

    case DML_TENSOR_DATA_TYPE_UINT64:
      scalar.UInt64 = static_cast<uint64_t>(value);
      break;

    case DML_TENSOR_DATA_TYPE_FLOAT32:
      scalar.Float32 = static_cast<float>(value);
      break;

    case DML_TENSOR_DATA_TYPE_FLOAT64:
      scalar.Float64 = static_cast<double>(value);
      break;

    case DML_TENSOR_DATA_TYPE_FLOAT16: {
      Eigen::half float16_value = static_cast<Eigen::half>(value);
      const BYTE* float16_bytes = reinterpret_cast<const BYTE*>(&float16_value);
      std::copy(float16_bytes, float16_bytes + sizeof(float16_value),
                scalar.Bytes);
    } break;

    default:
      DML_CHECK_SUCCEEDED(E_INVALIDARG);
      break;
  }

  return scalar;
}
}  // namespace dml
