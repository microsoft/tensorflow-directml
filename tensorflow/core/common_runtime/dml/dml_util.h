/* Copyright (c) Microsoft Corporation.

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

#pragma once

#include "tensorflow/core/common_runtime/dml/dml_buffer_region.h"
#include "tensorflow/core/common_runtime/dml/dml_common.h"
#include "tensorflow/core/common_runtime/dml/dml_device.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

enum class DxCallErrorHandling { Silent, Warning, Fatal };

// Calls D3D12CreateDevice after lazily loading the D3D12 module. Returns an
// empty ComPtr if unsuccessful. By default this helper will log warnings if the
// module is not found, and it will log nothing if the
// call to D3D12CreateDevice itself fails.
Microsoft::WRL::ComPtr<ID3D12Device> TryCreateD3d12Device(
    IUnknown* adapter, D3D_FEATURE_LEVEL minimum_feature_level,
    DxCallErrorHandling call_error_handling = DxCallErrorHandling::Silent,
    DxCallErrorHandling module_error_handling = DxCallErrorHandling::Warning);

// Calls D3D12CreateDevice after lazily loading the D3D12 module. The call must
// succeed or a fatal message will be logged and the program will abort.
inline Microsoft::WRL::ComPtr<ID3D12Device> CreateD3d12Device(
    IUnknown* adapter, D3D_FEATURE_LEVEL minimum_feature_level) {
  return TryCreateD3d12Device(adapter, minimum_feature_level,
                              DxCallErrorHandling::Fatal,
                              DxCallErrorHandling::Fatal);
}

#ifdef _WIN32
// Calls CreateDXGIFactory after lazily loading the DXGI module. Returns an
// empty ComPtr if unsuccessful. By default this helper will log warnings if the
// module is not found, and it will log nothing if the
// call to CreateDXGIFactory itself fails.
Microsoft::WRL::ComPtr<IDXGIFactory4> TryCreateDxgiFactory(
    DxCallErrorHandling call_error_handling = DxCallErrorHandling::Silent,
    DxCallErrorHandling module_error_handling = DxCallErrorHandling::Warning);

// Calls CreateDXGIFactory after lazily loading the DXGI module. The call must
// succeed or a fatal message will be logged and the program will abort.
inline Microsoft::WRL::ComPtr<IDXGIFactory4> CreateDxgiFactory() {
  return TryCreateDxgiFactory(DxCallErrorHandling::Fatal,
                              DxCallErrorHandling::Fatal);
}
#endif

#ifndef _WIN32
// Calls DXCoreCreateAdapterFactory after lazily loading the DXCore module.
// Returns an empty ComPtr if unsuccessful. By default this helper will log
// warnings if the module is not found, and it will log nothing if the call to
// DXCoreCreateAdapterFactory itself fails.
Microsoft::WRL::ComPtr<IDXCoreAdapterFactory> TryCreateDxCoreAdapterFactory(
    DxCallErrorHandling call_error_handling = DxCallErrorHandling::Silent,
    DxCallErrorHandling module_error_handling = DxCallErrorHandling::Warning);

// Calls DXCoreCreateAdapterFactory after lazily loading the DXCore module. The
// call must succeed or a fatal message will be logged and the program will
// abort.
inline Microsoft::WRL::ComPtr<IDXCoreAdapterFactory>
CreateDxCoreAdapterFactory() {
  return TryCreateDxCoreAdapterFactory(DxCallErrorHandling::Fatal,
                                       DxCallErrorHandling::Fatal);
}
#endif

// Calls DMLCreateDevice after lazily loading the DirectML module. Returns an
// empty ComPtr if unsuccessful. By default this helper will log warnings if the
// module is not found, and it will log nothing if the
// call to DMLCreateDevice itself fails.
Microsoft::WRL::ComPtr<IDMLDevice> TryCreateDmlDevice(
    ID3D12Device* d3d12_device, DML_CREATE_DEVICE_FLAGS dml_flags,
    DxCallErrorHandling call_error_handling = DxCallErrorHandling::Silent,
    DxCallErrorHandling module_error_handling = DxCallErrorHandling::Warning);

// Calls DMLCreateDevice after lazily loading the DirectML module. The
// call must succeed or a fatal message will be logged and the program will
// abort.
inline Microsoft::WRL::ComPtr<IDMLDevice> CreateDmlDevice(
    ID3D12Device* d3d12_device, DML_CREATE_DEVICE_FLAGS dml_flags) {
  return TryCreateDmlDevice(d3d12_device, dml_flags, DxCallErrorHandling::Fatal,
                            DxCallErrorHandling::Fatal);
}

// Converts a DML tensor data type to a TF tensor data type and vice versa.
DataType GetTfDataTypeFromDmlDataType(DML_TENSOR_DATA_TYPE type);
DML_TENSOR_DATA_TYPE GetDmlDataTypeFromTfDataType(DataType type);

// Converts a TF TensorShape into an array of uint32_t, and validates that the
// shape is representable as uint32_t. This is useful because shapes in TF are
// logically represented as int64, whereas DML requires uint32.
template <int dim_count = 4>
absl::InlinedVector<uint32_t, dim_count> NarrowTensorShape(
    const TensorShape& shape) {
  CHECK(shape.dims() >= 0);  // No partial tensor shapes allowed

  absl::InlinedVector<uint32_t, dim_count> narrowed_shape;
  for (int i = 0; i < shape.dims(); ++i) {
    int64_t dim = shape.dim_size(i);

    CHECK(dim >= 0 && dim <= UINT32_MAX);
    narrowed_shape.push_back(static_cast<uint32_t>(dim));
  }

  return narrowed_shape;
}

// Retrieves the index in canonical DML order (NCHW/NCDHW) of the specified
// axis. For example, the index of the 'H' dimension is 2 for 4D tensors descs,
// and 3 for 5D tensor descs.
uint32_t GetDmlDimensionIndex(DmlTensorAxis axis, uint32_t dml_dimension_count);

// Converts a TF-style TensorFormat and rank (or "dimension count") into an
// equivalent DmlTensorLayout. If rank < 4, this function defaults to dropping
// dimensions from the left. For example, a format of NHWC and rank of 2 results
// in a DML tensor layout of WC.
DmlTensorLayout GetDmlTensorLayout(TensorFormat format, uint32_t rank);

// Converts a TF-style TensorFormat into the equivalent DirectMLX enum value.
dml::TensorPolicy GetDmlXTensorPolicy(TensorFormat format);

dml::TensorStrides ComputePackedStrides(const dml::Span<const uint32_t>& sizes);

namespace dml_util {

// Kicks off a copy from the `src` GPU tensor to `dst` without waiting for it to
// complete.
void CopyTensorInSameDevice(OpKernelContext* op_ctx, Tensor* dst,
                            const Tensor& src);

// Calls D3D12BufferRegion::GetBufferBinding on each of the buffers and returns
// the result.
absl::InlinedVector<absl::optional<DML_BUFFER_BINDING>, 8> GetBufferBindings(
    absl::Span<const D3D12BufferRegion> buffers);

template <typename T, typename... TArgs>
Microsoft::WRL::ComPtr<T> MakeOrAbort(TArgs&&... args) {
  auto obj = Microsoft::WRL::Make<T>(std::forward<TArgs>(args)...);

  if (!obj.Get()) {
    DML_CHECK_SUCCEEDED(E_OUTOFMEMORY);
  }

  return obj;
}

}  // namespace dml_util

}  // namespace tensorflow