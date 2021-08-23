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

#include "tensorflow/core/common_runtime/dml/dml_util.h"

#include "tensorflow/core/common_runtime/dml/dml_bfc_allocator.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/stream_executor/platform/default/dso_loader.h"

namespace tensorflow {

Microsoft::WRL::ComPtr<ID3D12Device> TryCreateD3d12Device(
    IUnknown* adapter, D3D_FEATURE_LEVEL minimum_feature_level,
    bool log_failures) {
  auto d3d12_handle_or =
      stream_executor::internal::CachedDsoLoader::GetD3d12DsoHandle();
  if (!d3d12_handle_or.ok()) {
    LOG(WARNING) << "Could not load D3D12.";
    return nullptr;
  }

  using D3D12CreateDeviceFn = decltype(D3D12CreateDevice);

  D3D12CreateDeviceFn* d3d12CreateDevice;
  auto get_symbol_status = Env::Default()->GetSymbolFromLibrary(
      d3d12_handle_or.ValueOrDie(), "D3D12CreateDevice",
      (void**)&d3d12CreateDevice);
  if (!get_symbol_status.ok()) {
    LOG(WARNING) << "Could not find symbol D3D12CreateDevice. ";
    return nullptr;
  }

  Microsoft::WRL::ComPtr<ID3D12Device> d3d12_device;
  HRESULT create_device_hr = d3d12CreateDevice(adapter, minimum_feature_level,
                                               IID_PPV_ARGS(&d3d12_device));
  if (FAILED(create_device_hr)) {
    if (log_failures) {
      LOG(WARNING) << "D3D12CreateDevice failed with HRESULT "
                   << create_device_hr;
    }
    return nullptr;
  }

  return d3d12_device;
}

Microsoft::WRL::ComPtr<ID3D12Device> CreateD3d12Device(
    IUnknown* adapter, D3D_FEATURE_LEVEL minimum_feature_level) {
  auto d3d_device = TryCreateD3d12Device(adapter, minimum_feature_level, true);
  if (!d3d_device) {
    LOG(FATAL) << "Could not load D3D12.";
  }
  return d3d_device;
}

#ifdef _WIN32
Microsoft::WRL::ComPtr<IDXGIFactory4> TryCreateDxgiFactory() {
  auto dxgi_handle_or =
      stream_executor::internal::CachedDsoLoader::GetDxgiDsoHandle();
  if (!dxgi_handle_or.ok()) {
    LOG(WARNING) << "Could not load DXGI.";
    return nullptr;
  }

  using CreateDXGIFactoryFn = decltype(CreateDXGIFactory);

  CreateDXGIFactoryFn* createDxgiFactory;
  auto get_symbol_status = Env::Default()->GetSymbolFromLibrary(
      dxgi_handle_or.ValueOrDie(), "CreateDXGIFactory",
      (void**)&createDxgiFactory);
  if (!get_symbol_status.ok()) {
    LOG(WARNING) << "Could not find symbol CreateDXGIFactory. ";
    return nullptr;
  }

  Microsoft::WRL::ComPtr<IDXGIFactory4> dxgi_factory;
  HRESULT create_factory_hr = createDxgiFactory(IID_PPV_ARGS(&dxgi_factory));
  if (FAILED(create_factory_hr)) {
    LOG(WARNING) << "CreateDXGIFactory failed with HRESULT "
                 << create_factory_hr;
    return nullptr;
  }

  return dxgi_factory;
}
#endif // _WIN32

#ifndef _WIN32

// DXCoreCreateAdapterFactory has a C++ template function overload, so this helper exists
// to disambiguate when resolving the symbol.
template<typename... Ts>
using DXCoreCreateAdapterFactoryFnType = auto(Ts...) -> decltype(DXCoreCreateAdapterFactory(std::declval<Ts>()...));

Microsoft::WRL::ComPtr<IDXCoreAdapterFactory> CreateDxCoreAdapterFactory() {
  auto dxcore_handle_or =
      stream_executor::internal::CachedDsoLoader::GetDxCoreDsoHandle();
  if (!dxcore_handle_or.ok()) {
    LOG(FATAL) << "Could not load DXCore.";
    return nullptr;
  }

  using DXCoreCreateAdapterFactoryFn = DXCoreCreateAdapterFactoryFnType<REFIID,void**>;

  DXCoreCreateAdapterFactoryFn* createDxCoreAdapterFactory;
  auto get_symbol_status = Env::Default()->GetSymbolFromLibrary(
      dxcore_handle_or.ValueOrDie(), "DXCoreCreateAdapterFactory",
      (void**)&createDxCoreAdapterFactory);
  if (!get_symbol_status.ok()) {
    LOG(FATAL) << "Could not find symbol DXCoreCreateAdapterFactory. ";
    return nullptr;
  }

  Microsoft::WRL::ComPtr<IDXCoreAdapterFactory> dxcore_factory;
  HRESULT create_factory_hr =
      createDxCoreAdapterFactory(IID_PPV_ARGS(&dxcore_factory));
  if (FAILED(create_factory_hr)) {
    LOG(FATAL) << "DXCoreCreateAdapterFactory failed with HRESULT "
               << create_factory_hr;
    return nullptr;
  }

  return dxcore_factory;
}
#endif // !_WIN32

Microsoft::WRL::ComPtr<IDMLDevice> TryCreateDmlDevice(
    ID3D12Device* d3d12_device, DML_CREATE_DEVICE_FLAGS dml_flags) {
  auto dml_handle_or =
      stream_executor::internal::CachedDsoLoader::GetDirectMLDsoHandle();
  if (!dml_handle_or.ok()) {
    auto path = getenv("TF_DIRECTML_PATH");
    if (path) {
      LOG(WARNING) << "Could not load DirectML. TF_DIRECTML_PATH is set: "
                   << path;
    } else {
      LOG(WARNING) << "Could not load DirectML.";
    }

    return nullptr;
  }

  using DMLCreateDeviceFn = decltype(DMLCreateDevice);

  DMLCreateDeviceFn* dmlCreateDevice;
  auto get_symbol_status = Env::Default()->GetSymbolFromLibrary(
      dml_handle_or.ValueOrDie(), "DMLCreateDevice", (void**)&dmlCreateDevice);
  if (!get_symbol_status.ok()) {
    LOG(WARNING) << "Could not find symbol DMLCreateDevice. ";
    return nullptr;
  }

  Microsoft::WRL::ComPtr<IDMLDevice> dml_device;
  HRESULT create_device_hr =
      dmlCreateDevice(d3d12_device, dml_flags, IID_PPV_ARGS(&dml_device));
  if (FAILED(create_device_hr)) {
    LOG(WARNING) << "DMLCreateDevice failed with HRESULT " << create_device_hr;
    return {};
  }

  return dml_device;
}

Microsoft::WRL::ComPtr<IDMLDevice> CreateDmlDevice(
    ID3D12Device* d3d12_device, DML_CREATE_DEVICE_FLAGS dml_flags) {
  auto dml_device = TryCreateDmlDevice(d3d12_device, dml_flags);

  if (!dml_device) {
    auto path = getenv("TF_DIRECTML_PATH");
    if (path) {
      LOG(FATAL) << "Could not load DirectML. TF_DIRECTML_PATH is set: "
                 << path;
    } else {
      LOG(FATAL) << "Could not load DirectML.";
    }
  }

  return dml_device;
}

DataType GetTfDataTypeFromDmlDataType(DML_TENSOR_DATA_TYPE type) {
  switch (type) {
    case DML_TENSOR_DATA_TYPE_FLOAT32:
      return DT_FLOAT;
    case DML_TENSOR_DATA_TYPE_FLOAT16:
      return DT_HALF;
    case DML_TENSOR_DATA_TYPE_UINT32:
      return DT_UINT32;
    case DML_TENSOR_DATA_TYPE_UINT16:
      return DT_UINT16;
    case DML_TENSOR_DATA_TYPE_UINT8:
      return DT_UINT8;
    case DML_TENSOR_DATA_TYPE_INT32:
      return DT_INT32;
    case DML_TENSOR_DATA_TYPE_INT16:
      return DT_INT16;
    case DML_TENSOR_DATA_TYPE_INT8:
      return DT_INT8;
    default:
      LOG(FATAL) << "Invalid or unsupported data type.";
  }
}

// TFDML #24881131
bool Is64BitIntegerType(DataType type) {
  switch (type) {
    case DT_UINT64:
    case DT_INT64:
    case DT_UINT64_REF:
    case DT_INT64_REF:
      return true;
    default:
      return false;
  }
}

// TFDML #24881131
bool Is64BitSignedIntegerType(DataType type) {
  switch (type) {
    case DT_INT64:
    case DT_INT64_REF:
      return true;
    default:
      return false;
  }
}

// TFDML #24881131
bool Is64BitUnsignedIntegerType(DataType type) {
  switch (type) {
    case DT_UINT64:
    case DT_UINT64_REF:
      return true;
    default:
      return false;
  }
}

// TODO: 64-bit data support should be revisited once DML supports 64 bit
// datatypes
// TFDML #24881131
DML_TENSOR_DATA_TYPE GetDmlDataTypeFromTfDataType(DataType type) {
  switch (type) {
    case DT_UINT64:
    case DT_UINT64_REF:
      return DML_TENSOR_DATA_TYPE_UINT32;
    case DT_INT64:
    case DT_INT64_REF:
      return DML_TENSOR_DATA_TYPE_INT32;
    case DT_FLOAT:
    case DT_FLOAT_REF:
      return DML_TENSOR_DATA_TYPE_FLOAT32;
    case DT_HALF:
    case DT_HALF_REF:
      return DML_TENSOR_DATA_TYPE_FLOAT16;
    case DT_UINT32:
    case DT_UINT32_REF:
      return DML_TENSOR_DATA_TYPE_UINT32;
    case DT_UINT16:
    case DT_UINT16_REF:
      return DML_TENSOR_DATA_TYPE_UINT16;
    case DT_UINT8:
    case DT_UINT8_REF:
    case DT_BOOL:
    case DT_BOOL_REF:
      return DML_TENSOR_DATA_TYPE_UINT8;
    case DT_INT32:
    case DT_INT32_REF:
      return DML_TENSOR_DATA_TYPE_INT32;
    case DT_INT16:
    case DT_INT16_REF:
      return DML_TENSOR_DATA_TYPE_INT16;
    case DT_INT8:
    case DT_INT8_REF:
      return DML_TENSOR_DATA_TYPE_INT8;
    default:
      LOG(FATAL) << "Invalid or unsupported data type.";
  }
}

uint32_t GetDmlDimensionIndex(DmlTensorAxis axis,
                              uint32_t dml_dimension_count) {
  using namespace DmlTensorAxes;

  if (dml_dimension_count == kNchwDimensionCount) {
    switch (axis) {
      case N:
        return 0;
      case C:
        return 1;
      case H:
        return 2;
      case W:
        return 3;
      default:
        assert(false);
        LOG(FATAL) << "Invalid tensor axis";
    }
  } else {
    assert(dml_dimension_count == kNcdhwDimensionCount);

    switch (axis) {
      case N:
        return 0;
      case C:
        return 1;
      case D:
        return 2;
      case H:
        return 3;
      case W:
        return 4;
      default:
        assert(false);
        LOG(FATAL) << "Invalid tensor axis";
    }
  }
}

DmlTensorLayout GetDmlTensorLayout(TensorFormat format, uint32_t rank) {
  CHECK(rank <= DML_TENSOR_DIMENSION_COUNT_MAX);

  DmlTensorLayout tensor_layout;

  // When converting TF tensor formats to DML tensor layouts, we by default drop
  // dimensions from the left if the dimension count < 4. e.g. if the format is
  // NHWC and rank is 2, we return a layout of WC.

  switch (format) {
    case FORMAT_NHWC:
      if (rank >= 4) {
        tensor_layout.push_back(DmlTensorAxis::N);
      }
      if (rank >= 5) {
        tensor_layout.push_back(DmlTensorAxis::D);
      }
      if (rank >= 3) {
        tensor_layout.push_back(DmlTensorAxis::H);
      }
      if (rank >= 2) {
        tensor_layout.push_back(DmlTensorAxis::W);
      }
      if (rank >= 1) {
        tensor_layout.push_back(DmlTensorAxis::C);
      }
      break;
    case FORMAT_NCHW:
      if (rank >= 4) {
        tensor_layout.push_back(DmlTensorAxis::N);
      }
      if (rank >= 3) {
        tensor_layout.push_back(DmlTensorAxis::C);
      }
      if (rank >= 5) {
        tensor_layout.push_back(DmlTensorAxis::D);
      }
      if (rank >= 2) {
        tensor_layout.push_back(DmlTensorAxis::H);
      }
      if (rank >= 1) {
        tensor_layout.push_back(DmlTensorAxis::W);
      }
      break;
    default:
      LOG(FATAL) << "Unsupported tensor layout";
  }

  return tensor_layout;
}

dml::TensorPolicy GetDmlXTensorPolicy(TensorFormat format) {
  switch (format) {
    case FORMAT_NHWC:
      return dml::TensorPolicy::InterleavedChannel();
    case FORMAT_NCHW:
      return dml::TensorPolicy::Default();
    default:
      LOG(FATAL) << "Unsupported tensor layout";
  }
}

dml::TensorPolicy GetEmulatedInt64TensorPolicy() {
  return dml::TensorPolicy([](DML_TENSOR_DATA_TYPE dataType,
                              DML_TENSOR_FLAGS flags,
                              dml::Span<const uint32_t> sizes) {
    uint32_t dimension_count = static_cast<uint32_t>(sizes.size());

    // Compute strides
    dml::TensorDimensions strides(dimension_count);
    uint32_t stride = 2;  // double all strides
    for (int i = static_cast<int>(dimension_count) - 1; i >= 0; i--) {
      strides[i] = stride;
      stride *= sizes[i];
    }

    dml::TensorProperties props = {};
    props.guaranteedBaseOffsetAlignment = 0;
    props.strides = std::move(strides);
    props.totalTensorSizeInBytes = DMLCalcBufferTensorSize(
        dataType, dimension_count, sizes.data(), props.strides->data());
    return props;
  });
}

namespace dml_util {

void CopyTensorInSameDevice(OpKernelContext* op_ctx, Tensor* dst,
                            const Tensor& src) {
  auto* device = static_cast<Device*>(op_ctx->device());
  op_ctx->op_device_context()->CopyTensorInSameDevice(
      &src, device, dst,
      [op_ctx](const Status& s) { OP_REQUIRES_OK(op_ctx, s); });
}

absl::InlinedVector<absl::optional<DML_BUFFER_BINDING>, 8> GetBufferBindings(
    absl::Span<const D3D12BufferRegion> buffers) {
  absl::InlinedVector<absl::optional<DML_BUFFER_BINDING>, 8> bindings;
  bindings.reserve(buffers.size());

  for (const auto& buffer : buffers) {
    if (buffer) {
      bindings.push_back(buffer.GetBufferBinding());
    } else {
      bindings.push_back(absl::nullopt);
    }
  }

  return bindings;
}

}  // namespace dml_util

}  // namespace tensorflow