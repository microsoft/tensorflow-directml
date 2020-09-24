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

Microsoft::WRL::ComPtr<IDMLDevice> CreateDmlDevice(
    ID3D12Device* d3d12_device, DML_CREATE_DEVICE_FLAGS dml_flags) {
  auto dml_handle_or =
      stream_executor::internal::CachedDsoLoader::GetDirectMLDsoHandle();
  if (!dml_handle_or.ok()) {
    LOG(FATAL) << "Could not load DirectML. TF_DIRECTML_PATH="
               << getenv("TF_DIRECTML_PATH");
  }

  using DMLCreateDeviceFn = decltype(DMLCreateDevice);

  DMLCreateDeviceFn* dmlCreateDevice;
  Env::Default()->GetSymbolFromLibrary(
      dml_handle_or.ValueOrDie(), "DMLCreateDevice", (void**)&dmlCreateDevice);

  Microsoft::WRL::ComPtr<IDMLDevice> dml_device;
  DML_CHECK_SUCCEEDED(
      dmlCreateDevice(d3d12_device, dml_flags, IID_PPV_ARGS(&dml_device)));

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

DML_TENSOR_DATA_TYPE GetDmlDataTypeFromTfDataType(DataType type) {
  switch (type) {
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

D3D12BufferRegion CreateBufferForTensor(const DmlDevice* device,
                                        const Tensor& tensor) {
  DmlAllocator* allocator = device->GetAllocator();
  const void* p = tensor.tensor_data().data();

  // Important: we must use AllocatedBytes() here and not TotalBytes() because
  // AllocatedBytes includes the necessary padding and alignment, whereas
  // TotalBytes is exactly equal to the number of elements multiplied by the
  // element size.
  uint64_t size_in_bytes = tensor.AllocatedBytes();

  auto region = allocator->CreateBufferRegion(p, size_in_bytes);

  // DML always requires at least 4 byte alignment in all cases, so both the
  // offset and size must certainly be divisible by 4
  DCHECK(region.Offset() % 4 == 0);
  DCHECK(region.SizeInBytes() % 4 == 0);

  return region;
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