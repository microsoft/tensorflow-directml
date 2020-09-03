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

#include "tensorflow/core/common_runtime/dml/dml_common.h"
#include "tensorflow/core/framework/types.pb.h"

namespace tensorflow {

class TensorShape;

class DmlTensorDesc {
 public:
  // Constructs an invalid / optional tensor desc.
  DmlTensorDesc() = default;

  DmlTensorDesc(
      DML_TENSOR_DATA_TYPE data_type, absl::Span<const uint32_t> sizes,
      absl::optional<absl::Span<const uint32_t>> strides = absl::nullopt,
      uint32_t guaranteed_base_offset_alignment = 0,
      uint64_t end_padding_in_bytes = 0);

  // Creates a DML tensor desc given a TF data type and shape. This applies
  // broadcasting and padding of shapes, and is used to convert framework
  // currency into DML currency.
  //
  // data_type:
  //   The data type of the tensor. This must be one of the data types supported
  //   by DML.
  //
  // shape:
  //   The desired shape of the tensor. The tensor will be broadcast into this
  //   shape if necessary, if the non_broadcast_shape differs.
  //
  // non_broadcast_shape:
  //   The physical shape of the tensor, i.e. the tensor shape without
  //   broadcasting applied.
  //
  // tensor_layout:
  //   The logical layout of the dimensions in `shape`. If provided, this is
  //   used to shuffle the dimensions in `shape` into a canonical layout. This
  //   method assumes the desired canonical layout is NC[D]HW, which is true for
  //   most DML operators that care about layout. For example, a `shape` of {2,
  //   3, 4} and tensor_layout of {N, W, C} means that N=2, W=3, and C=4, and
  //   will produce a 4D DML tensor desc of sizes {2, 4, 1, 3}. If one of the
  //   dimensions is 'D', then this method returns a 5D tensor desc, otherwise
  //   it returns a 4D tensor desc. This parameter is optional. If not supplied,
  //   no shuffling of dimensions will occur.
  //
  // guaranteed_base_offset_alignment:
  //   The value of DML_TENSOR_DESC::GuaranteedBaseOffsetAlignment to set, or 0
  //   for the default.
  static DmlTensorDesc Create(DataType data_type, const TensorShape& shape,
                              const TensorShape& non_broadcast_shape,
                              absl::Span<const DmlTensorAxis> tensor_layout,
                              uint32_t guaranteed_base_offset_alignment = 0);

  // Same as above, but takes shapes as a span of uint32_t instead of
  // TensorShape.
  static DmlTensorDesc Create(
      DataType data_type, absl::Span<const uint32_t> dimensions,
      absl::Span<const uint32_t> non_broadcast_dimensions,
      absl::Span<const DmlTensorAxis> tensor_layout,
      uint32_t guaranteed_base_offset_alignment = 0);

  // Same as the tensor_layout version, but for tensors that don't need any
  // special layout (i.e. null strides). This overload accepts up to 8
  // dimensions (inclusive), while the tensor_layout overload only accepts up to
  // 5 dimensions.
  static DmlTensorDesc Create(DataType data_type, const TensorShape& shape,
                              const TensorShape& non_broadcast_shape,
                              uint32_t guaranteed_base_offset_alignment = 0);

  // Same as above, but takes shapes as a span of uint32_t instead of
  // TensorShape.
  static DmlTensorDesc Create(
      DataType data_type, absl::Span<const uint32_t> dimensions,
      absl::Span<const uint32_t> non_broadcast_dimensions,
      uint32_t guaranteed_base_offset_alignment = 0);

  DML_TENSOR_DESC GetDmlDesc();

  DML_TENSOR_DATA_TYPE GetDmlDataType() const {
    return buffer_tensor_desc_.DataType;
  }
  DataType GetTfDataType() const { return tf_tensor_type_; }

  void ForceUnsignedDataType();

  bool IsValid() const { return tensor_type_ != DML_TENSOR_TYPE_INVALID; }

  uint32_t GetDimensionCount() const {
    return buffer_tensor_desc_.DimensionCount;
  }

  absl::Span<const uint32_t> GetSizes() const {
    return absl::MakeSpan(sizes_, sizes_ + buffer_tensor_desc_.DimensionCount);
  }

  absl::Span<const uint32_t> GetStrides() const;

  UINT64 GetBufferSizeInBytes() const {
    assert(tensor_type_ == DML_TENSOR_TYPE_BUFFER);
    return buffer_tensor_desc_.TotalTensorSizeInBytes;
  }

 private:
  DML_TENSOR_TYPE tensor_type_ = DML_TENSOR_TYPE_INVALID;
  DataType tf_tensor_type_ = DT_INVALID;
  uint32_t sizes_[DML_TENSOR_DIMENSION_COUNT_MAX1] = {};
  uint32_t strides_[DML_TENSOR_DIMENSION_COUNT_MAX1] = {};
  DML_BUFFER_TENSOR_DESC buffer_tensor_desc_ = {};
};

}  // namespace tensorflow