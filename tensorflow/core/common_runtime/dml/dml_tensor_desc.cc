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

#include "tensorflow/core/common_runtime/dml/dml_tensor_desc.h"

#include "tensorflow/core/common_runtime/dml/dml_util.h"

namespace tensorflow {

// Constructs a std::set from a span of values.
template <typename T>
std::set<T> AsSet(absl::Span<const T> vals) {
  return std::set<T>(vals.begin(), vals.end());
}

DmlTensorDesc::DmlTensorDesc(DML_TENSOR_DATA_TYPE data_type,
                             absl::Span<const uint32_t> sizes,
                             absl::optional<absl::Span<const uint32_t>> strides,
                             uint32_t guaranteed_base_offset_alignment,
                             uint64_t end_padding_in_bytes) {
  tensor_type_ = DML_TENSOR_TYPE_BUFFER;
  buffer_tensor_desc_.DataType = data_type;
  tf_tensor_type_ = GetTfDataTypeFromDmlDataType(data_type);

  CHECK(sizes.size() <= ABSL_ARRAYSIZE(sizes_));
  std::copy(sizes.begin(), sizes.end(), sizes_);
  buffer_tensor_desc_.Sizes = sizes_;

  if (strides) {
    CHECK(strides->size() == sizes.size());
    std::copy(strides->begin(), strides->end(), strides_);
    buffer_tensor_desc_.Strides = strides_;
  }

  buffer_tensor_desc_.DimensionCount = static_cast<uint32_t>(sizes.size());
  buffer_tensor_desc_.Flags = DML_TENSOR_FLAG_NONE;

  buffer_tensor_desc_.GuaranteedBaseOffsetAlignment =
      guaranteed_base_offset_alignment;
  buffer_tensor_desc_.TotalTensorSizeInBytes = DMLCalcBufferTensorSize(
      buffer_tensor_desc_.DataType, buffer_tensor_desc_.DimensionCount, sizes_,
      strides ? strides_ : nullptr) + end_padding_in_bytes;
}

/*static*/ DmlTensorDesc DmlTensorDesc::Create(
    DataType data_type, const TensorShape& shape,
    const TensorShape& non_broadcast_shape,
    absl::Span<const DmlTensorAxis> tensor_layout,
    uint32_t guaranteed_base_offset_alignment) {
  const auto& dimensions = NarrowTensorShape(shape);
  const auto& non_broadcast_dimensions = NarrowTensorShape(non_broadcast_shape);

  return Create(data_type, dimensions, non_broadcast_dimensions, tensor_layout,
                guaranteed_base_offset_alignment);
}

/*static*/ DmlTensorDesc DmlTensorDesc::Create(
    DataType data_type, absl::Span<const uint32_t> dimensions,
    absl::Span<const uint32_t> non_broadcast_dimensions,
    absl::Span<const DmlTensorAxis> tensor_layout,
    uint32_t guaranteed_base_offset_alignment) {
  // Broadcasting can never remove dimensions, it can only add them
  CHECK(dimensions.size() >= non_broadcast_dimensions.size());

  // DML supports only up to 5 dimensions
  const uint32_t rank = static_cast<uint32_t>(dimensions.size());
  CHECK(rank <= DML_TENSOR_DIMENSION_COUNT_MAX);

  // Provide a default layout if none was supplied
  DmlTensorLayout default_layout;
  if (tensor_layout.empty()) {
    // This method is implemented with the assumption that the desired canonical
    // layout is NC[D]HW. Therefore we default it to NC[D]HW if no tensor_layout
    // is specified, as this will result in no change to the dimension order.
    default_layout = GetDmlTensorLayout(FORMAT_NCHW, rank);
    tensor_layout = default_layout;
  }

  // Sanity
  CHECK(tensor_layout.size() == dimensions.size());

  // Ensure that each axis supplied in the tensor_layout occurs only once in
  // that list (you can't have more than one dimension value per axis)
  DCHECK(AsSet(tensor_layout).size() == tensor_layout.size());

  ////////////////////////////////////////
  // Compute the strides for the resulting tensor, taking into account
  // broadcasting. Broadcasting stretches any dimensions with a single element.
  //
  // e.g. physical [2,1,4]
  //       desired [2,3,4]
  //       output  [2,3,4]
  //       strides [4,0,1]
  //
  // Note that the strides we compute here are in the order as specified by
  // `tensor_layout`; these are later swizzled into the canonical NCHW order as
  // required by DML.

  absl::InlinedVector<uint32_t, DML_TENSOR_DIMENSION_COUNT_MAX> strides(rank);
  uint32_t current_stride = 1;

  for (uint32_t i = 0; i < rank; ++i) {
    // Walk backwards through our dimensions and set the corresponding DML
    // dimension (as determined by the supplied tensor_layout). We walk the
    // dimensions backward so that we can broadcast as we go, as numpy-style
    // broadcasting is defined to start from trailing (fastest-changing)
    // dimensions and move upward.
    int dim_index = static_cast<int>(dimensions.size()) - i - 1;
    int non_broadcast_dim_index =
        static_cast<int>(non_broadcast_dimensions.size()) - i - 1;

    uint32_t dim_size = dimensions[dim_index];

    // If the non-broadcast shape has fewer dimensions, the missing dimensions
    // are equivalent to having a size of 1
    uint32_t non_broadcast_dim_size =
        non_broadcast_dim_index >= 0
            ? non_broadcast_dimensions[non_broadcast_dim_index]
            : 1;

    bool is_dim_broadcast = (dim_size != non_broadcast_dim_size);

    // Broadcasting is only valid if the non-broadcast dimension is 1
    CHECK(!is_dim_broadcast || non_broadcast_dim_size == 1);

    if (is_dim_broadcast) {
      strides[dim_index] = 0;
    } else {
      strides[dim_index] = current_stride;
      current_stride *= dim_size;
    }
  }

  ////////////////////////////////////////
  // Swizzle and pad the dimensions/strides into the order and dimension count
  // required by DML (either NCHW or NCDHW)

  // DML only supports two dimension counts: 4 and 5. A dimension count of 5 is
  // used only if one of the dimensions is D (for depth)
  const bool has_depth_dim =
      std::find(tensor_layout.begin(), tensor_layout.end(), DmlTensorAxis::D) !=
      tensor_layout.end();
  const uint32_t dml_dimension_count =
      has_depth_dim ? kNcdhwDimensionCount : kNchwDimensionCount;

  // When padding out dimensions to the count required by DML, fill unused
  // dimensions with a size of 1 and stride of 0.
  const uint32_t default_dimension_size = 1;
  const uint32_t default_dimension_stride = 0;

  absl::InlinedVector<uint32_t, DML_TENSOR_DIMENSION_COUNT_MAX> dml_sizes(
      dml_dimension_count, default_dimension_size);
  absl::InlinedVector<uint32_t, DML_TENSOR_DIMENSION_COUNT_MAX> dml_strides(
      dml_dimension_count, default_dimension_stride);

  assert(dimensions.size() == rank);
  assert(strides.size() == rank);
  assert(tensor_layout.size() == rank);

  // Walk through the sizes and strides and assign them to the correct dimension
  // index for the DML tensor desc
  for (uint32_t i = 0; i < rank; ++i) {
    using namespace DmlTensorAxes;

    DmlTensorAxis axis = tensor_layout[i];
    uint32_t dim_size = dimensions[i];
    uint32_t dim_stride = strides[i];

    // The index of a particular dimension (e.g. 'H') can vary depending on the
    // dimension count
    uint32_t dml_dim_index = GetDmlDimensionIndex(axis, dml_dimension_count);

    dml_sizes[dml_dim_index] = dim_size;
    dml_strides[dml_dim_index] = dim_stride;
  }

  ////////////////////////////////////////
  // Handle 64-bit tensors.

  // TODO #24881131: 64-bit data support should be revisited once DML supports
  // these types
  // TFDML #24881131

  uint64_t end_padding_in_bytes = 0;
  DML_TENSOR_DATA_TYPE dml_data_type = DML_TENSOR_DATA_TYPE_UNKNOWN;

  if (!Is64BitIntegerType(data_type)) {
    dml_data_type = GetDmlDataTypeFromTfDataType(data_type);
  } else {
    // DirectML doesn't support tensor of int64 because Direct3D doesn't support
    // the data type. A workaround is to use strides to fake 64-bit memory
    // access while only the lower 32 bits contains the data. This trick
    // obviously doesn't work if the data element is genuine 64-bit.
    dml_data_type = DML_TENSOR_DATA_TYPE_UINT32;

    // Double the stride values to emulate 64-bit integer support.
    for (uint32_t& stride : dml_strides) {
      stride *= 2;
    }

    // The physical size of the tensor will have an extra 4 bytes at the end.
    // DMLCalcBufferTensorSize calculates the minimum implied size, which is
    // based on the last addressable element of the tensor plus the space for
    // the last element. However, the size of the last element is now halved
    // from 8 bytes to 4 bytes.
    //
    // Example:
    // Original Tensor: size={2,3}, strides={3,1}, type=int64, size =
    // (1+{1,2}*{3,1})*sizeof(int64) = 6 * 8 = 48 Emulated Tensor: size={2,3},
    // strides={6,2}, type=int32, size = (1+{1,2}*{6,2})*sizeof(int32) = 11 * 4
    // = 44
    //
    // DirectML itself won't read/write the last 4 bytes, but we want the total
    // size to be accurate so that the entire region can be zeroed.
    end_padding_in_bytes = sizeof(uint32_t);
  }

  auto dml_desc = DmlTensorDesc(dml_data_type, dml_sizes, dml_strides,
                                guaranteed_base_offset_alignment);

  // Massage the end padding, if any, before returning
  dml_desc.buffer_tensor_desc_.TotalTensorSizeInBytes += end_padding_in_bytes;

  return dml_desc;
}  // namespace tensorflow

absl::Span<const uint32_t> DmlTensorDesc::GetStrides() const {
  if (buffer_tensor_desc_.Strides == nullptr) {
    return {};
  }
  return absl::MakeSpan(strides_,
                        strides_ + buffer_tensor_desc_.DimensionCount);
}

DML_TENSOR_DESC DmlTensorDesc::GetDmlDesc() {
  if (tensor_type_ == DML_TENSOR_TYPE_INVALID) {
    return {tensor_type_, nullptr};
  }

  buffer_tensor_desc_.Sizes = sizes_;
  if (buffer_tensor_desc_.Strides) {
    buffer_tensor_desc_.Strides = strides_;
  }

  // Only buffer tensors are supported right now.
  assert(tensor_type_ == DML_TENSOR_TYPE_BUFFER);
  return {tensor_type_, &buffer_tensor_desc_};
}

void DmlTensorDesc::ForceUnsignedDataType() {
  switch (buffer_tensor_desc_.DataType) {
    case DML_TENSOR_DATA_TYPE_INT32:
      buffer_tensor_desc_.DataType = DML_TENSOR_DATA_TYPE_UINT32;
      break;

    case DML_TENSOR_DATA_TYPE_INT16:
      buffer_tensor_desc_.DataType = DML_TENSOR_DATA_TYPE_UINT16;
      break;

    case DML_TENSOR_DATA_TYPE_INT8:
      buffer_tensor_desc_.DataType = DML_TENSOR_DATA_TYPE_UINT8;
      break;

      // Nothing to do if already unsigned
    case DML_TENSOR_DATA_TYPE_UINT32:
    case DML_TENSOR_DATA_TYPE_UINT16:
    case DML_TENSOR_DATA_TYPE_UINT8:
      break;

    default:
      LOG(FATAL) << "Can't coerce unknown or non-integral data type";
  }
}
}  // namespace tensorflow