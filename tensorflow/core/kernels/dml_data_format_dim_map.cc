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
#include "tensorflow/core/kernels/dml_kernel_wrapper.h"
#include "tensorflow/core/kernels/dml_ops_common.h"

namespace tensorflow {

// Ensure that `src` and `dst` define a valid permutation.
// Ops defined in this file assume that user specifies a permutation via two
// string attributes. This check validates that these attributes properly define
// it to prevent security vulnerabilities.
static bool IsValidPermutation(const std::string& src, const std::string& dst) {
  if (src.size() != dst.size()) {
    return false;
  }

  std::map<char, bool> characters;

  // Every character in `src` must be present only once
  for (const auto c : src) {
    if (characters[c]) {
      return false;
    }
    characters[c] = true;
  }

  // Every character in `dst` must show up in `src` exactly once
  for (const auto c : dst) {
    if (!characters[c]) {
      return false;
    }
    characters[c] = false;
  }

  // At this point, characters[] has been switched to true and false exactly
  // once for all character in `src` (and `dst`) so we have a valid permutation
  return true;
}

class DataFormatDimMapInitHelper : public InitializationHelper {
 public:
  struct Attributes {
    explicit Attributes(OpKernelConstruction* ctx) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("src_format", &src_format));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("dst_format", &dst_format));
      OP_REQUIRES(ctx, src_format.size() == 4 || src_format.size() == 5,
                  errors::InvalidArgument(strings::StrCat(
                      "Source format must be of length 4 or 5, received "
                      "src_format = ",
                      src_format)));
      OP_REQUIRES(ctx, dst_format.size() == 4 || dst_format.size() == 5,
                  errors::InvalidArgument(
                      strings::StrCat("Destination format must be of length "
                                      "4 or 5, received dst_format = ",
                                      dst_format)));
      OP_REQUIRES(ctx, IsValidPermutation(src_format, dst_format),
                  errors::InvalidArgument("Destination and source format must "
                                          "determine a permutation, got ",
                                          src_format, " and ", dst_format));
    }

    std::string src_format;
    std::string dst_format;
  };

  DataFormatDimMapInitHelper(OpKernelContext* ctx,
                             std::shared_ptr<const Attributes> attr)
      : attr_(attr) {}

  absl::string_view GetSrcFormat() const { return attr_->src_format; }
  absl::string_view GetDstFormat() const { return attr_->dst_format; }

 private:
  const std::shared_ptr<const Attributes> attr_;
};

template <typename T>
class DmlDataFormaDimMapKernel : public DmlKernel {
 public:
  using InitHelper = DataFormatDimMapInitHelper;

  explicit DmlDataFormaDimMapKernel(DmlKernelConstruction* ctx,
                                    const InitHelper* init_helper) {
    auto src_format = init_helper->GetSrcFormat();
    auto dst_format = init_helper->GetDstFormat();

    // Put all the indices into a single uint32 scalar that we use to fill the
    // buffer. Since the indices are forced to be within the [0, 3] range and it
    // has been validated earlier, we can represent them as 4 uint8 values. We
    // can then reinterpret them as a tensor of 4 uint8 values before doing the
    // gather operation.
    uint32_t src_dst_mapping_packed = 0;
    uint32_t left_shift = 0;
    uint8_t additional_src_dst_mapping = 0;

    for (uint8_t i = 0; i < src_format.size(); ++i) {
      for (uint8_t j = 0; j < dst_format.size(); ++j) {
        if (dst_format[j] == src_format[i]) {
          if (left_shift == 32) {
            // If we are dealing with a 5D format, we need an additional uint8_t
            // to store the last mapping
            additional_src_dst_mapping = j;
          } else {
            src_dst_mapping_packed |= j << left_shift;
            left_shift += 8;
          }
          break;
        }
      }
    }

    TensorShape collapsed_shape({ctx->GetInputTensorShape(0).num_elements()});

    DmlTensorInfo in_out;
    in_out.kernel_index = 0;
    in_out.desc = DmlTensorDesc::Create(ctx->GetInputDataType(0),
                                        collapsed_shape, collapsed_shape);

    DmlKernelTensors tensors;
    tensors.inputs = {in_out};
    tensors.outputs = {in_out};

    auto scope = dml::Graph(ctx->GetDmlDevice());
    auto inputs = GetDmlTensorDescs(tensors.inputs);
    auto indices = dml::InputTensor(scope, 0, inputs[0]);
    auto params = dml::ScalarTensor<uint32_t>(scope, src_dst_mapping_packed,
                                              {1, 1, 1, 1});

    params =
        dml::Reinterpret(params, DML_TENSOR_DATA_TYPE_UINT8, {1, 1, 1, 4}, {});

    constexpr uint32_t gather_axis = 3;

    if (src_format.size() == 5) {
      auto additional_params = dml::ScalarTensor<uint8_t>(
          scope, additional_src_dst_mapping, {1, 1, 1, 1});

      params = dml::Join({params, additional_params}, gather_axis);
    }

    // We need strides of 4 for int32 and strides of 8 for int64 since the
    // params are uint8
    // TFDML #24881131
    const uint32_t element_stride =
        Is64BitIntegerType(ctx->GetOutputDataType(0)) ? 8 : 4;

    const auto out_policy = dml::TensorPolicy(
        [element_stride](DML_TENSOR_DATA_TYPE dataType, DML_TENSOR_FLAGS flags,
                         dml::Span<const uint32_t> sizes) {
          uint32_t dimension_count = static_cast<uint32_t>(sizes.size());

          const uint32_t num_elements = std::accumulate(
              sizes.begin(), sizes.end(), 1u, std::multiplies<uint32_t>());

          dml::TensorDimensions strides(dimension_count);
          strides.back() = element_stride;

          dml::TensorProperties props = {};
          props.guaranteedBaseOffsetAlignment = 0;
          props.strides = std::move(strides);
          props.totalTensorSizeInBytes = DMLCalcBufferTensorSize(
              dataType, dimension_count, sizes.data(), props.strides->data());
          return props;
        });

    scope.SetTensorPolicy(out_policy);

    constexpr uint32_t index_dimensions = 1;
    auto result = dml::Gather(params, indices, gather_axis, index_dimensions);

    Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
        scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

    Initialize(ctx, std::move(tensors), compiled_op.Get());
  }

  StatusOr<DmlGpuEvent> Compute(DmlKernelContext* ctx) const override {
    // Since we gather uint8 values with strides of 4 or 8, we always need to
    // zero the buffer
    Tensor* output = ctx->GetOutputTensor(0);
    ctx->ZeroBuffer(ctx->CreateBufferForTensor(*output));
    return DmlKernel::Compute(ctx);
  }
};

#define REGISTER_KERNEL(T)                                                \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("DataFormatDimMap").Device(DEVICE_DML).TypeConstraint<T>("T"), \
      DmlKernelWrapper<DmlDataFormaDimMapKernel<T>,                       \
                       GetOutputShapeAsInputShapeHelper>);

TF_CALL_int32(REGISTER_KERNEL);
TF_CALL_int64(REGISTER_KERNEL);
#undef REGISTER_KERNEL

}  // namespace tensorflow