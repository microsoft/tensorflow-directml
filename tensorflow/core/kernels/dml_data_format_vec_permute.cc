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

  std::array<bool, 256> characters{};

  // Every character in `src` must be present only once
  for (const auto c : src) {
    const uint8_t char_index = static_cast<uint8_t>(c);
    if (characters[char_index]) {
      return false;
    }
    characters[char_index] = true;
  }

  // Every character in `dst` must show up in `src` exactly once
  for (const auto c : dst) {
    const uint8_t char_index = static_cast<uint8_t>(c);
    if (!characters[char_index]) {
      return false;
    }
    characters[char_index] = false;
  }

  // At this point, characters[] has been switched to true and false exactly
  // once for all character in `src` (and `dst`) so we have a valid permutation
  return true;
}

class DmlDataFormatVecPermuteKernel : public OpKernel {
 public:
  explicit DmlDataFormatVecPermuteKernel(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    std::string src_format;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("src_format", &src_format));
    OP_REQUIRES(ctx, src_format.size() == 4 || src_format.size() == 5,
                errors::InvalidArgument(
                    "Source format must be of length 4 or 5, received "
                    "src_format = ",
                    src_format));
    std::string dst_format;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dst_format", &dst_format));
    OP_REQUIRES(ctx, dst_format.size() == 4 || dst_format.size() == 5,
                errors::InvalidArgument("Destination format must be of length "
                                        "4 or 5, received dst_format = ",
                                        dst_format));
    OP_REQUIRES(
        ctx, IsValidPermutation(src_format, dst_format),
        errors::InvalidArgument(
            "Destination and source format must determine a permutation, got ",
            src_format, " and ", dst_format));

    src_format_ = src_format;
    dst_format_ = dst_format;
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);
    const TensorShape& input_shape = input.shape();

    OP_REQUIRES(ctx, input_shape.dims() == 1 || input_shape.dims() == 2,
                errors::InvalidArgument(
                    "input must be a vector or 2D tensor, but got shape ",
                    input_shape.DebugString()));

    const int full_dim_count = src_format_.size();
    const int spatial_dim_count = full_dim_count - 2;

    if (input_shape.dims() == 1) {
      OP_REQUIRES(ctx,
                  input.NumElements() == spatial_dim_count ||
                      input.NumElements() == full_dim_count,
                  errors::InvalidArgument("1D input must be of size ",
                                          spatial_dim_count, " or ",
                                          full_dim_count, ", but got shape ",
                                          input.shape().DebugString()));
    } else if (input_shape.dims() == 2) {
      OP_REQUIRES(ctx,
                  input.dim_size(0) == spatial_dim_count ||
                      input.dim_size(0) == full_dim_count,
                  errors::InvalidArgument("First dimension of 2D input must be "
                                          "of size ",
                                          spatial_dim_count, " or ",
                                          full_dim_count, ", but got shape ",
                                          input.shape().DebugString()));
      OP_REQUIRES(
          ctx, input_shape.dim_size(1) == 2,
          errors::InvalidArgument(
              "Second dimension of 2D input must be of size 2, but got shape ",
              input_shape.DebugString()));
    }
    std::string src_format = src_format_;
    std::string dst_format = dst_format_;

    if (input.dim_size(0) == spatial_dim_count) {
      // If the input is a vector of size 2, treat the two elements as spatial
      // dimensions.
      auto keep_only_spatial_dimensions =
          [spatial_dim_count](std::string* format_str) -> void {
        auto new_end =
            std::remove_if(format_str->begin(), format_str->end(),
                           [spatial_dim_count](const char dim) {
                             return dim != 'H' && dim != 'W' &&
                                    (spatial_dim_count == 2 || dim != 'D');
                           });
        format_str->erase(new_end, format_str->end());
      };
      keep_only_spatial_dimensions(&src_format);
      keep_only_spatial_dimensions(&dst_format);

      if (spatial_dim_count == 3) {
        OP_REQUIRES(
            ctx, src_format.size() == 3 && dst_format.size() == 3,
            errors::InvalidArgument(
                "Format specifier must contain D, H and W for 2D case"));
      } else {
        DCHECK(spatial_dim_count == 2);
        OP_REQUIRES(ctx, src_format.size() == 2 && dst_format.size() == 2,
                    errors::InvalidArgument(
                        "Format specifier must contain H and W for 2D case"));
      }
    }

    absl::InlinedVector<uint32_t, 5> permutations;

    for (size_t dst_index = 0; dst_index < dst_format.length(); ++dst_index) {
      for (size_t src_index = 0; src_index < src_format.length(); ++src_index) {
        if (dst_format[dst_index] == src_format[src_index]) {
          permutations.push_back(src_index);
          break;
        }
      }
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input_shape, &output));

    auto device_context =
        static_cast<DMLDeviceContext*>(ctx->op_device_context());

    D3D12BufferRegion input_buffer = device_context->GetBufferForTensor(input);

    D3D12BufferRegion output_buffer =
        device_context->GetBufferForTensor(*output);

    const int perm_stride = DataTypeSize(input.dtype()) * input_shape.dims();

    for (uint32_t i = 0; i < permutations.size(); ++i) {
      device_context->CopyBufferToBuffer(
          output_buffer.Subregion(i * perm_stride),
          input_buffer.Subregion(permutations[i] * perm_stride, perm_stride));
    }
  }

 private:
  std::string src_format_;
  std::string dst_format_;
};

#define REGISTER_KERNEL(type)                             \
  REGISTER_KERNEL_BUILDER(Name("DataFormatVecPermute")    \
                              .Device(DEVICE_DML)         \
                              .TypeConstraint<type>("T"), \
                          DmlDataFormatVecPermuteKernel);
TF_CALL_int32(REGISTER_KERNEL);
TF_CALL_int64(REGISTER_KERNEL);
#undef REGISTER_KERNEL

}  // namespace tensorflow