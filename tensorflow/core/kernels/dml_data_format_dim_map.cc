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

class DataFormatDimMapInitHelper : public InitializationHelper {
 public:
  struct Attributes {
    explicit Attributes(OpKernelConstruction* ctx) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("src_format", &src_format));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("dst_format", &dst_format));
      OP_REQUIRES(ctx, src_format.size() == 4,
                  errors::InvalidArgument(strings::StrCat(
                      "Source format must of length 4, received src_format = ",
                      src_format)));
      OP_REQUIRES(
          ctx, dst_format.size() == 4,
          errors::InvalidArgument(strings::StrCat(
              "Destination format must of length 4, received dst_format = ",
              dst_format)));
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

    for (uint8_t i = 0; i < src_format.size(); ++i) {
      for (uint8_t j = 0; j < dst_format.size(); ++j) {
        if (dst_format[j] == src_format[i]) {
          src_dst_mapping_packed |= j << left_shift;
          left_shift += 8;
          break;
        }
      }
    }

    auto scope = dml::Graph(ctx->GetDmlDevice());

    DmlKernelTensors tensors = GetTensorInfos(ctx, {});
    auto inputs = GetDmlTensorDescs(tensors.inputs);
    auto indices = dml::InputTensor(scope, 0, inputs[0]);

    const bool is_64_bit_integer = Is64BitIntegerType(ctx->GetInputDataType(0));

    // Reinterpreting to int32 is necessary for Gather to recognize the negative
    // indices, since int64 is simply a stride hack with the uint32 datatype.
    // Ones' complement will make sure that the sign bit is preserved since we
    // don't need the highest 32 bits of int64 to represent values in the range
    // [-4, 3].
    if (is_64_bit_integer) {
      indices = dml::Reinterpret(indices, DML_TENSOR_DATA_TYPE_INT32,
                                 {1, 1, 1, 4}, indices.GetOutputDesc().strides);
    }

    DML_SCALAR_UNION bits_scalar;
    bits_scalar.UInt32 = src_dst_mapping_packed;

    auto params = dml::FillValueConstant(
        scope, {1, 1, 1, 1}, DML_TENSOR_DATA_TYPE_UINT32, bits_scalar);

    params =
        dml::Reinterpret(params, DML_TENSOR_DATA_TYPE_UINT8, {1, 1, 1, 4}, {});

    constexpr uint32_t gather_axis = 3;
    constexpr uint32_t index_dimensions = 1;

    // We need strides of 4 for int32 and strides of 8 for int64 since the
    // params are uint8
    dml::TensorPolicy out_policy = dml::TensorPolicy::Default();
    if (is_64_bit_integer) {
      out_policy = GetEmulatedInt64TensorPolicy();
    }

    scope.SetTensorPolicy(out_policy);
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
      DmlKernelWrapper<DmlDataFormaDimMapKernel,                          \
                       GetOutputShapeAsInputShapeHelper>);

TF_CALL_int32(REGISTER_KERNEL);
TF_CALL_int64(REGISTER_KERNEL);
#undef REGISTER_KERNEL

}  // namespace tensorflow