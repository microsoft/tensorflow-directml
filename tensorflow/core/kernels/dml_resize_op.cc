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
#include "tensorflow/core/kernels/image_resizer_state.h"

namespace tensorflow {

class ResizeInitHelper : public InitializationHelper {
 public:
  struct Attributes {
    explicit Attributes(OpKernelConstruction* ctx) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("align_corners", &align_corners));
      OP_REQUIRES_OK(ctx,
                     ctx->GetAttr("half_pixel_centers", &half_pixel_centers));
    }

    bool align_corners;
    bool half_pixel_centers;
  };

  ResizeInitHelper(OpKernelContext* ctx, std::shared_ptr<const Attributes> attr)
      : attr_(attr) {
    image_resizer_state_.emplace(
        ImageResizerState(attr->align_corners, attr->half_pixel_centers));

    image_resizer_state_->ValidateAndCalculateOutputSize(ctx, ctx->input(0));
  }

  bool AlignCorners() const { return attr_->align_corners; }
  bool HalfPixelCenters() const { return attr_->half_pixel_centers; }

  const ImageResizerState& GetImageResizerState() const {
    DCHECK(image_resizer_state_.has_value());
    return *image_resizer_state_;
  }

 private:
  const std::shared_ptr<const Attributes> attr_;
  absl::optional<ImageResizerState> image_resizer_state_;

};  // namespace tensorflow

using InitHelper = ResizeInitHelper;

class GetResizeShapeHelper : public ShapeHelper {
 public:
  std::vector<TensorShape> GetOutputShapes(
      OpKernelContext* ctx,
      const InitializationHelper* initialization_helper) const {
    const Tensor& input = ctx->input(0);

    auto init_helper = static_cast<const InitHelper*>(initialization_helper);

    const ImageResizerState& image_resizer_state =
        init_helper->GetImageResizerState();

    // TF's Resize tensors are always in NHWC format
    TensorShape output_shape(
        {image_resizer_state.batch_size, image_resizer_state.out_height,
         image_resizer_state.out_width, image_resizer_state.channels});

    return {std::move(output_shape)};
  }
};

static bool CastInputToFloat(DML_INTERPOLATION_MODE mode, DataType data_type,
                             bool is_identity) {
  switch (mode) {
    case DML_INTERPOLATION_MODE_LINEAR:
      return data_type != DT_FLOAT;
    case DML_INTERPOLATION_MODE_NEAREST_NEIGHBOR:
      return !is_identity && data_type != DT_FLOAT && data_type != DT_HALF;
    default:
      return false;
  }
}

template <DML_INTERPOLATION_MODE interpolation_mode>
class DmlResizeKernel : public DmlKernel {
 public:
  using InitHelper = tensorflow::InitHelper;

  explicit DmlResizeKernel(DmlKernelConstruction* ctx,
                           const InitHelper* init_helper) {
    CHECK(ctx->GetInputCount() == 2);
    CHECK(ctx->GetOutputCount() == 1);

    TensorShape input_tensor_shape = ctx->GetInputTensorShape(0);
    TensorShape output_tensor_shape = ctx->GetOutputTensorShape(0);

    DmlTensorInfo input;
    input.kernel_index = 0;
    input.desc = DmlTensorDesc::Create(ctx->GetInputDataType(0),
                                       input_tensor_shape, input_tensor_shape);

    DmlTensorInfo output;
    output.kernel_index = 0;
    output.desc = DmlTensorDesc::Create(
        ctx->GetOutputDataType(0), output_tensor_shape, output_tensor_shape);

    DmlKernelTensors tensors;
    tensors.inputs = {input};
    tensors.outputs = {output};

    auto inputs = GetDmlTensorDescs(tensors.inputs);
    auto scope = dml::Graph(ctx->GetDmlDevice());
    auto result = dml::InputTensor(scope, 0, inputs[0]);

    DataType tf_input_data_type = ctx->GetInputDataType(0);
    DML_TENSOR_DATA_TYPE dml_input_data_type = result.GetOutputDesc().dataType;

    bool is_identity = input_tensor_shape == output_tensor_shape;

    // ResizeBilinear only supports DT_FLOAT as an output type, and DML doesn't
    // support mixed data types for Resample. Therefore, we need to cast all
    // non-float input types to DML_TENSOR_DATA_TYPE_FLOAT32.
    bool cast_to_float =
        CastInputToFloat(interpolation_mode, tf_input_data_type, is_identity);

    if (cast_to_float) {
      result = dml::Cast(result, DML_TENSOR_DATA_TYPE_FLOAT32);
    } else if (is_identity) {
      result = dml::Identity(result);
    }

    // If the input and output shapes are identical, we don't need to do
    // anything here
    if (!is_identity) {
      bool align_corners = init_helper->AlignCorners();
      bool half_pixel_centers = init_helper->HalfPixelCenters();
      float input_offset = 0.5f;
      float output_offset = -0.5f;

      if (interpolation_mode == DML_INTERPOLATION_MODE_LINEAR) {
        input_offset = half_pixel_centers ? 0.5f : 0.0f;
        output_offset = half_pixel_centers ? -0.5f : 0.0f;
      } else {
        input_offset = align_corners ? 0.0f : 0.5f;
        output_offset = half_pixel_centers ? -0.5f : 0.0f;
      }

      const ImageResizerState& image_resizer_state =
          init_helper->GetImageResizerState();

      float height_scale = 1.0f / image_resizer_state.height_scale;
      float width_scale = 1.0f / image_resizer_state.width_scale;
      float scales[] = {1.0f, height_scale, width_scale, 1.0f};

      float input_pixel_offsets[] = {0.5f, input_offset, input_offset, 0.5f};
      float output_pixel_offsets[] = {-0.5f, output_offset, output_offset,
                                      -0.5f};

      auto output_sizes = NarrowTensorShape(output_tensor_shape);

      result = dml::Resample(
          result,
          dml::TensorDesc::Dimensions(output_sizes.begin(), output_sizes.end()),
          interpolation_mode, scales, input_pixel_offsets,
          output_pixel_offsets);

      if (interpolation_mode == DML_INTERPOLATION_MODE_NEAREST_NEIGHBOR &&
          dml_input_data_type != result.GetOutputDesc().dataType) {
        result = dml::Cast(result, dml_input_data_type);
      }
    }

    Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
        scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

    Initialize(ctx, std::move(tensors), compiled_op.Get());
  }
};

#define DML_REGISTER_KERNEL(type)                                      \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("ResizeBilinear")                                           \
          .Device(DEVICE_DML)                                          \
          .TypeConstraint<type>("T")                                   \
          .HostMemory("size"),                                         \
      DmlKernelWrapper<DmlResizeKernel<DML_INTERPOLATION_MODE_LINEAR>, \
                       GetResizeShapeHelper>);

// Although DML's resample only supports floats and halfs, models sometimes use
// other types as well for the input type
TF_CALL_DML_FLOAT_TYPES(DML_REGISTER_KERNEL);
TF_CALL_uint8(DML_REGISTER_KERNEL);
TF_CALL_int8(DML_REGISTER_KERNEL);
TF_CALL_uint16(DML_REGISTER_KERNEL);
TF_CALL_int16(DML_REGISTER_KERNEL);
TF_CALL_int32(DML_REGISTER_KERNEL);
TF_CALL_int64(DML_REGISTER_KERNEL);
#undef DML_REGISTER_KERNEL

#define DML_REGISTER_KERNEL(type)                                   \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("ResizeNearestNeighbor")                                 \
          .Device(DEVICE_DML)                                       \
          .TypeConstraint<type>("T")                                \
          .HostMemory("size"),                                      \
      DmlKernelWrapper<                                             \
          DmlResizeKernel<DML_INTERPOLATION_MODE_NEAREST_NEIGHBOR>, \
          GetResizeShapeHelper>);

TF_CALL_DML_FLOAT_TYPES(DML_REGISTER_KERNEL);
TF_CALL_uint8(DML_REGISTER_KERNEL);
TF_CALL_int8(DML_REGISTER_KERNEL);
TF_CALL_uint16(DML_REGISTER_KERNEL);
TF_CALL_int16(DML_REGISTER_KERNEL);
TF_CALL_int32(DML_REGISTER_KERNEL);
#undef DML_REGISTER_KERNEL

}  // namespace tensorflow