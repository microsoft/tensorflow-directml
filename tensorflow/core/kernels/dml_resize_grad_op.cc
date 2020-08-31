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

template <DML_INTERPOLATION_MODE interpolation_mode>
class ResizeGradInitHelper : public InitializationHelper {
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

  ResizeGradInitHelper(OpKernelContext* ctx,
                       std::shared_ptr<const Attributes> attr)
      : attr_(attr) {
    if (interpolation_mode == DML_INTERPOLATION_MODE_LINEAR) {
      const Tensor& input = ctx->input(0);
      const Tensor& original_image = ctx->input(1);

      ImageResizerGradientState st(attr->align_corners,
                                   attr->half_pixel_centers);
      st.ValidateAndCalculateScales(ctx, input, original_image);

      if (!ctx->status().ok()) {
        return;
      }

      batch_size_ = st.batch_size;
      channels_ = st.channels;
      out_height_ = st.original_height;
      out_width_ = st.original_width;
      height_scale_ = st.height_scale;
      width_scale_ = st.width_scale;
    } else {
      DCHECK(interpolation_mode == DML_INTERPOLATION_MODE_NEAREST_NEIGHBOR);

      // Grab and validate the input:
      const Tensor& input = ctx->input(0);
      OP_REQUIRES(ctx, input.dims() == 4,
                  errors::InvalidArgument("input must be 4-dimensional",
                                          input.shape().DebugString()));

      // Grab and validate the output shape:
      const Tensor& shape_t = ctx->input(1);
      OP_REQUIRES(ctx, shape_t.dims() == 1,
                  errors::InvalidArgument("shape_t must be 1-dimensional",
                                          shape_t.shape().DebugString()));
      OP_REQUIRES(ctx, shape_t.NumElements() == 2,
                  errors::InvalidArgument("shape_t must have two elements",
                                          shape_t.shape().DebugString()));

      auto sizes = shape_t.vec<int32>();
      OP_REQUIRES(
          ctx, sizes(0) > 0 && sizes(1) > 0,
          errors::InvalidArgument("shape_t's elements must be positive"));

      batch_size_ = input.dim_size(0);
      const int64 in_height = input.dim_size(1);
      const int64 in_width = input.dim_size(2);
      channels_ = input.dim_size(3);

      out_height_ = sizes(0);
      out_width_ = sizes(1);
      height_scale_ =
          CalculateResizeScale(out_height_, in_height, attr->align_corners);
      width_scale_ =
          CalculateResizeScale(out_width_, in_width, attr->align_corners);
    }
  }

  bool AlignCorners() const { return attr_->align_corners; }
  bool HalfPixelCenters() const { return attr_->half_pixel_centers; }
  int64 GetBatchSize() const { return batch_size_; }
  int64 GetChannels() const { return channels_; }
  int64 GetOutputHeight() const { return out_height_; }
  int64 GetOutputWidth() const { return out_width_; }
  float GetHeightScale() const { return height_scale_; }
  float GetWidthScale() const { return width_scale_; }

 private:
  const std::shared_ptr<const Attributes> attr_;
  int64 batch_size_;
  int64 channels_;
  int64 out_height_;
  int64 out_width_;
  float height_scale_;
  float width_scale_;
};

template <DML_INTERPOLATION_MODE interpolation_mode>
using InitHelper = ResizeGradInitHelper<interpolation_mode>;

template <DML_INTERPOLATION_MODE interpolation_mode>
class ResizeGradShapeHelper : public ShapeHelper {
 public:
  std::vector<TensorShape> GetOutputShapes(
      OpKernelContext* ctx,
      const InitializationHelper* initialization_helper) const {
    const Tensor& input = ctx->input(0);

    auto init_helper = static_cast<const InitHelper<interpolation_mode>*>(
        initialization_helper);

    // TF's Resize tensors are always in NHWC format
    TensorShape output_shape(
        {init_helper->GetBatchSize(), init_helper->GetOutputHeight(),
         init_helper->GetOutputWidth(), init_helper->GetChannels()});

    return {std::move(output_shape)};
  }
};

static bool CastInputToFloat(DML_INTERPOLATION_MODE mode, DataType data_type) {
  switch (mode) {
    case DML_INTERPOLATION_MODE_LINEAR:
      return data_type != DT_FLOAT;
    case DML_INTERPOLATION_MODE_NEAREST_NEIGHBOR:
      return data_type != DT_FLOAT && data_type != DT_HALF;
    default:
      return false;
  }
}

template <DML_INTERPOLATION_MODE interpolation_mode>
class DmlResizeGradKernel : public DmlKernel {
 public:
  using InitHelper = tensorflow::InitHelper<interpolation_mode>;

  explicit DmlResizeGradKernel(DmlKernelConstruction* ctx,
                               const InitHelper* init_helper) {
    DmlKernelParams params;
    params.kernel_input_indices = {0};
    DmlKernelTensors tensors = GetTensorInfos(ctx, params);

    auto inputs = GetDmlTensorDescs(tensors.inputs);
    auto scope = dml::Scope(ctx->GetDmlDevice());
    auto result = dml::InputTensor(scope, 0, inputs[0]);

    if (CastInputToFloat(interpolation_mode, ctx->GetInputDataType(0))) {
      result = dml::Cast(result, DML_TENSOR_DATA_TYPE_FLOAT32);
    }

    bool align_corners = init_helper->AlignCorners();
    bool half_pixel_centers = init_helper->HalfPixelCenters();
    float input_offset = -0.5f;
    float output_offset = 0.5f;

    if (interpolation_mode == DML_INTERPOLATION_MODE_LINEAR) {
      input_offset = half_pixel_centers ? -0.5f : 0.0f;
      output_offset = half_pixel_centers ? 0.5f : 0.0f;
    } else {
      input_offset = half_pixel_centers ? -0.5f : 0.0f;
      output_offset = align_corners ? 0.0f : 0.5f;
    }

    float height_scale = init_helper->GetHeightScale();
    float width_scale = init_helper->GetWidthScale();

    // A scale of 0 is essentially the same as a scale of 1 / input_size. It
    // means that the input size in the forward pass was 1, and align_corners ==
    // true. It also means that we have a single pixel that contributed to all
    // the output pixels in the forward pass. The CPU algorithm handles this by
    // clamping the scale, but DML's validation doesn't allow scales of 0.
    // Therefore, if we have a scale of 0, we simply use a scale of 1 /
    // input_size instead.
    const TensorShape& input_shape = ctx->GetInputTensorShape(0);

    if (height_scale == 0.0f) {
      height_scale = 1.0f / input_shape.dim_size(1);
    }

    if (width_scale == 0.0f) {
      width_scale = 1.0f / input_shape.dim_size(2);
    }

    float scales[] = {1.0f, height_scale, width_scale, 1.0f};

    float input_pixel_offsets[] = {-0.5, input_offset, input_offset, -0.5};
    float output_pixel_offsets[] = {0.5, output_offset, output_offset, 0.5};

    const TensorShape& output_shape = ctx->GetOutputTensorShape(0);
    auto output_sizes = NarrowTensorShape(output_shape);

    result = dml::ResampleGrad(
        result,
        dml::TensorDesc::Dimensions(output_sizes.begin(), output_sizes.end()),
        interpolation_mode, scales, input_pixel_offsets, output_pixel_offsets);

    DataType tf_output_data_type = ctx->GetOutputDataType(0);
    auto dml_out_data_type = GetDmlDataTypeFromTfDataType(tf_output_data_type);

    if (result.GetOutputDesc().dataType != dml_out_data_type) {
      result = dml::Cast(result, dml_out_data_type);
    }

    Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
        scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

    Initialize(ctx, std::move(tensors), compiled_op.Get());
  }
};

#define DML_REGISTER_KERNEL(type)                                              \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("ResizeBilinearGrad").Device(DEVICE_DML).TypeConstraint<type>("T"), \
      DmlKernelWrapper<DmlResizeGradKernel<DML_INTERPOLATION_MODE_LINEAR>,     \
                       ResizeGradShapeHelper<DML_INTERPOLATION_MODE_LINEAR>>);

TF_CALL_float(DML_REGISTER_KERNEL);
TF_CALL_half(DML_REGISTER_KERNEL);
#undef DML_REGISTER_KERNEL

#define DML_REGISTER_KERNEL(type)                                       \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("ResizeNearestNeighborGrad")                                 \
          .Device(DEVICE_DML)                                           \
          .TypeConstraint<type>("T")                                    \
          .HostMemory("size"),                                          \
      DmlKernelWrapper<                                                 \
          DmlResizeGradKernel<DML_INTERPOLATION_MODE_NEAREST_NEIGHBOR>, \
          ResizeGradShapeHelper<DML_INTERPOLATION_MODE_NEAREST_NEIGHBOR>>);

TF_CALL_uint8(DML_REGISTER_KERNEL);
TF_CALL_int8(DML_REGISTER_KERNEL);
TF_CALL_int32(DML_REGISTER_KERNEL);
TF_CALL_half(DML_REGISTER_KERNEL);
TF_CALL_float(DML_REGISTER_KERNEL);
#undef DML_REGISTER_KERNEL

}  // namespace tensorflow