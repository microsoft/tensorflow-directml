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

class CropAndResizeInitHelper : public InitializationHelper {
 public:
  struct Attributes {
    explicit Attributes(OpKernelConstruction* ctx) {
      std::string method_attr;

      OP_REQUIRES_OK(ctx, ctx->GetAttr("method", &method_attr));
      OP_REQUIRES(ctx, method_attr == "bilinear" || method_attr == "nearest",
                  errors::InvalidArgument(
                      "method must be 'bilinear' or 'nearest'", method_attr));

      interpolation_mode = method_attr == "bilinear"
                               ? DML_INTERPOLATION_MODE_LINEAR
                               : DML_INTERPOLATION_MODE_NEAREST_NEIGHBOR;

      OP_REQUIRES_OK(ctx,
                     ctx->GetAttr("extrapolation_value", &extrapolation_value));
    }

    DML_INTERPOLATION_MODE interpolation_mode;
    float extrapolation_value;
  };

  CropAndResizeInitHelper(OpKernelContext* ctx,
                          std::shared_ptr<const Attributes> attr)
      : attr_(std::move(attr)) {
    const TensorShape& image_shape = ctx->input(0).shape();

    OP_REQUIRES(ctx, image_shape.dims() == 4,
                errors::InvalidArgument("input image must be 4-D",
                                        image_shape.DebugString()));

    const int image_height = image_shape.dim_size(1);
    const int image_width = image_shape.dim_size(2);

    OP_REQUIRES(ctx, image_height > 0 && image_width > 0,
                errors::InvalidArgument("image dimensions must be positive"));

    int num_boxes = 0;
    const TensorShape& boxes_shape = ctx->input(1).shape();
    const TensorShape& box_index_shape = ctx->input(2).shape();

    if (boxes_shape.num_elements() != 0 || boxes_shape.num_elements() != 0) {
      OP_REQUIRES(ctx, boxes_shape.dims() == 2,
                  errors::InvalidArgument("boxes must be 2-D",
                                          boxes_shape.DebugString()));

      num_boxes = boxes_shape.dim_size(0);

      OP_REQUIRES(ctx, boxes_shape.dim_size(1) == 4,
                  errors::InvalidArgument("boxes must have 4 columns"));

      OP_REQUIRES(ctx, box_index_shape.dims() == 1,
                  errors::InvalidArgument("box_index must be 1-D",
                                          boxes_shape.DebugString()));

      OP_REQUIRES(ctx, box_index_shape.dim_size(0) == num_boxes,
                  errors::InvalidArgument("box_index has incompatible shape"));
    }

    const Tensor& crop_size = ctx->input(3);
    OP_REQUIRES(ctx, crop_size.dims() == 1,
                errors::InvalidArgument("crop_size must be 1-D",
                                        crop_size.shape().DebugString()));

    OP_REQUIRES(ctx, crop_size.dim_size(0) == 2,
                errors::InvalidArgument("crop_size must have two elements",
                                        crop_size.shape().DebugString()));

    // Copy and validate crop sizes.
    auto crop_size_vec = crop_size.vec<int32>();
    const int crop_height = internal::SubtleMustCopy(crop_size_vec(0));
    const int crop_width = internal::SubtleMustCopy(crop_size_vec(1));
    const int depth = image_shape.dim_size(3);

    OP_REQUIRES(ctx, crop_height > 0 && crop_width > 0,
                errors::InvalidArgument("crop dimensions must be positive"));

    output_shape_ = TensorShape({num_boxes, crop_height, crop_width, depth});
  }

  TensorShape GetOutputShape() const { return output_shape_; }
  float GetExtrapolationValue() const { return attr_->extrapolation_value; }

  DML_INTERPOLATION_MODE GetInterpolationMode() const {
    return attr_->interpolation_mode;
  }

 private:
  TensorShape output_shape_;
  std::shared_ptr<const Attributes> attr_;
};

class CropAndResizeShapeHelper : public ShapeHelper {
 public:
  std::vector<TensorShape> GetOutputShapes(
      OpKernelContext* ctx,
      const InitializationHelper* initialization_helper) const override {
    auto init_helper =
        static_cast<const CropAndResizeInitHelper*>(initialization_helper);

    return {init_helper->GetOutputShape()};
  }
};

class DmlCropAndResizeKernel : public DmlKernel {
 public:
  using InitHelper = CropAndResizeInitHelper;

  explicit DmlCropAndResizeKernel(DmlKernelConstruction* ctx,
                                  const InitHelper* init_helper) {
    const TensorShape& image_shape = ctx->GetInputTensorShape(0);
    const TensorShape& output_tensor_shape = ctx->GetOutputTensorShape(0);

    DmlKernelParams params;
    params.kernel_input_indices = {0, 1, 2};

    using namespace DmlTensorAxes;
    auto layout = {N, H, W, C};

    DmlKernelTensors tensors = GetTensorInfos(ctx, params);
    tensors.inputs[0]->desc = CreateTensorDescFromInput(ctx, 0, layout);

    // RoiAlign only supports uint32 for the batch indices
    tensors.inputs[2]->desc.ForceUnsignedDataType();
    tensors.outputs[0]->desc = CreateTensorDescFromOutput(ctx, 0, layout);

    auto inputs = GetDmlTensorDescs(tensors.inputs);
    auto scope = dml::Graph(ctx->GetDmlDevice(),
                            dml::TensorPolicy::InterleavedChannel());
    auto input = dml::InputTensor(scope, 0, inputs[0]);

    // The output of CropAndResize is always DT_FLOAT, so cast the input to
    // DT_FLOAT before doing the operation
    if (ctx->GetOutputDataType(0) != ctx->GetInputDataType(0)) {
      auto dml_dtype = GetDmlDataTypeFromTfDataType(ctx->GetOutputDataType(0));
      input = dml::Cast(input, dml_dtype);
    }

    auto roi = dml::InputTensor(scope, 1, inputs[1]);
    auto batch_indices = dml::InputTensor(scope, 2, inputs[2]);

    // TF's ROIs are in {y1, x1, y2, x2} order, but DML's are in {x1, y1, x2,
    // y2} order
    auto roi_sizes = roi.GetOutputDesc().sizes;
    roi = dml::Reinterpret(roi, {1, roi_sizes[2], 2, 2}, {});

    auto seq_lengths =
        dml::ScalarTensor<uint32_t>(scope, 2, {1, roi_sizes[2], 2, 1});
    roi = dml::ReverseSubsequences(roi, seq_lengths, 3);

    // NHWC stides for sizes [1, 1, roiCount, 4]
    dml::TensorDimensions roiStrides{
        roi_sizes[2] * 4,
        roi_sizes[2] * 4,
        1,
        roi_sizes[2],
    };

    roi = dml::Reinterpret(roi, roi_sizes, roiStrides);

    float spatial_scale_x = image_shape.dim_size(2) - 1;
    float spatial_scale_y = image_shape.dim_size(1) - 1;
    const uint32_t crop_height = output_tensor_shape.dim_size(1);
    const uint32_t crop_width = output_tensor_shape.dim_size(2);
    constexpr uint32_t minimum_samples_per_output = 1;
    constexpr uint32_t maximum_samples_per_output = 1;
    constexpr float input_pixel_offset = 0;
    constexpr float output_pixel_offset = 0;
    constexpr bool align_region_to_corners = true;

    auto result =
        dml::RoiAlign(input, roi, batch_indices, DML_REDUCE_FUNCTION_AVERAGE,
                      init_helper->GetInterpolationMode(), spatial_scale_x,
                      spatial_scale_y, input_pixel_offset, output_pixel_offset,
                      init_helper->GetExtrapolationValue(),
                      minimum_samples_per_output, maximum_samples_per_output,
                      align_region_to_corners, crop_height, crop_width);

    Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
        scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

    Initialize(ctx, std::move(tensors), compiled_op.Get());
  }
};

#define DML_REGISTER_KERNEL(type)    \
  REGISTER_KERNEL_BUILDER(           \
      Name("CropAndResize")          \
          .Device(DEVICE_DML)        \
          .TypeConstraint<type>("T") \
          .HostMemory("crop_size"),  \
      DmlKernelWrapper<DmlCropAndResizeKernel, CropAndResizeShapeHelper>);

TF_CALL_half(DML_REGISTER_KERNEL);
TF_CALL_float(DML_REGISTER_KERNEL);
#undef DML_REGISTER_KERNEL

}  // namespace tensorflow