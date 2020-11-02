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

#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/common_runtime/dml/dml_operator_helper.h"
#include "tensorflow/core/common_runtime/dml/dml_util.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/kernels/dml_kernel_wrapper.h"
#include "tensorflow/core/kernels/dml_ops_common.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {

class NonzeroCoordinatesInitHelper : public InitializationHelper {
 public:
  using Attributes = EmptyAttributes;

  NonzeroCoordinatesInitHelper(OpKernelContext* ctx,
                               std::shared_ptr<const Attributes> attr) {
    int input_dims = ctx->input(0).dims();
    OP_REQUIRES(ctx, input_dims <= kNcdhwDimensionCount,
                errors::InvalidArgument(
                    "DML doesn't support more than 5D for Where, but ",
                    input_dims, " dimensions were provided."));
  }
};

class NonzeroCoordinatesShapeHelper : public ShapeHelper {
 public:
  std::vector<TensorShape> GetOutputShapes(
      OpKernelContext* ctx,
      const InitializationHelper* initialization_helper) const override {
    int input_dims = ctx->input(0).dims();

    TensorShape output_count_shape;
    for (int i = 0; i < input_dims; ++i) {
      output_count_shape.AddDim(1);
    }

    TensorShape output_coordinates_shape;
    for (int i = 0; i < input_dims - 2; ++i) {
      output_coordinates_shape.AddDim(1);
    }

    output_coordinates_shape.AddDim(ctx->input(0).NumElements());
    output_coordinates_shape.AddDim(input_dims);

    return {std::move(output_count_shape), std::move(output_coordinates_shape)};
  }
};

class DmlNonzeroCoordinatesKernel : public DmlKernel {
 public:
  using InitHelper = NonzeroCoordinatesInitHelper;

  explicit DmlNonzeroCoordinatesKernel(DmlKernelConstruction* ctx,
                                       const InitHelper* init_helper) {
    CHECK(ctx->GetInputCount() == 1);
    CHECK(ctx->GetOutputCount() == 2);

    DmlKernelTensors tensors = GetTensorInfos(ctx, DmlKernelParams{});
    auto inputs = GetDmlTensorDescs(tensors.inputs);
    auto outputs = GetDmlTensorDescs(tensors.outputs);

    DML_NONZERO_COORDINATES_OPERATOR_DESC nonzero_coordinates_desc = {};
    nonzero_coordinates_desc.InputTensor = &inputs[0];
    nonzero_coordinates_desc.OutputCountTensor = &outputs[0];
    nonzero_coordinates_desc.OutputCoordinatesTensor = &outputs[1];

    DML_OPERATOR_DESC op_desc = {DML_OPERATOR_NONZERO_COORDINATES,
                                 &nonzero_coordinates_desc};

    Initialize(ctx, std::move(tensors), op_desc);
  }
};

// Since DML_OPERATOR_NONZERO_COORDINATES has different outputs and output
// shapes than TF's Where, we register our own dml-only operator that can be
// called during initialization to get the output shape. The DML version of the
// operator can easily be cached because its output shape isn't dependent on the
// data, but TF's version cannot. Therefore, all the heavylifting is done inside
// the DmlNonzeroCoordinates op, as opposed to Where that only needs to copy the
// region containing the first N elements. This registration allows us to reuse
// the kernel wrapper helper and all the DML boilerplate inside the
// initialization helper, which is necessary to infer the output shapes.
REGISTER_OP("DmlNonzeroCoordinates")
    .Input("input: T")
    .Attr("T: {numbertype, bool} = DT_BOOL")
    .Output("nonzero_count: uint32")
    .Output("nonzero_coordinates: uint32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle input = c->input(0);
      const Tensor* input_tensor = c->input_tensor(0);
      int num_elements = input_tensor->NumElements();
      int input_dims = input_tensor->dims();

      c->set_output(0, c->Scalar());
      c->set_output(1, c->Matrix(num_elements, input_dims));
      return Status::OK();
    });

#define DML_REGISTER_KERNEL(type)                                       \
  REGISTER_KERNEL_BUILDER(Name("DmlNonzeroCoordinates")                 \
                              .Device(DEVICE_DML)                       \
                              .TypeConstraint<type>("T"),               \
                          DmlKernelWrapper<DmlNonzeroCoordinatesKernel, \
                                           NonzeroCoordinatesShapeHelper>)

TF_CALL_DML_ALL_TYPES(DML_REGISTER_KERNEL);
#undef DML_REGISTER_KERNEL

static Status ComputeNonzeroCoordinates(OpKernelContext* ctx,
                                        const NodeDef& node_def,
                                        uint32_t& num_nonzero_elements,
                                        TensorValue& nonzero_coordinates) {
  Status s;
  auto op_kernel = CreateOpKernel(DEVICE_DML, ctx->device(),
                                  ctx->get_allocator(AllocatorAttributes()),
                                  node_def, TF_GRAPH_DEF_VERSION, &s);
  TF_RETURN_IF_ERROR(s);

  absl::InlinedVector<TensorValue, 4> inputs = {
      TensorValue(const_cast<Tensor*>(&ctx->input(0)))};

  AllocatorAttributes output_attrs[] = {AllocatorAttributes(),
                                        AllocatorAttributes()};

  OpKernelContext::Params op_ctx_params;
  op_ctx_params.op_kernel = op_kernel.get();
  op_ctx_params.device = ctx->device();
  op_ctx_params.inputs = &inputs;
  op_ctx_params.output_attr_array = output_attrs;

  OpKernelContext op_ctx(&op_ctx_params, 2);
  op_kernel->Compute(&op_ctx);
  TF_RETURN_IF_ERROR(op_ctx.status());

  // Copy the nonzero count to the CPU
  Tensor num_nonzero_elements_tensor_cpu;
  AllocatorAttributes attr;
  attr.set_on_host(true);
  TF_RETURN_IF_ERROR(
      ctx, ctx->allocate_temp(DT_UINT32, TensorShape({}),
                              &num_nonzero_elements_tensor_cpu, attr));

  TensorValue num_nonzero_elements_tensor = op_ctx.release_output(0);
  nonzero_coordinates = op_ctx.release_output(1);

  Notification note;
  ctx->op_device_context()->CopyDeviceTensorToCPU(
      num_nonzero_elements_tensor.tensor, "",
      static_cast<Device*>(ctx->device()), &num_nonzero_elements_tensor_cpu,
      [&note, ctx](const Status& copy_status) {
        note.Notify();
        OP_REQUIRES_OK(ctx, copy_status);
      });

  note.WaitForNotification();
  TF_RETURN_IF_ERROR(ctx->status());

  num_nonzero_elements = num_nonzero_elements_tensor_cpu.scalar<uint32_t>()();

  return Status::OK();
}

static Status ComputeCast(OpKernelContext* ctx, const NodeDef& node_def,
                          TensorValue& nonzero_coordinates) {
  Status s;
  auto op_kernel = CreateOpKernel(DEVICE_DML, ctx->device(),
                                  ctx->get_allocator(AllocatorAttributes()),
                                  node_def, TF_GRAPH_DEF_VERSION, &s);
  TF_RETURN_IF_ERROR(s);

  absl::InlinedVector<TensorValue, 4> inputs = {
      TensorValue(const_cast<Tensor*>(nonzero_coordinates.tensor))};

  AllocatorAttributes output_attrs[] = {AllocatorAttributes()};

  OpKernelContext::Params op_ctx_params;
  op_ctx_params.op_kernel = op_kernel.get();
  op_ctx_params.device = ctx->device();
  op_ctx_params.inputs = &inputs;
  op_ctx_params.output_attr_array = output_attrs;

  OpKernelContext op_ctx(&op_ctx_params, 1);
  op_kernel->Compute(&op_ctx);
  TF_RETURN_IF_ERROR(op_ctx.status());

  nonzero_coordinates = op_ctx.release_output(0);

  return Status::OK();
}

class DmlWhereKernel : public OpKernel {
 public:
  explicit DmlWhereKernel(OpKernelConstruction* ctx) : OpKernel(ctx) {
    nonzero_coordinates_node_def_.set_op("DmlNonzeroCoordinates");
    nonzero_coordinates_node_def_.add_input("input");
    SetAttrValue(ctx->input_type(0),
                 &(*nonzero_coordinates_node_def_.mutable_attr())["T"]);

    cast_node_def_.set_op("Cast");
    cast_node_def_.add_input("x");
    SetAttrValue(DT_UINT32, &(*cast_node_def_.mutable_attr())["SrcT"]);
    SetAttrValue(DT_INT64, &(*cast_node_def_.mutable_attr())["DstT"]);
    SetAttrValue(false, &(*cast_node_def_.mutable_attr())["Truncate"]);
  }

  void Compute(OpKernelContext* ctx) override {
    int input_dims = ctx->input(0).dims();
    OP_REQUIRES(ctx, input_dims <= kNcdhwDimensionCount,
                errors::InvalidArgument(
                    "DML doesn't support more than 5D for Where, but ",
                    input_dims, " dimensions were provided."));

    uint32_t num_nonzero_elements;
    TensorValue nonzero_coordinates;

    OP_REQUIRES_OK(ctx, ComputeNonzeroCoordinates(
                            ctx, nonzero_coordinates_node_def_,
                            num_nonzero_elements, nonzero_coordinates));

    // Where only supports int64 as an output, but DML outputs uint32
    OP_REQUIRES_OK(ctx, ComputeCast(ctx, cast_node_def_, nonzero_coordinates));

    // Now that we know the number of nonzero elements, create the output shape
    // and allocate the output
    TensorShape output_shape({num_nonzero_elements, input_dims});
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));

    DeviceContext* device_context = ctx->op_device_context();
    Device* device = static_cast<Device*>(ctx->device());

    device_context->CopyTensorInSameDevice(
        nonzero_coordinates.tensor, device, output,
        [ctx](const Status& s) { OP_REQUIRES_OK(ctx, s); });
  }

 private:
  NodeDef nonzero_coordinates_node_def_;
  NodeDef cast_node_def_;
};

// TF's Where can never be cached because its output shape is dependent on the
// data, but this is fine since we don't compile anything here. We merely copy
// from one buffer to another.
#define REGISTER_KERNEL(type)                                     \
  REGISTER_KERNEL_BUILDER(                                        \
      Name("Where").Device(DEVICE_DML).TypeConstraint<type>("T"), \
      DmlWhereKernel);

TF_CALL_DML_ALL_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL

}  // namespace tensorflow