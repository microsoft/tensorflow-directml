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

#include "tensorflow/core/kernels/dml_kernel_wrapper.h"

#include "tensorflow/core/common_runtime/dml/dml_operator_helper.h"
#include "tensorflow/core/common_runtime/dml/dml_util.h"

namespace tensorflow {

DmlKernelWrapperBase::DmlKernelWrapperBase(OpKernelConstruction* ctx,
                                           DmlKernelCachePolicy cache_policy)
    : OpKernel(ctx),
      cache_policy_(cache_policy),
      node_def_(std::make_shared<NodeDef>(ctx->def())) {}

void DmlKernelWrapperBase::Compute(OpKernelContext* ctx) {
  const DmlDevice* dml_device = static_cast<const DmlDevice*>(ctx->device());
  const DmlKernelManager& kernel_manager = *dml_device->GetKernelManager();

  // Compute the output shapes
  const ShapeHelper* shape_helper = GetShapeHelper();

  std::shared_ptr<DmlKernel> kernel;
  std::vector<TensorShape> output_shapes;
  const InitializationHelper* init_helper = nullptr;
  DmlKernelKey key;

  if (cache_policy_ != DmlKernelCachePolicy::Never) {
    // Construct a kernel key which uniquely identifies the kernel instance we
    // need
    key = CreateKernelKey(ctx);

    // Retrieve an appropriate DmlKernel from the cache. If the kernel hasn't
    // been cached yet, it will be null
    kernel = TryGetCachedKernel(kernel_manager, key);
  }

  // If we found a cached kernel, simply retrieve its initialization helper
  if (kernel) {
    init_helper = kernel->GetInitializationHelper();
    output_shapes = shape_helper->GetOutputShapes(ctx, init_helper);
  } else {
    auto shared_helper = CreateInitializationHelper(ctx);
    init_helper = shared_helper.get();

    if (!ctx->status().ok()) {
      return;
    }

    output_shapes = shape_helper->GetOutputShapes(ctx, init_helper);

    // Check that the number of output shapes matches the number of outputs
    OP_REQUIRES(
        ctx, ctx->num_outputs() == output_shapes.size(),
        errors::InvalidArgument(
            "The shape helper supplied an incorrect number of output shapes. ",
            ctx->num_outputs(), " were expected, but ", output_shapes.size(),
            " were provided."));

    if (shared_helper->IsNoOpKernel(ctx, output_shapes)) {
      // Don't bother constructing/executing no-op'd kernels. Instead, just
      // construct empty output tensors and return immediately.
      for (int i = 0; i < ctx->num_outputs(); ++i) {
        // Kernels which output ref types aren't allowed to be no-op'd, because
        // there's no generic way to handle these outputs. Ref outputs need to
        // be forwarded from some input, but at this point we don't know
        // anything about how to map ref inputs to ref outputs..
        CHECK(!IsRefType(ctx->expected_output_dtype(i)));

        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(
            ctx, ctx->allocate_output(i, output_shapes[i], &output_tensor));

        // If the tensor is nonempty, fill it with zero's
        if (output_tensor->NumElements() != 0) {
          D3D12BufferRegion buffer =
              dml_util::CreateBufferForTensor(dml_device, *output_tensor);

          const uint8_t fill_pattern[] = {0};
          dml_device->GetExecutionContext()->FillBufferWithPattern(
              buffer.Resource(), buffer.Offset(), buffer.SizeInBytes(),
              fill_pattern);
        }
      }
      return;
    }

    DmlKernelConstruction dml_construction(dml_device, ctx, node_def_.get(),
                                           shape_helper, output_shapes,
                                           shared_helper);

    if (cache_policy_ == DmlKernelCachePolicy::Never) {
      // This kernel has requested to never be cached; create a new one
      // directly
      kernel = CreateKernel(&dml_construction, init_helper);
    } else {
      kernel = CreateCachedKernel(&dml_construction, kernel_manager, key,
                                  init_helper);
    }

    // Check for validation done during kernel construction
    if (!ctx->status().ok()) {
      return;
    }
  }

  assert(kernel != nullptr);

  // Execute the kernel
  DmlKernelContext dml_ctx(dml_device, ctx, init_helper, output_shapes,
                           kernel->GetOutputRefsForwarding());

  // Check for errors triggered during the kernel context's constructor (e.g.
  // OOM when allocating the output buffers)
  if (!ctx->status().ok()) {
    return;
  }

  auto status_or_event = ComputeKernel(kernel.get(), &dml_ctx);
  OP_REQUIRES_OK(ctx, status_or_event.status());

  // Keep this kernel alive at least until it's completed execution on the GPU
  kernel_manager.QueueReference(kernel, status_or_event.ConsumeValueOrDie());
}

DmlKernelKey DmlKernelWrapperBase::CreateKernelKey(OpKernelContext* ctx) const {
  DmlKernelKey key = {};
  key.op_type_name = this->type_string();
  key.node_def = node_def_;

  for (int i = 0; i < ctx->num_inputs(); ++i) {
    MemoryType memory_type = ctx->input_memory_type(i);

    // Resource types cannot be hashed or copied, so they cannot form part of a
    // kernel key. Therefore, resource tensors cannot be used as constant CPU
    // inputs. This is okay because it's unlikely a kernel would ever want to
    // take a dependency on the value of a *resource handle*, rather than the
    // contents of the tensor the handle refers to.
    const bool is_resource_type =
        (BaseType(ctx->input_dtype(i)) == DT_RESOURCE);

    Tensor tensor =
        ctx->input_is_ref(i) ? ctx->mutable_input(i, false) : ctx->input(i);

    DmlInputTensorKey tensor_key = {};
    tensor_key.is_constant_cpu_input =
        (memory_type == HOST_MEMORY && !is_resource_type);

    if (tensor_key.is_constant_cpu_input) {
      tensor_key.tensor = std::move(tensor);
    } else {
      tensor_key.tensor = TensorShapeAndType{tensor.shape(), tensor.dtype()};
    }

    key.input_tensors.push_back(std::move(tensor_key));
  }

  return key;
}

}  // namespace tensorflow