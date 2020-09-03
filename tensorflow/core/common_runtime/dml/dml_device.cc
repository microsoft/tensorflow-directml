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

#include "dml_device.h"

#include "dml_adapter_impl.h"
#include "dml_bfc_allocator.h"
#include "dml_device_context.h"
#include "dml_event_queue.h"
#include "dml_kernel_manager.h"
#include "dml_readback_heap.h"
#include "dml_upload_heap.h"
#include "tensorflow/core/common_runtime/dml/dml_util.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/variant_op_registry.h"

// {D113B493-BBA2-4993-8608-D706A73B91CE}
static const GUID PIX_EVAL_CAPTURABLE_WORK_GUID = {
    0xd113b493,
    0xbba2,
    0x4993,
    {0x86, 0x08, 0xd7, 0x06, 0xa7, 0x3b, 0x91, 0xce}};

namespace tensorflow {

DmlDevice::DmlDevice(const DmlDeviceState* state, const SessionOptions& options,
                     const DeviceAttributes& attributes)
    : LocalDevice(options, attributes), state_(state) {
  // Create the DML BFC allocator
  D3D12HeapAllocator* heap_allocator = state_->heap_allocator.get();
  uint64_t memory_limit_in_bytes = attributes.memory_limit();
  const GPUOptions& gpu_options = options.config.gpu_options();

  cpu_allocator_ = cpu_allocator();

  device_context_ = new DMLDeviceContext(
      state_->execution_context.get(), state_->event_queue.get(),
      state_->upload_heap.get(), state_->readback_heap.get());
  set_dml_device_context(device_context_);
}

Status DmlDevice::Sync() {
  VLOG(2) << "DirectML device: performing GPU sync.";

  auto start_time = std::chrono::high_resolution_clock::now();

  auto status_or_event = state_->execution_context->Flush();
  TF_RETURN_IF_ERROR(status_or_event.status());
  status_or_event.ConsumeValueOrDie().WaitForSignal();
  auto end_time = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> wait_seconds = end_time - start_time;
  VLOG(2) << "DirectML device: GPU sync took " << (wait_seconds.count() * 1e3)
          << "ms.";

  // Take the opportunity to free some memory if needed
  state_->kernel_manager->ReleaseCompletedReferences();
  return Status::OK();
}

Allocator* DmlDevice::GetAllocator(AllocatorAttributes attributes) {
  return attributes.on_host() ? cpu_allocator_ : state_->dml_allocator.get();
}

Status DmlDevice::MakeTensorFromProto(const TensorProto& tensor_proto,
                                      const AllocatorAttributes alloc_attrs,
                                      Tensor* tensor) {
  Tensor parsed(tensor_proto.dtype());
  if (!parsed.FromProto(cpu_allocator_, tensor_proto)) {
    return errors::InvalidArgument("Cannot parse tensor from proto: ",
                                   tensor_proto.DebugString());
  }

  Status copy_status;

  if (parsed.dtype() == DT_VARIANT) {
    const Variant* from = parsed.flat<Variant>().data();
    Tensor copy(cpu_allocator_, DT_VARIANT, parsed.shape());
    Variant* copy_variant = copy.flat<Variant>().data();

    std::list<Notification> notifications;

    auto copier = [&](const Tensor& from, Tensor* to) {
      // Copier isn't run in a multithreaded environment, so we don't
      // have to worry about the notifications list being modified in parallel.
      notifications.emplace_back();
      Notification& n = *notifications.rbegin();
      return MaybeCopyTensorToDML(alloc_attrs, from, *to, n, copy_status);
    };

    Status s;

    for (int64 ix = 0; ix < parsed.NumElements(); ++ix) {
      s = VariantDeviceCopy(VariantDeviceCopyDirection::HOST_TO_DEVICE,
                            from[ix], &copy_variant[ix], copier);
      if (!s.ok()) {
        break;
      }
    }

    for (auto& n : notifications) {
      n.WaitForNotification();
    }

    if (!s.ok()) {
      return s;
    }

    *tensor = std::move(copy);
    return copy_status;
  } else {
    Notification n;
    Status status =
        MaybeCopyTensorToDML(alloc_attrs, parsed, *tensor, n, copy_status);

    if (!status.ok()) {
      return status;
    }

    n.WaitForNotification();

    return copy_status;
  }
}

Status DmlDevice::MaybeCopyTensorToDML(const AllocatorAttributes alloc_attrs,
                                       const Tensor& from, Tensor& to,
                                       Notification& note,
                                       Status& copy_status) {
  if (alloc_attrs.on_host()) {
    to = from;
    note.Notify();
  } else {
    // If the tensor is not initialized, we likely ran out of memory.
    if (!to.IsInitialized()) {
      return errors::ResourceExhausted("OOM when allocating tensor of shape ",
                                       from.shape().DebugString(), " and type ",
                                       DataTypeString(from.dtype()));
    }

    Tensor copy(GetAllocator(alloc_attrs), from.dtype(), from.shape());

    device_context_->CopyCPUTensorToDevice(
        &from, this, &copy,
        [&note, copy, &to, &copy_status](const Status& s) {
          to = std::move(copy);
          copy_status.Update(s);
          note.Notify();
        },
        true);
  }

  return Status::OK();
}

Status DmlDevice::FillContextMap(const Graph* graph,
                                 DeviceContextMap* device_context_map) {
  // Fill in the context map. It is OK for this map to contain
  // duplicate DeviceContexts so long as we increment the refcount.
  device_context_map->resize(graph->num_node_ids());
  for (Node* n : graph->nodes()) {
    device_context_->Ref();
    (*device_context_map)[n->id()] = device_context_;
  }

  return Status::OK();
}

void DmlDevice::DebugOnSessionRunStart() {
  if (state_->sharing_contract) {
    state_->sharing_contract->BeginCapturableWork(
        PIX_EVAL_CAPTURABLE_WORK_GUID);
  }
}

void DmlDevice::DebugOnSessionRunEnd() {
  if (state_->sharing_contract) {
    state_->sharing_contract->EndCapturableWork(PIX_EVAL_CAPTURABLE_WORK_GUID);
  }
}

}  // namespace tensorflow