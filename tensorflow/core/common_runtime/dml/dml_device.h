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

#pragma once

#include "dml_common.h"
#include "dml_device_context.h"
#include "dml_device_state.h"
#include "tensorflow/core/common_runtime/local_device.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

// Implements tensorflow::Device using a shared DmlDeviceState.
class DmlDevice : public LocalDevice {
 public:  // Methods
  DmlDevice(const DmlDeviceState* state, const SessionOptions& options,
            const DeviceAttributes& attributes);

  ID3D12Device* GetD3D12Device() const { return state_->d3d_device.Get(); }
  IDMLDevice* GetDmlDevice() const { return state_->dml_device.Get(); }

  DmlAllocator* GetAllocator() const { return state_->dml_allocator.get(); }

  DmlDescriptorAllocator* GetDescriptorAllocator() const {
    return state_->descriptor_allocator.get();
  }

  DmlKernelManager* GetKernelManager() const {
    return state_->kernel_manager.get();
  }

  DmlExecutionContext* GetExecutionContext() const {
    return state_->execution_context.get();
  }

  DmlUploadHeap* GetUploadHeap() const { return state_->upload_heap.get(); }

  DmlReadbackHeap* GetReadbackHeap() const {
    return state_->readback_heap.get();
  }

  DmlEventQueue* GetEventQueue() const {
    return state_->event_queue.get();
  }

 public:  // tensorflow::Device overrides
  Status Sync() override;
  Allocator* GetAllocator(AllocatorAttributes attributes) override;

  Status MakeTensorFromProto(const TensorProto& tensor_proto,
                             const AllocatorAttributes alloc_attrs,
                             Tensor* tensor) override;

  Status FillContextMap(const Graph* graph,
                        DeviceContextMap* device_context_map) override;

  void CopyTensorInSameDevice(const Tensor* input_tensor, Tensor* output_tensor,
                              const DeviceContext* device_context,
                              StatusCallback done) override {
    // Forward to the device context where the real implementation lives
    device_context->CopyTensorInSameDevice(input_tensor, this, output_tensor,
                                           done);
  }

  void DebugOnSessionRunStart() override;
  void DebugOnSessionRunEnd() override;

 private:
  const DmlDeviceState* state_;  // Weak; owned by the device factory

  Allocator* cpu_allocator_;          // not owned
  DMLDeviceContext* device_context_;  // ref-counted

  Status MaybeCopyTensorToDML(const AllocatorAttributes alloc_attrs,
                              const Tensor& from, Tensor& to,
                              Notification& note, Status& copy_status);
};

}  // namespace tensorflow