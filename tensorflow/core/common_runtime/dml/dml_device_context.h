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

#include "dml_event_queue.h"
#include "dml_execution_context.h"
#include "dml_readback_heap.h"
#include "dml_upload_heap.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/device_base.h"

namespace tensorflow {

class DMLDeviceContext : public DeviceContext {
 private:
  // These are all weak pointers; owned by the DML device factory
  DmlExecutionContext* execution_context_;
  DmlEventQueue* event_queue_;
  DmlUploadHeap* upload_heap_;
  DmlReadbackHeap* readback_heap_;

 public:
  DMLDeviceContext(DmlExecutionContext* execution_context,
                   DmlEventQueue* event_queue, DmlUploadHeap* upload_heap,
                   DmlReadbackHeap* readback_heap)
      : execution_context_(execution_context),
        event_queue_(event_queue),
        upload_heap_(upload_heap),
        readback_heap_(readback_heap) {}

  void CopyCPUTensorToDevice(const Tensor* cpu_tensor, Device* device,
                             Tensor* device_tensor, StatusCallback done,
                             bool sync_dst_compute) const override;

  void CopyTensorInSameDevice(const Tensor* input_tensor, Device* device,
                              Tensor* output_tensor,
                              StatusCallback done) const override;

  void CopyDeviceTensorToCPU(const Tensor* device_tensor, StringPiece edge_name,
                             Device* device, Tensor* cpu_tensor,
                             StatusCallback done) override;
};

}  // namespace tensorflow