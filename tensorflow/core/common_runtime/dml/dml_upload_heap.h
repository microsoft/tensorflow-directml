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
#include "dml_execution_context.h"
#include "dml_pooled_heap.h"

namespace tensorflow {

class DmlExecutionContext;

// Implements a non-blocking, ring-buffer style upload heap for copying CPU data
// to GPU resources. This class is thread-safe.
class DmlUploadHeap : public DmlPooledHeap {
 public:
  DmlUploadHeap(ID3D12Device* device, DmlExecutionContext* execution_context);

  // Makes a copy of the source data and begins copying it into the destination
  // resource, and returns a DmlGpuEvent which will become signaled when the
  // copy is complete. The destination resource must be a default or readback
  // buffer.
  StatusOr<DmlGpuEvent> BeginUploadToGpu(const D3D12BufferRegion& dst,
                                         absl::Span<const uint8_t> src);

 private:
  std::mutex mutex_;
  DmlExecutionContext* execution_context_;  // weak; owned by DmlDeviceState
};

}  // namespace tensorflow
