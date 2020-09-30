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

#include "dml_bfc_allocator.h"

#include "dml_util.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {

static uint64_t GetMaxAllocationSize() {
  int64 override_value = 0;
  Status s =
      ReadInt64FromEnvVar("TF_DIRECTML_MAX_ALLOC_SIZE", 0, &override_value);

  if (s.ok() && override_value > 0) {
    return static_cast<uint64_t>(override_value);
  }

  return D3D12HeapAllocator::kDefaultMaxAllocationSizeInBytes;
}

DmlAllocator::DmlAllocator(D3D12HeapAllocator* heap_allocator,
                           uint64_t memory_limit_in_bytes,
                           const GPUOptions& gpu_options, const string& name,
                           DmlDeviceRemovedEvent* device_removed_event)
    : GPUBFCAllocator(new SubAllocatorWrapper(heap_allocator),
                      memory_limit_in_bytes, gpu_options, name,
                      GetMaxAllocationSize()),
      heap_allocator_(heap_allocator) {
  constexpr size_t rounded_bytes = 0;
  constexpr bool force_deallocation = true;
  device_removed_event->AddListener(
      std::bind(&DmlAllocator::DeallocateFreeRegions, this, rounded_bytes,
                force_deallocation));
}

D3D12BufferRegion DmlAllocator::CreateBufferRegion(const void* ptr,
                                                   uint64_t size_in_bytes) {
  return heap_allocator_->CreateBufferRegion(ptr, size_in_bytes);
}

}  // namespace tensorflow
