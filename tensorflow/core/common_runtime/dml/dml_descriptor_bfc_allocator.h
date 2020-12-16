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
#include "dml_descriptor_heap_allocator.h"
#include "tensorflow/core/common_runtime/bfc_allocator.h"

namespace tensorflow {

class DmlDescriptorAllocator : public BFCAllocator {
  // We allocate our descriptor heaps in fixed-sized blocks of 64k descriptors.
  // This is more than sufficient to accommodate even the largest of DML
  // operators or graphs.
  static constexpr uint64_t kMaxAllocationSizeInDescriptors = 65536;

  // We can allocate as many descriptors as we want (across all allocations).
  static constexpr size_t kMaxTotalDescriptors = static_cast<size_t>(-1);

  // Always dynamically grow the descriptor pool if necessary.
  static constexpr bool kAllowGrowth = true;

  // Always enable garbage collection.
  static constexpr bool kEnableGarbageCollection = true;

  // A SubAllocator that wraps a D3D12DescriptorHeapAllocator
  class SubAllocatorWrapper final : public tensorflow::SubAllocator {
   public:
    SubAllocatorWrapper(D3D12DescriptorHeapAllocator* impl)
        : SubAllocator({}, {}), impl_(impl) {}

   public:  // SubAllocator overrides
    void* Alloc(size_t alignment, size_t num_bytes) override {
      void* p = impl_->Alloc(num_bytes);
      VisitAlloc(p, 0, num_bytes);
      return p;
    }

    void Free(void* ptr, size_t num_bytes) override {
      VisitFree(ptr, 0, num_bytes);
      impl_->Free(ptr, num_bytes);
    }

   private:
    D3D12DescriptorHeapAllocator* impl_;
  };

 public:
  DmlDescriptorAllocator(D3D12DescriptorHeapAllocator* heap_allocator,
                         const string& name);

  D3D12DescriptorHandles GetDescriptorHandles(const void* ptr) const;

 private:
  D3D12DescriptorHeapAllocator* heap_allocator_;  // not owned
};

}  // namespace tensorflow
