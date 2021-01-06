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

#include "dml_adapter.h"
#include "dml_adapter_impl.h"
#include "dml_common.h"

namespace tensorflow {

class DmlAdapter;
class DmlExecutionContext;
class DmlEventQueue;
class D3D12HeapAllocator;
class DmlAllocator;
class D3D12DescriptorHeapAllocator;
class DmlDescriptorAllocator;
class DmlUploadHeap;
class DmlReadbackHeap;
class DmlKernelManager;
class GPUOptions;

// Holds device state that is shared across one or more DmlDevice instances.
// Instances of these state objects are owned by the DML device factory.
// Typically one of these state objects exists for each physical D3D adapter,
// but multiple TF DmlDevice instances can share this state. All objects owned
// by this state object are thread-safe.
struct DmlDeviceState {
 public:
  static std::unique_ptr<DmlDeviceState> Create(const DmlAdapter& adapter,
                                                const GPUOptions& gpu_options,
                                                uint64_t memory_limit_in_bytes);

  DmlDeviceState();
  ~DmlDeviceState();

  std::unique_ptr<DmlAdapter> adapter;
  Microsoft::WRL::ComPtr<ID3D12Device> d3d_device;
  Microsoft::WRL::ComPtr<ID3D12CommandQueue> command_queue;
  Microsoft::WRL::ComPtr<ID3D12SharingContract> sharing_contract;
  Microsoft::WRL::ComPtr<IDMLDevice> dml_device;
  std::unique_ptr<DmlExecutionContext> execution_context;
  std::unique_ptr<DmlEventQueue> event_queue;
  std::unique_ptr<D3D12HeapAllocator> heap_allocator;
  std::unique_ptr<DmlAllocator> dml_allocator;
  std::unique_ptr<D3D12DescriptorHeapAllocator> descriptor_heap_allocator;
  std::unique_ptr<DmlDescriptorAllocator> descriptor_allocator;
  std::unique_ptr<DmlUploadHeap> upload_heap;
  std::unique_ptr<DmlReadbackHeap> readback_heap;
  std::unique_ptr<DmlKernelManager> kernel_manager;
};

}  // namespace tensorflow