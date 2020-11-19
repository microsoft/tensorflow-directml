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

#include "dml_device_state.h"

#include "dml_adapter_impl.h"
#include "dml_bfc_allocator.h"
#include "dml_device_context.h"
#include "dml_event_queue.h"
#include "dml_kernel_manager.h"
#include "dml_readback_heap.h"
#include "dml_upload_heap.h"
#include "dml_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/stream_executor/platform/default/dso_loader.h"

using Microsoft::WRL::ComPtr;

static constexpr GUID kTensorflowDirectmlCreatorId = {
    0xcb7490ac,
    0x8a0f,
    0x44ec,
    {0x9b, 0x7b, 0x6f, 0x4c, 0xaf, 0xe8, 0xe9, 0xab}};

namespace tensorflow {

/*static*/ std::unique_ptr<DmlDeviceState> DmlDeviceState::Create(
    const DmlAdapter& adapter, const GPUOptions& gpu_options,
    uint64_t memory_limit_in_bytes) {
  D3D_FEATURE_LEVEL feature_level = adapter.IsComputeOnly()
                                        ? D3D_FEATURE_LEVEL_1_0_CORE
                                        : D3D_FEATURE_LEVEL_11_0;

  ComPtr<ID3D12Device> d3d_device;
  DML_CHECK_SUCCEEDED(D3D12CreateDevice(adapter.Impl()->Get(), feature_level,
                                        IID_PPV_ARGS(&d3d_device)));

  DML_CREATE_DEVICE_FLAGS dml_flags = DML_CREATE_DEVICE_FLAG_NONE;

#ifdef DML_BUILD_WINDOWS
  // If the D3D12 debug layer is enabled, enable the DML debug layer too. This
  // is only allowed when using an explicit path to DirectML.dll, since the
  // python wheels do not contain a copy of the debug layer.
  Microsoft::WRL::ComPtr<ID3D12InfoQueue> info_queue;
  if (getenv("TF_DIRECTML_PATH") &&
      SUCCEEDED(d3d_device->QueryInterface(IID_PPV_ARGS(&info_queue)))) {
    dml_flags |= DML_CREATE_DEVICE_FLAG_DEBUG;
    info_queue = nullptr;

    // Manually load DirectML.Debug.dll prior to creating the DML device, as a
    // way to override the DLL search path in case TF_DIRECTML_PATH is set.
    // We just need the DLL loaded into memory which is why we can throw away
    // the result of this function.
    stream_executor::internal::CachedDsoLoader::GetDirectMLDebugDsoHandle();
  }
#endif

  ComPtr<IDMLDevice> dml_device;
  dml_device = CreateDmlDevice(d3d_device.Get(), dml_flags);

  // Default to using compute queues for AMD since it seems to mitigate TDRs and
  // improve performance
  const bool use_compute_queue_default = adapter.VendorID() == VendorID::kAmd;

  bool use_compute_queue;
  Status s = ReadBoolFromEnvVar("TF_DIRECTML_USE_COMPUTE_QUEUE",
                                use_compute_queue_default, &use_compute_queue);

  D3D12_COMMAND_LIST_TYPE queue_type = use_compute_queue
                                           ? D3D12_COMMAND_LIST_TYPE_COMPUTE
                                           : D3D12_COMMAND_LIST_TYPE_DIRECT;

  D3D12_COMMAND_QUEUE_DESC command_queue_desc = {};
  command_queue_desc.Type = queue_type;
  command_queue_desc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
  command_queue_desc.Flags = D3D12_COMMAND_QUEUE_FLAG_DISABLE_GPU_TIMEOUT;
  command_queue_desc.NodeMask = 0;

  ComPtr<ID3D12CommandQueue> command_queue;

#ifdef DIRECTML_ENABLE_TELEMETRY
  Microsoft::WRL::ComPtr<ID3D12Device9> d3d_device9;
  if (SUCCEEDED(d3d_device->QueryInterface(IID_PPV_ARGS(&d3d_device9)))) {
    // If ID3D12Device9 is available, create with the DMLTF CreatorID for
    // telemetry. This call should succeed even if the ID is unrecognized.
    DML_CHECK_SUCCEEDED(d3d_device9->CreateCommandQueue1(
        &command_queue_desc, kTensorflowDirectmlCreatorId,
        IID_PPV_ARGS(&command_queue)));
  }
#endif

  // Create a queue without a client hint if telemetry is disabled or we're
  // running on an older version of D3D.
  if (!command_queue) {
    DML_CHECK_SUCCEEDED(d3d_device->CreateCommandQueue(
        &command_queue_desc, IID_PPV_ARGS(&command_queue)));
  }

  // Retrieve the sharing contract which we use to delimit capture boundaries.
  // This may fail, so sharing_contract_ may end up null.
  ComPtr<ID3D12SharingContract> sharing_contract;
  (void)command_queue->QueryInterface(IID_PPV_ARGS(&sharing_contract));

  auto heap_allocator = absl::make_unique<D3D12HeapAllocator>(
      d3d_device.Get(), CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
      D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS,
      D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
      D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

  auto dml_allocator = absl::make_unique<DmlAllocator>(
      heap_allocator.get(), memory_limit_in_bytes, gpu_options, "DmlAllocator");

  auto execution_context = absl::make_unique<DmlExecutionContext>(
      d3d_device.Get(), dml_device.Get(), command_queue.Get(),
      dml_allocator.get());

  auto event_queue = absl::make_unique<DmlEventQueue>();

  auto upload_heap = absl::make_unique<DmlUploadHeap>(d3d_device.Get(),
                                                      execution_context.get());

  auto readback_heap = absl::make_unique<DmlReadbackHeap>(
      d3d_device.Get(), execution_context.get(), event_queue.get());

  auto kernel_manager = absl::make_unique<DmlKernelManager>();

  // Construct the final state object
  auto state = absl::make_unique<DmlDeviceState>();
  state->adapter = absl::make_unique<DmlAdapter>(adapter);
  state->d3d_device = std::move(d3d_device);
  state->command_queue = std::move(command_queue);
  state->sharing_contract = std::move(sharing_contract);
  state->dml_device = std::move(dml_device);
  state->execution_context = std::move(execution_context);
  state->event_queue = std::move(event_queue);
  state->heap_allocator = std::move(heap_allocator);
  state->dml_allocator = std::move(dml_allocator);
  state->upload_heap = std::move(upload_heap);
  state->readback_heap = std::move(readback_heap);
  state->kernel_manager = std::move(kernel_manager);
  return state;
}

DmlDeviceState::DmlDeviceState() = default;
DmlDeviceState::~DmlDeviceState() = default;

}  // namespace tensorflow