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
#include "dml_buffer.h"
#include "dml_execution_context.h"
#include "dml_tracing.h"
#include "dml_util.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {

DmlCommandList::DmlCommandList(ID3D12Device* d3d_device, IDMLDevice* dml_device,
                               std::shared_ptr<DmlCommandQueue> queue)
    : d3d_device_(d3d_device),
      dml_device_(dml_device),
      queue_(std::move(queue)),
      descriptor_pool_(d3d_device, 2048),
      command_allocator_ring_(d3d_device, queue_->GetType(),
                              queue_->GetCurrentCompletionEvent()) {
  DML_CHECK_SUCCEEDED(
      dml_device->CreateCommandRecorder(IID_PPV_ARGS(&recorder_)));
}

void DmlCommandList::CopyBufferRegion(
    ID3D12Resource* dst_buffer, uint64_t dst_offset,
    D3D12_RESOURCE_STATES dst_state, ID3D12Resource* src_buffer,
    uint64_t src_offset, D3D12_RESOURCE_STATES src_state, uint64_t byte_count) {
  DmlTracing::Instance().LogExecutionContextCopyBufferRegion();

  absl::InlinedVector<D3D12_RESOURCE_BARRIER, 3> barriers;

  if (!(dst_state & D3D12_RESOURCE_STATE_COPY_DEST)) {
    barriers.push_back(CD3DX12_RESOURCE_BARRIER::Transition(
        dst_buffer, dst_state, D3D12_RESOURCE_STATE_COPY_DEST));
  }
  if (!(src_state & D3D12_RESOURCE_STATE_COPY_SOURCE)) {
    barriers.push_back(CD3DX12_RESOURCE_BARRIER::Transition(
        src_buffer, src_state, D3D12_RESOURCE_STATE_COPY_SOURCE));
  }

  if (!barriers.empty()) {
    d3d_command_list_->ResourceBarrier(barriers.size(), barriers.data());
  }

  d3d_command_list_->CopyBufferRegion(dst_buffer, dst_offset, src_buffer,
                                      src_offset, byte_count);

  // Reset barrier state
  for (auto& barrier : barriers) {
    std::swap(barrier.Transition.StateBefore, barrier.Transition.StateAfter);
  }

  // Since this copy may write to GPU memory, we also need to perform an
  // aliasing barrier
  barriers.push_back(CD3DX12_RESOURCE_BARRIER::Aliasing(nullptr, nullptr));

  d3d_command_list_->ResourceBarrier(barriers.size(), barriers.data());
}

void DmlCommandList::FillBufferWithPattern(
    ID3D12Resource* dst, uint64_t dst_offset, uint64_t dst_size_in_bytes,
    absl::Span<const uint8_t>
        value /* Data type agnostic value, treated as raw bits */) {
  DmlTracing::Instance().LogExecutionContextFillBufferWithPattern();

  // The fill pattern for ClearUnorderedAccessViewUint is 16 bytes.
  union {
    uint32_t integers[4];
    uint8_t bytes[16];
  } fillPattern = {};

  assert(sizeof(fillPattern.bytes) == 16);
  assert(value.size() <=
         sizeof(fillPattern.bytes));  // No element is expected larger than
                                      // 128 bits (e.g. complex128).

  if (!value.empty()) {
    assert(sizeof(fillPattern.bytes) % value.size() ==
           0);  // Should fit evenly into 16 bytes (e.g. uint8, float16, uint32,
                // float64...).

    // Repeat the value multiple times into the pattern buffer.
    size_t valueIndex = 0;
    for (uint8_t& p : fillPattern.bytes) {
      p = value[valueIndex++];
      valueIndex = (valueIndex == value.size()) ? 0 : valueIndex;
    }
  }
  // Else just leave fill pattern as zeroes.

  // The destination must be appropriately aligned and padded
  assert(dst_offset % sizeof(uint32_t) == 0);
  assert(dst_size_in_bytes % sizeof(uint32_t) == 0);

  // Create a RAW buffer UAV over the resource.
  D3D12_UNORDERED_ACCESS_VIEW_DESC uav_desc = {};
  uav_desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
  uav_desc.Format = DXGI_FORMAT_R32_TYPELESS;
  uav_desc.Buffer.FirstElement =
      static_cast<uint32_t>(dst_offset / sizeof(uint32_t));
  uav_desc.Buffer.NumElements =
      static_cast<uint32_t>(dst_size_in_bytes / sizeof(uint32_t));
  uav_desc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;

  const uint32_t needed_descriptor_count = 1;
  DmlDescriptorRange descriptor_range_cpu = descriptor_pool_.AllocDescriptors(
      needed_descriptor_count, current_completion_event_,
      D3D12_DESCRIPTOR_HEAP_FLAG_NONE);
  DmlDescriptorRange descriptor_range_gpu = descriptor_pool_.AllocDescriptors(
      needed_descriptor_count, current_completion_event_,
      D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE);
  d3d_device_->CreateUnorderedAccessView(dst, nullptr, &uav_desc,
                                         descriptor_range_cpu.cpu_handle);
  d3d_device_->CreateUnorderedAccessView(dst, nullptr, &uav_desc,
                                         descriptor_range_gpu.cpu_handle);

  SetDescriptorHeap(descriptor_range_gpu.heap);

  // Record a ClearUAV onto the command list.
  d3d_command_list_->ClearUnorderedAccessViewUint(
      descriptor_range_gpu.gpu_handle, descriptor_range_cpu.cpu_handle, dst,
      fillPattern.integers, 0, nullptr);

  // Barrier all outputs.
  D3D12_RESOURCE_BARRIER barriers[] = {
      CD3DX12_RESOURCE_BARRIER::UAV(nullptr),
      CD3DX12_RESOURCE_BARRIER::Aliasing(nullptr, nullptr)};
  d3d_command_list_->ResourceBarrier(ABSL_ARRAYSIZE(barriers), barriers);
}

void DmlCommandList::InitializeOperator(IDMLOperatorInitializer* initializer,
                                        IDMLBindingTable* binding_table,
                                        ID3D12DescriptorHeap* descriptor_heap) {
  // Record the initialization work.
  SetDescriptorHeap(descriptor_heap);
  recorder_->RecordDispatch(d3d_command_list_.Get(), initializer,
                            binding_table);

  // Barrier if there's an output (i.e. persistent resource), or if any temps
  // are used.
  DML_BINDING_PROPERTIES binding_props = initializer->GetBindingProperties();
  if ((binding_props.PersistentResourceSize > 0) ||
      (binding_props.TemporaryResourceSize > 0)) {
    D3D12_RESOURCE_BARRIER barriers[] = {
        CD3DX12_RESOURCE_BARRIER::UAV(nullptr),
        CD3DX12_RESOURCE_BARRIER::Aliasing(nullptr, nullptr)};
    d3d_command_list_->ResourceBarrier(ABSL_ARRAYSIZE(barriers), barriers);
  }
}

void DmlCommandList::ExecuteOperator(IDMLCompiledOperator* op,
                                     IDMLBindingTable* binding_table,
                                     ID3D12DescriptorHeap* descriptor_heap) {
  DmlTracing::Instance().LogExecuteOperatorStart(op, d3d_command_list_.Get());

  // Record the execution work.
  SetDescriptorHeap(descriptor_heap);
  recorder_->RecordDispatch(d3d_command_list_.Get(), op, binding_table);

  // Barrier all outputs.
  D3D12_RESOURCE_BARRIER barriers[] = {
      CD3DX12_RESOURCE_BARRIER::UAV(nullptr),
      CD3DX12_RESOURCE_BARRIER::Aliasing(nullptr, nullptr)};
  d3d_command_list_->ResourceBarrier(ABSL_ARRAYSIZE(barriers), barriers);

  DmlTracing::Instance().LogExecuteOperatorEnd(d3d_command_list_.Get());
}

void DmlCommandList::ResourceBarrier(
    absl::Span<const D3D12_RESOURCE_BARRIER> barriers) {
  d3d_command_list_->ResourceBarrier(static_cast<uint32_t>(barriers.size()),
                                     barriers.data());
}

void DmlCommandList::UavBarrier() {
  D3D12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::UAV(nullptr);
  d3d_command_list_->ResourceBarrier(1, &barrier);
}

void DmlCommandList::SetDescriptorHeap(ID3D12DescriptorHeap* descriptor_heap) {
  if (descriptor_heap != nullptr &&
      descriptor_heap != current_descriptor_heap_) {
    current_descriptor_heap_ = descriptor_heap;
    ID3D12DescriptorHeap* descriptor_heaps[] = {descriptor_heap};
    d3d_command_list_->SetDescriptorHeaps(ABSL_ARRAYSIZE(descriptor_heaps),
                                          descriptor_heaps);
  }
}

void DmlCommandList::Open() {
  assert(current_descriptor_heap_ == nullptr);

  ID3D12CommandAllocator* allocator = command_allocator_ring_.GetNextAllocator(
      queue_->GetNextCompletionEvent());

  if (!d3d_command_list_) {
    // Lazily create underlying D3D command list.
    DML_CHECK_SUCCEEDED(d3d_device_->CreateCommandList(
        0, queue_->GetType(), , allocator, nullptr,
        IID_PPV_ARGS(&d3d_command_list_)));
  } else {
    DML_CHECK_SUCCEEDED(d3d_command_list_->Reset(allocator, nullptr));
  }
}

Status DmlCommandList::Close() {
  HRESULT hr = d3d_command_list_->Close();
  if (dml_util::HrIsOutOfMemory(hr)) {
    return errors::ResourceExhausted("OOM when closing the command list");
  } else {
    DML_CHECK_SUCCEEDED(hr);
  }

  // The descriptor heap must be set on the command list the next time it's
  // opened.
  current_descriptor_heap_ = nullptr;

  // Fail early if something horrifying happens
  DML_CHECK_SUCCEEDED(dml_device_->GetDeviceRemovedReason());
  DML_CHECK_SUCCEEDED(d3d_device_->GetDeviceRemovedReason());

  return Status::OK();
}

}  // namespace tensorflow