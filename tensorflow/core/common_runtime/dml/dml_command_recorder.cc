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

#include "dml_command_recorder.h"

#include "dml_bfc_allocator.h"
#include "dml_buffer.h"
#include "dml_util.h"

namespace tensorflow {

DmlCommandRecorder::DmlCommandRecorder(
    ID3D12Device* d3d_device, IDMLDevice* dml_device,
    std::shared_ptr<DmlCommandQueue> command_queue, DmlAllocator* allocator)
    : queue_(std::move(command_queue)),
      d3d_device_(d3d_device),
      dml_device_(dml_device),
      descriptor_pool_(d3d_device, 2048),
      allocator_(allocator),
      command_allocator_ring_(d3d_device, queue_->GetType(),
                              queue_->GetCurrentCompletionEvent()) {
  DML_CHECK_SUCCEEDED(dml_device->CreateOperatorInitializer(
      0, nullptr, IID_PPV_ARGS(&initializer_)));
  DML_CHECK_SUCCEEDED(
      dml_device->CreateCommandRecorder(IID_PPV_ARGS(&recorder_)));

  Open();
}

void DmlCommandRecorder::InitializeOperator(
    IDMLCompiledOperator* op,
    const DML_BINDING_DESC& persistent_resource_binding,
    const DML_BINDING_DESC& input_array_binding) {
  if (!status_.ok()) return;

  // Reset the initializer to reference the input operator.
  IDMLCompiledOperator* ops[] = {op};
  DML_CHECK_SUCCEEDED(initializer_->Reset(ABSL_ARRAYSIZE(ops), ops));

  DML_BINDING_PROPERTIES init_binding_props =
      initializer_->GetBindingProperties();
  DML_BINDING_PROPERTIES exec_binding_props = op->GetBindingProperties();

  const uint32_t num_descriptors = init_binding_props.RequiredDescriptorCount;
  DmlDescriptorRange descriptor_range = descriptor_pool_.AllocDescriptors(
      num_descriptors, queue_->GetNextCompletionEvent());

  // Create a binding table for initialization.
  DML_BINDING_TABLE_DESC binding_table_desc = {};
  binding_table_desc.Dispatchable = initializer_.Get();
  binding_table_desc.CPUDescriptorHandle = descriptor_range.cpu_handle;
  binding_table_desc.GPUDescriptorHandle = descriptor_range.gpu_handle;
  binding_table_desc.SizeInDescriptors = num_descriptors;

  Microsoft::WRL::ComPtr<IDMLBindingTable> binding_table;
  DML_CHECK_SUCCEEDED(dml_device_->CreateBindingTable(
      &binding_table_desc, IID_PPV_ARGS(&binding_table)));

  // Create a temporary resource for initializing the op, if it's required.
  UINT64 temporary_resource_size = init_binding_props.TemporaryResourceSize;
  DmlBuffer temp_resource;
  if (temporary_resource_size > 0) {
    // Allocate a temporary buffer and keep a use on it until the end of this
    // method. The buffer resource will still be alive (managed by the pool);
    // freeing allows the resource to be shared with other operators, but
    // because the allocator is multi-threaded we need to at least keep a use on
    // it until we're done with it locally to prevent the buffer being reused.
    temp_resource = DmlBuffer(allocator_, temporary_resource_size);
    if (!temp_resource) {
      status_ = errors::ResourceExhausted("OOM when allocating a buffer of ",
                                          temporary_resource_size, " bytes");
      return;
    }

    // Bind the temporary resource.
    DML_BUFFER_BINDING buffer_binding = temp_resource.GetBufferBinding();
    DML_BINDING_DESC binding_desc = {DML_BINDING_TYPE_BUFFER, &buffer_binding};
    binding_table->BindTemporaryResource(&binding_desc);
  }

  // Bind inputs, if provided.
  if (input_array_binding.Type != DML_BINDING_TYPE_NONE) {
    // An operator with inputs to bind MUST use a BUFFER_ARRAY.
    assert(input_array_binding.Type == DML_BINDING_TYPE_BUFFER_ARRAY);
    binding_table->BindInputs(1, &input_array_binding);
  }

  // Bind the persistent resource, which is an output of initialization.
  if (persistent_resource_binding.Type != DML_BINDING_TYPE_NONE) {
    // Persistent resources MUST be bound as buffers.
    assert(persistent_resource_binding.Type == DML_BINDING_TYPE_BUFFER);
    binding_table->BindOutputs(1, &persistent_resource_binding);
  }

  // Record the initialization work.
  SetDescriptorHeap(descriptor_range.heap);
  recorder_->RecordDispatch(current_command_list_.Get(), initializer_.Get(),
                            binding_table.Get());

  // Barrier if there's an output (i.e. persistent resource), or if any temps
  // are used.
  if ((persistent_resource_binding.Type != DML_BINDING_TYPE_NONE) ||
      (temporary_resource_size > 0)) {
    D3D12_RESOURCE_BARRIER barriers[] = {
        CD3DX12_RESOURCE_BARRIER::UAV(nullptr),
        CD3DX12_RESOURCE_BARRIER::Aliasing(nullptr, nullptr)};
    current_command_list_->ResourceBarrier(ABSL_ARRAYSIZE(barriers), barriers);
  }

  OnCommandRecorded();
}

void DmlCommandRecorder::ExecuteOperator(
    IDMLCompiledOperator* op,
    const DML_BINDING_DESC& persistent_resource_binding,
    absl::Span<const DML_BINDING_DESC> input_bindings,
    absl::Span<const DML_BINDING_DESC> output_bindings) {
  if (!status_.ok()) return;

  DML_BINDING_PROPERTIES exec_binding_props = op->GetBindingProperties();

  const uint32_t num_descriptors = exec_binding_props.RequiredDescriptorCount;
  DmlDescriptorRange descriptor_range = descriptor_pool_.AllocDescriptors(
      num_descriptors, queue_->GetNextCompletionEvent());

  // Create a binding table for execution.
  DML_BINDING_TABLE_DESC binding_table_desc = {};
  binding_table_desc.Dispatchable = op;
  binding_table_desc.CPUDescriptorHandle = descriptor_range.cpu_handle;
  binding_table_desc.GPUDescriptorHandle = descriptor_range.gpu_handle;
  binding_table_desc.SizeInDescriptors = num_descriptors;

  Microsoft::WRL::ComPtr<IDMLBindingTable> binding_table;
  DML_CHECK_SUCCEEDED(dml_device_->CreateBindingTable(
      &binding_table_desc, IID_PPV_ARGS(&binding_table)));

  // Create a temporary resource for executing the op, if it's required.
  UINT64 temporary_resource_size = exec_binding_props.TemporaryResourceSize;
  DmlBuffer temp_resource;
  if (temporary_resource_size > 0) {
    // Allocate a temporary buffer and keep a use on it until the end of this
    // method. The buffer resource will still be alive (managed by the pool);
    // freeing allows the resource to be shared with other operators, but
    // because the allocator is multi-threaded we need to at least keep a use on
    // it until we're done with it locally to prevent the buffer being reused.
    temp_resource = DmlBuffer(allocator_, temporary_resource_size);
    if (!temp_resource) {
      status_ = errors::ResourceExhausted("OOM when allocating a buffer of ",
                                          temporary_resource_size, " bytes");
      return;
    }

    // Bind the temporary resource.
    DML_BUFFER_BINDING buffer_binding = temp_resource.GetBufferBinding();
    DML_BINDING_DESC binding_desc = {DML_BINDING_TYPE_BUFFER, &buffer_binding};
    binding_table->BindTemporaryResource(&binding_desc);
  }

  if (persistent_resource_binding.Type != DML_BINDING_TYPE_NONE) {
    binding_table->BindPersistentResource(&persistent_resource_binding);
  }

  binding_table->BindInputs(static_cast<uint32_t>(input_bindings.size()),
                            input_bindings.data());
  binding_table->BindOutputs(static_cast<uint32_t>(output_bindings.size()),
                             output_bindings.data());

  // Record the execution work.
  SetDescriptorHeap(descriptor_range.heap);
  recorder_->RecordDispatch(current_command_list_.Get(), op,
                            binding_table.Get());

  // Barrier all outputs.
  D3D12_RESOURCE_BARRIER barriers[] = {
      CD3DX12_RESOURCE_BARRIER::UAV(nullptr),
      CD3DX12_RESOURCE_BARRIER::Aliasing(nullptr, nullptr)};
  current_command_list_->ResourceBarrier(ABSL_ARRAYSIZE(barriers), barriers);

  OnCommandRecorded();
}

void DmlCommandRecorder::CopyBufferRegion(
    ID3D12Resource* dst_buffer, uint64_t dst_offset,
    D3D12_RESOURCE_STATES dst_state, ID3D12Resource* src_buffer,
    uint64_t src_offset, D3D12_RESOURCE_STATES src_state, uint64_t byte_count) {
  if (!status_.ok()) return;

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
    current_command_list_->ResourceBarrier(barriers.size(), barriers.data());
  }

  current_command_list_->CopyBufferRegion(dst_buffer, dst_offset, src_buffer,
                                          src_offset, byte_count);

  // Reset barrier state
  for (auto& barrier : barriers) {
    std::swap(barrier.Transition.StateBefore, barrier.Transition.StateAfter);
  }

  // Since this copy may write to GPU memory, we also need to perform an
  // aliasing barrier
  barriers.push_back(CD3DX12_RESOURCE_BARRIER::Aliasing(nullptr, nullptr));

  current_command_list_->ResourceBarrier(barriers.size(), barriers.data());

  OnCommandRecorded();
}

void DmlCommandRecorder::FillBufferWithPattern(
    ID3D12Resource* dst, uint64_t dst_offset, uint64_t dst_size_in_bytes,
    absl::Span<const uint8_t>
        value /* Data type agnostic value, treated as raw bits */) {
  if (!status_.ok()) return;

  // The fill pattern for ClearUnorderedAccessViewUint is 16 bytes.
  union {
    uint32_t integers[4];
    uint8_t bytes[16];
  } fillPattern = {};

  assert(ARRAYSIZE(fillPattern.bytes) == 16);
  assert(value.size() <=
         ARRAYSIZE(fillPattern.bytes));  // No element is expected larger than
                                         // 128 bits (e.g. complex128).

  if (!value.empty()) {
    assert(ARRAYSIZE(fillPattern.bytes) % value.size() ==
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
      needed_descriptor_count, queue_->GetNextCompletionEvent(),
      D3D12_DESCRIPTOR_HEAP_FLAG_NONE);
  DmlDescriptorRange descriptor_range_gpu = descriptor_pool_.AllocDescriptors(
      needed_descriptor_count, queue_->GetNextCompletionEvent(),
      D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE);
  d3d_device_->CreateUnorderedAccessView(dst, nullptr, &uav_desc,
                                         descriptor_range_cpu.cpu_handle);
  d3d_device_->CreateUnorderedAccessView(dst, nullptr, &uav_desc,
                                         descriptor_range_gpu.cpu_handle);

  SetDescriptorHeap(descriptor_range_gpu.heap);

  // Record a ClearUAV onto the command list.
  current_command_list_->ClearUnorderedAccessViewUint(
      descriptor_range_gpu.gpu_handle, descriptor_range_cpu.cpu_handle, dst,
      fillPattern.integers, 0, nullptr);

  // Barrier all outputs.
  D3D12_RESOURCE_BARRIER barriers[] = {
      CD3DX12_RESOURCE_BARRIER::UAV(nullptr),
      CD3DX12_RESOURCE_BARRIER::Aliasing(nullptr, nullptr)};
  current_command_list_->ResourceBarrier(ABSL_ARRAYSIZE(barriers), barriers);

  OnCommandRecorded();
}

void DmlCommandRecorder::ResourceBarrier(
    absl::Span<const D3D12_RESOURCE_BARRIER> barriers) {
  if (!status_.ok()) return;

  current_command_list_->ResourceBarrier(static_cast<uint32_t>(barriers.size()),
                                         barriers.data());
  OnCommandRecorded();
}

void DmlCommandRecorder::Open() {
  // This should have been checked in one of the public functions before calling
  // Open()
  DCHECK(status_.ok());

  assert(current_descriptor_heap_ == nullptr);

  ID3D12CommandAllocator* allocator =
      command_allocator_ring_.GetCurrentAllocator();

  if (cached_command_lists_.empty()) {
    DML_CHECK_SUCCEEDED(d3d_device_->CreateCommandList(
        0, queue_->GetType(), command_allocator_ring_.GetCurrentAllocator(),
        nullptr, IID_PPV_ARGS(&current_command_list_)));
  } else {
    current_command_list_ = cached_command_lists_.front();
    cached_command_lists_.pop_front();
    DML_CHECK_SUCCEEDED(current_command_list_->Reset(allocator, nullptr));
  }

  // The current command allocator will become eligible for reset once this
  // command list completes execution
  command_allocator_ring_.AdvanceAllocator(queue_->GetNextCompletionEvent());
}

void DmlCommandRecorder::CloseAndExecute() {
  if (!status_.ok()) return;

  HRESULT hr = current_command_list_->Close();

  if (dml_util::HrIsOutOfMemory(hr)) {
    status_ = errors::ResourceExhausted("OOM when closing the command list");
  } else {
    DML_CHECK_SUCCEEDED(hr);

    if (operations_recorded_in_current_command_list_ != 0) {
      // Close and execute the command list
      ID3D12CommandList* commandLists[] = {current_command_list_.Get()};
      queue_->ExecuteCommandLists(commandLists);
    }

    cached_command_lists_.push_back(current_command_list_.Get());
  }

  current_command_list_ = nullptr;
  operations_recorded_in_current_command_list_ = 0;

  // The descriptor heap must be set on the command list the next time it's
  // opened.
  current_descriptor_heap_ = nullptr;

  // Fail early if something horrifying happens
  DML_CHECK_SUCCEEDED(dml_device_->GetDeviceRemovedReason());
  DML_CHECK_SUCCEEDED(d3d_device_->GetDeviceRemovedReason());

  // Always keep the command recorder in an opened state
  Open();
}

void DmlCommandRecorder::SetDescriptorHeap(
    ID3D12DescriptorHeap* descriptor_heap) {
  // This should have been checked in one of the public functions before calling
  // SetDescriptorHeap()
  DCHECK(status_.ok());

  if (descriptor_heap != nullptr &&
      descriptor_heap != current_descriptor_heap_) {
    current_descriptor_heap_ = descriptor_heap;

    ID3D12DescriptorHeap* descriptor_heaps_[] = {descriptor_heap};
    current_command_list_->SetDescriptorHeaps(ABSL_ARRAYSIZE(descriptor_heaps_),
                                              descriptor_heaps_);
  }
}

void DmlCommandRecorder::OnCommandRecorded() {
  // This should have been checked in one of the public functions before calling
  // OnCommandRecorded()
  DCHECK(status_.ok());

  ++operations_recorded_in_current_command_list_;

  if (operations_recorded_in_current_command_list_ >= 25) {
    CloseAndExecute();
    assert(operations_recorded_in_current_command_list_ == 0);
  }
}

bool DmlCommandRecorder::HasUnflushedWork() const {
  return operations_recorded_in_current_command_list_ != 0;
}

}  // namespace tensorflow