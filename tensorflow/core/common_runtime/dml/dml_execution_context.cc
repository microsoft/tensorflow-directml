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

#include "dml_execution_context.h"

#include "dml_bfc_allocator.h"
#include "dml_buffer.h"
#include "dml_tracing.h"
#include "dml_util.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {

DmlExecutionContext::DmlExecutionContext(ID3D12Device* d3d_device,
                                         IDMLDevice* dml_device,
                                         ID3D12CommandQueue* queue,
                                         DmlAllocator* allocator) {
  shared_state_ = std::make_shared<SharedState>();
  shared_state_->impl = absl::make_unique<DmlExecutionContextImpl>(
      d3d_device, dml_device, queue, allocator);

  int64 batch_flush_size = 0;
  Status s =
      ReadInt64FromEnvVar("TF_DIRECTML_BATCH_FLUSH_SIZE", 0, &batch_flush_size);
  if (s.ok() && batch_flush_size != 0) {
    shared_state_->batch_flush_size_ = static_cast<uint32_t>(batch_flush_size);
  }

  // Launch the thread, supplying it with a pointer to the shared state
  thread_ = std::thread(ThreadProc, shared_state_);
}

DmlExecutionContext::~DmlExecutionContext() {
  // Request exit of the background thread
  std::unique_lock<std::mutex> lock(shared_state_->mutex);
  shared_state_->exit_requested = true;
  shared_state_->new_function_enqueued.notify_all();  // wake the thread
  lock.unlock();

  // detach() rather than join(), because we don't want (or need) to wait for
  // it to complete. This prevents blocking in a destructor, which would be
  // bad.
  thread_.detach();
}

DmlExecutionContextImpl::DmlExecutionContextImpl(ID3D12Device* d3d_device,
                                                 IDMLDevice* dml_device,
                                                 ID3D12CommandQueue* queue,
                                                 DmlAllocator* allocator)
    : queue_(std::make_shared<DmlCommandQueue>(queue)),
      d3d_device_(d3d_device),
      dml_device_(dml_device),
      descriptor_pool_(d3d_device, 2048),
      allocator_(allocator),
      command_allocator_ring_(d3d_device, queue_->GetType(),
                              queue_->GetCurrentCompletionEvent()) {
  DML_CHECK_SUCCEEDED(
      dml_device->CreateCommandRecorder(IID_PPV_ARGS(&recorder_)));
  OpenCommandList();
}

DmlGpuEvent DmlExecutionContextImpl::CopyBufferRegion(
    ID3D12Resource* dst_buffer, uint64_t dst_offset,
    D3D12_RESOURCE_STATES dst_state, ID3D12Resource* src_buffer,
    uint64_t src_offset, D3D12_RESOURCE_STATES src_state, uint64_t byte_count) {
  assert(!closed_);
  if (!status_.ok()) {
    GetCurrentCompletionEvent();
  }

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

  return GetCurrentCompletionEvent();
}

DmlGpuEvent DmlExecutionContextImpl::FillBufferWithPattern(
    ID3D12Resource* dst, uint64_t dst_offset, uint64_t dst_size_in_bytes,
    absl::Span<const uint8_t>
        value /* Data type agnostic value, treated as raw bits */) {
  assert(!closed_);
  if (!status_.ok()) {
    GetCurrentCompletionEvent();
  }

  DmlTracing::Instance().LogExecutionContextFillBufferWithPattern();

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

  return GetCurrentCompletionEvent();
}

DmlGpuEvent DmlExecutionContextImpl::InitializeOperator(
    IDMLOperatorInitializer* initializer, IDMLBindingTable* binding_table,
    ID3D12DescriptorHeap* descriptor_heap) {
  assert(!closed_);
  if (!status_.ok()) {
    GetCurrentCompletionEvent();
  }

  // Record the initialization work.
  SetDescriptorHeap(descriptor_heap);
  recorder_->RecordDispatch(current_command_list_.Get(), initializer,
                            binding_table);

  // Barrier if there's an output (i.e. persistent resource), or if any temps
  // are used.
  DML_BINDING_PROPERTIES binding_props = initializer->GetBindingProperties();
  if ((binding_props.PersistentResourceSize > 0) ||
      (binding_props.TemporaryResourceSize > 0)) {
    D3D12_RESOURCE_BARRIER barriers[] = {
        CD3DX12_RESOURCE_BARRIER::UAV(nullptr),
        CD3DX12_RESOURCE_BARRIER::Aliasing(nullptr, nullptr)};
    current_command_list_->ResourceBarrier(ABSL_ARRAYSIZE(barriers), barriers);
  }

  OnCommandRecorded();

  return GetCurrentCompletionEvent();
}

DmlGpuEvent DmlExecutionContextImpl::ExecuteOperator(
    IDMLCompiledOperator* op, IDMLBindingTable* binding_table,
    ID3D12DescriptorHeap* descriptor_heap) {
  assert(!closed_);
  if (!status_.ok()) {
    GetCurrentCompletionEvent();
  }

  // Record the execution work.
  SetDescriptorHeap(descriptor_heap);
  recorder_->RecordDispatch(current_command_list_.Get(), op, binding_table);

  // Barrier all outputs.
  D3D12_RESOURCE_BARRIER barriers[] = {
      CD3DX12_RESOURCE_BARRIER::UAV(nullptr),
      CD3DX12_RESOURCE_BARRIER::Aliasing(nullptr, nullptr)};
  current_command_list_->ResourceBarrier(ABSL_ARRAYSIZE(barriers), barriers);

  OnCommandRecorded();

  return GetCurrentCompletionEvent();
}

DmlGpuEvent DmlExecutionContextImpl::ResourceBarrier(
    absl::Span<const D3D12_RESOURCE_BARRIER> barriers) {
  assert(!closed_);
  if (!status_.ok()) {
    GetCurrentCompletionEvent();
  }

  current_command_list_->ResourceBarrier(static_cast<uint32_t>(barriers.size()),
                                         barriers.data());
  OnCommandRecorded();

  return GetCurrentCompletionEvent();
}

DmlGpuEvent DmlExecutionContextImpl::UavBarrier() {
  assert(!closed_);
  if (!status_.ok()) {
    GetCurrentCompletionEvent();
  }

  D3D12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::UAV(nullptr);
  current_command_list_->ResourceBarrier(1, &barrier);
  OnCommandRecorded();

  return GetCurrentCompletionEvent();
}

StatusOr<DmlGpuEvent> DmlExecutionContextImpl::Flush() {
  assert(!closed_);
  DmlTracing::Instance().LogExecutionContextFlush();
  VLOG(1) << "DML EC IMPL: Begin Flush ("
          << operations_recorded_in_current_command_list_ << " cmds)";

  if (operations_recorded_in_current_command_list_ == 0) {
    // Nothing to flush
    return GetCurrentCompletionEvent();
  }

  CloseCommandListAndExecute();

  if (!status_.ok()) {
    // "Unknown" represents device removals, which are uncoverable failures
    if (!errors::IsUnknown(status_)) {
      status_ = Status::OK();
    }
    return status_;
  }

  auto event = GetCurrentCompletionEvent();
  VLOG(1) << "DML EC IMPL: End Flush; completion fv = " << event.fence_value;

  return event;
}

void DmlExecutionContextImpl::Close() {
  assert(!closed_);
  queue_->Close();
  closed_ = true;
}

DmlGpuEvent DmlExecutionContextImpl::GetCurrentCompletionEvent() {
  assert(!closed_);

  DmlGpuEvent event = queue_->GetCurrentCompletionEvent();

  // If something has been recorded into a command list but not submitted yet,
  // it means that the *next* fence value is the one to signal completion.
  if (operations_recorded_in_current_command_list_ != 0) {
    ++event.fence_value;
  }

  return event;
}

D3D12_COMMAND_LIST_TYPE DmlExecutionContextImpl::GetCommandListTypeForQueue()
    const {
  assert(!closed_);
  return queue_->GetType();
}

void DmlExecutionContextImpl::SetDescriptorHeap(
    ID3D12DescriptorHeap* descriptor_heap) {
  // This should have been checked in one of the public functions before calling
  // SetDescriptorHeap()
  DCHECK(status_.ok());

  if (descriptor_heap != nullptr &&
      descriptor_heap != current_descriptor_heap_) {
    current_descriptor_heap_ = descriptor_heap;

    ID3D12DescriptorHeap* descriptor_heaps[] = {descriptor_heap};
    current_command_list_->SetDescriptorHeaps(ABSL_ARRAYSIZE(descriptor_heaps),
                                              descriptor_heaps);
  }
}

void DmlExecutionContextImpl::OnCommandRecorded() {
  // This should have been checked in one of the public functions before calling
  // OnCommandRecorded()
  DCHECK(status_.ok());

  ++operations_recorded_in_current_command_list_;
}

void DmlExecutionContextImpl::OpenCommandList() {
  // This should have been checked in one of the public functions before calling
  // OpenCommandList()
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

void DmlExecutionContextImpl::CloseCommandListAndExecute() {
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

  // Always keep the command list in an opened state
  OpenCommandList();
}

DmlGpuEvent DmlExecutionContext::CopyBufferRegion(
    ID3D12Resource* dst_buffer, uint64_t dst_offset,
    D3D12_RESOURCE_STATES dst_state, ID3D12Resource* src_buffer,
    uint64_t src_offset, D3D12_RESOURCE_STATES src_state, uint64_t byte_count) {
  std::unique_lock<std::mutex> lock(shared_state_->mutex);

  auto event = shared_state_->impl->GetCurrentCompletionEvent();
  ++event.fence_value;

  shared_state_->batched_functions.emplace_back([=]() {
    return shared_state_->impl->CopyBufferRegion(
        dst_buffer, dst_offset, dst_state, src_buffer, src_offset, src_state,
        byte_count);
  });

  shared_state_->new_function_enqueued.notify_all();

  return event;
}

// TODO: this can be batched as well for small byte counts (typical). Larger
// copies should flush the batch and execute synchronously.
DmlGpuEvent DmlExecutionContext::FillBufferWithPattern(
    ID3D12Resource* dst, uint64_t dst_offset, uint64_t dst_size_in_bytes,
    absl::Span<const uint8_t> value) {
  std::unique_lock<std::mutex> lock(shared_state_->mutex);

  auto event = shared_state_->impl->GetCurrentCompletionEvent();
  ++event.fence_value;

  absl::InlinedVector<uint8_t, 16> value_copy(value.size());
  std::copy(value.begin(), value.end(), value_copy.begin());

  shared_state_->batched_functions.emplace_back(
      [=, value_copy = std::move(value_copy)]() {
        shared_state_->impl->FillBufferWithPattern(
            dst, dst_offset, dst_size_in_bytes, value_copy);
      });

  return event;
}

DmlGpuEvent DmlExecutionContext::InitializeOperator(
    IDMLOperatorInitializer* initializer, IDMLBindingTable* binding_table,
    ID3D12DescriptorHeap* descriptor_heap) {
  std::unique_lock<std::mutex> lock(shared_state_->mutex);

  auto event = shared_state_->impl->GetCurrentCompletionEvent();
  ++event.fence_value;

  // The caller may not keep the binding table alive for longer than this
  // function call, so take a reference and transfer ownership to the lambda.
  Microsoft::WRL::ComPtr<IDMLBindingTable> binding_table_ref{binding_table};
  shared_state_->batched_functions.emplace_back(
      [=, binding_table = std::move(binding_table_ref)]() {
        shared_state_->impl->InitializeOperator(
            initializer, binding_table.Get(), descriptor_heap);
      });

  shared_state_->new_function_enqueued.notify_all();

  return event;
}

DmlGpuEvent DmlExecutionContext::ExecuteOperator(
    IDMLCompiledOperator* op, IDMLBindingTable* binding_table,
    ID3D12DescriptorHeap* descriptor_heap) {
  std::unique_lock<std::mutex> lock(shared_state_->mutex);

  auto event = shared_state_->impl->GetCurrentCompletionEvent();
  ++event.fence_value;

  // The caller may not keep the binding table alive for longer than this
  // function call, so take a reference and transfer ownership to the lambda.
  // TODO: consider r-value param to avoid unnecessary addref/release.
  Microsoft::WRL::ComPtr<IDMLBindingTable> binding_table_ref{binding_table};
  shared_state_->batched_functions.emplace_back(
      [=, binding_table = std::move(binding_table_ref)]() {
        shared_state_->impl->ExecuteOperator(op, binding_table.Get(),
                                             descriptor_heap);
      });

  shared_state_->new_function_enqueued.notify_all();

  return event;
}

DmlGpuEvent DmlExecutionContext::ResourceBarrier(
    absl::Span<const D3D12_RESOURCE_BARRIER> barriers) {
  std::unique_lock<std::mutex> lock(shared_state_->mutex);

  auto event = shared_state_->impl->GetCurrentCompletionEvent();
  ++event.fence_value;

  // The caller may not keep the barriers referenced by the span alive for
  // longer than this function call, so make a copy and transfer ownership to
  // the lambda.
  absl::InlinedVector<D3D12_RESOURCE_BARRIER, 4> barriers_copy;
  shared_state_->batched_functions.emplace_back(
      [=, barriers = std::move(barriers_copy)]() {
        shared_state_->impl->ResourceBarrier(barriers);
      });

  shared_state_->new_function_enqueued.notify_all();

  return event;
}

DmlGpuEvent DmlExecutionContext::UavBarrier() {
  std::unique_lock<std::mutex> lock(shared_state_->mutex);

  auto event = shared_state_->impl->GetCurrentCompletionEvent();
  ++event.fence_value;

  shared_state_->batched_functions.emplace_back(
      [=]() { shared_state_->impl->UavBarrier(); });

  shared_state_->new_function_enqueued.notify_all();

  return event;
}

StatusOr<DmlGpuEvent> DmlExecutionContext::Flush() {
  std::unique_lock<std::mutex> lock(shared_state_->mutex);
  return InvokeBatchedFunctionsAndExecute();
}

DmlGpuEvent DmlExecutionContext::GetCurrentCompletionEvent() {
  std::unique_lock<std::mutex> lock(shared_state_->mutex);
  auto event = shared_state_->impl->GetCurrentCompletionEvent();

  // If something has been batched but not submitted yet,
  // it means that the *next* fence value is the one to signal completion.
  if (!shared_state_->batched_functions.empty()) {
    ++event.fence_value;
  }

  return event;
}

StatusOr<DmlGpuEvent> DmlExecutionContext::InvokeBatchedFunctionsAndExecute() {
  for (auto& f : shared_state_->batched_functions) {
    f();
  }
  shared_state_->batched_functions.clear();
  return shared_state_->impl->Flush();
}

/*static*/ void DmlExecutionContext::ThreadProc(
    std::shared_ptr<SharedState> state) {
  auto last_flush = std::chrono::high_resolution_clock::now();

  while (true) {
    std::unique_lock<std::mutex> lock(state->mutex);
    if (state->exit_requested) {
      break;
    }

    std::chrono::duration<double> elapsed =
        std::chrono::high_resolution_clock::now() - last_flush;
    auto elapsed_ms = elapsed.count() * 1e3;

    if ((state->batched_functions.size() >= state->batch_flush_size_) ||
        (!state->batched_functions.empty() && elapsed_ms >= 1)) {
      for (auto& f : state->batched_functions) {
        f();
      }
      state->batched_functions.clear();
      state->impl->Flush();
      last_flush = std::chrono::high_resolution_clock::now();
    } else {
      // Wait for new functions
      state->new_function_enqueued.wait(lock);

      // No need for a loop around the wait() in case of spurious wakeup; just
      // return to the top. This also handles the case where exit is
      // requested.
      continue;
    }

    lock.unlock();
  }
}

}  // namespace tensorflow
