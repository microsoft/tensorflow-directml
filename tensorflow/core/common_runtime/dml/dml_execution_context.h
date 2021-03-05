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

#include <condition_variable>
#include <functional>
#include <thread>
#include <vector>

#include "dml_command_allocator_ring.h"
#include "dml_command_queue.h"
#include "dml_common.h"
#include "dml_descriptor_pool.h"
#include "dml_status.h"

namespace tensorflow {
class DmlAllocator;
class DmlCommandQueue;

// Asynchronously performs GPU work, and automatically manages command list
// recording and submission to queues. Work submitted to the DmlExecutionContext
// is typically recorded onto a command list and may not immediately begin
// execution on the GPU. Call Flush() to force all recorded work to be submitted
// to the command queue for execution on the GPU. This class is NOT thread-safe.
class DmlExecutionContextImpl {
 public:
  // Constructs an DmlExecutionContext that executes on the supplied queue.
  DmlExecutionContextImpl(ID3D12Device* d3d12_device, IDMLDevice* dml_device,
                          ID3D12CommandQueue* queue, DmlAllocator* allocator);

  // Waits for flushed work, discards unflushed work, and discards associated
  // references to prevent circular references. Must be the last call on the
  // object before destruction.
  void Close();

  // Queues a CopyBufferRegion (see ID3D12GraphicsCommandList::CopyBufferRegion)
  // for execution. Transition barriers are automatically inserted to transition
  // the source and destination resources to COPY_SOURCE and COPY_DEST if
  // necessary.
  DmlGpuEvent CopyBufferRegion(ID3D12Resource* dst_buffer, uint64_t dst_offset,
                               D3D12_RESOURCE_STATES dst_state,
                               ID3D12Resource* src_buffer, uint64_t src_offset,
                               D3D12_RESOURCE_STATES src_state,
                               uint64_t byte_count);

  DmlGpuEvent FillBufferWithPattern(
      ID3D12Resource* dst, uint64_t dst_offset, uint64_t dst_size_in_bytes,
      absl::Span<const uint8_t>
          value /* Data type agnostic value, treated as raw bits */);

  DmlGpuEvent InitializeOperator(IDMLOperatorInitializer* initializer,
                                 IDMLBindingTable* binding_table,
                                 ID3D12DescriptorHeap* descriptor_heap);

  DmlGpuEvent ExecuteOperator(IDMLCompiledOperator* op,
                              IDMLBindingTable* binding_table,
                              ID3D12DescriptorHeap* descriptor_heap);

  DmlGpuEvent ResourceBarrier(
      absl::Span<const D3D12_RESOURCE_BARRIER> barriers);

  DmlGpuEvent UavBarrier();

  // Forces all queued work to begin executing on the GPU. This method returns
  // immediately and does not wait for the submitted work to complete execution
  // on the GPU.
  StatusOr<DmlGpuEvent> Flush();

  Status GetCommandRecorderStatus() const { return status_; }

  // Returns an event which will become signaled when everything submitted to
  // the execution context thus far has completed execution on the GPU,
  // including work that has yet to be flushed to the queue.
  DmlGpuEvent GetCurrentCompletionEvent();

  D3D12_COMMAND_LIST_TYPE GetCommandListTypeForQueue() const;

 private:
  Microsoft::WRL::ComPtr<ID3D12Device> d3d_device_;

  std::shared_ptr<DmlCommandQueue> queue_;

  bool closed_ = false;

  Microsoft::WRL::ComPtr<IDMLDevice> dml_device_;
  Microsoft::WRL::ComPtr<IDMLCommandRecorder> recorder_;

  // Descriptors are allocated from a pool. The current heap pointer is only
  // used to avoid redundantly setting the same heap; it does not have ownership
  // of the heap object.
  DmlDescriptorPool descriptor_pool_;
  ID3D12DescriptorHeap* current_descriptor_heap_ = nullptr;

  DmlAllocator* allocator_ = nullptr;
  DmlCommandAllocatorRing<2> command_allocator_ring_;

  // The command list currently being recorded into, and whether any command
  // have been recorded yet.
  Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> current_command_list_;
  uint32_t operations_recorded_in_current_command_list_ = 0;

  // A pool of cached command lists which may be re-used.
  std::deque<Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList>>
      cached_command_lists_;

  // Status of the first error encountered when closing the command list.
  // Operations that flush the command list or readback from the GPU should make
  // sure that this status doesn't contain an error before doing so.
  Status status_ = Status::OK();

  void SetDescriptorHeap(ID3D12DescriptorHeap* descriptor_heap);

  // Increments operations_recorded_in_current_command_list_. If the size of the
  // current command list exceeds a certain value (based on heuristic), the
  // command list is flushed.
  void OnCommandRecorded();

  void OpenCommandList();
  void CloseCommandListAndExecute();
};

// A thread-safe wrapper over DmlExecutionContextImpl. Calls to this class are
// batched to minimize work within the lock, and the batched calls are
// periodically flushed by a background thread (or by explicitly calling Flush).
class DmlExecutionContext {
 public:
  DmlExecutionContext(ID3D12Device* d3d12_device, IDMLDevice* dml_device,
                      ID3D12CommandQueue* queue, DmlAllocator* allocator);

  ~DmlExecutionContext();

  void Close() {
    std::unique_lock<std::mutex> lock(shared_state_->mutex);
    shared_state_->impl->Close();
  }

  // NOTE: the caller is responsible for keeping the resources alive until the
  // returned GPU event has completed.
  DmlGpuEvent CopyBufferRegion(ID3D12Resource* dst_buffer, uint64_t dst_offset,
                               D3D12_RESOURCE_STATES dst_state,
                               ID3D12Resource* src_buffer, uint64_t src_offset,
                               D3D12_RESOURCE_STATES src_state,
                               uint64_t byte_count);

  // NOTE: the caller is responsible for keeping the resources alive until the
  // returned GPU event has completed.
  DmlGpuEvent FillBufferWithPattern(ID3D12Resource* dst, uint64_t dst_offset,
                                    uint64_t dst_size_in_bytes,
                                    absl::Span<const uint8_t> value);

  // NOTE: the caller is responsible for keeping the initializer and
  // descriptor_heap alive until the returned GPU event has completed.
  DmlGpuEvent InitializeOperator(IDMLOperatorInitializer* initializer,
                                 Microsoft::WRL::ComPtr<IDMLBindingTable>&& binding_table,
                                 ID3D12DescriptorHeap* descriptor_heap);

  // NOTE: the caller is responsible for keeping the op and descriptor_heap
  // alive until the returned GPU event has completed.
  DmlGpuEvent ExecuteOperator(IDMLCompiledOperator* op,
                              Microsoft::WRL::ComPtr<IDMLBindingTable>&& binding_table,
                              ID3D12DescriptorHeap* descriptor_heap);

  DmlGpuEvent ResourceBarrier(
      absl::Span<const D3D12_RESOURCE_BARRIER> barriers);

  // A slightly more efficient version of ResourceBarrier when the barrier span
  // only includes a UAV barrier (elides an extra copy).
  DmlGpuEvent UavBarrier();

  StatusOr<DmlGpuEvent> Flush();

  Status GetCommandRecorderStatus() const {
    return shared_state_->impl->GetCommandRecorderStatus();
  }

  DmlGpuEvent GetCurrentCompletionEvent();

  D3D12_COMMAND_LIST_TYPE GetCommandListTypeForQueue() const {
    return shared_state_->impl->GetCommandListTypeForQueue();
  }

 private:
  static constexpr uint32_t default_batch_flush_size = 100;

  struct SharedState {
    std::mutex mutex;
    uint32_t batch_flush_size_ = default_batch_flush_size;
    std::unique_ptr<DmlExecutionContextImpl> impl;
    std::condition_variable new_function_enqueued;
    absl::InlinedVector<std::function<void()>, default_batch_flush_size>
        batched_functions;
    bool exit_requested = false;
  };

  std::shared_ptr<SharedState> shared_state_;
  std::thread thread_;

  static void ThreadProc(std::shared_ptr<SharedState> state);

  StatusOr<DmlGpuEvent> InvokeBatchedFunctionsAndExecute();
};

}  // namespace tensorflow
