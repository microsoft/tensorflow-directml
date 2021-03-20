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
  dml_command_queue_ = std::make_shared<DmlCommandQueue>(queue);

  batch_state_ = std::make_shared<BatchState>();
  batch_state_->next_flush_event =
      dml_command_queue_->GetCurrentCompletionEvent();
  ++batch_state_->next_flush_event.fence_value;

  uint32_t batch_flush_size = default_batch_flush_size;
  {
    int64 batch_flush_size_int64 = 0;
    Status s = ReadInt64FromEnvVar("TF_DIRECTML_BATCH_FLUSH_SIZE", 0,
                                   &batch_flush_size_int64);
    if (s.ok() && batch_flush_size_int64 != 0) {
      batch_flush_size = static_cast<uint32_t>(batch_flush_size_int64);
    }
  }

  uint32_t batch_flush_time_us = default_batch_flush_time_us;
  {
    int64 batch_flush_time_us_int64 = 0;
    Status s = ReadInt64FromEnvVar("TF_DIRECTML_BATCH_FLUSH_TIME", 0,
                                   &batch_flush_time_us_int64);
    if (s.ok() && batch_flush_time_us_int64 != 0) {
      batch_flush_time_us = static_cast<uint32_t>(batch_flush_time_us_int64);
    }
  }

  dml_command_list_ = std::make_shared<DmlCommandList>(
      d3d_device, dml_device, dml_command_queue_->GetType(), allocator);

  execution_thread_ =
      std::thread(ExecutionThreadProc, batch_state_, dml_command_list_,
                  dml_command_queue_, batch_flush_size, batch_flush_time_us);
}

DmlExecutionContext::~DmlExecutionContext() {
  // Request exit of the background thread
  std::unique_lock<std::mutex> lock(batch_state_->mutex);
  batch_state_->exit_requested = true;
  batch_state_->command_added.notify_all();  // wake the thread
  lock.unlock();

  // detach() rather than join(), because we don't want (or need) to wait for
  // it to complete. This prevents blocking in a destructor, which would be
  // bad.
  execution_thread_.detach();
}

DmlGpuEvent DmlExecutionContext::CopyBufferRegion(
    ID3D12Resource* dst_buffer, uint64_t dst_offset,
    D3D12_RESOURCE_STATES dst_state, ID3D12Resource* src_buffer,
    uint64_t src_offset, D3D12_RESOURCE_STATES src_state, uint64_t byte_count) {
  std::unique_lock<std::mutex> lock(batch_state_->mutex);

  batch_state_->WriteBatch().emplace_back([=](DmlCommandList& command_list) {
    command_list.CopyBufferRegion(dst_buffer, dst_offset, dst_state, src_buffer,
                                  src_offset, src_state, byte_count);
  });

  batch_state_->command_added.notify_all();

  return batch_state_->next_flush_event;
}

DmlGpuEvent DmlExecutionContext::FillBufferWithPattern(
    ID3D12Resource* dst, uint64_t dst_offset, uint64_t dst_size_in_bytes,
    absl::Span<const uint8_t> value) {
  std::unique_lock<std::mutex> lock(batch_state_->mutex);

  absl::InlinedVector<uint8_t, 16> value_copy(value.begin(), value.end());
  batch_state_->WriteBatch().emplace_back(
      [=, value = std::move(value_copy)](DmlCommandList& command_list) {
        command_list.FillBufferWithPattern(dst, dst_offset, dst_size_in_bytes,
                                           value);
      });

  batch_state_->command_added.notify_all();

  return batch_state_->next_flush_event;
}

DmlGpuEvent DmlExecutionContext::InitializeOperator(
    IDMLOperatorInitializer* initializer,
    Microsoft::WRL::ComPtr<IDMLBindingTable>&& binding_table,
    ID3D12DescriptorHeap* descriptor_heap) {
  std::unique_lock<std::mutex> lock(batch_state_->mutex);

  batch_state_->WriteBatch().emplace_back(
      [=,
       binding_table = std::move(binding_table)](DmlCommandList& command_list) {
        command_list.InitializeOperator(initializer, binding_table.Get(),
                                        descriptor_heap);
      });

  batch_state_->command_added.notify_all();

  return batch_state_->next_flush_event;
}

DmlGpuEvent DmlExecutionContext::ExecuteOperator(
    IDMLCompiledOperator* op,
    Microsoft::WRL::ComPtr<IDMLBindingTable>&& binding_table,
    ID3D12DescriptorHeap* descriptor_heap) {
  std::unique_lock<std::mutex> lock(batch_state_->mutex);

  batch_state_->WriteBatch().emplace_back(
      [=,
       binding_table = std::move(binding_table)](DmlCommandList& command_list) {
        command_list.ExecuteOperator(op, binding_table.Get(), descriptor_heap);
      });

  batch_state_->command_added.notify_all();

  return batch_state_->next_flush_event;
}

DmlGpuEvent DmlExecutionContext::ResourceBarrier(
    absl::Span<const D3D12_RESOURCE_BARRIER> barriers) {
  std::unique_lock<std::mutex> lock(batch_state_->mutex);

  // The caller may not keep the barriers referenced by the span alive for
  // longer than this function call, so make a copy and transfer ownership to
  // the lambda.
  absl::InlinedVector<D3D12_RESOURCE_BARRIER, 4> barriers_copy(barriers.begin(),
                                                               barriers.end());
  batch_state_->WriteBatch().emplace_back(
      [=, barriers = std::move(barriers_copy)](DmlCommandList& command_list) {
        command_list.ResourceBarrier(barriers);
      });

  batch_state_->command_added.notify_all();

  return batch_state_->next_flush_event;
}

DmlGpuEvent DmlExecutionContext::UavBarrier() {
  std::unique_lock<std::mutex> lock(batch_state_->mutex);

  batch_state_->WriteBatch().emplace_back(
      [=](DmlCommandList& command_list) { command_list.UavBarrier(); });

  batch_state_->command_added.notify_all();

  return batch_state_->next_flush_event;
}

StatusOr<DmlGpuEvent> DmlExecutionContext::Flush() {
  std::unique_lock<std::mutex> lock(batch_state_->mutex);
  auto event = batch_state_->next_flush_event;
  if (batch_state_->WriteBatch().empty()) {
    --event.fence_value;
  }

  batch_state_->flush_requested = true;
  batch_state_->command_added.notify_all();
  return event;
}

Status DmlExecutionContext::GetCommandRecorderStatus() const {
  std::unique_lock<std::mutex> lock(batch_state_->mutex);
  return batch_state_->status;
}

DmlGpuEvent DmlExecutionContext::GetCurrentCompletionEvent() {
  std::unique_lock<std::mutex> lock(batch_state_->mutex);
  auto event = batch_state_->next_flush_event;
  if (batch_state_->WriteBatch().empty()) {
    --event.fence_value;
  }
  return event;
}

D3D12_COMMAND_LIST_TYPE DmlExecutionContext::GetCommandListTypeForQueue()
    const {
  // No need to acquire the lock since the queue type is immutable once the
  // queue is constructed.
  return dml_command_queue_->GetType();
}

/*static*/ void DmlExecutionContext::ExecutionThreadProc(
    std::shared_ptr<BatchState> state,
    std::shared_ptr<DmlCommandList> command_list,
    std::shared_ptr<DmlCommandQueue> command_queue, uint32_t batch_flush_size,
    uint32_t batch_flush_time_us) {
#if _WIN32
  SetThreadDescription(GetCurrentThread(), L"TFDML Execution Thread");
#endif

  auto last_flush_time = std::chrono::steady_clock::now();

  while (true) {
    std::chrono::duration<double> elapsed =
        std::chrono::steady_clock::now() - last_flush_time;
    auto elapsed_us = elapsed.count() * 1e6;

    std::unique_lock<std::mutex> lock(state->mutex);
    if (state->exit_requested) {
      break;
    }

    auto& batch = state->WriteBatch();

    if (batch.empty()) {
      // Wait for new work to be batched.
      state->command_added.wait(lock);

      // Return to the top in case of spurious wakeup.
      continue;
    }

    // Check if it's time to swap the write/execute batches and flush work to
    // the GPU: this occurs if a flush is explicitly requested, the batch has
    // reached a certain size, or enough time has elapsed since the last flush.
    // The goal here is to balance feeding the GPU work while the CPU is
    // processing more commands and avoiding many small packets.
    bool flush = false;
    DmlGpuEvent batch_completion_event = state->next_flush_event;
    if (state->flush_requested || batch.size() >= batch_flush_size ||
        elapsed_us >= batch_flush_time_us) {
      state->write_batch_index = (state->write_batch_index + 1) % 2;
      flush = true;
      ++state->next_flush_event.fence_value;
    }
    state->flush_requested = false;

    // Unlock to allow kernels to resume writing to the new write batch.
    lock.unlock();

    if (flush) {
      DmlTracing::Instance().LogExecutionContextFlush();
      // Record the commands into the command list.
      command_list->Open(batch_completion_event);
      for (auto& command : batch) {
        command(*command_list);
      }
      auto status = command_list->Close();

      if (!status.ok()) {
        lock.lock();
        state->status = status;
        lock.unlock();
        break;
      }

      ID3D12CommandList* command_lists[] = {command_list->Get()};
      command_queue->ExecuteCommandLists(command_lists);

      batch.clear();
      last_flush_time = std::chrono::steady_clock::now();
    }
  }
}

}  // namespace tensorflow
