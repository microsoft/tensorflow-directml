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

#include "dml_event_queue.h"

namespace tensorflow {

DmlEventQueue::DmlEventQueue(ID3D12Fence* fence) {
  shared_state_ = std::make_shared<SharedState>();
  shared_state_->fence = fence;

  // Launch the thread, supplying it with a pointer to the shared state
  thread_ = std::thread(ThreadProc, shared_state_);
}

DmlEventQueue::~DmlEventQueue() {
  // Request exit of the background thread
  std::unique_lock<std::mutex> lock(shared_state_->mutex);
  shared_state_->exit_requested = true;
  shared_state_->new_event_enqueued.notify_all();  // wake the thread
  lock.unlock();

  // detach() rather than join(), because we don't want (or need) to wait for
  // it to complete. This prevents blocking in a destructor, which would be
  // bad.
  thread_.detach();
}

void DmlEventQueue::Enqueue(DmlGpuEvent gpu_event, DoneCallback done_callback) {
  std::unique_lock<std::mutex> lock(shared_state_->mutex);
  const auto& state = shared_state_;

  // Sanity: double check that we're only using one fence at a time, because
  // otherwise monotonically increasing fence values no longer describe a
  // linear timeline...
  CHECK(state->fence.Get() == gpu_event.fence.Get());
  CHECK(state->fence->GetCompletedValue() + 1 >=
        state->current_awaited_fence_value);

  // If the thread is currently waiting on a fence value less or equal than the
  // supplied gpu_event, it means the fence may or may not be signaled, and
  // therefore should be queued to be processed by the thread. Otherwise, if the
  // fence value is greater, it means it's definitely already been signaled and
  // the callback should be invoked immediately.
  if (state->current_awaited_fence_value <= gpu_event.fence_value) {
    // Queue the event and notify the thread.
    state->events_by_fence_value.emplace(gpu_event.fence_value,
                                         Event{std::move(done_callback)});
    state->new_event_enqueued.notify_all();
  } else {
    // Sanity: all prior events in the queue should have been processed already
    DCHECK(state->events_by_fence_value.lower_bound(gpu_event.fence_value) ==
           state->events_by_fence_value.begin());

    done_callback();
  }
}

/*static*/ void DmlEventQueue::ThreadProc(std::shared_ptr<SharedState> state) {
  std::vector<Event> events_to_process;

  while (true) {
    std::unique_lock<std::mutex> lock(state->mutex);
    if (state->exit_requested) {
      break;
    }

    if (state->events_by_fence_value.empty()) {
      // Wait for new events
      state->new_event_enqueued.wait(lock);

      // No need for a loop around the wait() in case of spurious wakeup; just
      // return to the top. This also handles the case where exit is
      // requested.
      continue;
    }

    DCHECK(!state->events_by_fence_value.empty());
    DCHECK(events_to_process.empty());

    // Decide the next fence value to wait on. Using GetCompletedValue + 1
    // ensures that *any* signal wakes this thread (assuming monotonically
    // increasing fence values, which we require.)
    //
    // Note that because fences are asynchronous, this next_fence_value could
    // become signaled at any time (even immediately). This is okay; it just
    // means that the wait we perform below will return immediately, and the
    // loop will continue.
    uint64_t next_fence_value = state->fence->GetCompletedValue() + 1;
    state->current_awaited_fence_value = next_fence_value;

    // Find all the events that have fence values < next_fence_values. These
    // events, by definition, have had their fence values signaled and can now
    // be processed.
    auto begin = state->events_by_fence_value.begin();
    auto end = state->events_by_fence_value.lower_bound(next_fence_value);

    // Move the signaled events into the vector
    events_to_process.reserve(std::distance(begin, end));
    for (auto it = begin; it != end; ++it) {
      DCHECK(it->first < next_fence_value);
      events_to_process.push_back(std::move(it->second));
    }
    state->events_by_fence_value.erase(begin, end);

    // Process the events by invoking their done callback
    for (const auto& event : events_to_process) {
      event.done_callback();
    }

    // We've finished processing the events, so we can unlock the mutex now.
    lock.unlock();

    events_to_process.clear();

    // Now wait for the fence to become signaled again. Recall that the choice
    // of next_fence_value ensures this wait will complete no matter what value
    // is signaled.
    DmlGpuEvent next_event{next_fence_value, state->fence};
    next_event.WaitForSignal();

    // We require monotonically increasing fence values; time is not allowed to
    // go backward!
    CHECK(state->fence->GetCompletedValue() >= next_fence_value);
  }
}

}  // namespace tensorflow
