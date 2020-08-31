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

DmlEventQueue::DmlEventQueue() {
  shared_state_ = std::make_shared<SharedState>();

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

void DmlEventQueue::Enqueue(DmlGpuEvent gpu_event,
                            std::function<void()> done_callback) {
  std::unique_lock<std::mutex> lock(shared_state_->mutex);
  shared_state_->events.push({std::move(gpu_event), std::move(done_callback)});
  shared_state_->new_event_enqueued.notify_all();
}

/*static*/ void DmlEventQueue::ThreadProc(std::shared_ptr<SharedState> state) {
  while (true) {
    std::unique_lock<std::mutex> lock(state->mutex);
    if (state->exit_requested) {
      break;
    }

    if (state->events.empty()) {
      // Wait for new events
      state->new_event_enqueued.wait(lock);

      // No need for a loop around the wait() in case of spurious wakeup; just
      // return to the top. This also handles the case where exit is
      // requested.
      continue;
    }

    assert(!state->events.empty());
    Event event = std::move(state->events.front());
    state->events.pop();

    // We've taken ownership of the event, which means we can now unlock the
    // shared state
    lock.unlock();

    // Handle the event by blocking until it's signaled, then invoking the
    // "done" callback
    event.gpu_event.WaitForSignal();
    event.done_callback();
  }
}

}  // namespace tensorflow
