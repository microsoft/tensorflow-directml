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
#include <map>
#include <thread>

#include "dml_common.h"
#include "dml_gpu_event.h"

namespace tensorflow {

// Allows for queueing CPU work in response to a signaled GPU event. Each
// instance of this queue can only be used with a single fence, and the fence's
// signaled values are assumed to only ever increase in a monotonic fashion.
// This class is thread-safe.
class DmlEventQueue {
 public:
  using DoneCallback = std::function<void()>;

  explicit DmlEventQueue(ID3D12Fence* fence);
  ~DmlEventQueue();

  // Enqueues an arbitrary callback to fire once the given GPU event becomes
  // signaled. The callback is invoked asynchronously, on an arbitrary thread.
  // If there are multiple callbacks enqueued for a single fence value, those
  // callbacks are executed in the order they were queued. This method is
  // thread-safe.
  void Enqueue(DmlGpuEvent gpu_event, DoneCallback done_callback);

 private:
  struct Event {
    DoneCallback done_callback;
  };

  // State shared with the background thread. Protected by `mutex`.
  struct SharedState {
    // The fence associated with this queue.
    Microsoft::WRL::ComPtr<ID3D12Fence> fence;
    std::mutex mutex;
    std::condition_variable new_event_enqueued;  // An event that fires whenever
                                                 // a new event is added.
    std::multimap<uint64_t, Event> events_by_fence_value;

    // The current fence value that the thread is waiting to be signaled. This
    // value is guaranteed to be <= fence->GetCompletedValue()+1.
    uint64_t current_awaited_fence_value = 0;

    bool exit_requested = false;
  };

  static void ThreadProc(std::shared_ptr<SharedState> state);

  std::shared_ptr<SharedState> shared_state_;
  std::thread thread_;
};

}  // namespace tensorflow
