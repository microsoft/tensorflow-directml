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

#include <queue>
#include <thread>
#include <condition_variable>
#include <functional>

#include "dml_common.h"
#include "dml_gpu_event.h"

namespace tensorflow {

// Allows for queueing CPU work in response to a signaled GPU event.
// This class is thread-safe.
class DmlEventQueue {
 public:
  DmlEventQueue();
  ~DmlEventQueue();

  // Enqueues an arbitrary callback to fire once the given GPU event becomes
  // signaled. The callback is invoked asynchronously, on an arbitrary thread.
  void Enqueue(DmlGpuEvent gpu_event, std::function<void()> done_callback);

 private:
  struct Event {
    DmlGpuEvent gpu_event;
    std::function<void()> done_callback;
  };

  // State shared with the background thread. Protected by `mutex`.
  struct SharedState {
    std::mutex mutex;
    std::condition_variable new_event_enqueued;
    std::queue<Event> events;
    bool exit_requested = false;
  };

  static void ThreadProc(std::shared_ptr<SharedState> state);

  std::shared_ptr<SharedState> shared_state_;
  std::thread thread_;
};

}  // namespace tensorflow
