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

#include "dml_common.h"
#include "dml_gpu_event.h"

namespace tensorflow {

// A fixed-size ring of command allocators. Each time an allocator is retrieved,
// the allocator will be reset if its previously recorded commands have finished
// executing on the GPU.
template <size_t allocator_count>
class DmlCommandAllocatorRing {
 public:
  DmlCommandAllocatorRing(ID3D12Device* device,
                          D3D12_COMMAND_LIST_TYPE command_list_type,
                          DmlGpuEvent initial_event) {
    for (auto& info : command_allocators_) {
      DML_CHECK_SUCCEEDED(device->CreateCommandAllocator(
          command_list_type, IID_PPV_ARGS(&info.allocator)));

      info.completion_event = initial_event;
    }
  }

  ID3D12CommandAllocator* GetNextAllocator(DmlGpuEvent next_completion_event) {
    size_t earliest_other_allocator =
        (current_command_allocator_ + 1) % allocator_count;

    assert(!command_allocators_[current_command_allocator_]
                .completion_event.IsSignaled() ||
           command_allocators_[earliest_other_allocator]
               .completion_event.IsSignaled());

    if (command_allocators_[earliest_other_allocator]
            .completion_event.IsSignaled()) {
      DML_CHECK_SUCCEEDED(
          command_allocators_[earliest_other_allocator].Get()->Reset());
      current_command_allocator_ = earliest_other_allocator;
    }

    // Set the completion event for the current allocator so it can be reset
    // eventually.
    command_allocators_[current_command_allocator_].completion_event =
        std::move(next_completion_event);

    return command_allocators_[current_command_allocator_].Get();
  }

 private:
  struct CommandAllocatorInfo {
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> allocator;

    // The event which will be signaled when the last command list submitted
    // using this allocator completes execution on the GPU.
    DmlGpuEvent completion_event = {};

    ID3D12CommandAllocator* Get() const { return allocator.Get(); }
  };

  std::array<CommandAllocatorInfo, allocator_count> command_allocators_;
  size_t current_command_allocator_ = 0;
};

}  // namespace tensorflow