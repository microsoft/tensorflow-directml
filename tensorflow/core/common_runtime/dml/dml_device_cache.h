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

#include "tensorflow/core/common_runtime/dml/dml_util.h"

namespace tensorflow {

// Maintains a static cache of device singletons, one per adapter. This class is
// thread-safe.
class DmlDeviceCache {
 public:
  static DmlDeviceCache& Instance();
  uint32_t GetAdapterCount() const;

  // It is a little odd that we require GPUOptions and memory_limit here, as
  // those can vary per TF device instance - they're not process-global. We
  // handle this by using the options and memory limit that are provided to the
  // first device created on this adapter. If subsequent devices are created on
  // the same adapter but with different options/memory_limit, they are ignored.
  // This is unusual, but matches the behavior of the CUDA device.
  const DmlDeviceState* GetOrCreateDeviceState(uint32_t adapter_index,
                                               const GPUOptions& gpu_options,
                                               uint64_t memory_limit_in_bytes);

  const DmlAdapter& GetAdapter(uint32_t adapter_index) const;

 private:
  DmlDeviceCache();

  mutable std::mutex mutex_;

  std::vector<DmlAdapter> adapters_;

  // Lazily constructed
  std::vector<std::unique_ptr<DmlDeviceState>> device_states_;
};

}  // namespace tensorflow