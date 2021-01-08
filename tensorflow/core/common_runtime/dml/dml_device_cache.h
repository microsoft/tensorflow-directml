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

  // Parts of the grappler will look up physical hardware properties using a TF
  // device specification (e.g. '/device:DML:0'). However, TF device IDs are not
  // directly equivalent to the adapter index. Device IDs to adapter index
  // mappings are influenced by two separate mechanisms:
  //
  // 1. ConfigProto.gpu_options.visible_device_list : can hide or permute the
  //    order of adapter indices. For example, "1,0" maps "DML:0" to
  //    adapter index 1 and "DML:1" to adapter index 0.
  //
  // 2. ConfigProto.gpu_options.experimental.virtual_devices : can partition a
  //    single physical adapter into multiple logical devices. For example,
  //    "DML:0" and "DML:1" may both use the same adapter 0.
  //
  // These two methods serve as a way to translate a device ID to the correct
  // adapter index, but there is a problem: the visible_device_list is scoped to
  // a TF session, and a single process can have multiple sessions. This would
  // be manageable if sessions had unique IDs and the grappler code propagated
  // this ID to its hardware lookup functions, but alas this is not the case.
  // Consequently, these helpers are on a process-wide singleton and they may
  // fail if multiple sessions use different visible device lists (same as GPU
  // helpers).
  //
  // NOTE: the virtual_devices can only be configured once, so they're
  // effectively scoped to the process and do not cause problems with this
  // singleton approach.
  Status MapDeviceIdToAdapterIndex(int device_id, uint32_t adapter_index);
  Status GetAdapterIndexFromDeviceId(int device_id, uint32_t* adapter_index);

 private:
  DmlDeviceCache();

  mutable std::mutex mutex_;

  std::vector<DmlAdapter> adapters_;

  using TDeviceToAdapterMap = std::unordered_map<int, uint32_t>;
  TDeviceToAdapterMap device_id_to_adapter_index_map_;

  // Lazily constructed
  std::vector<std::unique_ptr<DmlDeviceState>> device_states_;
};

}  // namespace tensorflow