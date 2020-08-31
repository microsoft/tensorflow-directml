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

#include "tensorflow/core/common_runtime/dml/dml_kernel_manager.h"

#include "tensorflow/core/util/env_var.h"

namespace tensorflow {

static size_t GetMaxCacheSize() {
  int64 env_var = -1;
  Status s = ReadInt64FromEnvVar("TF_DIRECTML_KERNEL_CACHE_SIZE", -1, &env_var);

  if (s.ok() && env_var >= 0) {
    return env_var;
  }

  return DmlKernelManager::kDefaultMaxCacheSize;
}

DmlKernelManager::DmlKernelManager() : max_cache_size_(GetMaxCacheSize()) {}

void DmlKernelManager::TrimCache() const {
  assert(lru_list_.size() == kernel_cache_.size());

  if (kernel_cache_.size() <= max_cache_size_) {
    // The cache is within the maximum size; nothing to do
    return;
  }

  // Find the least-recently used element
  const DmlKernelKey* lru_key = lru_list_.back();

  VLOG(3) << "DmlKernelManager: evicting '" << lru_key->op_type_name
          << "' from cache, key=0x" << lru_key;

  // Erase it from the LRU list and the cache
  lru_list_.pop_back();
  size_t elements_removed = kernel_cache_.erase(*lru_key);

  assert(elements_removed == 1);
}

void DmlKernelManager::OnRecentlyUsed(const DmlKernelKey* key,
                                      CacheEntry* entry) const {
  if (entry->lru_iter == lru_list_.begin()) {
    return;  // This entry is already the most-recently used
  }

  // Remove it from the LRU list and re-insert it at the beginning (the list is
  // ordered from most-recently to least-recently used)
  lru_list_.erase(entry->lru_iter);
  lru_list_.push_front(key);
  entry->lru_iter = lru_list_.begin();
}

void DmlKernelManager::OnKernelCreation(const DmlKernelKey* key,
                                        DmlKernel* kernel) const {
  VLOG(3) << "DmlKernelManager: instantating '" << key->op_type_name
          << "' kernel, key=0x" << key << ", kernel=0x" << kernel;
}

void DmlKernelManager::QueueReference(std::shared_ptr<DmlKernel> kernel,
                                      DmlGpuEvent gpu_event) const {
  std::unique_lock<std::mutex> lock(mutex_);

  QueuedReference ref = {};
  ref.kernel = std::move(kernel);
  ref.gpu_event = std::move(gpu_event);
  queued_references_.push_back(std::move(ref));
}

void DmlKernelManager::ReleaseCompletedReferences() const {
  std::unique_lock<std::mutex> lock(mutex_);

  std::vector<QueuedReference> references_to_free;

  // Search the `queued_references_` for references that can be freed, and move
  // them into `references_to_free`. While searching for freeable references,
  // this loop also compacts the elements of `queued_references_` such that all
  // remaining valid objects are at the beginning of the vector. The invalid,
  // moved-from objects at the end are then erase()'d after the loop.
  auto dst = queued_references_.begin();
  for (auto it = dst; it != queued_references_.end(); ++it) {
    if (it->gpu_event.IsSignaled()) {
      // Move this reference into references_to_free
      references_to_free.push_back(std::move(*it));
    } else {
      // Compact the queued_references_ vector
      if (it != dst) {
        *dst = std::move(*it);
      }
      ++dst;
    }
  }

  queued_references_.erase(dst, queued_references_.end());
  lock.unlock();

  VLOG(2) << "DmlKernelManager: cleared " << references_to_free.size()
          << " references.";

  // Clearing this vector releases the references. This is done outside the
  // lock because kernel destructors are invoked when the reference count
  // reaches zero, and we don't want to hold locks over arbitrary code.
  references_to_free.clear();
}

size_t DmlKernelManager::GetCacheSize() const {
  std::unique_lock<std::mutex> lock(mutex_);
  return kernel_cache_.size();
}

void DmlKernelManager::ClearCache() {
  std::unique_lock<std::mutex> lock(mutex_);
  lru_list_.clear();
  kernel_cache_.clear();
}

}  // namespace tensorflow