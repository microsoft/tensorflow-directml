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

#include <memory>

#include "tensorflow/core/common_runtime/dml/dml_common.h"
#include "tensorflow/core/common_runtime/dml/dml_gpu_event.h"
#include "tensorflow/core/common_runtime/dml/dml_kernel_context.h"
#include "tensorflow/core/common_runtime/dml/dml_kernel_key.h"

namespace tensorflow {

class DmlKernel;
class DmlKernelConstruction;
class NoOpInitializationHelper;

// Creates, caches, and manages GPU lifetime of DirectML operator kernel
// instances. Note that this class manages instances of DmlKernel-derived
// objects, which are different from tensorflow::OpKernels. The DmlKernelWrapper
// (which does implement OpKernel) uses the kernel manager to find/create the
// correct DmlKernel to execute in OpKernel::Compute.
//
// The major difference between DmlKernel-derived objects and OpKernel-derived
// objects is that DmlKernel offers a much simplified interface. Implementers of
// DmlKernels may assume that:
//   - All shapes are always static, and available at kernel construction
//   - Computation of output shapes is done for you
//   - Constant CPU tensors have their contents available at kernel construction
//
// The DmlKernelWrapper and DmlKernelManager abstract away all of the
// boilerplate of worrying about dynamic shapes, caching DML operators, managing
// lifetime on the GPU timeline, etc.
//
// This class is thread-safe.
class DmlKernelManager {
 public:
  // Can be overridden by the TF_DIRECTML_KERNEL_CACHE_SIZE environment variable
  static constexpr size_t kDefaultMaxCacheSize = 1536;

  DmlKernelManager();

  template <typename TKernel>
  std::shared_ptr<TKernel> CreateCachedKernel(
      DmlKernelConstruction* ctx, const DmlKernelKey& key,
      const typename TKernel::InitHelper* init_helper) const {
    static_assert(std::is_base_of<DmlKernel, TKernel>::value,
                  "Kernel type does not inherit from DmlKernel");

    // Create a new kernel. Because this can potentially be
    // slow, we don't hold the lock over the kernel creation.
    auto kernel = std::make_shared<TKernel>(ctx, init_helper);
    OnKernelCreation(&key, kernel.get());

    // Make a deep copy of the key so that we own the memory
    auto key_copy = key.Clone();

    std::unique_lock<std::mutex> lock(mutex_);

    CacheEntry entry = {};
    entry.kernel = kernel;

    // Another thread may have already inserted an instance of this kernel
    // into the cache while we weren't holding the lock. That's okay; in this
    // case, the .emplace() is a no-op and the kernel will not be cached.
    auto result = kernel_cache_.emplace(key_copy, std::move(entry));

    // Retrieve the iterator to the newly-inserted element, or the existing
    // element if another thread beat us to it.
    auto it = result.first;

    bool insertion_succeeded = result.second;
    if (insertion_succeeded) {
      // If this was a newly-inserted cache entry, also add an LRU entry
      lru_list_.push_front(&it->first);
      it->second.lru_iter = lru_list_.begin();
    }

    // Update the LRU cache
    OnRecentlyUsed(&it->first, &it->second);

    if (insertion_succeeded) {
      TrimCache();
    }

    return kernel;
  }

  template <typename TKernel>
  std::shared_ptr<TKernel> TryGetCachedKernel(const DmlKernelKey& key) const {
    static_assert(std::is_base_of<DmlKernel, TKernel>::value,
                  "Kernel type does not inherit from DmlKernel");

    std::unique_lock<std::mutex> lock(mutex_);

    auto it = kernel_cache_.find(key);

    if (it == kernel_cache_.end()) {
      return nullptr;
    }

    // Update the LRU cache
    OnRecentlyUsed(&it->first, &it->second);

    auto kernel = std::static_pointer_cast<TKernel>(it->second.kernel);
    return kernel;
  }

  // Ensures that a reference is maintained on a kernel at least until the given
  // GPU event enters the signaled state.
  void QueueReference(std::shared_ptr<DmlKernel> kernel,
                      DmlGpuEvent gpu_event) const;

  // Releases all shared_ptrs supplied to QueueReference which have had their
  // GPU event signaled.
  void ReleaseCompletedReferences() const;

  // Returns the number of cached kernels.
  size_t GetCacheSize() const;

  // Frees all cached kernels which have completed execution on the GPU.
  void ClearCache();

 private:
  // A non-owning pointer to the key for the kernel which is used to keep track
  // of the least-recently-used kernel. This is a pointer into a kernel_cache_
  // element. This is okay because std::unordered_map is guaranteed never to
  // invalidate pointers/references to elements.
  using LruEntry = const DmlKernelKey*;

  struct CacheEntry {
    std::shared_ptr<DmlKernel> kernel;

    // An iterator into the lru_list_. The position of this iterator in the list
    // indicates how recently used this cache entry is.
    std::list<LruEntry>::iterator lru_iter;
  };

  struct QueuedReference {
    std::shared_ptr<DmlKernel> kernel;
    DmlGpuEvent gpu_event;
  };

  // Trims the cache by least recently used until it's below the max cache size.
  void TrimCache() const;

  // Marks the cache entry as being recently used, for the purposes of the LRU
  // cache. `key` and `entry` must be pointers into elements of the
  // kernel_cache_ map.
  void OnRecentlyUsed(const DmlKernelKey* key, CacheEntry* entry) const;

  void OnKernelCreation(const DmlKernelKey* key, DmlKernel* kernel) const;

  mutable std::mutex mutex_;
  const size_t max_cache_size_;

  // All of these members are protected by mutex_

  mutable std::unordered_map<DmlKernelKey, CacheEntry, hash<DmlKernelKey>>
      kernel_cache_;

  // Ordered by most-recently to least-recently used.
  mutable std::list<LruEntry> lru_list_;

  mutable std::vector<QueuedReference> queued_references_;
};

}  // namespace tensorflow