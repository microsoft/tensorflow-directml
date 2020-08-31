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

#include <random>
#include <thread>

#include "tensorflow/core/common_runtime/dml/dml_adapter.h"
#include "tensorflow/core/common_runtime/dml/dml_adapter_impl.h"
#include "tensorflow/core/common_runtime/dml/dml_operator_helper.h"
#include "tensorflow/core/common_runtime/dml/dml_util.h"
#include "tensorflow/core/kernels/dml_ops_common.h"
#include "tensorflow/core/platform/test.h"

using Microsoft::WRL::ComPtr;

namespace tensorflow {

class DmlKernelManagerTest : public ::testing::Test {
 public:
  DmlKernelManagerTest()
      : ctx_(nullptr, nullptr, nullptr, nullptr, {}, nullptr),
        init_helper_(nullptr, nullptr),
        kernel_manager_() {}

 protected:
  DmlKernelKey CreateKey(const char* identifier) {
    DmlKernelKey key;
    key.op_type_name = identifier;
    key.node_def = std::make_shared<NodeDef>();
    return key;
  }

  DmlKernelConstruction ctx_;
  const NoOpInitializationHelper init_helper_;
  DmlKernelManager kernel_manager_;
};

class DmlMockKernel : public DmlKernel {
 public:
  using InitHelper = NoOpInitializationHelper;

  explicit DmlMockKernel(DmlKernelConstruction* ctx,
                         const InitHelper* init_helper) {}
};

TEST_F(DmlKernelManagerTest, Cache) {
  // Cache should start out empty
  EXPECT_TRUE(kernel_manager_.GetCacheSize() == 0);

  DmlKernelKey key0 = CreateKey("key0");
  DmlKernelKey key1 = CreateKey("key1");

  // Creating a kernel should cache it
  auto kernel0 = kernel_manager_.CreateCachedKernel<DmlMockKernel>(
      &ctx_, key0, &init_helper_);
  EXPECT_TRUE(kernel_manager_.GetCacheSize() == 1);

  // Retrieving the same kernel key a second time should hit the cache and
  // return the same object
  auto kernel0_copy = kernel_manager_.TryGetCachedKernel<DmlMockKernel>(key0);
  EXPECT_TRUE(kernel0 == kernel0_copy);

  // Creating a second kernel should increase the cache size
  auto kernel1 = kernel_manager_.CreateCachedKernel<DmlMockKernel>(
      &ctx_, key1, &init_helper_);
  EXPECT_TRUE(kernel_manager_.GetCacheSize() == 2);
  EXPECT_TRUE(kernel0 != kernel1);

  // Clearing the cache should reset its size to 0
  kernel_manager_.ClearCache();
  EXPECT_TRUE(kernel_manager_.GetCacheSize() == 0);

  // Now that the cache has been cleared, ensure that retrieving an existing key
  // results in a different object
  auto kernel1_new = kernel_manager_.CreateCachedKernel<DmlMockKernel>(
      &ctx_, key1, &init_helper_);
  EXPECT_TRUE(kernel1_new != kernel1);
}

// Ensure that the kernel manager never exceeds its maximum size
TEST_F(DmlKernelManagerTest, CacheEviction) {
  static const size_t kKernelCount = DmlKernelManager::kDefaultMaxCacheSize * 2;

  for (size_t i = 0; i < kKernelCount; ++i) {
    std::string key_name = std::to_string(i);
    DmlKernelKey key = CreateKey(key_name.c_str());

    auto kernel = kernel_manager_.CreateCachedKernel<DmlMockKernel>(
        &ctx_, key, &init_helper_);
    EXPECT_TRUE(kernel_manager_.GetCacheSize() <=
                DmlKernelManager::kDefaultMaxCacheSize);
  }
}

// Tests LRU cache eviction
TEST_F(DmlKernelManagerTest, LeastRecentlyUsed) {
  // First fill up the cache and keep weak pointers to them (so we can tell when
  // they are released)
  std::vector<std::weak_ptr<DmlMockKernel>> kernels;
  for (size_t i = 0; i < DmlKernelManager::kDefaultMaxCacheSize; ++i) {
    std::string key_name = "key" + std::to_string(i);
    DmlKernelKey key = CreateKey(key_name.c_str());

    auto kernel = kernel_manager_.CreateCachedKernel<DmlMockKernel>(
        &ctx_, key, &init_helper_);
    EXPECT_TRUE(kernel_manager_.GetCacheSize() <=
                DmlKernelManager::kDefaultMaxCacheSize);

    kernels.push_back(kernel);  // weak
  }

  // At this point all kernels should be alive
  for (const auto& kernel : kernels) {
    EXPECT_TRUE(!kernel.expired());
  }

  // Touch key0 and key 2 which makes them most-recently-used
  kernel_manager_.TryGetCachedKernel<DmlMockKernel>(CreateKey("key0"));
  kernel_manager_.TryGetCachedKernel<DmlMockKernel>(CreateKey("key2"));

  for (const auto& kernel : kernels) {
    EXPECT_TRUE(!kernel.expired());
  }

  // Add a bunch more kernels, which should cause the kernel manager to evict
  // the least recently used kernels (which should start with key1, then key3)

  kernel_manager_.CreateCachedKernel<DmlMockKernel>(&ctx_, CreateKey("foo"),
                                                    &init_helper_);
  EXPECT_TRUE(!kernels[0].expired());
  EXPECT_TRUE(kernels[1].expired());
  EXPECT_TRUE(!kernels[2].expired());
  EXPECT_TRUE(!kernels[3].expired());
  for (size_t i = 4; i < kernels.size(); ++i) {
    EXPECT_TRUE(!kernels[i].expired());
  }

  kernel_manager_.CreateCachedKernel<DmlMockKernel>(&ctx_, CreateKey("bar"),
                                                    &init_helper_);
  EXPECT_TRUE(!kernels[0].expired());
  EXPECT_TRUE(kernels[1].expired());
  EXPECT_TRUE(!kernels[2].expired());
  EXPECT_TRUE(kernels[3].expired());
  for (size_t i = 4; i < kernels.size(); ++i) {
    EXPECT_TRUE(!kernels[i].expired());
  }
}

// Spawn off a bunch of threads and randomly call methods on the kernel manager.
// The kernel manager is expected to always be thread-safe.
TEST_F(DmlKernelManagerTest, ThreadSafety) {
  static const size_t kThreadCount = 16;
  static const size_t kIterationCount = 1000;

  static const char* kKeyNames[] = {"key0", "key1", "key2", "key3"};

  auto thread_proc = [this](uint32_t thread_id) {
    std::mt19937 rng(thread_id);
    std::uniform_int_distribution<> action_dist(0, 2);

    for (size_t i = 0; i < kIterationCount; ++i) {
      // Pick a random action
      switch (action_dist(rng)) {
        case 0: {
          // Randomly choose a key name
          std::uniform_int_distribution<> key_dist(
              0, ABSL_ARRAYSIZE(kKeyNames) - 1);
          const char* key_name = kKeyNames[key_dist(rng)];

          DmlKernelKey key = CreateKey(key_name);

          std::shared_ptr<DmlMockKernel> kernel;

          EXPECT_NO_FATAL_FAILURE(
              kernel = kernel_manager_.TryGetCachedKernel<DmlMockKernel>(key));

          if (!kernel) {
            EXPECT_NO_FATAL_FAILURE(
                kernel = kernel_manager_.CreateCachedKernel<DmlMockKernel>(
                    &ctx_, key, &init_helper_));
          }
        } break;
        case 1:
          EXPECT_NO_FATAL_FAILURE(kernel_manager_.GetCacheSize());
          break;
        case 2:
          EXPECT_NO_FATAL_FAILURE(kernel_manager_.ClearCache());
          break;
        default:
          FAIL();
      }
    }
  };

  std::vector<std::thread> threads;
  for (size_t i = 0; i < kThreadCount; ++i) {
    threads.push_back(std::thread(thread_proc, i));
  }

  // Wait for all threads to complete
  for (auto& thread : threads) {
    thread.join();
  }
}

// Tests that QueueReference/ReleaseCompletedReferences correctly holds onto
// references based on fence signal state.
TEST_F(DmlKernelManagerTest, QueueReference) {
  ComPtr<ID3D12Device> device;

#ifdef DML_BUILD_WINDOWS
  DML_CHECK_SUCCEEDED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0,
                                        IID_PPV_ARGS(&device)));
#else
  auto adapters = EnumerateAdapters();
  EXPECT_FALSE(adapters.empty());

  D3D_FEATURE_LEVEL dxcore_feature_level = adapters[0].IsComputeOnly()
                                               ? D3D_FEATURE_LEVEL_1_0_CORE
                                               : D3D_FEATURE_LEVEL_11_0;

  DML_CHECK_SUCCEEDED(D3D12CreateDevice(
      adapters[0].Impl()->Get(), dxcore_feature_level, IID_PPV_ARGS(&device)));
#endif

  ComPtr<ID3D12Fence> fence;
  DML_CHECK_SUCCEEDED(
      device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));

  // Enters the signaled state when the fence's completed value >= 1
  DmlGpuEvent event1 = {1, fence};

  // Enters the signaled state when the fence's completed value >= 2
  DmlGpuEvent event2 = {2, fence};

  auto kernel1 = kernel_manager_.CreateCachedKernel<DmlMockKernel>(
      &ctx_, CreateKey("key1"), &init_helper_);
  auto kernel2 = kernel_manager_.CreateCachedKernel<DmlMockKernel>(
      &ctx_, CreateKey("key2"), &init_helper_);

  // Tell the kernel manager to keep these kernels alive until their respective
  // events become signaled
  kernel_manager_.QueueReference(kernel1, event1);
  kernel_manager_.QueueReference(kernel2, event2);

  // Now that the kernel manager has taken ownership, release our references
  // (but keep a weak_ptr so we can detect when they go away)
  std::weak_ptr<DmlMockKernel> kernel1_weak = kernel1;
  std::weak_ptr<DmlMockKernel> kernel2_weak = kernel2;
  kernel1.reset();
  kernel2.reset();
  kernel_manager_.ClearCache();

  EXPECT_TRUE(!kernel1);
  EXPECT_TRUE(!kernel2);
  EXPECT_TRUE(!kernel1_weak.expired());
  EXPECT_TRUE(!kernel2_weak.expired());
  EXPECT_TRUE(kernel_manager_.GetCacheSize() == 0);

  // fence = 0
  // ReleaseCompletedReferences should be a no-op
  kernel_manager_.ReleaseCompletedReferences();
  EXPECT_TRUE(!kernel1_weak.expired());
  EXPECT_TRUE(!kernel2_weak.expired());

  // fence = 1
  // kernel1 should be released
  DML_CHECK_SUCCEEDED(fence->Signal(1));
  kernel_manager_.ReleaseCompletedReferences();
  EXPECT_TRUE(kernel1_weak.expired());
  EXPECT_TRUE(!kernel2_weak.expired());

  // fence = 2
  // Both kernels should be released
  DML_CHECK_SUCCEEDED(fence->Signal(2));
  kernel_manager_.ReleaseCompletedReferences();
  EXPECT_TRUE(kernel1_weak.expired());
  EXPECT_TRUE(kernel2_weak.expired());
}

}  // namespace tensorflow