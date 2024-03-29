/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.h"

#include "tensorflow/core/common_runtime/gpu/gpu_mem_allocator.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace tensorflow {

bool GPUBFCAllocator::GetAllowGrowthValue(const GPUOptions& gpu_options) {
  const char* force_allow_growth_string =
      std::getenv("TF_FORCE_GPU_ALLOW_GROWTH");
  if (force_allow_growth_string == nullptr) {
    return gpu_options.allow_growth();
  }

  if (strcmp("false", force_allow_growth_string) == 0) {
    if (gpu_options.allow_growth()) {
      LOG(WARNING)
          << "Overriding allow_growth setting because the"
          << " TF_FORCE_GPU_ALLOW_GROWTH environment variable is set. Original"
          << " config value was " << gpu_options.allow_growth() << ".";
    }
    return false;
  } else if (strcmp("true", force_allow_growth_string) == 0) {
    if (!gpu_options.allow_growth()) {
      LOG(WARNING)
          << "Overriding allow_growth setting because the"
          << " TF_FORCE_GPU_ALLOW_GROWTH environment variable is set. Original"
          << " config value was " << gpu_options.allow_growth() << ".";
    }
    return true;
  }

  LOG(ERROR)
      << "The TF_FORCE_GPU_ALLOW_GROWTH environment variable is set but could"
      << " not be parsed: \"" << force_allow_growth_string << "\". Valid"
      << " values are \"true\" or \"false\". Using original config value"
      << " of " << gpu_options.allow_growth() << ".";
  return gpu_options.allow_growth();
}

bool GPUBFCAllocator::GetGarbageCollectionValue(
    bool default_garbage_collection_value) {
  const char* enable_gpu_garbage_collection =
      std::getenv("TF_ENABLE_GPU_GARBAGE_COLLECTION");
  if (enable_gpu_garbage_collection == nullptr) {
    // Use the default value from the caller if no environment variable exists,
    // as some subclasses of the BFC allocator may default to disabling garbage
    // collection to avoid freeing allocations in the middle of GPU processing.
    return default_garbage_collection_value;
  }
  if (strcmp("false", enable_gpu_garbage_collection) == 0) {
    return false;
  } else if (strcmp("true", enable_gpu_garbage_collection) == 0) {
    return true;
  }

  LOG(ERROR)
      << "The TF_ENABLE_GPU_GARBAGE_COLLECTION environment variable is set but"
      << " could not be parsed: \"" << enable_gpu_garbage_collection << "\"."
      << " Valid values are \"true\" or \"false\"."
      << " Using the default value \"true\".";
  return true;
}

GPUBFCAllocator::GPUBFCAllocator(SubAllocator* sub_allocator,
                                 size_t total_memory, const string& name,
                                 size_t max_allocation_size)
    : GPUBFCAllocator(sub_allocator, total_memory, GPUOptions(), name,
                      max_allocation_size) {}

GPUBFCAllocator::GPUBFCAllocator(SubAllocator* sub_allocator,
                                 size_t total_memory,
                                 const GPUOptions& gpu_options,
                                 const string& name, size_t max_allocation_size,
                                 bool default_garbage_collection_value)
    : BFCAllocator(sub_allocator, total_memory,
                   GPUBFCAllocator::GetAllowGrowthValue(gpu_options), name,
                   GPUBFCAllocator::GetGarbageCollectionValue(
                       default_garbage_collection_value),
                   8, max_allocation_size) {}

}  // namespace tensorflow
