/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.
Portions Copyright (c) Microsoft Corporation.

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

#include "tensorflow/core/common_runtime/dml/dml_common.h"
#include "tensorflow/core/common_runtime/dml/dml_device.h"
#include "tensorflow/core/common_runtime/dml/dml_kernel_manager.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/dml_ops_common.h"

namespace tensorflow {

class ShapeHelper;
class InitializationHelper;

enum class DmlKernelCachePolicy {
  Always,
  Never,
  Default = Always,
};

// Wraps a DmlKernel and implements the OpKernel interface on its behalf. This
// wrapper forms the boundary between the DML kernel implementations and
// tensorflow's kernel interfaces, and presents a simpler abstraction to the
// wrapped DmlKernel than what is supplied in the full OpKernel.
class DmlKernelWrapperBase : public OpKernel {
 public:
  explicit DmlKernelWrapperBase(OpKernelConstruction* ctx,
                                DmlKernelCachePolicy cache_policy);

  void Compute(OpKernelContext* ctx) override;

 protected:
  virtual const ShapeHelper* GetShapeHelper() const = 0;
  virtual std::shared_ptr<const InitializationHelper>
  CreateInitializationHelper(OpKernelContext* ctx) const = 0;

  virtual std::shared_ptr<DmlKernel> CreateCachedKernel(
      DmlKernelConstruction* ctx, const DmlKernelManager& kernel_manager,
      const DmlKernelKey& key,
      const InitializationHelper* initialized_helper) const = 0;

  virtual std::shared_ptr<DmlKernel> TryGetCachedKernel(
      const DmlKernelManager& kernel_manager,
      const DmlKernelKey& key) const = 0;

  virtual std::shared_ptr<DmlKernel> CreateKernel(
      DmlKernelConstruction* ctx,
      const InitializationHelper* initialized_helper) const = 0;

  virtual StatusOr<DmlGpuEvent> ComputeKernel(
      DmlKernel* kernel, DmlKernelContext* context) const = 0;

 protected:
  // Creates a key which uniquely identifies the kernel instance we need. The
  // returned key can be used to retrieve an appropriate DML kernel from the
  // cache.
  virtual DmlKernelKey CreateKernelKey(OpKernelContext* ctx) const;

  DmlKernelCachePolicy cache_policy_;
  std::shared_ptr<const NodeDef> node_def_;
};

// Implements a (templated) GetOrCreateKernel and output shape computation for
// the kernel wrapper.
//
// `TKernel` must be a DmlKernel implementation.
// `TShapeHelper` must be a type that matches the following signature:
//     struct S {
//       S(OpKernelConstruction* ctx); // constructor
//
//       // Computes the shapes of each output tensor. Must be thread-safe.
//       std::vector<TensorShape> GetOutputShapes(OpKernelContext* ctx) const;
//     }
//
// This class is intended to be used when registering DML kernels for an op.
// Example:
//     REGISTER_KERNEL_BUILDER(Name("Add").Device(DEVICE_DML),
//                             DmlKernelWrapper<DmlAddKernel, AddShapeHelper>);
//
template <typename TKernel, typename TShapeHelper,
          DmlKernelCachePolicy cache_policy = DmlKernelCachePolicy::Default>
class DmlKernelWrapper : public DmlKernelWrapperBase {
 public:
  using Attributes = typename TKernel::InitHelper::Attributes;

  explicit DmlKernelWrapper(OpKernelConstruction* ctx)
      : DmlKernelWrapperBase(ctx, cache_policy),
        attr_(std::make_shared<Attributes>(ctx)) {}

 protected:
  const ShapeHelper* GetShapeHelper() const final { return &shape_helper_; }

  std::shared_ptr<DmlKernel> CreateCachedKernel(
      DmlKernelConstruction* ctx, const DmlKernelManager& kernel_manager,
      const DmlKernelKey& key,
      const InitializationHelper* initialized_helper) const final {
    // If the cache policy is "Never", the kernel wrapper should simply create
    // the kernel directly instead of delegating to the kernel manager
    assert(cache_policy != DmlKernelCachePolicy::Never);

    // Create the kernel and cache it
    return kernel_manager.CreateCachedKernel<TKernel>(
        ctx, key,
        static_cast<const typename TKernel::InitHelper*>(initialized_helper));
  }

  std::shared_ptr<DmlKernel> TryGetCachedKernel(
      const DmlKernelManager& kernel_manager,
      const DmlKernelKey& key) const final {
    // If the cache policy is "Never", the kernel wrapper should never try to
    // retrieved a cached kernel
    assert(cache_policy != DmlKernelCachePolicy::Never);

    // Retrieve the kernel from the cache
    return kernel_manager.TryGetCachedKernel<TKernel>(key);
  }

  std::shared_ptr<DmlKernel> CreateKernel(
      DmlKernelConstruction* ctx,
      const InitializationHelper* initialized_helper) const final {
    return std::make_shared<TKernel>(
        ctx,
        static_cast<const typename TKernel::InitHelper*>(initialized_helper));
  }

  std::shared_ptr<const InitializationHelper> CreateInitializationHelper(
      OpKernelContext* ctx) const final {
    return std::make_shared<const typename TKernel::InitHelper>(ctx, attr_);
  }

  StatusOr<DmlGpuEvent> ComputeKernel(
      DmlKernel* kernel, DmlKernelContext* context) const override {
    return kernel->Compute(context);
  }

 private:
  const std::shared_ptr<const Attributes> attr_;
  const TShapeHelper shape_helper_;
};

}  // namespace tensorflow