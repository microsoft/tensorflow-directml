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

#include "absl/types/variant.h"
#include "tensorflow/core/common_runtime/dml/dml_common.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"

namespace tensorflow {

struct TensorShapeAndType {
  TensorShape shape;
  DataType dtype;
};

// Used to identify/hash an input tensor for a DML kernel. A DML kernel may
// choose to register certain input tensors as requiring to be stored in host
// memory. (This is achieved by using .HostMemory("input_name") during
// REGISTER_KERNEL_BUILDER). Such tensors are known as constant CPU input
// tensors, otherwise they're just regular input tensors.
//
// Since constant CPU inputs are provided during construction of DML kernels,
// the contents of the tensor (as well as its shape and type) forms part of the
// signature that uniquely identifies a DML kernel instance. Otherwise, just the
// shape and data type form part of the key.
struct DmlInputTensorKey {
  // If is_constant_cpu_input is false, this stores just the TensorShape and
  // type. Otherwise, for constant CPU inputs, this stores the entire tensor
  // (i.e. the shape/dtype as well as the data itself.)
  absl::variant<Tensor, TensorShapeAndType> tensor;
  bool is_constant_cpu_input;

  DmlInputTensorKey Clone() const;  // Performs a deep copy
  bool operator==(const DmlInputTensorKey& other) const;
};

// Uniquely identifes a DML kernel instance. This is used for caching of
// kernels, since DML kernels are immutable once constructed.
struct DmlKernelKey {
  string op_type_name;  // e.g. "Conv2D"

  // The attributes and their values for the kernel
  std::shared_ptr<const NodeDef> node_def;

  absl::InlinedVector<DmlInputTensorKey, 6> input_tensors;

  DmlKernelKey Clone() const;  // Performs a deep copy
  bool operator==(const DmlKernelKey& other) const;
};

uint64 DmlKernelKeyHash(const DmlKernelKey& k);

// Template specialization of std::hash for DmlKernelKey
template <>
struct hash<DmlKernelKey> {
  size_t operator()(const DmlKernelKey& k) const { return DmlKernelKeyHash(k); }
};

}  // namespace tensorflow