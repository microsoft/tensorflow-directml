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

#include "tensorflow/core/common_runtime/dml/dml_kernel_key.h"

#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/lib/hash/hash.h"

namespace tensorflow {

static uint64 TensorShapeHash(const TensorShape& s) {
  uint64 hash = s.dims();
  for (int i = 0; i < s.dims(); ++i) {
    hash = Hash64Combine(hash, s.dim_size(i));
  }
  return hash;
}

static uint64 DmlInputTensorKeyHash(const DmlInputTensorKey& k) {
  uint64 hash = 0;

  if (k.is_constant_cpu_input) {
    const Tensor& tensor = absl::get<Tensor>(k.tensor);

    hash = TensorShapeHash(tensor.shape());
    hash = Hash64Combine(hash, tensor.dtype());

    // Hash the contents of the tensor too. Note that this only works for
    // primitive types (i.e. where DataTypeCanUseMemcpy is true)
    hash = Hash64Combine(
        hash, Hash64(tensor.tensor_data().data(), tensor.tensor_data().size()));
  } else {
    const TensorShapeAndType& tensor = absl::get<TensorShapeAndType>(k.tensor);

    hash = TensorShapeHash(tensor.shape);
    hash = Hash64Combine(hash, tensor.dtype);
  }

  return hash;
}

uint64 DmlKernelKeyHash(const DmlKernelKey& k) {
  uint64 hash = Hash64(k.op_type_name);

  AttrSlice attributes(*k.node_def);
  for (const auto& kvp : attributes) {
    const AttrValue& attr = kvp.second;

    // Need to combine these hashes in an order-independent way, because the map
    // of attributes is unordered.
    hash = Hash64CombineUnordered(hash, AttrValueHash(attr));
  }

  for (const DmlInputTensorKey& input : k.input_tensors) {
    hash = Hash64Combine(hash, DmlInputTensorKeyHash(input));
  }

  return hash;
}

DmlInputTensorKey DmlInputTensorKey::Clone() const {
  DmlInputTensorKey clone = {};

  // If the input is a CPU initializer, we need to deep copy its content
  // since it uniquely identifies it. Otherwise, only the shape and datatype
  // need to be part of the signature.
  if (this->is_constant_cpu_input) {
    clone.tensor = tensor::DeepCopy(absl::get<Tensor>(this->tensor));
  } else {
    clone.tensor = absl::get<TensorShapeAndType>(this->tensor);
  }

  clone.is_constant_cpu_input = this->is_constant_cpu_input;
  return clone;
}

DmlKernelKey DmlKernelKey::Clone() const {
  DmlKernelKey clone = {};
  clone.op_type_name = this->op_type_name;
  clone.node_def = this->node_def;

  for (const auto& input : this->input_tensors) {
    clone.input_tensors.push_back(input.Clone());
  }

  return clone;
}

bool DmlKernelKey::operator==(const DmlKernelKey& other) const {
  if (this->op_type_name != other.op_type_name) {
    return false;
  }

  AttrSlice my_attributes(*this->node_def);
  AttrSlice other_attributes(*other.node_def);

  AttrSlice::Scratch scratch = {};
  if (!my_attributes.EqualAttrs(other_attributes, &scratch)) {
    return false;
  }

  if (this->input_tensors != other.input_tensors) {
    return false;
  }

  return true;
}

bool DmlInputTensorKey::operator==(const DmlInputTensorKey& other) const {
  if (this->is_constant_cpu_input != other.is_constant_cpu_input) {
    return false;
  }

  // Compare the tensors

  if (is_constant_cpu_input) {
    const auto& tensor0 = absl::get<Tensor>(this->tensor);
    const auto& tensor1 = absl::get<Tensor>(other.tensor);

    if (tensor0.shape() != tensor1.shape()) {
      return false;
    }

    if (tensor0.dtype() != tensor1.dtype()) {
      return false;
    }

    // If this is a constant CPU input, the tensor contents also form part of
    // the key, so we need to compare those too
    if (this->is_constant_cpu_input) {
      auto data_0 = tensor0.tensor_data();
      auto data_1 = tensor1.tensor_data();
      if (data_0.size() != data_1.size()) {
        return false;
      }

      if (memcmp(data_0.data(), data_1.data(), data_0.size())) {
        return false;
      }
    }
  } else {
    const auto& tensor0 = absl::get<TensorShapeAndType>(this->tensor);
    const auto& tensor1 = absl::get<TensorShapeAndType>(other.tensor);

    if (tensor0.shape != tensor1.shape) {
      return false;
    }

    if (tensor0.dtype != tensor1.dtype) {
      return false;
    }
  }

  return true;
}

}  // namespace tensorflow