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

//
// Helpers for implementing TensorArray kernels for DML.
//

#pragma once

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/tensor_array.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace tensor_array {

// DML's helpers don't need the device template parameter, so we use this tag
// simply as a way of specializing our templates.
struct DMLDeviceTag {};

Status DmlAddToTensor(OpKernelContext* ctx, Tensor* sum, const Tensor* current,
                      const Tensor* add);

void DmlTensorSetZero(OpKernelContext* ctx, Tensor* value);

void DmlConcatTensors(OpKernelContext* ctx, Tensor* output_tensor,
                      absl::Span<PersistentTensor> values);

void DmlSplitTensor(OpKernelContext* ctx, Tensor* output_tensor,
                    const Tensor& input_tensor, int64 start_element,
                    int64 element_count);

Status DmlTensorCopy(OpKernelContext* ctx, Tensor* src, Tensor* dst);

// These provide template instantiations of AddToTensor, TensorSetZero,
// ConcatTensors, and SplitTensors for DMLDeviceTag, which simply forward to the
// non-templated version. See tensor_array.h for details.
#define DML_TENSOR_ARRAY_SPECIALIZATIONS(T)                                   \
  template <>                                                                 \
  inline Status AddToTensor<DMLDeviceTag, T>(                                 \
      OpKernelContext * ctx, Tensor * sum, const Tensor* current,             \
      const Tensor* add) {                                                    \
    return DmlAddToTensor(ctx, sum, current, add);                            \
  }                                                                           \
  template <>                                                                 \
  inline Status TensorSetZero<DMLDeviceTag, T>(OpKernelContext * ctx,         \
                                               Tensor * value) {              \
    DmlTensorSetZero(ctx, value);                                             \
    return Status::OK();                                                      \
  }                                                                           \
  template <>                                                                 \
  inline Status TensorCopyUnaligned<DMLDeviceTag, T>(                         \
      OpKernelContext * ctx, Tensor * src, Tensor * dst) {                    \
    return DmlTensorCopy(ctx, src, dst);                                      \
  }                                                                           \
  template <>                                                                 \
  inline void ConcatTensors<DMLDeviceTag, T>(                                 \
      OpKernelContext * ctx, Tensor * output_tensor,                          \
      absl::Span<PersistentTensor> values) {                                  \
    DmlConcatTensors(ctx, output_tensor, values);                             \
  }                                                                           \
  template <>                                                                 \
  inline void SplitTensor<DMLDeviceTag, T>(                                   \
      OpKernelContext * ctx, Tensor * output_tensor,                          \
      const Tensor& input_tensor, int64 start_element, int64 element_count) { \
    DmlSplitTensor(ctx, output_tensor, input_tensor, start_element,           \
                   element_count);                                            \
  }

TF_CALL_ALL_TYPES(DML_TENSOR_ARRAY_SPECIALIZATIONS);

#undef DML_TENSOR_ARRAY_SPECIALIZATIONS

}  // namespace tensor_array
}  // namespace tensorflow