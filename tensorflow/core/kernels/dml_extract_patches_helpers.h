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

#include "tensorflow/core/common_runtime/dml/dml_common.h"

namespace dml {
dml::Expression ExtractPatches(dml::Graph& scope, dml::Expression input,
                               absl::Span<const uint32_t> window_sizes,
                               absl::Span<const uint32_t> window_strides,
                               absl::Span<const uint32_t> window_rates,
                               absl::Span<const uint32_t> start_padding,
                               absl::Span<const uint32_t> end_padding,
                               absl::Span<const uint32_t> output_sizes);
}  // namespace dml
