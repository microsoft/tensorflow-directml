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

#include "tensorflow/core/kernels/dml_extract_patches_helpers.h"

#include "tensorflow/core/common_runtime/dml/dml_operator_helper.h"
#include "tensorflow/core/common_runtime/dml/dml_util.h"
#include "tensorflow/core/kernels/dml_ops_common.h"

namespace dml {
dml::Expression ExtractPatches(dml::Graph& scope, dml::Expression input,
                               absl::Span<const uint32_t> window_sizes,
                               absl::Span<const uint32_t> window_strides,
                               absl::Span<const uint32_t> window_rates,
                               absl::Span<const uint32_t> start_padding,
                               absl::Span<const uint32_t> end_padding,
                               absl::Span<const uint32_t> output_sizes) {
  assert(window_sizes.size() == 4 || window_sizes.size() == 5);
  assert(window_sizes.size() == window_strides.size());
  assert(window_sizes.size() == window_rates.size());
  assert(window_sizes.size() == start_padding.size());
  assert(window_sizes.size() == end_padding.size());
  assert(window_sizes.size() == input.GetOutputDesc().sizes.size());

  bool is_identity = std::all_of(window_sizes.begin(), window_sizes.end(),
                                 [](uint32_t val) { return val == 1; }) &&
                     std::all_of(window_strides.begin(), window_strides.end(),
                                 [](uint32_t val) { return val == 1; });

  if (is_identity) {
    return dml::Identity(input);
  }

  bool need_padding = std::any_of(start_padding.begin(), start_padding.end(),
                                  [](uint32_t val) { return val != 0; }) ||
                      std::any_of(end_padding.begin(), end_padding.end(),
                                  [](uint32_t val) { return val != 0; });

  if (need_padding) {
    input = dml::Padding(input, DML_PADDING_MODE_CONSTANT, 0.0f, start_padding,
                         end_padding);
  }

  auto padded_input_sizes = input.GetOutputDesc().sizes;
  int dim_count = padded_input_sizes.size();
  int spatial_dim_count = dim_count - 2;
  uint32_t step_multiplier = 1;

  absl::InlinedVector<dml::Expression, 3> global_indices(spatial_dim_count);
  dml::TensorDesc::Dimensions broadcasted_global_indices_sizes(dim_count, 1);

  for (int i = spatial_dim_count; i > 0; --i) {
    uint32_t patch_step = step_multiplier * window_strides[i];
    uint32_t element_step = step_multiplier * window_rates[i];
    uint32_t out_size = output_sizes[i];

    if (out_size > 1 && window_sizes[i] > 1) {
      // Indices of the patches in the input (first patch is at index 0, and
      // last patch is at index out_size - 1)
      auto patch_indices =
          dml::Sequence<uint32_t>(scope, 0, patch_step, {1, 1, out_size, 1});

      patch_indices = dml::Tile(patch_indices, {1, 1, 1, window_sizes[i]});

      // Indices relative to the patch (first element of the patch is at index
      // 0, and last element is at index ksize - 1)
      auto relative_element_indices = dml::Sequence<uint32_t>(
          scope, 0, element_step, {1, 1, 1, window_sizes[i]});

      relative_element_indices = dml::Reinterpret(
          relative_element_indices, {1, 1, out_size, window_sizes[i]},
          dml::TensorDesc::Dimensions({0, 0, 0, 1}));

      global_indices[i - 1] = patch_indices + relative_element_indices;
    } else if (out_size > 1) {
      global_indices[i - 1] =
          dml::Sequence<uint32_t>(scope, 0, patch_step, {1, 1, out_size, 1});
    } else {
      DCHECK(window_sizes[i] > 1);
      global_indices[i - 1] = dml::Sequence<uint32_t>(
          scope, 0, element_step, {1, 1, 1, window_sizes[i]});
    }

    broadcasted_global_indices_sizes[i] = out_size * window_sizes[i];

    // Increase the step multiplier for the next dimension
    step_multiplier *= padded_input_sizes[i];
  }

  dml::Expression broadcasted_global_indices;
  dml::TensorDesc::Dimensions broadcasted_sizes(dim_count, 1);
  dml::TensorDesc::Dimensions broadcasted_strides(dim_count, 0);
  uint32_t running_stride = 1;

  for (int i = spatial_dim_count - 1; i >= 0; --i) {
    auto sizes = global_indices[i].GetOutputDesc().sizes;
    broadcasted_sizes[i + 1] = sizes[3] * sizes[2];

    dml::TensorDesc::Dimensions strides(dim_count, 0);
    strides[i + 1] = 1;
    auto broadcasted =
        dml::Reinterpret(global_indices[i], broadcasted_sizes, strides);

    if (i == spatial_dim_count - 1) {
      broadcasted_global_indices = broadcasted;
    } else {
      broadcasted_global_indices = dml::Reinterpret(
          broadcasted_global_indices, broadcasted_sizes, broadcasted_strides);
      broadcasted_global_indices += broadcasted;
    }

    // Set the strides for the next iteration
    broadcasted_strides[i + 1] = running_stride;
    running_stride *= sizes[3] * sizes[2];
  }

  broadcasted_global_indices = dml::Reinterpret(broadcasted_global_indices,
                                                {1, 1, 1, running_stride}, {});

  uint32_t spatial_input_size = 1;
  for (int i = 1; i < padded_input_sizes.size() - 1; ++i) {
    spatial_input_size *= padded_input_sizes[i];
  }

  dml::TensorDesc::Dimensions reshaped_input_sizes({
      1,
      padded_input_sizes.front(),
      spatial_input_size,
      padded_input_sizes.back(),
  });

  dml::TensorDesc::Dimensions gathered_sizes({
      1,
      padded_input_sizes.front(),
      running_stride,
      padded_input_sizes.back(),
  });

  auto reshaped_input = dml::Reinterpret(input, reshaped_input_sizes, {});
  auto gather_indices =
      dml::Reinterpret(broadcasted_global_indices, gathered_sizes,
                       dml::TensorDesc::Dimensions({0, 0, 1, 0}));

  // Gather the elements to construct the patches
  auto gathered = dml::GatherElements(reshaped_input, gather_indices, 2);

  // After gathering the elements, the patches are in the space dimensions. We
  // need to move them to the depth dimensions instead.
  // Reshape gathered into [batch] + [out_size[1], ksize[1], ...,
  // out_size[N-2], ksize[N-2]] + [depth]
  dml::TensorDesc::Dimensions reshaped_gathered_sizes;
  reshaped_gathered_sizes.push_back(padded_input_sizes.front());

  for (int i = 0; i < spatial_dim_count; ++i) {
    reshaped_gathered_sizes.push_back(output_sizes[i + 1]);
    reshaped_gathered_sizes.push_back(window_sizes[i + 1]);
  }

  reshaped_gathered_sizes.push_back(padded_input_sizes.back());

  // Permute reshaped_gathered into [batch] + [out_size[1], ...,
  // out_size[N-2]] + [depth] + [ksize[1], ..., ksize[N-2]]
  uint32_t stride = 1;
  dml::TensorDesc::Dimensions reshaped_gathered_strides(
      reshaped_gathered_sizes.size());

  for (int i = reshaped_gathered_sizes.size() - 1; i >= 0; --i) {
    reshaped_gathered_strides[i] = stride;
    stride *= reshaped_gathered_sizes[i];
  }

  dml::TensorDesc::Dimensions perm_strides;
  perm_strides.reserve(reshaped_gathered_strides.size());

  dml::TensorDesc::Dimensions perm_sizes;
  perm_sizes.reserve(reshaped_gathered_sizes.size());

  perm_strides.push_back(reshaped_gathered_strides.front());
  perm_sizes.push_back(reshaped_gathered_sizes.front());

  for (int i = 1; i < spatial_dim_count * 2; i += 2) {
    perm_strides.push_back(reshaped_gathered_strides[i]);
    perm_sizes.push_back(reshaped_gathered_sizes[i]);
  }

  for (int i = 2; i <= spatial_dim_count * 2; i += 2) {
    perm_strides.push_back(reshaped_gathered_strides[i]);
    perm_sizes.push_back(reshaped_gathered_sizes[i]);
  }

  perm_strides.push_back(reshaped_gathered_strides.back());
  perm_sizes.push_back(reshaped_gathered_sizes.back());

  auto gathered_reshaped = dml::Reinterpret(gathered, perm_sizes, perm_strides);
  gathered_reshaped = dml::Identity(gathered_reshaped);

  return gathered_reshaped;
}
}  // namespace dml
