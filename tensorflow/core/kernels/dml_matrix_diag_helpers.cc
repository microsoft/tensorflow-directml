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

#include "tensorflow/core/kernels/dml_matrix_diag_helpers.h"

#include <numeric>

#include "tensorflow/core/common_runtime/dml/dml_operator_helper.h"
#include "tensorflow/core/common_runtime/dml/dml_util.h"
#include "tensorflow/core/kernels/dml_ops_common.h"

namespace dml {

dml::Expression MatrixDiag(dml::Graph& scope, dml::Expression diag,
                           int32_t k_min, int32_t k_max, float padding_value,
                           int64_t out_height, int64_t out_width) {
  assert(k_min <= k_max);

  dml::TensorDesc::Dimensions diag_shape = diag.GetOutputDesc().sizes;
  uint32_t diag_depth = diag_shape[diag_shape.size() - 2];
  uint32_t diag_width = diag_shape[diag_shape.size() - 1];

  std::vector<int32_t> k_range;
  std::generate_n(std::back_inserter(k_range), k_max - k_min + 1,
                  [val = k_min]() mutable { return std::abs(val++); });

  int32_t min_k2zero = *std::min_element(k_range.begin(), k_range.end());
  int32_t max_diag_len = min_k2zero + diag_width;
  int64_t k_btm = 1 - out_height;

  // Get K lengths
  uint32_t rwcl_min = std::min(out_height, out_width);
  uint32_t rwcl_gap = std::abs(out_height - out_width);

  // Build k_lens by inserting series in the following order:
  //   Left bottom
  //   Middle
  //   Top right
  // The result will be a vector that looks like [1 2 3 4 1 1 1 4 3 2 1]
  std::vector<int> k_lens_cpu(rwcl_min - 1);
  std::iota(k_lens_cpu.begin(), k_lens_cpu.end(), 1);
  k_lens_cpu.resize(k_lens_cpu.size() + rwcl_gap + 1, rwcl_min);
  std::generate_n(std::back_inserter(k_lens_cpu), rwcl_min - 1,
                  [val = rwcl_min - 1]() mutable { return val--; });

  // MatrixDiag and MatrixDiagV2 align both the superdiagonal and the
  // subdiagonal to the left
  int k_sup_stt = std::max(0, k_min) - k_btm;
  int k_sup_end = std::max<int>(k_max + 1 - k_btm, k_sup_stt);
  int k_sub_stt = k_min - k_btm;
  int k_sub_end = std::max<int>(std::min(0, k_max + 1) - k_btm, k_sub_stt);

  std::vector<int> all_k_lens(k_lens_cpu.begin() + k_sub_stt,
                              k_lens_cpu.begin() + k_sub_end);

  std::copy(k_lens_cpu.begin() + k_sup_stt, k_lens_cpu.begin() + k_sup_end,
            std::back_inserter(all_k_lens));

  int max_k_len = *std::max_element(all_k_lens.begin(), all_k_lens.end());
  int top_k_len = all_k_lens.back();
  int btm_k_len = all_k_lens.front();

  uint32_t diag_elem_count = std::accumulate(
      diag_shape.begin(), diag_shape.end(), 1u, std::multiplies<uint32_t>());

  auto diag_strides = diag.GetOutputDesc().strides;

  // Diag's stride should either be a broadcasted scalar or empty
  DCHECK(!diag_strides.has_value() ||
         std::all_of(diag_strides->begin(), diag_strides->end(),
                     [](uint32_t stride) { return stride == 0; }));

  dml::TensorDesc::Dimensions diag_rev_shape(
      {1u, 1u, diag_elem_count / diag_width, diag_width});
  auto reshaped_diag = dml::Reinterpret(diag, diag_rev_shape, diag_strides);

  uint32_t k_sub_len = std::max(0, k_sub_end - k_sub_stt);
  uint32_t k_sup_len = std::max(0, k_sup_end - k_sup_stt);

  dml::Expression sub_rev_len_1;

  if (k_sub_len != 0) {
    auto left_btm =
        dml::Sequence<int32_t>(scope, 1, 1, {1, 1, 1, rwcl_min - 1});

    auto right_top = dml::Sequence<int32_t>(scope, rwcl_min - 1, -1,
                                            {1, 1, 1, rwcl_min - 1});

    auto klen_mid =
        dml::ScalarTensor<int32_t>(scope, rwcl_min, {1, 1, 1, rwcl_gap + 1});

    auto k_lens = dml::Join({left_btm, klen_mid, right_top}, 3);

    // Only slice if we don't want the entire tensor
    if (k_sub_len != k_lens.GetOutputDesc().sizes.back()) {
      sub_rev_len_1 =
          dml::Slice(k_lens, {0, 0, 0, static_cast<uint32_t>(k_sub_stt)},
                     {1, 1, 1, k_sub_len}, {1, 1, 1, 1});
    }

    sub_rev_len_1 =
        dml::Reinterpret(sub_rev_len_1, DML_TENSOR_DATA_TYPE_UINT32);
  }

  auto sup_rev_len_1 = k_sup_len == 0 ? dml::Expression()
                                      : dml::ScalarTensor<uint32_t>(
                                            scope, 1, {1, 1, 1, k_sup_len});

  // Build cnt_rev_len_1
  dml::Expression cnt_rev_len_1;

  if (k_sub_len == 0) {
    cnt_rev_len_1 = sup_rev_len_1;
  } else if (k_sup_len == 0) {
    cnt_rev_len_1 = sub_rev_len_1;
  } else {
    cnt_rev_len_1 = dml::Join({sub_rev_len_1, sup_rev_len_1}, 3);
  }

  auto sub_rev_len_2 = k_sub_len == 0
                           ? dml::Expression()
                           : dml::ScalarTensor<uint32_t>(scope, diag_width,
                                                         {1, 1, 1, k_sub_len});

  // MatrixDiag and MatrixDiagV2's alignment is always LEFT_LEFT, so
  // sup_rev_len_2 is the same as sup_rev_len_1
  auto sup_rev_len_2 = sup_rev_len_1;

  // Build cnt_rev_len_2
  dml::Expression cnt_rev_len_2;

  if (k_sub_len == 0) {
    cnt_rev_len_2 = sup_rev_len_2;
  } else if (k_sup_len == 0) {
    cnt_rev_len_2 = sub_rev_len_2;
  } else {
    cnt_rev_len_2 = dml::Join({sub_rev_len_2, sup_rev_len_2}, 3);
  }

  auto cnt_rev_len_length = cnt_rev_len_1.GetOutputDesc().sizes.back();
  auto exp_rev_len_1 = cnt_rev_len_1;
  auto exp_rev_len_2 = cnt_rev_len_2;

  // Build exp_rev_len_1
  if (cnt_rev_len_length > 1) {
    auto exp_rev_len_seqs =
        dml::ScalarTensor<uint32_t>(scope, cnt_rev_len_length, {1, 1, 1, 1});

    exp_rev_len_1 =
        dml::ReverseSubsequences(exp_rev_len_1, exp_rev_len_seqs, 3);

    exp_rev_len_2 =
        dml::ReverseSubsequences(exp_rev_len_2, exp_rev_len_seqs, 3);
  }

  // Broadcast exp_rev_len_1 to match reshaped_diag
  dml::TensorDesc::Dimensions rev_shape({
      1,
      1,
      diag_elem_count / diag_width / cnt_rev_len_length,
      cnt_rev_len_length,
  });

  dml::TensorDesc::Dimensions rev_strides({0, 0, 0, 1});

  dml::TensorDesc::Dimensions reshaped_rev_sizes({
      1,
      1,
      diag_elem_count / diag_width,
      1,
  });

  // Broadcast and reshape exp_rev_len_1, which specifies the length to
  // reverse each row of each batch
  exp_rev_len_1 = dml::Reinterpret(exp_rev_len_1, rev_shape, rev_strides);
  exp_rev_len_1 = dml::Identity(exp_rev_len_1);
  exp_rev_len_1 = dml::Reinterpret(exp_rev_len_1, reshaped_rev_sizes, {});

  auto reversed_diag_1 =
      dml::ReverseSubsequences(reshaped_diag, exp_rev_len_1, 3);

  // Broadcast and reshape exp_rev_len_2, which specifies the length to
  // reverse each row of each batch
  exp_rev_len_2 = dml::Reinterpret(exp_rev_len_2, rev_shape, rev_strides);
  exp_rev_len_2 = dml::Identity(exp_rev_len_2);
  exp_rev_len_2 = dml::Reinterpret(exp_rev_len_2, reshaped_rev_sizes, {});

  auto sorted_diag =
      dml::ReverseSubsequences(reversed_diag_1, exp_rev_len_2, 3);

  // DML only supports until 5D for Identity, but we can coalesce together the
  // dimensions that don't need to be transposed
  uint32_t head_shape_elem_count =
      std::accumulate(diag_shape.begin(), diag_shape.end() - 2, 1u,
                      std::multiplies<uint32_t>());

  dml::TensorDesc::Dimensions tran_diag_sizes({
      head_shape_elem_count,
      1,
      diag_width,
      diag_depth,
  });

  dml::TensorDesc::Dimensions tran_diag_strides({
      diag_width * diag_depth,
      diag_width * diag_depth,
      1,
      diag_width,
  });

  auto tran_diag =
      dml::Reinterpret(sorted_diag, tran_diag_sizes, tran_diag_strides);

  tran_diag = dml::Identity(tran_diag);

  // Make the diagonal
  uint32_t width = tran_diag_sizes.back();
  uint32_t height = diag_elem_count / width;
  auto reshaped_tran_diag =
      dml::Reinterpret(tran_diag, {1, 1, height, width}, {});

  uint32_t top_pad = max_k_len - top_k_len;
  uint32_t btm_pad = max_k_len - btm_k_len;
  uint32_t left_pad = top_pad;
  uint32_t right_pad = btm_pad + diag_width;

  auto diag_pad =
      dml::Padding(reshaped_tran_diag, DML_PADDING_MODE_CONSTANT, padding_value,
                   {0, 0, 0, left_pad}, {0, 0, 0, right_pad});

  int diag_pad_width = width + left_pad + right_pad;

  dml::TensorDesc::Dimensions exp_shape({
      1,
      1,
      head_shape_elem_count,
      diag_width,
  });

  dml::TensorDesc::Dimensions exp_shape_reshaped(
      {1, 1, head_shape_elem_count * diag_width, 1});

  auto rg =
      dml::Sequence<float>(scope, left_pad * 2, -1, {1, 1, 1, diag_width});

  rg = dml::ActivationRelu(rg);
  rg = dml::Cast(rg, DML_TENSOR_DATA_TYPE_UINT32);

  auto expanded_range = dml::Reinterpret(
      rg, exp_shape, dml::TensorDesc::Dimensions({0, 0, 0, 1}));

  expanded_range = dml::Identity(expanded_range);

  auto reshaped_range =
      dml::Reinterpret(expanded_range, exp_shape_reshaped, {});

  auto pad_left = dml::ReverseSubsequences(diag_pad, reshaped_range, 3);
  auto pad_left_shape = pad_left.GetOutputDesc().sizes;
  pad_left_shape[3] = diag_pad_width - left_pad;

  if (left_pad > 0) {
    pad_left =
        dml::Slice(pad_left, {0, 0, 0, left_pad}, pad_left_shape, {1, 1, 1, 1});
  }

  uint32_t pad_left_depth = pad_left_shape[2];
  uint32_t pad_left_width = pad_left_shape[3];

  auto pad_full_length = dml::ScalarTensor<uint32_t>(scope, pad_left_width,
                                                     {1, 1, pad_left_depth, 1});

  auto rev = dml::ReverseSubsequences(pad_left, pad_full_length, 3);

  auto rg2 = dml::Sequence<int32_t>(scope, right_pad + btm_pad, -1,
                                    {1, 1, 1, diag_width});

  auto expanded_range2 = dml::Reinterpret(
      rg2, exp_shape, dml::TensorDesc::Dimensions({0, 0, 0, 1}));

  expanded_range2 = dml::Identity(expanded_range2);

  auto reshaped_range2 = dml::Reinterpret(
      expanded_range2, DML_TENSOR_DATA_TYPE_UINT32, exp_shape_reshaped, {});

  auto raw_pad_right = dml::ReverseSubsequences(rev, reshaped_range2, 3);
  auto raw_pad_right_shape = raw_pad_right.GetOutputDesc().sizes;
  int raw_pad_right_width = raw_pad_right_shape.back();

  auto sliced_raw_pad_right = raw_pad_right;

  if (btm_pad > 0) {
    raw_pad_right_shape.back() = raw_pad_right_width - btm_pad;

    sliced_raw_pad_right = dml::Slice(raw_pad_right, {0, 0, 0, btm_pad},
                                      raw_pad_right_shape, {1, 1, 1, 1});
  }

  // Build all_width
  auto all_width = dml::ScalarTensor<uint32_t>(
      scope, raw_pad_right_width - btm_pad, exp_shape_reshaped);

  auto pad_right = dml::ReverseSubsequences(sliced_raw_pad_right, all_width, 3);

  // Diagonalize
  auto rg3 = dml::Sequence<uint32_t>(scope, diag_depth - btm_pad, 1,
                                     {1, 1, 1, diag_width});

  auto expanded_range3 = dml::Reinterpret(
      rg3, exp_shape, dml::TensorDesc::Dimensions({0, 0, 0, 1}));

  expanded_range3 = dml::Identity(expanded_range3);

  auto reshaped_range3 =
      dml::Reinterpret(expanded_range3, exp_shape_reshaped, {});

  auto rev2 = dml::ReverseSubsequences(pad_right, reshaped_range3, 3);

  int k_max_idx = k_max - k_btm;
  int k_max_len = k_lens_cpu[k_max_idx];
  int k_gap = std::abs(k_max) - min_k2zero;
  int diagonalize_width = k_max_len + k_gap;

  auto sliced_rev2_shape = rev2.GetOutputDesc().sizes;
  auto new_diag = rev2;

  if (diagonalize_width != sliced_rev2_shape.back()) {
    sliced_rev2_shape.back() = diagonalize_width;
    new_diag =
        dml::Slice(new_diag, {0, 0, 0, 0}, sliced_rev2_shape, {1, 1, 1, 1});
  }

  int new_diag_elem_count =
      std::accumulate(sliced_rev2_shape.begin(), sliced_rev2_shape.end(), 1u,
                      std::multiplies<uint32_t>());

  uint32_t new_depth = diag_width;
  uint32_t new_width = new_diag_elem_count / head_shape_elem_count / diag_width;

  dml::TensorDesc::Dimensions new_diag_shape = {1, head_shape_elem_count,
                                                new_depth, new_width};

  new_diag = dml::Reinterpret(new_diag, new_diag_shape, {});

  // Finally, pad to output shape
  uint32_t pad_row = out_height - new_depth;
  uint32_t pad_col = out_width - new_width;
  uint32_t pad_top = std::max(0, -k_max);
  uint32_t pad_lft = std::max(0, k_min);
  uint32_t pad_btm = pad_row - pad_top;
  uint32_t pad_rht = pad_col - pad_lft;
  auto result =
      dml::Padding(new_diag, DML_PADDING_MODE_CONSTANT, padding_value,
                   {0, 0, pad_top, pad_lft}, {0, 0, pad_btm, pad_rht});

  return result;
}

}  // namespace dml
