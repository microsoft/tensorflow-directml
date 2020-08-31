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

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/cc/ops/math_ops_internal.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/platform/test.h"

#ifdef WIN32
// This is necessary for the linker to find htonl and friends
#pragma comment(lib, "Ws2_32.lib")
#endif

namespace tensorflow {

template <typename TOperator>
Tensor RunBinaryCwiseOp(Scope scope, const Input::Initializer& aData,
                        const Input::Initializer& bData) {
  assert(aData.tensor.dtype() == bData.tensor.dtype());
  DataType data_type = aData.tensor.dtype();

  TensorShape a_shape = aData.tensor.shape();
  TensorShape b_shape = bData.tensor.shape();
  assert(a_shape.IsSameSize(b_shape));

  auto a = ops::Placeholder(scope, data_type, ops::Placeholder::Shape(a_shape));
  auto b = ops::Placeholder(scope, data_type, ops::Placeholder::Shape(b_shape));
  auto result = TOperator(scope, a, b);

  ClientSession::FeedType inputs;
  inputs.emplace(a, aData);
  inputs.emplace(b, bData);

  // Run the model and print the output
  ClientSession session(scope);
  std::vector<Tensor> outputs;
  TF_CHECK_OK(session.Run(inputs, {result}, &outputs));

  return outputs[0];
}

template <typename TOperator, typename TDataType>
void TestBinaryCwiseOp(const Input::Initializer& aData,
                       const Input::Initializer& bData) {
  assert(aData.tensor.dtype() == bData.tensor.dtype());
  DataType data_type = aData.tensor.dtype();

  Scope cpu_scope = Scope::NewRootScope().WithDevice("/device:CPU:0");
  Scope dml_scope = Scope::NewRootScope().WithDevice("/device:DML:0");

  Tensor cpu_result = RunBinaryCwiseOp<TOperator>(cpu_scope, aData, bData);
  Tensor dml_result = RunBinaryCwiseOp<TOperator>(dml_scope, aData, bData);

  EXPECT_EQ(cpu_result.NumElements(), dml_result.NumElements());
  EXPECT_TRUE(cpu_result.shape().IsSameSize(dml_result.shape()));

  auto cpu_matrix = cpu_result.matrix<TDataType>();
  auto dml_matrix = dml_result.matrix<TDataType>();

  if (data_type == DT_FLOAT || data_type == DT_HALF) {
    for (int64 i = 0; i < cpu_result.NumElements(); i++) {
      EXPECT_FLOAT_EQ(static_cast<float>(cpu_matrix(i)),
                      static_cast<float>(dml_matrix(i)));
    }
  } else {
    for (int64 i = 0; i < cpu_result.NumElements(); i++) {
      EXPECT_EQ(cpu_matrix(i), dml_matrix(i));
    }
  }
}

TEST(DmlKernelTests, SimpleExample) {
  using namespace tensorflow::ops;

  Scope scope = Scope::NewRootScope();
  scope = scope.WithDevice("/device:DML:0");

  // Compute clamp(a + b, min, max)

  // Scalar tensor which will be broadcast to the 2x2 size of `b`
  auto a = Placeholder(scope, DT_FLOAT, Placeholder::Shape({1}));

  // 2x2 tensor
  auto b = Placeholder(scope, DT_FLOAT, Placeholder::Shape({2, 2}));

  // Unlike DML, TF's ClipByValue takes its min and max as tensors.
  // This exercises handling of constant CPU input data.
  auto min = Placeholder(scope, DT_FLOAT, Placeholder::Shape({1}));
  auto max = Placeholder(scope, DT_FLOAT, Placeholder::Shape({1}));

  auto result = ClipByValue(scope, AddV2(scope, a, b), min, max);

  // Set up inputs
  //   a = 0.4
  //   b = [[0.1, 0.3],
  //        [0.5, 0.7]]
  //   min = 0.0
  //   max = 1.0
  //
  // Expected output:
  //   clamp(a + b, 0.0, 1.0) =
  //     [[0.5, 0.7],
  //      [0.9, 1.0]]
  //
  ClientSession::FeedType inputs;
  inputs.emplace(a, Input::Initializer({{0.4f}}));
  inputs.emplace(b, Input::Initializer({{0.1f, 0.3f}, {0.5f, 0.7f}}));
  inputs.emplace(min, 0.0f);
  inputs.emplace(max, 1.0f);

  // Run the model and print the output
  ClientSession session(scope);
  std::vector<Tensor> outputs;
  TF_CHECK_OK(session.Run(inputs, {result}, &outputs));

  auto output = outputs[0].matrix<float>();
  LOG(INFO) << output;
  EXPECT_FLOAT_EQ(output(0, 0), 0.5f);
  EXPECT_FLOAT_EQ(output(0, 1), 0.7f);
  EXPECT_FLOAT_EQ(output(1, 0), 0.9f);
  EXPECT_FLOAT_EQ(output(1, 1), 1.0f);

  // Run it again with different inputs
  //   a = 0.5
  //   b = [[0.2, 0.4],
  //        [0.6, 0.8]]
  //   min = 0.0
  //   max = 1.2
  //
  // Expected output:
  //   clamp(a + b, 0.0, 1.2) =
  //     [[0.7, 0.9],
  //      [1.1, 1.2]]
  //
  inputs.clear();
  outputs.clear();
  inputs.emplace(a, Input::Initializer({{0.5f}}));
  inputs.emplace(b, Input::Initializer({{0.2f, 0.4f}, {0.6f, 0.8f}}));
  inputs.emplace(min, 0.0f);
  inputs.emplace(max, 1.2f);
  TF_CHECK_OK(session.Run(inputs, {result}, &outputs));

  output = outputs[0].matrix<float>();
  LOG(INFO) << output;
  EXPECT_FLOAT_EQ(output(0, 0), 0.7f);
  EXPECT_FLOAT_EQ(output(0, 1), 0.9f);
  EXPECT_FLOAT_EQ(output(1, 0), 1.1f);
  EXPECT_FLOAT_EQ(output(1, 1), 1.2f);
}

TEST(DmlKernelTests, Relu) {
  TensorShape shape({2, 2});
  auto input_data = {2.0f, -2.0f, 0.0f, 0.1f};

  Scope scope = Scope::NewRootScope().WithDevice("/device:DML:0");
  ClientSession session(scope);

  {
    Tensor half_data(DT_HALF, shape);
    test::FillValues<Eigen::half>(&half_data, input_data);

    auto input =
        ops::Placeholder(scope, DT_HALF, ops::Placeholder::Shape(shape));

    ClientSession::FeedType inputs;
    inputs.emplace(input, half_data);

    auto result = ops::Relu(scope, input);
    std::vector<Tensor> outputs;
    TF_CHECK_OK(session.Run(inputs, {result}, &outputs));

    auto output = outputs[0].matrix<Eigen::half>();

    EXPECT_EQ(output(0, 0), Eigen::half(2.0f));
    EXPECT_EQ(output(0, 1), Eigen::half(0.0f));
    EXPECT_EQ(output(1, 0), Eigen::half(0.0f));
    EXPECT_EQ(output(1, 1), Eigen::half(0.1f));
  }

  {
    Tensor float_data(DT_FLOAT, shape);
    test::FillValues<float>(&float_data, input_data);

    auto input =
        ops::Placeholder(scope, DT_FLOAT, ops::Placeholder::Shape(shape));

    ClientSession::FeedType inputs;
    inputs.emplace(input, float_data);

    auto result = ops::Relu(scope, input);
    std::vector<Tensor> outputs;
    TF_CHECK_OK(session.Run(inputs, {result}, &outputs));

    auto output = outputs[0].matrix<float>();

    EXPECT_EQ(output(0, 0), 2.0f);
    EXPECT_EQ(output(0, 1), 0.0f);
    EXPECT_EQ(output(1, 0), 0.0f);
    EXPECT_EQ(output(1, 1), 0.1f);
  }
}

TEST(DmlKernelTests, RealDiv) {
  TensorShape shape({2, 2});
  auto a_data = {2.0f, 7.0f, 9.0f, 50.0f};
  auto b_data = {1.0f, 2.0f, 3.0f, 4.0f};

  {
    Tensor half_a(DT_HALF, shape);
    test::FillValues<Eigen::half>(&half_a, a_data);

    Tensor half_b(DT_HALF, shape);
    test::FillValues<Eigen::half>(&half_b, b_data);

    TestBinaryCwiseOp<tensorflow::ops::RealDiv, Eigen::half>(half_a, half_b);
  }

  {
    Tensor float_a(DT_FLOAT, shape);
    test::FillValues<float>(&float_a, a_data);

    Tensor float_b(DT_FLOAT, shape);
    test::FillValues<float>(&float_b, b_data);

    TestBinaryCwiseOp<tensorflow::ops::RealDiv, float>(float_a, float_b);
  }
}

TEST(DmlKernelTests, Mul) {
  TensorShape shape({2, 2});
  auto a_data = {2.0f, 7.0f, 9.0f, 50.0f};
  auto b_data = {1.0f, 2.0f, 3.0f, 4.0f};

  {
    Tensor half_a(DT_HALF, shape);
    test::FillValues<Eigen::half>(&half_a, a_data);

    Tensor half_b(DT_HALF, shape);
    test::FillValues<Eigen::half>(&half_b, b_data);

    TestBinaryCwiseOp<tensorflow::ops::Mul, Eigen::half>(half_a, half_b);
  }

  {
    Tensor float_a(DT_FLOAT, shape);
    test::FillValues<float>(&float_a, a_data);

    Tensor float_b(DT_FLOAT, shape);
    test::FillValues<float>(&float_b, b_data);

    TestBinaryCwiseOp<tensorflow::ops::Mul, float>(float_a, float_b);
  }
}

TEST(DmlKernelTests, LogicalAnd) {
  auto a_data = Input::Initializer({{true, false}, {true, false}});
  auto b_data = Input::Initializer({{true, true}, {false, false}});
  TestBinaryCwiseOp<tensorflow::ops::LogicalAnd, bool>(a_data, b_data);
}

TEST(DmlKernelTests, TanhGrad) {
  Input::Initializer y = {{1.0f, 2.0f}, {3.0f, 4.0f}};
  Input::Initializer dy = {{0.1f, 0.3f}, {0.5f, 0.7f}};

  TestBinaryCwiseOp<tensorflow::ops::internal::TanhGrad, float>(y, dy);
}

TEST(DmlKernelTests, SigmoidGrad) {
  Input::Initializer y = {{1.0f, 2.0f}, {3.0f, 4.0f}};
  Input::Initializer dy = {{0.1f, 0.3f}, {0.5f, 0.7f}};

  TestBinaryCwiseOp<tensorflow::ops::internal::SigmoidGrad, float>(y, dy);
}

}  // namespace tensorflow