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

#include "tensorflow/core/common_runtime/dml/dml_operator_helper.h"
#include "tensorflow/core/common_runtime/dml/dml_util.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/dml_kernel_wrapper.h"
#include "tensorflow/core/kernels/dml_ops_common.h"
#include "tensorflow/core/kernels/random_op.h"
#include "tensorflow/core/kernels/stateless_random_ops.h"
#include "tensorflow/core/util/guarded_philox_random.h"

namespace tensorflow {
using CPUDevice = Eigen::ThreadPoolDevice;

// Helpers to convert random uniform bits to a real uniform distribution. This
// approach outputs a floating-point value with sign=0 (positive), exponent=2^0,
// and mantissa set to the lowest-order M bits from the random generator output
// (M = bits in the floating-point mantissa). For example, FP32 consumes the
// lowest 23 bits from each 32-bit generator value; FP16 consumes the lowest 10
// bits from each 32-bit generator value. FP64 (not implemented) would require
// 2 generator values per output vaule, and it would use the lowest 52 bits.
dml::Expression UniformFloat(dml::Graph& scope, dml::Expression input_state,
                             uint32_t element_count) {
  // FP32 has 1 sign bit, 8 exponent bits, and 23 mantissa bits.
  constexpr uint32_t sign_and_exponent_value = ((1 << (8 - 1)) - 1) << 23;
  constexpr uint32_t mantissa_mask_value = (1 << 23) - 1;

  auto generator_outputs =
      dml::RandomGenerator(input_state, {1, 1, 1, element_count}, false);
  auto random_bits = generator_outputs.values;

  auto sign_and_exponent = dml::ScalarTensor(scope, sign_and_exponent_value,
                                             random_bits.GetOutputDesc().sizes);

  auto mantissa_mask = dml::ScalarTensor(scope, mantissa_mask_value,
                                         random_bits.GetOutputDesc().sizes);

  auto result = sign_and_exponent | (random_bits & mantissa_mask);

  return dml::Reinterpret(result, DML_TENSOR_DATA_TYPE_FLOAT32) - 1.0f;
}

dml::Expression UniformHalf(dml::Graph& scope, dml::Expression input_state,
                            uint32_t element_count) {
  // FP16 has 1 sign bit, 5 exponent bits, and 10 mantissa bits.
  constexpr uint32_t sign_and_exponent_value = ((1 << (5 - 1)) - 1) << 10;
  constexpr uint32_t mantissa_mask_value = (1 << 10) - 1;

  auto generator_outputs =
      dml::RandomGenerator(input_state, {1, 1, 1, element_count}, false);
  auto random_bits = generator_outputs.values;

  auto sign_and_exponent = dml::ScalarTensor(scope, sign_and_exponent_value,
                                             random_bits.GetOutputDesc().sizes);

  auto mantissa_mask = dml::ScalarTensor(scope, mantissa_mask_value,
                                         random_bits.GetOutputDesc().sizes);

  auto result = sign_and_exponent | (random_bits & mantissa_mask);

  result = dml::Cast(result, DML_TENSOR_DATA_TYPE_UINT16);

  return dml::Reinterpret(result, DML_TENSOR_DATA_TYPE_FLOAT16) - 1.0f;
}

class StatelessRandomUniformInitHelper : public InitializationHelper {
 public:
  using Attributes = EmptyAttributes;

  StatelessRandomUniformInitHelper(OpKernelContext* ctx,
                                   std::shared_ptr<const Attributes> attr) {
    const Tensor& shape_t = ctx->input(0);
    const Tensor& seed_t = ctx->input(1);
    TensorShape shape;

    OP_REQUIRES_OK(ctx, ctx->op_kernel().MakeShape(shape_t, &shape));
    OP_REQUIRES(ctx, seed_t.dims() == 1 && seed_t.dim_size(0) == 2,
                errors::InvalidArgument("seed must have shape [2], not ",
                                        seed_t.shape().DebugString()));

    OP_REQUIRES_OK(ctx, GenerateKey(seed_t, &key_, &counter_));

    output_shape_ = std::move(shape);
  }

  const TensorShape& GetOutputShape() const { return output_shape_; }
  const random::PhiloxRandom::Key GetKey() const { return key_; }
  const random::PhiloxRandom::ResultType GetCounter() const { return counter_; }

 private:
  TensorShape output_shape_;
  random::PhiloxRandom::Key key_;
  random::PhiloxRandom::ResultType counter_;
};

using InitHelper = StatelessRandomUniformInitHelper;

class StatelessRandomUniformShapeHelper : public ShapeHelper {
 public:
  std::vector<TensorShape> GetOutputShapes(
      OpKernelContext* ctx,
      const InitializationHelper* initialization_helper) const override {
    auto init_helper = static_cast<const InitHelper*>(initialization_helper);
    return {init_helper->GetOutputShape()};
  }
};

class DmlStatelessRandomUniformKernel : public DmlKernel {
  std::array<uint32_t, 6> input_state_;

 public:
  using InitHelper = tensorflow::InitHelper;

  explicit DmlStatelessRandomUniformKernel(DmlKernelConstruction* ctx,
                                           const InitHelper* init_helper) {
    auto num_elements =
        static_cast<uint32_t>(init_helper->GetOutputShape().num_elements());

    // Copy counter & key into the input_state_ buffer.
    auto counter = init_helper->GetCounter();
    auto key = init_helper->GetKey();
    input_state_[0] = counter[0];
    input_state_[1] = counter[1];
    input_state_[2] = counter[2];
    input_state_[3] = counter[3];
    input_state_[4] = key[0];
    input_state_[5] = key[1];

    // Reserve an input binding, even though TF doesn't provide a (device)
    // input tensor. We will swap in a temporary buffer and upload the CPU
    // state at compute time.
    DmlTensorInfo state_info;
    state_info.kernel_index = 0;
    std::array<uint32_t, 4> state_sizes = {1, 1, 1, 6};
    state_info.desc =
        DmlTensorDesc::Create(DT_UINT32, state_sizes, state_sizes);

    // Flatten output shape for DirectML.
    DmlTensorInfo output_info;
    output_info.kernel_index = 0;
    std::array<uint32_t, 4> output_sizes = {1, 1, 1, num_elements};
    output_info.desc = DmlTensorDesc::Create(ctx->GetOutputDataType(0),
                                             output_sizes, output_sizes);

    DmlKernelTensors tensors;
    tensors.inputs = {state_info};
    tensors.outputs = {output_info};

    auto inputs = GetDmlTensorDescs(tensors.inputs);
    auto scope = dml::Graph(ctx->GetDmlDevice());
    auto input_state = dml::InputTensor(scope, 0, inputs[0]);

    dml::Expression result;
    if (ctx->GetOutputDataType(0) == DT_FLOAT) {
      result = UniformFloat(scope, input_state, num_elements);
    } else {
      DCHECK(ctx->GetOutputDataType(0) == DT_HALF);
      result = UniformHalf(scope, input_state, num_elements);
    }

    Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
        scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

    Initialize(ctx, std::move(tensors), compiled_op.Get());
  }

  StatusOr<DmlGpuEvent> Compute(DmlKernelContext* ctx) const override {
    DmlBuffer input_state_buffer =
        ctx->AllocateDefaultBuffer(6 * sizeof(uint32_t));
    D3D12BufferRegion output_buffer =
        ctx->CreateBufferForTensor(*ctx->GetOutputTensor(0));

    if (!input_state_buffer) {
      return errors::ResourceExhausted("OOM when allocating a buffer of ",
                                       6 * sizeof(uint32_t), " bytes");
    }

    absl::InlinedVector<absl::optional<DML_BUFFER_BINDING>, 1> input_bindings;
    input_bindings.push_back(input_state_buffer.GetBufferBinding());

    absl::InlinedVector<absl::optional<DML_BUFFER_BINDING>, 1> output_bindings;
    output_bindings.push_back(output_buffer.GetBufferBinding());

    // Upload generator input state.
    auto byte_ptr = reinterpret_cast<const uint8_t*>(input_state_.data());
    auto byte_span =
        absl::MakeSpan(byte_ptr, input_state_.size() * sizeof(input_state_[0]));

    ctx->CopyHostToBuffer(input_state_buffer.Resource(),
                          input_state_buffer.Offset(), byte_span);

    return ctx->ExecuteOperator(GetCompiledOp(), GetPersistentResourceBinding(),
                                input_bindings, output_bindings);
  }
};

#define DML_REGISTER_KERNEL(type)                       \
  REGISTER_KERNEL_BUILDER(                              \
      Name("StatelessRandomUniform")                    \
          .Device(DEVICE_DML)                           \
          .HostMemory("shape")                          \
          .HostMemory("seed")                           \
          .TypeConstraint<type>("dtype"),               \
      DmlKernelWrapper<DmlStatelessRandomUniformKernel, \
                       StatelessRandomUniformShapeHelper>);
TF_CALL_DML_FLOAT_TYPES(DML_REGISTER_KERNEL);
#undef DML_REGISTER_KERNEL

// ----------------------------------------------------------------------------

template <typename TKernel, typename TShapeHelper,
          DmlKernelCachePolicy cache_policy = DmlKernelCachePolicy::Default>
class DmlPhiloxWrapper
    : public DmlKernelWrapper<TKernel, TShapeHelper, cache_policy> {
 public:
  explicit DmlPhiloxWrapper(OpKernelConstruction* ctx)
      : DmlKernelWrapper<TKernel, TShapeHelper, cache_policy>(ctx) {
    OP_REQUIRES_OK(ctx, generator_.Init(ctx));
  }

  StatusOr<DmlGpuEvent> ComputeKernel(
      DmlKernel* kernel, DmlKernelContext* context) const override {
    return static_cast<TKernel*>(kernel)->Compute(context, generator_);
  }

 protected:
  mutable GuardedPhiloxRandom generator_;
};

class RandomUniformInitHelper : public InitializationHelper {
 public:
  using Attributes = EmptyAttributes;

  RandomUniformInitHelper(OpKernelContext* ctx,
                          std::shared_ptr<const Attributes> attr) {
    const Tensor& shape_t = ctx->input(0);
    TensorShape shape;
    OP_REQUIRES_OK(ctx, ctx->op_kernel().MakeShape(shape_t, &shape));
    output_shape_ = std::move(shape);
  }

  const TensorShape& GetOutputShape() const { return output_shape_; }

 private:
  TensorShape output_shape_;
};

class RandomUniformShapeHelper : public ShapeHelper {
 public:
  std::vector<TensorShape> GetOutputShapes(
      OpKernelContext* ctx,
      const InitializationHelper* initialization_helper) const override {
    auto init_helper =
        static_cast<const RandomUniformInitHelper*>(initialization_helper);
    return {init_helper->GetOutputShape()};
  }
};

class DmlRandomUniformKernel : public DmlKernel {
  absl::optional<DmlBuffer> state_buffer_;
  uint32_t num_output_elements_;

 public:
  using InitHelper = tensorflow::RandomUniformInitHelper;

  explicit DmlRandomUniformKernel(DmlKernelConstruction* ctx,
                                  const RandomUniformInitHelper* init_helper) {
    num_output_elements_ =
        static_cast<uint32_t>(init_helper->GetOutputShape().num_elements());

    state_buffer_ = ctx->AllocateDefaultBuffer(6 * sizeof(uint32_t));

    OP_REQUIRES(ctx->GetOpKernelContext(), state_buffer_,
                errors::ResourceExhausted("OOM when allocating a buffer of ",
                                          6 * sizeof(uint32_t), " bytes"));

    // Reserve input state binding. This will point at state_buffer_.
    DmlTensorInfo state_info;
    state_info.kernel_index = 0;
    std::array<uint32_t, 4> state_sizes = {1, 1, 1, 6};
    state_info.desc =
        DmlTensorDesc::Create(DT_UINT32, state_sizes, state_sizes);

    // Flatten output shape for DirectML.
    DmlTensorInfo output_info;
    output_info.kernel_index = 0;
    std::array<uint32_t, 4> output_sizes = {1, 1, 1, num_output_elements_};
    output_info.desc = DmlTensorDesc::Create(ctx->GetOutputDataType(0),
                                             output_sizes, output_sizes);

    DmlKernelTensors tensors;
    tensors.inputs = {state_info};
    tensors.outputs = {output_info};

    auto inputs = GetDmlTensorDescs(tensors.inputs);
    auto scope = dml::Graph(ctx->GetDmlDevice());
    auto input_state = dml::InputTensor(scope, 0, inputs[0]);

    dml::Expression result;
    if (ctx->GetOutputDataType(0) == DT_FLOAT) {
      result = UniformFloat(scope, input_state, num_output_elements_);
    } else {
      DCHECK(ctx->GetOutputDataType(0) == DT_HALF);
      result = UniformHalf(scope, input_state, num_output_elements_);
    }

    Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op =
        scope.Compile(DML_EXECUTION_FLAG_NONE, {result});

    Initialize(ctx, std::move(tensors), compiled_op.Get());
  }

  StatusOr<DmlGpuEvent> Compute(DmlKernelContext* ctx,
                                GuardedPhiloxRandom& generator) const {
    D3D12BufferRegion output_buffer =
        ctx->CreateBufferForTensor(*ctx->GetOutputTensor(0));

    absl::InlinedVector<absl::optional<DML_BUFFER_BINDING>, 1> input_bindings;
    input_bindings.push_back(state_buffer_->GetBufferBinding());

    absl::InlinedVector<absl::optional<DML_BUFFER_BINDING>, 1> output_bindings;
    output_bindings.push_back(output_buffer.GetBufferBinding());

    // Upload generator state. Note that generator_.ReserveRandomOutputs() is
    // thread safe and doesn't actually invoke the Philox generator; it simply
    // returns the current counter and then advances its internal counter.
    std::array<uint32_t, 6> state_buf;
    auto philox_state =
        generator.ReserveRandomOutputs(num_output_elements_, 256);
    state_buf[0] = philox_state.counter()[0];
    state_buf[1] = philox_state.counter()[1];
    state_buf[2] = philox_state.counter()[2];
    state_buf[3] = philox_state.counter()[3];
    state_buf[4] = philox_state.key()[0];
    state_buf[5] = philox_state.key()[1];

    auto byte_ptr = reinterpret_cast<const uint8_t*>(state_buf.data());
    auto byte_span =
        absl::MakeSpan(byte_ptr, state_buf.size() * sizeof(state_buf[0]));

    ctx->CopyHostToBuffer(state_buffer_->Resource(), state_buffer_->Offset(),
                          byte_span);

    return ctx->ExecuteOperator(GetCompiledOp(), GetPersistentResourceBinding(),
                                input_bindings, output_bindings);
  }
};

#define DML_REGISTER_KERNEL(type)         \
  REGISTER_KERNEL_BUILDER(                \
      Name("RandomUniform")               \
          .Device(DEVICE_DML)             \
          .HostMemory("shape")            \
          .TypeConstraint<type>("dtype"), \
      DmlPhiloxWrapper<DmlRandomUniformKernel, RandomUniformShapeHelper>);
TF_CALL_DML_FLOAT_TYPES(DML_REGISTER_KERNEL);
#undef DML_REGISTER_KERNEL

// ----------------------------------------------------------------------------

// Emulates a DML philox PRNG+distribution by executing it on the CPU and
// copying the results to the GPU.
template <class Distribution>
class DmlEmulatedPhiloxRandomKernel : public OpKernel {
 public:
  typedef typename Distribution::ResultElementType T;
  explicit DmlEmulatedPhiloxRandomKernel(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, generator_.Init(ctx));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& shape = ctx->input(0);

    TensorShape output_shape;
    OP_REQUIRES_OK(ctx, MakeShape(shape, &output_shape));

    AllocatorAttributes host_attrs;
    host_attrs.set_on_host(true);

    Tensor host_output;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_temp(ctx->expected_output_dtype(0), output_shape,
                                &host_output, host_attrs));

    auto output_flat = host_output.flat<T>();
    functor::FillPhiloxRandom<CPUDevice, Distribution>()(
        ctx, ctx->eigen_device<CPUDevice>(),
        // Multiplier 256 is the same as in FillPhiloxRandomTask; do not change
        // it just here.
        generator_.ReserveRandomOutputs(output_flat.size(), 256),
        output_flat.data(), output_flat.size(), Distribution());

    Tensor* output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));

    Device* device = static_cast<Device*>(ctx->device());
    ctx->op_device_context()->CopyCPUTensorToDevice(
        &host_output, device, output, [](const Status& s) { TF_CHECK_OK(s); });
  }

 private:
  GuardedPhiloxRandom generator_;
};

#define DML_REGISTER_KERNEL(type)                                        \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("RandomStandardNormal")                                       \
          .Device(DEVICE_DML)                                            \
          .HostMemory("shape")                                           \
          .TypeConstraint<type>("dtype"),                                \
      DmlEmulatedPhiloxRandomKernel<                                     \
          random::NormalDistribution<random::PhiloxRandom, type>>);      \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("TruncatedNormal")                                            \
          .Device(DEVICE_DML)                                            \
          .HostMemory("shape")                                           \
          .TypeConstraint<type>("dtype"),                                \
      DmlEmulatedPhiloxRandomKernel<random::TruncatedNormalDistribution< \
          random::SingleSampleAdapter<random::PhiloxRandom>, type>>);

TF_CALL_DML_FLOAT_TYPES(DML_REGISTER_KERNEL);
#undef DML_REGISTER_KERNEL

}  // namespace tensorflow