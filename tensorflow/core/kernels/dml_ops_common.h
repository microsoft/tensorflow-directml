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
#include "tensorflow/core/common_runtime/dml/dml_kernel_context.h"
#include "tensorflow/core/common_runtime/dml/dml_operator_helper.h"
#include "tensorflow/core/common_runtime/dml/dml_tensor_desc.h"

namespace tensorflow {

struct DmlKernelParams {
  // For each input to the DML operator, indicates the index of the
  // corresponding input tensor to the kernel. For example, if the DML operator
  // takes 3 inputs in the following order:
  //   A
  //   B
  //   {null}
  //   C
  //
  // But the kernel receives its inputs in this order:
  //   C
  //   A
  //   B
  //   D
  //
  // The kernel_input_indices should be an array of size 4, with the following
  // values: {1, 2, nullopt, 0}
  //
  // This parameter is optional. If the kernel and the DML operator have the
  // same number of inputs, and they're in the same order, then this vector can
  // be empty.
  absl::InlinedVector<absl::optional<uint32_t>, 8> kernel_input_indices;

  // Same as above, but for output tensors.
  absl::InlinedVector<absl::optional<uint32_t>, 4> kernel_output_indices;

  // Specifies a tensor shape to broadcast/coerce all input tensors to. This is
  // often used to handle the numpy-style broadcasting as required by TF. For
  // example, DML requires that all tensors supplied to ELEMENT_WISE_ADD must
  // have identical sizes and dimension counts. However due to broadcasting, a
  // kernel for Add might receive one input with shape {1, 2}, to be added to a
  // tensor with shape {5, 1, 2}. In this case the expectation is that the
  // {1, 2} tensor should be broadcast into a tensor of shape {5, 1, 2}.
  //
  // In the above example, supplying a tensor shape of {5, 1, 2} will cause the
  // {1, 2} tensor's desc to be automatically broadcasted to {5, 1, 2}.
  absl::optional<TensorShape> input_shape;

  // Same as above, but for the output shape(s) of the kernel.
  absl::optional<TensorShape> output_shape;

  // OutputIndex -> InputIndex ref forwarding mapping
  absl::InlinedVector<absl::optional<uint32_t>, 8> output_refs_forwarding;
};

struct DmlTensorInfo {
  DmlTensorDesc desc;

  // The index of the corresponding tensor as supplied to the kernel.
  // See the comment in DmlKernelParams for an explanation for what this does.
  uint32_t kernel_index;
};

struct DmlKernelTensors {
  absl::InlinedVector<absl::optional<DmlTensorInfo>, 8> inputs;
  absl::InlinedVector<absl::optional<DmlTensorInfo>, 4> outputs;
  absl::InlinedVector<absl::optional<uint32_t>, 8> output_refs_forwarding;
};

// Abstract base class of all DirectML kernel implementations. Note that this
// does not implement OpKernel; see the DmlKernelWrapper class for details.
class DmlKernel {
 public:
  virtual ~DmlKernel() = default;

  // Computes this kernel. By default, this simply submits the compiled DML
  // operator for execution. A DmlGpuEvent is returned which becomes signaled
  // when the kernel completes execution on the GPU. This method is thread-safe.
  virtual StatusOr<DmlGpuEvent> Compute(DmlKernelContext* ctx) const;

  absl::Span<const absl::optional<uint32_t>> GetOutputRefsForwarding() const {
    return output_refs_forwarding_;
  }

  const InitializationHelper* GetInitializationHelper() const {
    return init_helper_.get();
  }

 protected:
  // A helper to set up the input and output tensor descs for this kernel in a
  // DirectML-canonical format (i.e. DmlTensorDesc). DmlTensorDesc handles
  // broadcasting and translation/coercion of TF-style shapes into DML-style
  // shapes. The returned tensor descs are also helpful for setting up the
  // DML_OPERATOR_DESC which is necessary for initializing the kernel.
  static DmlKernelTensors GetTensorInfos(DmlKernelConstruction* ctx,
                                         const DmlKernelParams& params);

  static DmlTensorDesc CreateTensorDescFromInput(
      DmlKernelConstruction* ctx, uint32_t kernel_index,
      absl::Span<const DmlTensorAxis> tensor_layout,
      const absl::optional<TensorShape>& tensor_shape = absl::nullopt);

  static DmlTensorDesc CreateTensorDescFromOutput(
      DmlKernelConstruction* ctx, uint32_t kernel_index,
      absl::Span<const DmlTensorAxis> tensor_layout,
      const absl::optional<TensorShape>& tensor_shape = absl::nullopt);

  static DmlTensorDesc CreateTensorDescFromInput(
      DmlKernelConstruction* ctx, uint32_t kernel_index,
      const absl::optional<TensorShape>& tensor_shape = absl::nullopt);

  static DmlTensorDesc CreateTensorDescFromOutput(
      DmlKernelConstruction* ctx, uint32_t kernel_index,
      const absl::optional<TensorShape>& tensor_shape = absl::nullopt);

  static absl::InlinedVector<DML_TENSOR_DESC, 8> GetDmlTensorDescs(
      absl::Span<absl::optional<DmlTensorInfo>> tensor_infos);

  template <size_t Size>
  static absl::InlinedVector<DML_TENSOR_DESC, 8> GetDmlTensorDescs(
      absl::InlinedVector<absl::optional<DmlTensorInfo>, Size>& tensor_infos) {
    return GetDmlTensorDescs(absl::Span<absl::optional<DmlTensorInfo>>(
        tensor_infos.data(), tensor_infos.size()));
  }

  // Initializes the kernel. Use the GetTensorInfos helper to compute the
  // required tensor descs. This takes ownership of the DmlKernelTensors so when
  // this method returns, the `tensor_descs` object will be empty. This method
  // must be called exactly once.
  void Initialize(DmlKernelConstruction* ctx, DmlKernelTensors&& tensor_descs,
                  const DML_OPERATOR_DESC& op_desc);

  // Same as above, but for callers who want to compile their own DML op.
  void Initialize(DmlKernelConstruction* ctx, DmlKernelTensors&& tensor_descs,
                  IDMLCompiledOperator* compiled_op);

  // For ops that skip the DML graph (e.g. BlockLSTM in seq_len_max==0 case)
  void InitializeAsNoOp(DmlKernelConstruction* ctx) {
    init_helper_ = ctx->GetInitializationHelper();
  }

  // Creates buffers over all input tensors, whose bindings can be given to
  // ExecuteOperator. This should only be called by children classes if
  // overriding Compute and not calling DmlKernel::Compute
  absl::InlinedVector<D3D12BufferRegion, 8> CreateInputBuffers(
      DmlKernelContext* ctx) const;

  // Creates buffers over all output tensors, whose bindings can be given to
  // ExecuteOperator. This should only be called by children classes if
  // overriding Compute and not calling DmlKernel::Compute
  absl::InlinedVector<D3D12BufferRegion, 4> CreateOutputBuffers(
      DmlKernelContext* ctx) const;

  const DML_BUFFER_BINDING* GetPersistentResourceBinding() const;

  IDMLCompiledOperator* GetCompiledOp() const { return compiled_op_.Get(); }

 private:
  Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_op_;
  DmlBuffer persistent_resource_;
  absl::optional<DML_BUFFER_BINDING> persistent_resource_binding_;
  std::shared_ptr<const InitializationHelper> init_helper_;

  // The order and count of these descs match the DML operator, which might be
  // different to the order/number of inputs TF supplies to the kernel. Each
  // DmlTensorInfo contains an index which maps the DML input to the correct
  // input on the TF kernel.
  //
  // A value of nullopt indicates an optional tensor that's empty.
  absl::InlinedVector<absl::optional<DmlTensorInfo>, 8> input_descs_;

  // Same as `input_descs_`, but for output tensors.
  absl::InlinedVector<absl::optional<DmlTensorInfo>, 4> output_descs_;

  // This parameter is optional. If the kernel doesn't have any ref outputs,
  // this vector can be empty. Note that the vector's index represents the index
  // of the output; not the index of the input.
  //
  // For example, if we need to forward the first input to the first output, we
  // would have a vector with a single element: {0}
  // On the other hand, if we need to forward the second input to the first
  // output, we would have the following vector: {1}
  // Finally, if we have 2 outputs but only the second one is a ref output and
  // needs the second input to be forwarded to it, we would have the following
  // vector: {nullopt, 1}
  absl::InlinedVector<absl::optional<uint32_t>, 8> output_refs_forwarding_;
};

template <typename T>
struct TfTensorTypeTraits;

template <>
struct TfTensorTypeTraits<float> {
  static constexpr DML_TENSOR_DATA_TYPE dml_type = DML_TENSOR_DATA_TYPE_FLOAT32;
  static float FromFloat(float val) { return val; }
  static DML_SCALAR_UNION ToDmlScalar(float val) {
    DML_SCALAR_UNION scalar;
    scalar.Float32 = val;
    return scalar;
  }
};

template <>
struct TfTensorTypeTraits<Eigen::half> {
  static constexpr DML_TENSOR_DATA_TYPE dml_type = DML_TENSOR_DATA_TYPE_FLOAT16;
  static Eigen::half FromFloat(float val) {
    return Eigen::half_impl::float_to_half_rtne(val);
  }
  static DML_SCALAR_UNION ToDmlScalar(Eigen::half val) {
    DML_SCALAR_UNION scalar;
    *reinterpret_cast<Eigen::half*>(scalar.Bytes) = val;
    return scalar;
  }
};

template <>
struct TfTensorTypeTraits<uint32_t> {
  static constexpr DML_TENSOR_DATA_TYPE dml_type = DML_TENSOR_DATA_TYPE_UINT32;
  static DML_SCALAR_UNION ToDmlScalar(uint32_t val) {
    DML_SCALAR_UNION scalar;
    scalar.UInt32 = val;
    return scalar;
  }
};

template <>
struct TfTensorTypeTraits<uint16_t> {
  static constexpr DML_TENSOR_DATA_TYPE dml_type = DML_TENSOR_DATA_TYPE_UINT16;
  static DML_SCALAR_UNION ToDmlScalar(uint16_t val) {
    DML_SCALAR_UNION scalar;
    scalar.UInt16 = val;
    return scalar;
  }
};

template <>
struct TfTensorTypeTraits<int32_t> {
  static constexpr DML_TENSOR_DATA_TYPE dml_type = DML_TENSOR_DATA_TYPE_INT32;
  static DML_SCALAR_UNION ToDmlScalar(int32_t val) {
    DML_SCALAR_UNION scalar;
    scalar.Int32 = val;
    return scalar;
  }
};

}  // namespace tensorflow

// Extends DirectMLX with tensorflow helpers
namespace dml {

DML_SCALAR_UNION ScalarUnion(double value, DML_TENSOR_DATA_TYPE data_type);

template <typename T>
Expression ScalarTensor(Graph& scope, T value, TensorDesc::Dimensions sizes) {
  dml::TensorDesc::Dimensions scalar_dims(sizes.size(), 1);
  dml::TensorDesc::Dimensions scalar_strides(sizes.size(), 0);

  auto scalar = dml::Reinterpret(
      dml::FillValueConstant(
          scope, scalar_dims, tensorflow::TfTensorTypeTraits<T>::dml_type,
          tensorflow::TfTensorTypeTraits<T>::ToDmlScalar(value)),
      sizes,         /* broadcast shape */
      scalar_strides /* broadcast strides */
  );
  return scalar;
}

template <typename T>
Expression Sequence(Graph& scope, T start, T step,
                    TensorDesc::Dimensions sizes) {
  auto seq = dml::FillValueSequence(
      scope, sizes, tensorflow::TfTensorTypeTraits<T>::dml_type,
      tensorflow::TfTensorTypeTraits<T>::ToDmlScalar(start),
      tensorflow::TfTensorTypeTraits<T>::ToDmlScalar(step));

  return seq;
}

// Zero is special since the bit representation is the same regardless of type,
// so there's no need to have a function template. The dataType is used only for
// the tensor desc.
inline Expression ZeroTensor(Graph& scope, DML_TENSOR_DATA_TYPE dataType,
                             TensorDesc::Dimensions size) {
  DML_SCALAR_UNION scalar_value{};
  auto scalar = dml::Reinterpret(
      dml::FillValueConstant(scope, dml::TensorDesc::Dimensions{1, 1, 1, 1},
                             dataType, scalar_value),
      size,                                   /* broadcast shape */
      dml::TensorDesc::Dimensions{0, 0, 0, 0} /* broadcast strides */
  );
  return scalar;
}

inline Expression ActivationRelu6(Expression input) {
  return dml::Clip(input, 0, 6);
}

}  // namespace dml