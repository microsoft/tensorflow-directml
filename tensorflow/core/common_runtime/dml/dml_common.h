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

#include <vector>
#include <deque>
#include <mutex>
#include <chrono>

#ifndef _WIN32
typedef uint64_t UINT64, ULONG_PTR; // TODO: move to dx headers repo
#include "winadapter.h"
#endif

#include <directx/d3d12.h>
#include <wrl/client.h>
#include <wrl/implements.h>

#if _WIN32
#include <dxgi1_6.h>
#else
#include <directx/dxcore.h>
#endif

#include "absl/types/optional.h"
#include "absl/types/variant.h"
#include "absl/types/span.h"
#include "absl/container/inlined_vector.h"

#ifndef DML_BUILD_WINDOWS
#include "dxcore.h"
#endif

#define DML_TARGET_VERSION_USE_LATEST 1
#define DMLX_USE_ABSEIL 1

#include "DirectMLConfig.h"
#if DIRECTML_PREVIEW_BUILD
#include "DirectMLPreview.h"
#else
#include "DirectML.h"
#endif

#include "DirectMLX.h"
#include "d3dx12.h"

// When building for Windows, this include will statically assert that the declared IIDs
// match those found in the Windows SDK.
#include "dxguids.h"
#include "dml_guids.h"

// Drop-in C++11-compatible replacements for optional, variant, and small_vector
// which are used by the external ApiHelpers.h header
namespace dml {
using absl::optional;
using absl::make_optional;
using absl::nullopt;
using absl::variant;
using absl::visit;

template <typename T, size_t N>
using small_vector = absl::InlinedVector<T, N>;
}  // namespace dml

using byte = unsigned char;

#include "dml_error_handling.h"

// This enum is deliberately not given contiguous integral values to prevent
// accidental use of these values as indices, because doing so is usually wrong.
// These axes should be thought of as being symbolic/logical, because the
// mapping between logical axis and index of the dimension depends on the tensor
// layout. For example, the location of Conv2D's 'C' dimension depends on that
// operator's data_layout attribute so it's usually wrong to assume that e.g.
// the 'C' dimension always lives at index 1. Use the `GetDmlDimensionIndex`
// helper utility to convert from this tensor axis to the corresponding index of
// the dimension in a DML_TENSOR_DESC.
enum class DmlTensorAxis : char {
  N = 'N',
  C = 'C',
  D = 'D',
  H = 'H',
  W = 'W',
};

// These are placed in a namespace for convenience so you can `using namespace`
// to save from typing DmlTensorAxis:: everywhere.
namespace DmlTensorAxes {
static constexpr auto N = DmlTensorAxis::N;
static constexpr auto C = DmlTensorAxis::C;
static constexpr auto D = DmlTensorAxis::D;
static constexpr auto H = DmlTensorAxis::H;
static constexpr auto W = DmlTensorAxis::W;
}  // namespace DmlTensorAxes

using DmlTensorLayoutBase = absl::InlinedVector<DmlTensorAxis, DML_TENSOR_DIMENSION_COUNT_MAX>;

struct DmlTensorLayout : public DmlTensorLayoutBase {
  DmlTensorLayout() = default;

  // Inherit constructors from base
  using DmlTensorLayoutBase::DmlTensorLayoutBase;

  static DmlTensorLayout Nchw() {
    return {DmlTensorAxis::N, DmlTensorAxis::C, DmlTensorAxis::H,
            DmlTensorAxis::W};
  };
  static DmlTensorLayout Nhwc() {
    return {DmlTensorAxis::N, DmlTensorAxis::H, DmlTensorAxis::W,
            DmlTensorAxis::C};
  };
};

// Some operators only handle 4 dimensions.
static constexpr uint32_t kNchwDimensionCount = 4;

static constexpr uint32_t kNchwSpatialDimensionCount = 2;
static constexpr uint32_t kNcdhwDimensionCount = 5;
static constexpr uint32_t kNcdhwSpatialDimensionCount = 3;

// 8 dimensions are supported for elementwise operators
static constexpr uint32_t kBinaryCwiseOpMaxDimCount = 8;

// The batch and channel dimensions of NCW, NCHW, NCDHW....
static constexpr uint32_t kNonspatialDimensionCount = 2;

namespace WRL {
#ifdef DML_BUILD_WINDOWS
// Helper wrapper over Microsoft::WRL::RuntimeClass. This is already implemented
// in wrladapter.h, so it is only declared for Windows builds.
template <typename... TInterfaces>
using Base = Microsoft::WRL::RuntimeClass<
    Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>,
    TInterfaces...>;
#endif
using namespace Microsoft::WRL;
}  // namespace WRL