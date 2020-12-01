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

#ifdef _WIN32
#include <Windows.h>
#else
#include "winadapter.h"
#endif

#include "tensorflow/core/platform/default/logging.h"

namespace tensorflow {
[[noreturn]] void DmlHandleFailedHr(HRESULT hr, const char* expression,
                                    const char* file, int line);
}  // namespace tensorflow

#define DML_CHECK_SUCCEEDED(x)                                    \
  do {                                                            \
    HRESULT _hr = (x);                                            \
    if (TF_PREDICT_FALSE(FAILED(_hr))) {                          \
      tensorflow::DmlHandleFailedHr(_hr, #x, __FILE__, __LINE__); \
    }                                                             \
  } while (0)
