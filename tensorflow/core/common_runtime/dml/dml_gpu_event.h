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

#include "dml_common.h"

namespace tensorflow {

// Represents a fence which will be signaled at some point (usually by the GPU).
struct DmlGpuEvent {
  uint64_t fence_value;
  Microsoft::WRL::ComPtr<ID3D12Fence> fence;

  bool IsSignaled() const { return fence->GetCompletedValue() >= fence_value; }

  // Blocks until IsSignaled returns true.
  void WaitForSignal() const {
    if (IsSignaled()) return;  // early-out

      // Workaround for Bug 25381212: Supplying a null HANDLE to
      // SetEventOnCompletion causes the capture layer to deadlock during GPU
      // capture
      // TFDML #25381212
      //
      // On Windows, use a Win32 event + WaitForSingleObject which prevents
      // the deadlock in PIX when you supply nullptr to SetEventOnCompletion.
#ifdef _WIN32
    class UniqueHandle {
     public:
      explicit UniqueHandle(HANDLE handle) : m_handle(handle) {}
      UniqueHandle(const UniqueHandle&) = delete;
      UniqueHandle(UniqueHandle&& other) {
        m_handle = std::move(other.m_handle);
        other.m_handle = nullptr;
      }
      ~UniqueHandle() {
        if (m_handle) {
          CloseHandle(m_handle);
          m_handle = nullptr;
        }
      }
      UniqueHandle& operator=(UniqueHandle& other) = delete;
      UniqueHandle& operator=(UniqueHandle&& other)
      {
          m_handle = std::move(other.m_handle);
          other.m_handle = nullptr;
          return *this;
      }
      HANDLE get() { return m_handle; }
      operator bool() const { return m_handle; }

     private:
      HANDLE m_handle = nullptr;
    };

    UniqueHandle h(CreateEvent(nullptr, TRUE, FALSE, nullptr));

    if (!h) {
      DML_CHECK_SUCCEEDED(HRESULT_FROM_WIN32(GetLastError()));
    }

    DML_CHECK_SUCCEEDED(fence->SetEventOnCompletion(fence_value, h.get()));

    WaitForSingleObject(h.get(), INFINITE);
#else
    // Nullptr event blocks CPU until completion (creates and waits on an event
    // internally)
    DML_CHECK_SUCCEEDED(fence->SetEventOnCompletion(fence_value, nullptr));
#endif
  }
};

}  // namespace tensorflow
