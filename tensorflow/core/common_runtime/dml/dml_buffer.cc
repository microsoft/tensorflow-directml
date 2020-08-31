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

#include "dml_buffer.h"

#include "dml_bfc_allocator.h"
#include "dml_device.h"

namespace tensorflow {

/*explicit*/ DmlBuffer::DmlBuffer(DmlAllocator* allocator,
                                  uint64_t size_in_bytes)
    : allocator_(allocator) {
  ptr_ = allocator_->AllocateRaw(0, size_in_bytes);

  // If the allocation fails, leave this buffer empty
  if (!ptr_) {
    return;
  }

  buffer_region_ = allocator_->CreateBufferRegion(ptr_, size_in_bytes);
}

DmlBuffer::~DmlBuffer() {
  // The only time buffer_region_ will be null is if this object was moved-from,
  // or if allocation failed in the constructor. In either case, we have nothing
  // to free.
  if (buffer_region_) {
    allocator_->DeallocateRaw(ptr_);
  }
}

ID3D12Resource* DmlBuffer::Resource() const {
  return buffer_region_ ? buffer_region_.Resource() : nullptr;
}

uint64_t DmlBuffer::Offset() const {
  return buffer_region_ ? buffer_region_.Offset() : 0;
}

uint64_t DmlBuffer::SizeInBytes() const {
  return buffer_region_ ? buffer_region_.SizeInBytes() : 0;
}

DML_BUFFER_BINDING DmlBuffer::GetBufferBinding() const {
  return buffer_region_ ? buffer_region_.GetBufferBinding()
                        : DML_BUFFER_BINDING{};
}

}  // namespace tensorflow
