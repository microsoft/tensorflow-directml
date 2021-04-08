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

// This file's sole purpose is to initialize the GUIDs declared using the DEFINE_GUID macro. This file
// is used instead of dxguids.cpp in the DirectX-Headers repository for two reasons:
// 1. DXGI IIDs aren't defined in DirectX-Headers
// 2. DirectML IIDs aren't defined in DirectX-Headers

#define INITGUID

#ifndef _WIN32
#include "winadapter.h"
#include <directx/d3d12.h>
#include <directx/dxcore.h>
#include "DirectML.h"
// #include "dml_guids.h"
#else
#include <dxgi1_6.h>
#include <directx/d3d12.h>
#endif