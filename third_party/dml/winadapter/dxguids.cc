// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

// This file's sole purpose is to initialize the GUIDs declared using the DEFINE_GUID macro.
#define INITGUID

#ifndef _WIN32
#include "winadapter.h"
#include <dxcore.h>
#else
#include <dxgi1_6.h>
#endif

#include <d3d12.h>