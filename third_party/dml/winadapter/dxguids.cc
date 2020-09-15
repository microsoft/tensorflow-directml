// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

// This file's sole purpose is to initialize the GUIDs declared using the DEFINE_GUID macro.
#define INITGUID
#ifndef _WIN32
#include "winadapter.h"
#endif
#include "dxcore.h"
#include "d3d12.h"