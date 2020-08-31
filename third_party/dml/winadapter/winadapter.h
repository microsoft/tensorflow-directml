// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

// These #defines prevent the idl-generated headers from trying to include
// Windows.h from the SDK rather than this one.
#define RPC_NO_WINDOWS_H
#define COM_NO_WINDOWS_H

// These #defines prevent the idl-generated headers from pulling in anything else.
#define __RPCNDR_H__
#define __RPC_H__
#define __oaidl_h__
#define __ocidl_h__
#define __objidl_h__

// Block real definitions of SAL
#define SPECSTRINGS_H

// If including d3dx12.h, use the ATL CComPtr rather than WRL ComPtr
#define D3DX12_USE_ATL

// Allcaps type definitions
#include <cstddef>
#include <cstdint>
using std::size_t;
typedef unsigned char BYTE, *PBYTE;
typedef unsigned char *LPBYTE;

typedef BYTE BOOLEAN;
typedef BOOLEAN *PBOOLEAN;

typedef unsigned char byte;
typedef unsigned char boolean;

typedef uint32_t BOOL, *LPBOOL;

// Note: using fixed-width here to match Windows widths
// Specifically this is different for 'long' vs 'LONG'
typedef int32_t INT, *PINT;
typedef int32_t LONG, *PLONG, *LPLONG;
typedef uint32_t UINT, *PUINT;
typedef uint32_t ULONG, *PULONG;
typedef int64_t LONGLONG, *PLONGLONG;
typedef int64_t LONG_PTR, *PLONG_PTR;
typedef uint64_t ULONGLONG, *PULONGLONG;
typedef uint64_t ULONG_PTR, *PULONG_PTR;
typedef uint64_t UINT_PTR, *PUINT_PTR;
typedef int64_t INT_PTR, *PINT_PTR;

typedef int16_t SHORT, *PSHORT;
typedef uint16_t USHORT, *PUSHORT;
typedef int16_t INT16, *PINT16;
typedef uint16_t UINT16, *PUINT16;
typedef uint8_t UCHAR, *PUCHAR;
typedef int8_t INT8, *PINT8;
typedef uint8_t UINT8, *PUINT8;

typedef uint16_t WORD;
typedef uint32_t DWORD, *PDWORD, *LPDWORD;
typedef uint64_t DWORD_PTR, DWORD64;

typedef int32_t INT32, *PINT32;
typedef uint32_t UINT32, *PUINT32;
typedef int32_t LONG32, *PLONG32;
typedef uint32_t ULONG32, *PULONG32;
typedef int64_t INT64, *PINT64;
typedef uint64_t UINT64, *PUINT64;
typedef int64_t LONG64, *PLONG64;
typedef uint64_t ULONG64, *PULONG64;

typedef void VOID, *PVOID, *LPVOID, *HANDLE, *RPC_IF_HANDLE, *HMODULE;
typedef const void *LPCVOID;
typedef size_t SIZE_T, *PSIZE_T;
typedef float FLOAT;
typedef double DOUBLE;

// Note: WCHAR is not the same between Windows and Linux, to enable
// string manipulation APIs to work with resulting strings.
// APIs to D3D/DXCore will work on Linux wchars, but beware with
// interactions directly with the Windows kernel.
typedef char CHAR, *PSTR, *LPSTR, TCHAR, *PTSTR;
typedef const char *LPCSTR, *PCSTR, *LPCTSTR, *PCTSTR;
typedef wchar_t WCHAR, *PWSTR, *LPWSTR, *PWCHAR;
typedef const wchar_t *LPCWSTR, *PCWSTR;

#undef LONG_MAX
#define LONG_MAX INT_MAX
#undef ULONG_MAX
#define ULONG_MAX UINT_MAX

#ifndef UINT_MAX
#define UINT_MAX 0xffffffffu
#endif

// Misc defines
#define interface struct
#define MIDL_INTERFACE(x) interface __declspec(uuid(x))
#define CONST const
#define MAX_PATH 260
#define C_ASSERT static_assert
#define __analysis_assume(x)

#define TRUE 1u
#define FALSE 0u

#define OUT
#define IN
#define NEAR
#define FAR
#define OPTIONAL

#define DECLARE_INTERFACE(iface)                interface iface
#define DECLARE_INTERFACE_(iface, baseiface)    interface iface : public baseiface
#define PURE = 0
#define THIS_
#define THIS void

#define DECLARE_HANDLE(name)                                                   \
  struct name##__ {                                                            \
    int unused;                                                                \
  };                                                                           \
  typedef struct name##__ *name
  
typedef struct _GUID {
    uint32_t Data1;
    uint16_t Data2;
    uint16_t Data3;
    uint8_t  Data4[ 8 ];
} GUID;

#define DECLSPEC_SELECTANY  __declspec(selectany)
#define EXTERN_C    extern "C"
#define DEFINE_GUID(name, l, w1, w2, b1, b2, b3, b4, b5, b6, b7, b8) \
        EXTERN_C const GUID DECLSPEC_SELECTANY name \
                = { l, w1, w2, { b1, b2,  b3,  b4,  b5,  b6,  b7,  b8 } }

typedef GUID IID;
typedef GUID CLSID;
typedef GUID UUID;

#define REFGUID const GUID &
#define REFIID const IID &
#define REFCLSID const IID &

#include <string.h>

__inline int IsEqualGUID(REFGUID rguid1, REFGUID rguid2)
{
    return !memcmp(&rguid1, &rguid2, sizeof(GUID));
}

__inline int InlineIsEqualGUID(REFGUID rguid1, REFGUID rguid2)
{
    return (
        ((uint32_t *)&rguid1)[0] == ((uint32_t *)&rguid2)[0] &&
        ((uint32_t *)&rguid1)[1] == ((uint32_t *)&rguid2)[1] &&
        ((uint32_t *)&rguid1)[2] == ((uint32_t *)&rguid2)[2] &&
        ((uint32_t *)&rguid1)[3] == ((uint32_t *)&rguid2)[3]);
}

// SAL annotations
#define _In_
#define _In_z_
#define _In_opt_
#define _In_reads_(x)
#define _In_reads_opt_(x)
#define _In_reads_bytes_(x)
#define _In_reads_bytes_opt_(x)
#define _In_range_(x, y)
#define _Out_
#define _Out_opt_
#define _Outptr_
#define _Outptr_opt_result_bytebuffer_(x)
#define _COM_Outptr_
#define _COM_Outptr_opt_
#define _Out_writes_(x)
#define _Out_writes_z_(x)
#define _Out_writes_opt_(x)
#define _Out_writes_all_(x)
#define _Out_writes_all_opt_(x)
#define _Out_writes_to_opt_(x, y)
#define _Out_writes_bytes_(x)
#define _Out_writes_bytes_all_(x)
#define _Out_writes_bytes_all_opt_(x)
#define _Out_writes_bytes_opt_(x)
#define _Inout_
#define _Inout_opt_
#define _Inout_updates_(x)
#define _Inout_updates_bytes_(x)
#define _Field_size_(x)
#define _Field_size_opt_(x)
#define _Field_size_bytes_(x)
#define _Field_size_full_(x)
#define _Field_size_bytes_full_(x)
#define _Field_size_bytes_full_opt_(x)
#define _Field_size_bytes_part_(x, y)
#define _Field_range_(x, y)
#define _Field_z_
#define _Check_return_
#define _IRQL_requires_(x)
#define _IRQL_requires_min_(x)
#define _IRQL_requires_max_(x)
#define _At_(x, y)
#define _Always_(x)
#define _Return_type_success_(x)
#define _Translates_Win32_to_HRESULT_(x)
#define _Maybenull_
#define _Outptr_result_maybenull_
#define _Outptr_result_nullonfailure_
#define _Analysis_assume_(x)
#define _Success_(x)
#define _In_count_(x)
#define __out
#define __in_ecount(x)
#define __in_ecount_opt(x)
#define __in_opt

// Calling conventions
#define __stdcall
#define STDMETHODCALLTYPE
#define STDAPICALLTYPE
#define STDAPI extern "C" HRESULT STDAPICALLTYPE
#define WINAPI
#define APIENTRY
#define CALLBACK
#define STDMETHOD(name) virtual HRESULT name
#define STDMETHOD_(type,name) virtual type name
#define IFACEMETHOD(method) /*__override*/ STDMETHOD(method)
#define IFACEMETHOD_(type, method) /*__override*/ STDMETHOD_(type, method)

// Error codes
#define _HRESULT_DEFINED
typedef _Return_type_success_(return >= 0) LONG HRESULT;
#define SUCCEEDED(hr) (((HRESULT)(hr)) >= 0)
#define FAILED(hr) (((HRESULT)(hr)) < 0)
#define S_OK ((HRESULT)0L)
#define E_OUTOFMEMORY 0x80000002L
#define E_INVALIDARG  0x80000003L
#define E_NOINTERFACE 0x80000004L
typedef LONG NTSTATUS;

// Declspecs
// #define __declspec_align(x) alignas(x)
// #define __declspec_uuid(x)
// #define __declspec_selectany
// #define __declspec(x) __declspec_##x
// #define __pragma(x) __pragma_##x
// #define DECLSPEC_ALIGN(x) __declspec(align(x))
#define DECLSPEC_UUID(x) __declspec(uuid(x))
#define DECLSPEC_NOVTABLE

// Full type definitions
typedef struct _LUID {
    ULONG LowPart;
    LONG HighPart;
} LUID, *PLUID;

typedef struct tagRECT
{
    int left;
    int top;
    int right;
    int bottom;
} RECT;
typedef RECT*      PRECT;
typedef RECT NEAR* NPRECT;
typedef RECT FAR*  LPRECT;

typedef struct tagRECTL
{
    LONG left;
    LONG top;
    LONG right;
    LONG bottom;
} RECTL;
typedef RECTL*      PRECTL;
typedef RECTL NEAR* NPRECTL;
typedef RECTL FAR*  LPRECTL;

typedef struct tagPOINT
{
    int x;
    int y;
} POINT;
typedef POINT*       PPOINT;
typedef POINT NEAR* NPPOINT;
typedef POINT FAR*  LPPOINT;

typedef struct tagSIZE
{
    LONG        cx;
    LONG        cy;
} SIZE, *PSIZE, *LPSIZE;
typedef SIZE               SIZEL;
typedef SIZE               *PSIZEL, *LPSIZEL;

typedef union _LARGE_INTEGER {
  struct {
    DWORD LowPart;
    DWORD HighPart;
  } u;
  LONGLONG QuadPart;
} LARGE_INTEGER;

typedef LARGE_INTEGER *PLARGE_INTEGER;

typedef union _ULARGE_INTEGER {
  struct {
    DWORD LowPart;
    DWORD HighPart;
  } u;
  ULONGLONG QuadPart;
} ULARGE_INTEGER;

typedef ULARGE_INTEGER *PULARGE_INTEGER;

typedef enum {
    PowerActionNone = 0,
    PowerActionReserved,
    PowerActionSleep,
    PowerActionHibernate,
    PowerActionShutdown,
    PowerActionShutdownReset,
    PowerActionShutdownOff,
    PowerActionWarmEject,
    PowerActionDisplayOff
} POWER_ACTION, *PPOWER_ACTION;

typedef enum _DEVICE_POWER_STATE {
    PowerDeviceUnspecified = 0,
    PowerDeviceD0,
    PowerDeviceD1,
    PowerDeviceD2,
    PowerDeviceD3,
    PowerDeviceMaximum
} DEVICE_POWER_STATE, *PDEVICE_POWER_STATE;

struct SECURITY_ATTRIBUTES;

DECLARE_HANDLE(HWND);
DECLARE_HANDLE(HDC);
DECLARE_HANDLE(PALETTEENTRY);


// ENUM_FLAG_OPERATORS

// Define operator overloads to enable bit operations on enum values that are
// used to define flags. Use DEFINE_ENUM_FLAG_OPERATORS(YOUR_TYPE) to enable these
// operators on YOUR_TYPE.
#ifdef __cplusplus
extern "C++" {
    template <size_t S>
    struct _ENUM_FLAG_INTEGER_FOR_SIZE;

    template <>
    struct _ENUM_FLAG_INTEGER_FOR_SIZE<1>
    {
        typedef int8_t type;
    };

    template <>
    struct _ENUM_FLAG_INTEGER_FOR_SIZE<2>
    {
        typedef int16_t type;
    };

    template <>
    struct _ENUM_FLAG_INTEGER_FOR_SIZE<4>
    {
        typedef int32_t type;
    };

    template <>
    struct _ENUM_FLAG_INTEGER_FOR_SIZE<8>
    {
        typedef int64_t type;
    };

    // used as an approximation of std::underlying_type<T>
    template <class T>
    struct _ENUM_FLAG_SIZED_INTEGER
    {
        typedef typename _ENUM_FLAG_INTEGER_FOR_SIZE<sizeof(T)>::type type;
    };

}
#define DEFINE_ENUM_FLAG_OPERATORS(ENUMTYPE) \
extern "C++" { \
inline constexpr ENUMTYPE operator | (ENUMTYPE a, ENUMTYPE b) { return ENUMTYPE(((_ENUM_FLAG_SIZED_INTEGER<ENUMTYPE>::type)a) | ((_ENUM_FLAG_SIZED_INTEGER<ENUMTYPE>::type)b)); } \
inline ENUMTYPE &operator |= (ENUMTYPE &a, ENUMTYPE b) { return (ENUMTYPE &)(((_ENUM_FLAG_SIZED_INTEGER<ENUMTYPE>::type &)a) |= ((_ENUM_FLAG_SIZED_INTEGER<ENUMTYPE>::type)b)); } \
inline constexpr ENUMTYPE operator & (ENUMTYPE a, ENUMTYPE b) { return ENUMTYPE(((_ENUM_FLAG_SIZED_INTEGER<ENUMTYPE>::type)a) & ((_ENUM_FLAG_SIZED_INTEGER<ENUMTYPE>::type)b)); } \
inline ENUMTYPE &operator &= (ENUMTYPE &a, ENUMTYPE b) { return (ENUMTYPE &)(((_ENUM_FLAG_SIZED_INTEGER<ENUMTYPE>::type &)a) &= ((_ENUM_FLAG_SIZED_INTEGER<ENUMTYPE>::type)b)); } \
inline constexpr ENUMTYPE operator ~ (ENUMTYPE a) { return ENUMTYPE(~((_ENUM_FLAG_SIZED_INTEGER<ENUMTYPE>::type)a)); } \
inline constexpr ENUMTYPE operator ^ (ENUMTYPE a, ENUMTYPE b) { return ENUMTYPE(((_ENUM_FLAG_SIZED_INTEGER<ENUMTYPE>::type)a) ^ ((_ENUM_FLAG_SIZED_INTEGER<ENUMTYPE>::type)b)); } \
inline ENUMTYPE &operator ^= (ENUMTYPE &a, ENUMTYPE b) { return (ENUMTYPE &)(((_ENUM_FLAG_SIZED_INTEGER<ENUMTYPE>::type &)a) ^= ((_ENUM_FLAG_SIZED_INTEGER<ENUMTYPE>::type)b)); } \
}
#else
#define DEFINE_ENUM_FLAG_OPERATORS(ENUMTYPE)
#endif

// D3DX12 uses these
#include <cstdlib>
#define HeapAlloc(heap, flags, size) malloc(size)
#define HeapFree(heap, flags, ptr) free(ptr)

// IUnknown

interface DECLSPEC_UUID("00000000-0000-0000-C000-000000000046") DECLSPEC_NOVTABLE IUnknown
{
   virtual HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void **ppvObject) = 0;
   virtual ULONG STDMETHODCALLTYPE AddRef() = 0;
   virtual ULONG STDMETHODCALLTYPE Release() = 0;
   
   template <class Q> HRESULT STDMETHODCALLTYPE QueryInterface(Q** pp) {
       return QueryInterface(__uuidof(Q), (void **)pp);
   }
};

extern "C++"
{
    template<typename T> void** IID_PPV_ARGS_Helper(T** pp)
    {
        static_cast<IUnknown*>(*pp);
        return reinterpret_cast<void**>(pp);
    }
}

#define IID_PPV_ARGS(ppType) __uuidof(**(ppType)), IID_PPV_ARGS_Helper(ppType)