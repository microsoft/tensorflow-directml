#if _WIN32
#include <Windows.h>
#include <TraceLoggingProvider.h>
#else
// No-op macros for WSL
#define TRACELOGGING_DECLARE_PROVIDER
#define TRACELOGGING_DEFINE_PROVIDER
#define TraceLoggingRegister
#define TraceLoggingUnregister
#define TraceLoggingWrite
#define TraceLoggingValue
#endif

#include <cstdio>

#include "dml_tracing.h"

TRACELOGGING_DECLARE_PROVIDER(g_providerHandle);

// {0E57B9AE-5CE1-4BEF-86BC-24152F6A9560}
TRACELOGGING_DEFINE_PROVIDER(g_providerHandle, "Microsoft.DirectML.TensorFlow",
                             (0xe57b9ae, 0x5ce1, 0x4bef, 0x86, 0xbc, 0x24, 0x15,
                              0x2f, 0x6a, 0x95, 0x60));

DmlTracing::DmlTracing() { TraceLoggingRegister(g_providerHandle); }

DmlTracing::~DmlTracing() { TraceLoggingUnregister(g_providerHandle); }

/*static*/ DmlTracing& DmlTracing::Instance() {
  static DmlTracing traceLogger;
  return traceLogger;
}

void DmlTracing::LogKernelCompute(const std::string& name) {
  TraceLoggingWrite(g_providerHandle, "KernelCompute",
                    TraceLoggingString(name.c_str(), "OpName"));
}