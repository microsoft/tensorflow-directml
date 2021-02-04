#if _WIN32
// clang-format off
#include <Windows.h>
#include <TraceLoggingProvider.h>
#include <evntrace.h>
// clang-format on
#else
// No-op macros for WSL
#define TRACELOGGING_DECLARE_PROVIDER
#define TRACELOGGING_DEFINE_PROVIDER
#define TraceLoggingRegister
#define TraceLoggingUnregister
#define TraceLoggingWrite
#define TraceLoggingValue
#endif

#include "dml_tracing.h"

TRACELOGGING_DECLARE_PROVIDER(g_providerHandle);

// {0E57B9AE-5CE1-4BEF-86BC-24152F6A9560}
TRACELOGGING_DEFINE_PROVIDER(
    g_providerHandle, "Microsoft.Windows.AI.MachineLearning.Dml.TensorFlow",
    (0xe57b9ae, 0x5ce1, 0x4bef, 0x86, 0xbc, 0x24, 0x15, 0x2f, 0x6a, 0x95,
     0x60));

DmlTracing::DmlTracing() { TraceLoggingRegister(g_providerHandle); }

DmlTracing::~DmlTracing() { TraceLoggingUnregister(g_providerHandle); }

/*static*/ DmlTracing& DmlTracing::Instance() {
  static DmlTracing traceLogger;
  return traceLogger;
}

void DmlTracing::LogSessionRunStart() {
  TraceLoggingWrite(g_providerHandle, "SessionRun",
                    TraceLoggingOpcode(EVENT_TRACE_TYPE_START));
}

void DmlTracing::LogSessionRunEnd() {
  TraceLoggingWrite(g_providerHandle, "SessionRun",
                    TraceLoggingOpcode(EVENT_TRACE_TYPE_STOP));
}

void DmlTracing::LogExecutionContextCopyBufferRegion() {
  TraceLoggingWrite(g_providerHandle, "ExecutionContextCopyBufferRegion");
}

void DmlTracing::LogExecutionContextFillBufferWithPattern() {
  TraceLoggingWrite(g_providerHandle, "ExecutionContextFillBufferWithPattern");
}

void DmlTracing::LogExecutionContextFlush() {
  TraceLoggingWrite(g_providerHandle, "ExecutionContextFlush");
}

void DmlTracing::LogKernelCompute(const std::string& op_type,
                                  const std::string& op_name) {
  TraceLoggingWrite(g_providerHandle, "KernelCompute",
                    TraceLoggingString(op_type.c_str(), "Type"),
                    TraceLoggingString(op_name.c_str(), "Name"));
}