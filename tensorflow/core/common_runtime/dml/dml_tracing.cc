#if _WIN32
// clang-format off
#include <Windows.h>
#include <TraceLoggingProvider.h>
#include <evntrace.h>
// clang-format on

#define USE_PIX
#include "pix3.h"
#include <d3d12.h>

typedef HRESULT(WINAPI* PIXBeginEventOnCommandListFn)(
    ID3D12GraphicsCommandList* commandList, UINT64 color, PCSTR formatString);
typedef HRESULT(WINAPI* PIXEndEventOnCommandListFn)(
    ID3D12GraphicsCommandList* commandList);
typedef HRESULT(WINAPI* PIXSetMarkerOnCommandListFn)(
    ID3D12GraphicsCommandList* commandList, UINT64 color, PCSTR formatString);

static decltype(PIXGetThreadInfo)* g_pixGetThreadInfo = nullptr;
static decltype(PIXEventsReplaceBlock)* g_pixEventsReplaceBlock = nullptr;
static PIXBeginEventOnCommandListFn g_pixBeginEventOnCommandList = nullptr;
static PIXEndEventOnCommandListFn g_pixEndEventOnCommandList = nullptr;
static PIXSetMarkerOnCommandListFn g_pixSetMarkerOnCommandList = nullptr;

void BeginEventOnCommandList(ID3D12GraphicsCommandList* command_list,
                             UINT64 color, PCSTR format_string) {
  if (g_pixBeginEventOnCommandList) {
    g_pixBeginEventOnCommandList(command_list, color, format_string);
  }
}

void EndEventOnCommandList(ID3D12GraphicsCommandList* command_list) {
  if (g_pixEndEventOnCommandList) {
    g_pixEndEventOnCommandList(command_list);
  }
}

void SetMarkerOnCommandList(ID3D12GraphicsCommandList* command_list,
                            UINT64 color, PCSTR format_string) {
  if (g_pixSetMarkerOnCommandList) {
    g_pixSetMarkerOnCommandList(command_list, color, format_string);
  }
}

extern "C" PIXEventsThreadInfo* PIXGetThreadInfo() noexcept {
  if (!g_pixGetThreadInfo) {
    return nullptr;
  }
  return g_pixGetThreadInfo();
}

extern "C" UINT64 WINAPI PIXEventsReplaceBlock(PIXEventsThreadInfo* threadInfo,
                                               bool getEarliestTime) noexcept {
  if (!g_pixEventsReplaceBlock) {
    return 0;
  }
  return g_pixEventsReplaceBlock(threadInfo, getEarliestTime);
}

#else
// No-op macros for WSL
#define TRACELOGGING_DECLARE_PROVIDER(...)
#define TRACELOGGING_DEFINE_PROVIDER(...)
#define TraceLoggingRegister(...)
#define TraceLoggingUnregister(...)
#define TraceLoggingWrite(...)
#define TraceLoggingValue(...)
#define TraceLoggingString(...)
#define TraceLoggingOpcode(...)
#define EVENT_TRACE_TYPE_START
#define EVENT_TRACE_TYPE_STOP

#define PIXBeginEvent(...)
#define PIXEndEvent(...)
#define PIXSetMarker(...)
#define PIX_COLOR(...)
#define BeginEventOnCommandList(...)
#define EndEventOnCommandList(...)
#define SetMarkerOnCommandList(...)
#endif

#include "dml_tracing.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/stream_executor/platform/default/dso_loader.h"

TRACELOGGING_DECLARE_PROVIDER(g_providerHandle);

// {0E57B9AE-5CE1-4BEF-86BC-24152F6A9560}
TRACELOGGING_DEFINE_PROVIDER(
    g_providerHandle, "Microsoft.Windows.AI.MachineLearning.Dml.TensorFlow",
    (0xe57b9ae, 0x5ce1, 0x4bef, 0x86, 0xbc, 0x24, 0x15, 0x2f, 0x6a, 0x95,
     0x60));

DmlTracing::DmlTracing() {
  TraceLoggingRegister(g_providerHandle);

  tensorflow::int64 trace_level = 0;
  tensorflow::Status s = tensorflow::ReadInt64FromEnvVar(
      "TF_DIRECTML_TRACE_LEVEL", trace_level_, &trace_level);
  if (s.ok()) {
    trace_level_ = static_cast<TraceLevel>(trace_level);
  }

#if _WIN32
  if (trace_level_ > None) {
    auto pix_handle_or =
        stream_executor::internal::CachedDsoLoader::GetPixDsoHandle();
    if (pix_handle_or.ok()) {
      tensorflow::Env::Default()->GetSymbolFromLibrary(
          pix_handle_or.ValueOrDie(), "PIXGetThreadInfo",
          reinterpret_cast<void**>(&g_pixGetThreadInfo));
      tensorflow::Env::Default()->GetSymbolFromLibrary(
          pix_handle_or.ValueOrDie(), "PIXEventsReplaceBlock",
          reinterpret_cast<void**>(&g_pixEventsReplaceBlock));
      tensorflow::Env::Default()->GetSymbolFromLibrary(
          pix_handle_or.ValueOrDie(), "PIXBeginEventOnCommandList",
          reinterpret_cast<void**>(&g_pixBeginEventOnCommandList));
      tensorflow::Env::Default()->GetSymbolFromLibrary(
          pix_handle_or.ValueOrDie(), "PIXEndEventOnCommandList",
          reinterpret_cast<void**>(&g_pixEndEventOnCommandList));
      tensorflow::Env::Default()->GetSymbolFromLibrary(
          pix_handle_or.ValueOrDie(), "PIXSetMarkerOnCommandList",
          reinterpret_cast<void**>(&g_pixSetMarkerOnCommandList));
    }
  }
#endif // _WIN32
}

DmlTracing::~DmlTracing() { TraceLoggingUnregister(g_providerHandle); }

/*static*/ DmlTracing& DmlTracing::Instance() {
  static DmlTracing traceLogger;
  return traceLogger;
}

void DmlTracing::LogSessionRunStart() {
  if (trace_level_ >= LowFrequency) {
    TraceLoggingWrite(g_providerHandle, "SessionRun",
                      TraceLoggingOpcode(EVENT_TRACE_TYPE_START));
    PIXBeginEvent(PIX_COLOR(255, 0, 0), "SessionRun");
  }
}

void DmlTracing::LogSessionRunEnd() {
  if (trace_level_ >= LowFrequency) {
    TraceLoggingWrite(g_providerHandle, "SessionRun",
                      TraceLoggingOpcode(EVENT_TRACE_TYPE_STOP));
    PIXEndEvent();
  }
}

void DmlTracing::LogExecutionContextCopyBufferRegion() {
  if (trace_level_ >= All) {
    TraceLoggingWrite(g_providerHandle, "ExecutionContextCopyBufferRegion");
  }
}

void DmlTracing::LogExecutionContextFillBufferWithPattern() {
  if (trace_level_ >= All) {
    TraceLoggingWrite(g_providerHandle,
                      "ExecutionContextFillBufferWithPattern");
  }
}

void DmlTracing::LogExecutionContextFlush() {
  if (trace_level_ >= All) {
    TraceLoggingWrite(g_providerHandle, "ExecutionContextFlush");
    PIXSetMarker(0, "EC Flush");
  }
}

void DmlTracing::LogKernelCompute(const std::string& op_type,
                                  const std::string& op_name) {
  if (trace_level_ >= All) {
    TraceLoggingWrite(g_providerHandle, "KernelCompute",
                      TraceLoggingString(op_type.c_str(), "Type"),
                      TraceLoggingString(op_name.c_str(), "Name"));
  }
}

void DmlTracing::LogExecuteOperatorStart(
    IDMLCompiledOperator* op, ID3D12GraphicsCommandList* command_list) {
#if _WIN32
  if (trace_level_ >= All) {
    std::vector<char> eventName(100);
    UINT data_size = (UINT)(eventName.size() * sizeof(char));
    op->GetPrivateData(kPixEventNameId, &data_size, eventName.data());
    BeginEventOnCommandList(command_list, PIX_COLOR(128, 255, 128),
                            eventName.data());
  }
#endif
}

void DmlTracing::LogExecuteOperatorEnd(
    ID3D12GraphicsCommandList* command_list) {
  if (trace_level_ >= All) {
    EndEventOnCommandList(command_list);
  }
}