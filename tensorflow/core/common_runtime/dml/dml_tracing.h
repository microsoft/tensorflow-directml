#pragma once

#include "dml_common.h"

// Helper for adding tracing events useful in ETW-based perf analysis
// (GPUView/WPA). No telemetry.
class DmlTracing {
 public:
  // GUID to associate a PIX event name with an IDMLObject (there is an existing
  // SetName method, but no public GetName method).
  constexpr static GUID kPixEventNameId = {0x191d7d5, 0x9fe1, 0x47cc, 0xb6,
                                           0x46,      0xf6,   0x78,   0xb6,
                                           0x33,      0x2f,   0x96};

  enum TraceLevel { None = 0, LowFrequency = 1, All = 10 };

 private:
  DmlTracing();
  ~DmlTracing();

  TraceLevel trace_level_ = None;

 public:
  static DmlTracing& Instance();

  void LogSessionRunStart();
  void LogSessionRunEnd();
  void LogExecutionContextCopyBufferRegion();
  void LogExecutionContextFillBufferWithPattern();
  void LogExecutionContextFlush();
  void LogKernelCompute(const std::string& op_type, const std::string& op_name);

  // GPU timeline
  void LogExecuteOperatorStart(IDMLCompiledOperator* op,
                               ID3D12GraphicsCommandList* command_list);
  void LogExecuteOperatorEnd(ID3D12GraphicsCommandList* command_list);
};