#pragma once

#include "dml_common.h"

// Helper for adding tracing events useful in ETW-based perf analysis
// (GPUView/WPA). No telemetry.
class DmlTracing {
 private:
  DmlTracing();
  ~DmlTracing();

 public:
  static DmlTracing& Instance();

  void LogSessionRunStart();
  void LogSessionRunEnd();
  void LogExecutionContextCopyBufferRegion();
  void LogExecutionContextFillBufferWithPattern();
  void LogExecutionContextFlush();
  void LogKernelCompute(const std::string& op_type, const std::string& op_name);
};