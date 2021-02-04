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

  void LogKernelCompute(const std::string& name);
};