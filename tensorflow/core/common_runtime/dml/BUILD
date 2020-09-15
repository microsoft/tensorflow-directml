load("//tensorflow:tensorflow.bzl", "tf_copts", "tf_dml_cc_test", "if_windows", "if_not_windows")

cc_library(
    name = "dml_core",
    srcs = [
        "dml_adapter.cc",
        "dml_adapter_impl.cc",
    ],
    hdrs = [
        "dml_common.h",
        "dml_error_handling.h",
        "dml_adapter.h",
        "dml_adapter_impl.h",
        "dml_adapter_heuristics.h",
    ],
    linkstatic = 1,
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//third_party/dml:dx_helpers",
        "@dml_redist//:headers",
        "//third_party/dml:winadapter",
    ],
    alwayslink = 1,
    visibility = ["//visibility:public"],
    
    copts = tf_copts() + if_windows([
        "-DDML_BUILD_WINDOWS",
        "-D_WIN32_WINNT=_WIN32_WINNT_WIN10"
    ]),
    linkopts = if_windows([
        "d3d12.lib", 
        "dxgi.lib", 
    ]) + if_not_windows([
        "-ld3d12",
        "-ldxcore",
        "-Wl,-rpath,$$ORIGIN/../_solib_directml"
    ]),
)

cc_library(
    name = "dml_runtime",
    srcs = [
        "dml_bfc_allocator.cc",
        "dml_buffer.cc",
        "dml_buffer_region.cc",
        "dml_command_queue.cc",
        "dml_command_recorder.cc",
        "dml_descriptor_pool.cc",
        "dml_device_factory.cc",
        "dml_device.cc",
        "dml_device_state.cc",
        "dml_device_context.cc",
        "dml_event_queue.cc",
        "dml_execution_context.cc",
        "dml_heap_allocator.cc",
        "dml_kernel_context.cc",
        "dml_kernel_key.cc",
        "dml_kernel_manager.cc",
        "dml_operator_helper.cc",
        "dml_pooled_heap.cc",
        "dml_readback_heap.cc",
        "dml_tensor_desc.cc",
        "dml_upload_heap.cc",
        "dml_util.cc",
    ],
    hdrs = [
        "dml_bfc_allocator.h",
        "dml_buffer.h",
        "dml_buffer_region.h",
        "dml_command_allocator_ring.h",
        "dml_command_queue.h",
        "dml_command_recorder.h",
        "dml_descriptor_pool.h",
        "dml_device.h",
        "dml_device_state.h",
        "dml_device_context.h",
        "dml_event_queue.h",
        "dml_execution_context.h",
        "dml_gpu_event.h",
        "dml_heap_allocator.h",
        "dml_kernel_context.h",
        "dml_kernel_key.h",
        "dml_kernel_manager.h",
        "dml_operator_helper.h",
        "dml_pooled_heap.h",
        "dml_readback_heap.h",
        "dml_tensor_desc.h",
        "dml_upload_heap.h",
        "dml_util.h",
        "dml_status.h",
    ],
    linkstatic = 1,
    deps = [
        ":dml_core",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:gpu_bfc_allocator",
        "//tensorflow/core:proto_text",
        "//tensorflow/stream_executor",
        "//tensorflow/stream_executor/platform:dso_loader",
    ],
    alwayslink = 1,
    visibility = ["//visibility:public"],
    
    copts = tf_copts() + if_windows([
        "-DDML_BUILD_WINDOWS",
        "-D_WIN32_WINNT=_WIN32_WINNT_WIN10"
    ]),
)

tf_dml_cc_test(
    name = "dml_runtime_test",
    size = "small",
    srcs = [
        "dml_kernel_manager_test.cc",
    ],
    linkstatic = 1,
    deps = [
        ":dml_runtime",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:dml_ops",
        "//tensorflow/core/kernels:stateless_random_ops",
    ],
    extra_copts = if_windows([
        "-DDML_BUILD_WINDOWS",
        "-D_WIN32_WINNT=_WIN32_WINNT_WIN10"
    ]),
)