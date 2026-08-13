# Getting Started

This walkthrough covers a release checkout, a Windows example build, the
smallest submission skeleton, and the three contracts new users need to know.
It assumes C++ and CMake knowledge, not Vulkan.

## 1. Pin a release

Izanagi is pre-1.0. Choose a tag from the
[releases page](https://github.com/Raikaru/izanagi/releases) and replace
`vX.Y.Z` below:

```sh
git clone --branch vX.Y.Z https://github.com/Raikaru/izanagi.git
cd izanagi
```

Do not ship against `main`; breaking changes land there before the next
release. See [Stability.md](Stability.md).

## 2. Choose a Vulkan profile

| Build selection | Compiled profile | Use when |
|---|---|---|
| `-DIZANAGI_VK_PROFILE=NATIVE` | `IZANAGI_VK_NATIVE_1` | The device provides Vulkan 1.4, descriptor heaps, untyped pointers, and unified image layouts. |
| `-DIZANAGI_VK_PROFILE=BINDLESS` | `IZANAGI_VK_BINDLESS_1` | The device satisfies the Vulkan 1.2+ descriptor-indexing profile. |

Neither Vulkan version nor GPU generation is sufficient by itself. The
capability report evaluates exact features and limits:

```sh
build/bin/Debug/izanagi_capability_report.exe
```

The full definitions are in [VulkanProfiles.md](VulkanProfiles.md); named test
evidence is in [HardwareSupport.md](HardwareSupport.md).

## 3. Build and run

Requirements: CMake 3.28+, Visual Studio 2022 with C++20 support, and a Vulkan
loader. The Bindless profile is used here because it covers the broader class
of existing drivers:

```sh
cmake --preset dev-windows-msvc -DIZANAGI_VK_PROFILE=BINDLESS
cmake --build --preset dev-windows-msvc
ctest --test-dir build -C Debug --output-on-failure
build/bin/Debug/izanagi_examples.exe
```

CMake acquires Vulkan-Headers, volk, VMA, and the host `slangc` tool. A Vulkan
SDK is needed only for validation layers. Other platforms, WSI choices, and
install builds are covered by [Build.md](Build.md).

The Win32 host can run a deterministic presentation smoke:

```sh
build/bin/Debug/izanagi_examples.exe --cycle-test 1 --no-vsync --frame-latency 1
```

Use `--help` for the current harness options rather than copying a fixed list
from documentation.

## 4. Minimal submission

```cpp
#include <izanagi/gpu.h>

using namespace gpu;

int main() {
    DeviceDesc desc{};
    desc.log_callback = [](LogLevel level, Span<const char> message,
                           uint32_t line, Span<const char> file, void*) {
        // Route level, file, line, and message to your logger.
    };
    desc.log_level = LogLevel::Debug;

    Device device = create_device(desc);
    if (device == nullptr) {
        return 1;
    }

    Queue queue = get_queue(device);
    CommandBuffer commands = queue_start_command_recording(queue);

    // Record copies, dispatches, or draws.

    cmd_finalize(commands);
    Submission submitted = queue_submit(queue, {&commands, 1});
    if (submitted.status != SubmitStatus::Success ||
        !wait_submission(submitted)) {
        destroy_device(device);
        return 1;
    }

    destroy_device(device);
    return 0;
}
```

The examples are the executable reference for pipeline creation, shader
arguments, rendering, presentation, and readback.

## 5. Core contracts

### Resources are pointer- or handle-addressed

`gpu::malloc` returns a `GpuPtr`, not a public buffer object. Textures,
texture views, samplers, and pipelines use generation-checked opaque handles.
Shaders resolve texture and sampler handles through `shaders/izanagi.slang`.
There are no public descriptor sets or binding layouts.

### Pipelines compile asynchronously

Use `request_compute_pipeline` or `request_graphics_pipeline` in frame-facing
code. `cmd_set_pipeline` returns `false` while a pipeline is pending or after
it fails; the application binds an explicit fallback or skips the work.
Blocking `create_*_pipeline` calls are for loading paths.

### Lifetime follows submissions

Immediate `free` is valid only when no recorded, pending, in-flight, or future
GPU access exists. For resources reachable through GPU pointers or descriptor
handles, use `free_after(..., submission)` with their last successful
submission. See [Architecture.md](Architecture.md) and
[IntegrationPatterns.md](IntegrationPatterns.md).

## Transfer uploads

```cpp
Queue upload_queue = get_queue(device, QueueType::Transfer);
Submission uploaded = queue_submit(upload_queue, {&upload_commands, 1});

SubmissionWait ready{
    .submission = uploaded,
    .stage = StageFlags::VertexShader, // first stage that consumes the upload
};
Submission rendered =
    queue_submit(graphics_queue, {&draw_commands, 1}, {}, {}, {&ready, 1});
```

`QueueType::Transfer` selects a dedicated transfer queue when available and
aliases the graphics queue otherwise. `SubmissionWait` performs the GPU-side
cross-queue memory and ownership handoff without a CPU wait.

## Presentation

Query `get_surface_capabilities`, choose a reported format and present mode,
then pass both through `SurfaceConfiguration`. The helper
`choose_present_mode(caps.present_modes, vsync)` provides deterministic
fallback when the preferred mode is unavailable. `frame_latency` must be in
the public in-flight range and is capped by the swapchain image count.

## Diagnostics

Keep a `ProcLogCallback` enabled during development. Deterministic failures
report their source location and relevant handle, command, or retained object
name. `enable_validation = true` forwards Vulkan validation messages through
the same callback when `VK_LAYER_KHRONOS_validation` is installed.

Use `set_debug_name` for buffers, textures, and pipelines. Bracket passes with
`cmd_push_debug_group` and `cmd_pop_debug_group`. The Vulkan backend forwards
those names and regions through `VK_EXT_debug_utils` when available.

## Next

- [PortingFromVulkan.md](PortingFromVulkan.md): mental-model translation.
- [Architecture.md](Architecture.md): threading, submissions, and retirement.
- [IntegrationPatterns.md](IntegrationPatterns.md): renderer ownership and
  transient data.
- [PipelineCompilation.md](PipelineCompilation.md): async compilation and
  persistent caches.
- [ShaderABI.md](ShaderABI.md): CPU/shader layout and profile artifact rules.
