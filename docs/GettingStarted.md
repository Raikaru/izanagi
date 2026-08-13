# Getting Started

This walkthrough takes you from a fresh checkout to a working triangle with
diagnostics. It assumes you know C++ and CMake; it does not assume Vulkan.

## 0. Check your GPU (30 seconds)

Izanagi needs either:

| Profile | Requirement |
|---|---|
| `IZANAGI_VK_NATIVE_1` | Vulkan 1.4 + `VK_EXT_descriptor_heap` + `VK_KHR_shader_untyped_pointers` + `VK_KHR_unified_image_layouts` (modern NVIDIA/AMD/Intel drivers) |
| `IZANAGI_VK_BINDLESS_1` | Vulkan 1.2+ with the descriptor-indexing feature set (nearly everything else: dzn, Turnip, RADV, …) |

The build includes `izanagi_capability_report`, which prints exactly what
your device provides and which profile it satisfies:

```sh
build/bin/Debug/izanagi_capability_report.exe
```

If you see `bindless profile: supported`, you are good with the default
profile. If you see a missing-requirement list, that is the complete answer
— see [VulkanProfiles.md](VulkanProfiles.md) for the definitions and
[HardwareSupport.md](HardwareSupport.md) for the qualified-device table.

## 1. Get the code — pin a tag

Never build from `main` (see [Stability.md](Stability.md)):

```sh
git clone --branch v0.2.0 https://github.com/Raikaru/izanagi.git
cd izanagi
```

## 2. Build

```sh
cmake -S . -B build --preset dev-windows-msvc
cmake --build build --config Debug
```

No Vulkan SDK needed: Vulkan-Headers, volk, VMA and slangc are fetched
automatically. Full details (other generators, Android, WSL/Linux,
validation layers) are in [Build.md](Build.md).

## 3. Run the triangle

```sh
build/bin/Debug/izanagi_examples.exe
```

Keys: `M`/`N` cycle examples, `ESC` quits. Headless verification:

```sh
build/bin/Debug/izanagi_examples.exe --example hello_triangle --screenshot out.png --frames 30
```

Presentation controls are available directly in the Win32 harness:

```sh
build/bin/Debug/izanagi_examples.exe --no-vsync --frame-latency 1
build/bin/Debug/izanagi_examples.exe --vsync --frame-latency 2
```

Applications use the same policy explicitly:

```cpp
SurfaceCapabilities caps = get_surface_capabilities(device);
SurfaceConfiguration surface{
    .format        = caps.formats[0],
    .usages        = UsageFlags::ColorAttachment,
    .width         = width,
    .height        = height,
    .present_mode  = choose_present_mode(caps.present_modes, vsync),
    .frame_latency = 2,
};
configure_surface(device, surface);
```

## 4. Your first app

A minimal device + submit loop:

```cpp
#include <izanagi/gpu.h>
#include <cstdio>

using namespace gpu;

int main() {
    // Diagnostics: every deterministic failure is reported here.
    DeviceDesc desc{};
    desc.log_callback = [](LogLevel lvl, Span<const char> msg, uint32_t line,
                           Span<const char> file, void*) {
        std::printf("[%d] %.*s:%u: %.*s\n", (int)lvl, (int)file.size(), file.data(),
                    line, (int)msg.size(), msg.data());
    };
    desc.log_level = LogLevel::Debug;

    Device device = create_device(desc);
    if (device == nullptr) { return 1; }

    // GPU memory is malloc: a 64-bit device address.
    GpuPtr vertices = malloc(device, sizeof(float) * 3 * 3, Memory::Gpu);

    Queue queue = get_queue(device);
    CommandBuffer cb = queue_start_command_recording(queue);

    // ... record commands (see the examples) ...

    cmd_finalize(cb);
    Submission s = queue_submit(queue, Span<const CommandBuffer>(&cb, 1));
    wait_submission(s);   // loading-phase style; see below for frame pacing

    free(device, vertices);
    destroy_device(device);
    return 0;
}
```

The three things every new user should internalize before writing real
code:

1. **There are no descriptor sets or bindings.** Textures and samplers are
   indices into a device-global heap; shaders resolve them through the
   `izanagi.slang` prelude. See [PortingFromVulkan.md](PortingFromVulkan.md).
2. **Pipelines compile asynchronously.** Use `request_graphics_pipeline` +
   `cmd_set_pipeline`'s false return (bind a fallback or skip) for frame
   code; use blocking `create_graphics_pipeline` only for loading screens.
3. **Lifetime is explicit.** `free` is only safe when the GPU cannot touch
   the resource; `free_after(resource, submission)` is the normal path for
   anything reachable by GPU pointers.

For streaming uploads, request `get_queue(device, QueueType::Transfer)`.
It returns a transfer-only queue when the device exposes one and otherwise
aliases the default queue. Pass the upload `Submission` back to a graphics
`queue_submit` as a `SubmissionWait`; this performs the GPU-side ownership
and memory handoff without blocking the CPU.

The examples (`examples/hello_triangle`, `examples/compute_texture`,
`examples/textured_cube`) are the reference for every pattern above.

## Diagnostics

Set `log_level = LogLevel::Debug` and keep the callback on during
development. Failure messages are actionable and specific:

```
[ERROR] resources.cpp:464: free_after(texture): invalid or stale handle
[ERROR] commands.cpp:138: command references a pointer outside a live allocation
```

Every message carries `file:line`. Handle failures report the handle value
and the generation mismatch. `enable_validation = true` additionally
forwards driver validation messages through the same callback (requires the
Vulkan SDK validation layer installed).

Use `set_debug_name` on buffers, textures, and pipelines, and bracket passes
with `cmd_push_debug_group` / `cmd_pop_debug_group`. Izanagi retains the
names for deterministic errors and forwards them to RenderDoc through
`VK_EXT_debug_utils` when available.

## Troubleshooting

- **"Failed to create Vulkan instance"** — no loader/driver, or the driver
  is below the profile floor. Run `izanagi_capability_report`.
- **"bindless profile missing requirement:" followed by names** — this is
  the complete, precise list; each name maps to a Vulkan feature or limit
  documented in [VulkanProfiles.md](VulkanProfiles.md).
- **Command buffer rejected at submit** — a command recorded a
  deterministic failure; the log callback names the exact reason.
- **A stale-handle error you don't understand** — you freed something
  twice, or used a handle after `free_after` invalidated it. Handles are
  generation-checked: the second use is always the bug.

## Next

- [PortingFromVulkan.md](PortingFromVulkan.md) — the mental-model map.
- [Architecture.md](Architecture.md) — retirement, pools, the compiler
  worker.
- [IntegrationPatterns.md](IntegrationPatterns.md) — renderer-owned frame
  transactions, transient rings, and indirect resource lifetimes.
- [PipelineCompilation.md](PipelineCompilation.md) — async compile and the
  persistent cache.
- [ShaderABI.md](ShaderABI.md) — the shader side of the pointer model.
