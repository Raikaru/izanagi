# Izanagi

A from-scratch bindless GPU API library following Sebastian Aaltonen's
"No Graphics API" model: `malloc`-style GPU memory returning 64-bit device
addresses, root arguments as raw pointers, a global indexable
texture/sampler heap, stage-mask-only barriers, timeline-semaphore sync,
minimal PSOs, and dynamic rendering.

**v1 scope:** Vulkan backends (this is a Windows machine), a Slang shader
toolchain compiled offline by `slangc`, three verified examples, a headless
API test suite, and a [Metal 4 mapping document](docs/Metal4Mapping.md)
proving the API ports 1:1 to Apple's Metal 4 core API (the Metal backend
itself is out of v1).

**Requirements, principle first:** Izanagi requires real shader-addressable
GPU pointers and a persistent GPU-indexed resource namespace. The modern
Vulkan profile (`IZANAGI_VK_NATIVE_1`) uses native descriptor heaps and
untyped pointers; the compatibility Vulkan profile (`IZANAGI_VK_BINDLESS_1`)
uses backend-private descriptor-indexing arrays and typed physical pointers
while preserving the same public programming model. See
[docs/VulkanProfiles.md](docs/VulkanProfiles.md) and
[docs/HardwareSupport.md](docs/HardwareSupport.md) — no GPU generation or
Vulkan version is a support guarantee; only exact feature bits, limits, ABI,
and conformance tests are.

Design reference: [rkevingibson/loon_gpu](https://github.com/rkevingibson/loon_gpu)
(loon targets Vulkan 1.3 + descriptor-indexing sets; Izanagi targets Vulkan 1.4
+ `VK_EXT_descriptor_heap`, the modern replacement).

## Design

- **GPU memory is `malloc`.** `gpu::malloc(device, bytes, Memory)` returns a
  64-bit device address (`GpuPtr`). `Memory::Gpu` = device-local,
  `Memory::Default` = host-visible + device-address, `Memory::Readback` =
  host-visible readback. `get_host_pointer` maps it on the CPU side.
- **Root arguments are raw pointers.** Pipelines take no binding layout.
  `cmd_dispatch`/`cmd_draw` take one or two GPU pointers to user structs. A
  shader receives `uniform Args { T* vert; T* frag; }` and reads everything
  through those pointers; the backend delivers them privately (native
  profile: `vkCmdPushDataEXT`; bindless profile: ordinary push constants).
- **One global resource namespace.** Textures, texture views, and samplers
  are created into a device-global indexable namespace (native profile:
  `VK_EXT_descriptor_heap`; bindless profile: backend-private
  descriptor-indexing arrays). Handles are opaque uint64s; shaders use the
  `izanagi.slang` prelude helpers (`getTexture2D`, `getSampler`) to turn them
  into resources on either profile.
- **Stage-mask barriers only.** `cmd_barrier(StageFlags, StageFlags)` is the
  entire sync model — no image layout transitions in the public API.
- **Timeline-semaphore sync.** Queue submissions carry optional wait/signal
  timeline values; `queue_on_submitted_work_completed` runs callbacks.
- **Minimal PSOs, dynamic rendering.** No render-pass objects, no pipeline
  layouts. Graphics pipelines are created from a `RasterDesc` (color targets +
  optional depth format) and use dynamic rendering (`vkCmdBeginRendering`).
- **MSAA render targets with resolve.** A color attachment may name a
  sample-count-1 resolve target; the pass resolves into it at end (average).
- **`cmd_generate_mipmaps`.** Successive linear blits build mips 1..N-1 from
  mip 0 in one command (layer 0 only).
- **Transparent pipeline dedup + persistent native cache.** Identical
  pipeline descriptions share one compiled pipeline (refcounted, handles stay
  distinct); an optional per-device native cache (`VkPipelineCache`) is
  seeded/saved via app-provided load/store callbacks keyed by an opaque
  `CacheIdentity`. The native blob is driver/GPU-specific, not transferable.
- **Asynchronous pipeline compilation.** `request_*_pipeline` deep-copies the
  description and compiles on a device-owned worker thread — never on the
  frame-critical thread. `get_pipeline_status`/`wait_pipeline` track
  Pending → Ready/Failed; `cmd_set_pipeline` returns false for non-Ready
  pipelines so the application explicitly binds a fallback or skips. On
  devices without extended dynamic state the baked graphics state is compiled
  into private static pipeline variants on the same worker (prewarm with
  `request_graphics_state`/`wait_graphics_state`; a not-yet-compiled variant
  fails the command buffer deterministically). See
  [docs/PipelineCompilation.md](docs/PipelineCompilation.md).
- **Explicit GPU lifetime.** `queue_submit` returns a `Submission` token
  (timeline value published only on success); command pools, buffers,
  textures, pipelines, and descriptor slots are retired by queue timeline —
  never by presentation frame counters. `free_after` retires pointer-reachable
  resources against a submission; command buffers auto-retain the objects they
  name. See [docs/Architecture.md](docs/Architecture.md).

## Requirements

- Windows 10/11 with a Vulkan 1.3+ driver (NVIDIA/AMD/Intel; the Native
  profile needs 1.4 + descriptor-heap support, the Bindless profile needs
  1.3 + the descriptor-indexing feature set — `izanagi_capability_report`
  reports exactly what a device provides)
- CMake 3.30+ and a C++20 compiler (VS 2022 toolset tested)
- No system Vulkan SDK required: Vulkan-Headers, volk, VMA are fetched
  automatically; `slangc` (2026.5.2) is downloaded at configure time.
- The Vulkan SDK is only needed for validation layers (`enable_validation`).

## Build

```sh
cmake -S . -B build --preset dev-windows-msvc
cmake --build build --config Debug
```

Outputs land in `build/bin/Debug/`:

| Target | Purpose |
|---|---|
| `izanagi.lib` | the library (Vulkan backend) |
| `izanagi_tests.exe` | headless API tests, exit 0 = pass |
| `izanagi_examples.exe` | all three examples + screenshot/cycle modes |

## Run

```sh
# Headless API tests (from any CWD)
build/bin/Debug/izanagi_tests.exe

# Examples, interactive
build/bin/Debug/izanagi_examples.exe            # hello_triangle
build/bin/Debug/izanagi_examples.exe --example textured_cube

# Keys: M / N cycle examples, ESC quits

# Screenshot verification (runs N frames, writes PNG, exits)
build/bin/Debug/izanagi_examples.exe --example hello_triangle \
    --screenshot out.png --frames 30

# Automated smoke: cycle through examples + resize, exit 0
build/bin/Debug/izanagi_examples.exe --cycle-test 3
```

Shaders are compiled at build time by `slangc` to SPIR-V and loaded from
`<exe_dir>/shaders/` — the executables are self-contained relative to their
own directory (no CWD dependence).

**Option A — FetchContent** (simplest; pulls the source, builds the static lib):

```cmake
include(FetchContent)
FetchContent_Declare(izanagi
    GIT_REPOSITORY https://github.com/Raikaru/izanagi.git
    GIT_TAG        v0.1.0)   # pin a tag — never main (see docs/Stability.md)
FetchContent_MakeAvailable(izanagi)

target_link_libraries(your_app PRIVATE Izanagi::izanagi)
```

Pin the newest tag from the [releases](https://github.com/Raikaru/izanagi/tags)
page and read the `### Breaking` entries in [CHANGELOG.md](CHANGELOG.md)
between your current tag and the new one when upgrading.

As a subproject, `IZANAGI_BUILD_TESTS` and `IZANAGI_BUILD_EXAMPLES` default to
OFF — only the library is built, and slangc is not downloaded.


**Option B — install + find_package**:

```sh
cmake -S . -B build --preset dev-windows-msvc
cmake --install build --config Debug --prefix <install-prefix>
```

```cmake
find_package(Izanagi CONFIG REQUIRED)
target_link_libraries(your_app PRIVATE Izanagi::izanagi)
```

The public header `izanagi/gpu.h` is Vulkan-free (no Vk* types, no volk
include); consumers only need the include dir + the compiled lib.

## Platform status

Windows (Vulkan Native, WIN32 WSI) is the certified baseline: the full suite
passes on both profiles, including the forced legacy-copy and forced
static-graphics-state configurations. Linux: the bindless profile's full
suite passes on the WSL dzn rig (mesa 26.2.0 dzn — D3D12-on-Vulkan, api 1.2,
no copy-commands2, no extended dynamic state) exercising exactly the private
fallbacks; headless builds + common tests run on CI, and a Lavapipe probe job
(non-authoritative) is in CI. Android has an arm64 NDK cross-build plus
phone-hosted Turnip/KGSL bindless conformance CI; support remains
experimental and the stock Qualcomm Vulkan path is a clean capability-rejection
path. RADV/NVK/ANV and Maxwell/Polaris/Skylake/GCN-class hardware are
**unqualified** (no hardware; the
`tools/run_hardware_qualification.sh` bundle + hardware-qualification issue
template exist for volunteered reports). Metal and iOS remain in incremental
phases. See [docs/PlatformSupport.md](docs/PlatformSupport.md) and
[docs/Build.md](docs/Build.md) — a platform is never listed as supported merely
because it compiles.

## Examples

1. **hello_triangle** — classic RGB triangle. No heap use; exercises
   graphics pipeline creation, dynamic rendering, present.
2. **compute_texture** — a compute shader writes an animated pattern into a
   storage texture; a fullscreen triangle samples it. Exercises heap storage
   + sampled slots, compute dispatch, stage barrier.
3. **textured_cube** — rotating checkerboard cube with depth testing.
   Exercises depth texture + depth-stencil state, index buffer, vertex
   buffers via device addresses, heap views, MVP via frame-ring args.

## Tests (`tests/api_tests.cpp`)

1. Device create/destroy
2. `malloc`/`get_host_pointer` roundtrip
3. Compute end-to-end (`memcpy_kernel.slang`: `dst[i] = src[i] * 2 + 1`)
4. Texture upload + readback
5. Heap slot recycling (indices reused, no leaks)
6. Semaphores: signal/wait + deferred completion callback
7. Indirect dispatch (`cmd_dispatch_indirect`)
8. Specialization constants (`kMul` overridden 1 → 5)
9. Indirect draws — single + multi
10. Mip-chain + cube-face subresource copies
11. BC1 block-compressed copy roundtrip
12. MSAA 4x render + resolve
13. `cmd_generate_mipmaps` (mip 0 → mip 3 chain)
14. Pipeline dedup (identical descs share one pipeline; every key field distinct)
15. Persistent pipeline cache (store/load round-trip + invalid-blob tolerance)
16. Async pipeline compile (non-blocking proof, in-flight lifetime)
17. Async Pending/Failed transitions + bind semantics
18. Async input ownership (caller storage destroyed after request)
19. Concurrent dedup (4 threads, one record)
20. Shutdown with queued/compiling work
21. Submission tokens (failure never advances the timeline)
22. Headless command-pool retirement (2000 submits, no presentation)
23. `free_after` for buffers/textures/pipelines + view/sampler retirement
24. Command-buffer texture retention (render after handle free)
25. Memory alignment + bounds validation + host coherence ops
26. Descriptor handle encoding (generation, stale rejection) + device limits + depth bias

GPU-independent container/arena tests (`tests/common_tests.cpp`, no Vulkan
device required): arena alignment/mark-rewind/overflow, nested scopes,
concurrent per-thread arenas, vector insert/growth/forced-allocation-failure,
slot generations, bitset bounds, enum bitwise operators.

## Repository layout

```
CMakeLists.txt / CMakePresets.json   build config + install/export rules
cmake/                               CompileSlangShader.cmake, IzanagiCompileShader.cmake, IzanagiConfig.cmake.in
include/izanagi/gpu.h                full public API (Vulkan-free header)
src/common/                          containers (Span, SlotMap, TwoLevelBitset)
src/vk/                              Vulkan 1.4 backend
shaders/izanagi.slang                shared prelude (heap access helpers)
examples/                            three examples + win32 framework
tests/                               headless API tests
docs/                                architecture, profiles, Stability.md, GettingStarted.md,
                                     PortingFromVulkan.md, Metal4Mapping.md (v2 backend design)
third_party/                         FetchContent deps (volk, VMA, Vulkan-Headers, slangc)
```

New to the project? Start at
[docs/GettingStarted.md](docs/GettingStarted.md), then the
[porting map](docs/PortingFromVulkan.md). Consuming it from another
project? Read [docs/Stability.md](docs/Stability.md) first. The
[ROADMAP.md](ROADMAP.md) is public and honest; contributions follow
[CONTRIBUTING.md](CONTRIBUTING.md).

## Notes

- `Format` keeps ETC2/ASTC entries for the future Metal backend; the Vulkan
  backend logs an error and returns a null handle for unsupported formats at
  `create_texture`.
- Slang shaders are compiled with `-matrix-layout-row-major`; CPU-side math
  (see `examples/common/math.h`) transposes at the upload boundary.

## License

MIT — see [LICENSE](LICENSE). The API design and enum value sets are derived
from [loon_gpu](https://github.com/rkevingibson/loon_gpu) (MIT); the
descriptor-heap mechanics follow NVIDIA's
[vk_minimal_latest](https://github.com/nvpro-samples/vk_minimal_latest)
(Apache-2.0) samples. `examples/common/stb_image_write.h` is public domain
(stb, nothings.org).
