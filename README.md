# Izanagi

A from-scratch bindless GPU API library following Sebastian Aaltonen's
"No Graphics API" model: `malloc`-style GPU memory returning 64-bit device
addresses, root arguments as raw pointers, a global indexable
texture/sampler heap, stage-mask-only barriers, timeline-semaphore sync,
minimal PSOs, and dynamic rendering.

**v1 scope:** Vulkan 1.4 backend only (this is a Windows machine), a Slang
shader toolchain compiled offline by `slangc`, three verified examples, a
headless API test suite, and a [Metal 4 mapping document](docs/Metal4Mapping.md)
proving the API ports 1:1 to Apple's Metal 4 core API (the Metal backend
itself is out of v1).

Design reference: [rkevingibson/loon_gpu](https://github.com/rkevingibson/loon_gpu)
(loon targets Vulkan 1.3 + descriptor-indexing sets; Izanagi targets Vulkan 1.4
+ `VK_EXT_descriptor_heap`, the modern replacement).

## Design

- **GPU memory is `malloc`.** `gpu::malloc(device, bytes, Memory)` returns a
  64-bit device address (`GpuPtr`). `Memory::Gpu` = device-local,
  `Memory::Default` = host-visible + device-address, `Memory::Readback` =
  host-visible readback. `get_host_pointer` maps it on the CPU side.
- **Root arguments are raw pointers.** Pipelines take no binding layout.
  `cmd_dispatch`/`cmd_draw` take one or two GPU pointers to user structs,
  delivered via `vkCmdPushDataEXT` (push constants). A shader receives
  `uniform Args { T* vert; T* frag; }` and reads everything through those.
- **One global resource heap.** Textures, texture views, and samplers are
  created into a device-global indexable heap (`VK_EXT_descriptor_heap`).
  Handles are opaque uint64s; shaders use the `izanagi.slang` prelude helpers
  (`getTexture2D`, `getSampler`) to turn them into resources.
- **Stage-mask barriers only.** `cmd_barrier(StageFlags, StageFlags)` is the
  entire sync model — no image layout transitions in the public API.
- **Timeline-semaphore sync.** Queue submissions carry optional wait/signal
  timeline values; `queue_on_submitted_work_completed` runs callbacks.
- **Minimal PSOs, dynamic rendering.** No render-pass objects, no pipeline
  layouts. Graphics pipelines are created from a `RasterDesc` (color targets +
  optional depth format) and use dynamic rendering (`vkCmdBeginRendering`).

## Requirements

- Windows 10/11 with a Vulkan 1.4-capable driver (NVIDIA/AMD/Intel)
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

## Repository layout

```
CMakeLists.txt / CMakePresets.json   build config
cmake/CompileSlangShader.cmake       slangc → SPIR-V build step
include/izanagi/gpu.h                full public API (Vulkan backend impl)
src/common/                          containers (Span, SlotMap, TwoLevelBitset)
src/vk/                              Vulkan 1.4 backend
shaders/izanagi.slang                shared prelude (heap access helpers)
examples/                            three examples + win32 framework
tests/                               headless API tests
docs/Metal4Mapping.md                API → Metal 4 mapping (v2 backend design)
third_party/                         FetchContent deps (volk, VMA, Vulkan-Headers, slangc)
```

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
