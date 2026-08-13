# Izanagi

Izanagi is a small C++20 GPU API built around shader-addressable memory:
GPU allocations return 64-bit addresses, shaders receive root-data pointers,
textures and samplers live in a device-global namespace, and synchronization
uses stage masks and timeline-backed submissions.

The public header is Vulkan-free. The current implementation is Vulkan, with
two compiled capability profiles:

- `IZANAGI_VK_NATIVE_1`: Vulkan 1.4 descriptor heaps and untyped pointers.
- `IZANAGI_VK_BINDLESS_1`: Vulkan 1.2+ descriptor indexing with the same public
  programming model.

Exact feature requirements matter more than API version or GPU generation.
See [Vulkan profiles](docs/VulkanProfiles.md) and
[hardware support](docs/HardwareSupport.md) before choosing a profile.

> Izanagi is pre-1.0. Breaking changes may ship in a `0.x` release. Pin a
> release tag and review the [changelog](CHANGELOG.md) when upgrading; do not
> consume `main`. See the [stability policy](docs/Stability.md).

## Programming model

- `gpu::malloc` returns a `GpuPtr`; root arguments are application structs
  reached through GPU pointers.
- Texture views and samplers are opaque handles resolved through
  `shaders/izanagi.slang`; there are no public descriptor sets or binding
  layouts.
- `cmd_barrier(before, after)` is the public synchronization primitive; image
  layouts stay backend-private.
- Pipeline requests compile asynchronously and are deduplicated. Applications
  decide whether to bind a fallback or skip work while a pipeline is pending.
- `queue_submit` returns a `Submission` token used for waits, completion, and
  deferred resource destruction.
- Presentation mode, frame latency, transfer-queue handoff, object names, and
  command debug groups are explicit API policies rather than hidden backend
  behavior.

The detailed model is in [Architecture](docs/Architecture.md). Vulkan users
should start with [Porting from Vulkan](docs/PortingFromVulkan.md).

## Quick start

Requirements: CMake 3.28+, a C++20 compiler, and a Vulkan driver satisfying the
selected profile. Dependencies and the host `slangc` tool are acquired by
CMake; a Vulkan SDK is needed only for validation layers.

Windows is the current example host:

```sh
cmake --preset dev-windows-msvc -DIZANAGI_VK_PROFILE=BINDLESS
cmake --build --preset dev-windows-msvc
ctest --test-dir build -C Debug --output-on-failure
build/bin/Debug/izanagi_examples.exe
```

For Native-profile builds, Linux, headless mode, XCB/Wayland, Android, install
rules, and consumer integration, use the [build guide](docs/Build.md).

## Use from another project

Pin a release tag with CMake `FetchContent`, or install Izanagi and use:

```cmake
find_package(Izanagi CONFIG REQUIRED)
target_link_libraries(your_app PRIVATE Izanagi::izanagi)
```

The target exports its C++20 requirement. Tests, examples, and the capability
report default to off when Izanagi is consumed as a subproject. Complete
dependency instructions are in the [stability policy](docs/Stability.md).

## Documentation

| Need | Document |
|---|---|
| First build and first submission | [Getting Started](docs/GettingStarted.md) |
| Build options, presets, WSI, install | [Build](docs/Build.md) |
| Vulkan capability requirements | [Vulkan Profiles](docs/VulkanProfiles.md) |
| Tested devices and driver evidence | [Hardware Support](docs/HardwareSupport.md) |
| Platform-level support claims | [Platform Support](docs/PlatformSupport.md) |
| Public lifetime and threading model | [Architecture](docs/Architecture.md) |
| Renderer integration patterns | [Integration Patterns](docs/IntegrationPatterns.md) |
| Async pipeline behavior | [Pipeline Compilation](docs/PipelineCompilation.md) |
| CPU/shader layout contract | [Shader ABI](docs/ShaderABI.md) |
| Raw Vulkan migration map | [Porting from Vulkan](docs/PortingFromVulkan.md) |
| Release compatibility | [Stability Policy](docs/Stability.md) |
| Metal design mapping | [Metal 4 Mapping](docs/Metal4Mapping.md) |

Project status lives in [ROADMAP.md](ROADMAP.md); release changes live in
[CHANGELOG.md](CHANGELOG.md); contribution requirements are in
[CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT — see [LICENSE](LICENSE). Izanagi's API design is derived from
[loon_gpu](https://github.com/rkevingibson/loon_gpu) (MIT). Descriptor-heap
mechanics draw from
[vk_minimal_latest](https://github.com/nvpro-samples/vk_minimal_latest)
(Apache-2.0).
