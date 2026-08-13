# Building Izanagi

This page is the source of truth for build configuration. Capability and
device requirements are documented separately in
[VulkanProfiles.md](VulkanProfiles.md).

## Requirements

- CMake 3.28 or newer.
- A C++20 compiler.
- A Vulkan loader and a driver satisfying the selected capability profile at
  runtime.

Vulkan-Headers, volk, VMA, and the required host shader tools are acquired by
CMake. A system Vulkan SDK is optional; install it only when Vulkan validation
layers are required.

## CMake options

| Option | Values | Default | Purpose |
|---|---|---|---|
| `IZANAGI_VK_PROFILE` | `NATIVE`, `BINDLESS` | `NATIVE` | Selects the Vulkan capability profile and shader ABI. |
| `IZANAGI_WSI` | `AUTO`, `HEADLESS`, `WIN32`, `XCB`, `WAYLAND`, `ANDROID`, `METAL` | `AUTO` | Selects window-system integration. Unsupported host/WSI pairs fail configuration. |
| `IZANAGI_BUILD_TESTS` | `ON`, `OFF` | top level: `ON`; subproject: `OFF` | Builds API and GPU-independent tests. |
| `IZANAGI_BUILD_EXAMPLES` | `ON`, `OFF` | top level: `ON`; subproject: `OFF` | Builds examples when a host exists; currently Win32 only. |
| `IZANAGI_BUILD_CAPABILITY_REPORT` | `ON`, `OFF` | top level: `ON`; subproject: `OFF` | Builds `izanagi_capability_report`. |
| `IZANAGI_SLANGC` | executable path | auto-discovered | Overrides host `slangc` discovery. |
| `IZANAGI_BACKEND` | `VULKAN_NATIVE` | `VULKAN_NATIVE` | Selects the private backend. Other declared values are rejected until implemented. |

`IZANAGI_PROFILE` is a derived artifact identity such as
`IZANAGI_VK_BINDLESS_1`; do not use it to select a profile. Use
`IZANAGI_VK_PROFILE=BINDLESS` or `NATIVE`.

`IZANAGI_WSI=AUTO` resolves to Win32 on Windows, Android on Android, and
headless elsewhere. Linux desktop builds must select `XCB` or `WAYLAND`
explicitly.

## Presets

| Preset | Configuration |
|---|---|
| `dev-windows-msvc` | Visual Studio 2022, Win32 WSI, Debug build preset |
| `windows-headless` | Visual Studio 2022, no WSI |
| `linux-clang-debug` | Clang + Ninja, headless Debug |
| `linux-gcc-release` | GCC + Ninja, headless Release |
| `linux-clang-asan` | Clang + Ninja, ASan/UBSan, GPU-independent tests |

The Vulkan profile can be appended to any configure preset:

```sh
cmake --preset dev-windows-msvc -DIZANAGI_VK_PROFILE=BINDLESS
cmake --build --preset dev-windows-msvc
```

## Common builds

### Windows with examples

```sh
cmake --preset dev-windows-msvc -DIZANAGI_VK_PROFILE=BINDLESS
cmake --build --preset dev-windows-msvc
ctest --test-dir build -C Debug --output-on-failure
```

Executables are written to `build/bin/Debug`. The Win32 example host supports
interactive, screenshot, resize-cycle, present-mode, and frame-latency smoke
runs; use `--help` for its current command-line options.

### Linux headless

```sh
cmake --preset linux-clang-debug -DIZANAGI_VK_PROFILE=BINDLESS
cmake --build out/linux-clang-debug
ctest --test-dir out/linux-clang-debug --output-on-failure
```

Set `IZANAGI_TESTS_ALLOW_SKIP=1` only on a build runner where the absence of a
Vulkan device is expected. It does not turn a failing GPU test into a pass.

### Linux XCB or Wayland

```sh
cmake -S . -B build-wayland -G Ninja \
  -DIZANAGI_WSI=WAYLAND -DIZANAGI_VK_PROFILE=BINDLESS \
  -DIZANAGI_BUILD_EXAMPLES=OFF
cmake --build build-wayland
```

Use `XCB` instead of `WAYLAND` for the XCB path. These configurations build
the surface implementation; the repository does not yet provide a Linux
example host.

### Android

Android uses the NDK toolchain, `arm64-v8a`, `IZANAGI_WSI=ANDROID`, and the
Bindless profile. The maintained command line and artifact checks live in
[`.github/workflows/ci.yml`](../.github/workflows/ci.yml). Physical Adreno
testing lives in
[`.github/workflows/android-phone-runner.yml`](../.github/workflows/android-phone-runner.yml);
the workflow, not this guide, owns runner setup and replacement-driver details.

## Shader tool and artifacts

`slangc` is always a build-host executable. Discovery order is:

1. `IZANAGI_SLANGC`;
2. `slangc` on `PATH`;
3. the repository-managed package for the host OS and architecture.

Cross-compiling never executes a target-architecture shader compiler. Shader
artifacts are profile-specific and are not interchangeable; see
[ShaderABI.md](ShaderABI.md) and [VulkanProfiles.md](VulkanProfiles.md).

## Consume with CMake

### FetchContent

Replace `vX.Y.Z` with a release from the
[releases page](https://github.com/Raikaru/izanagi/releases):

```cmake
include(FetchContent)
FetchContent_Declare(izanagi
    GIT_REPOSITORY https://github.com/Raikaru/izanagi.git
    GIT_TAG        vX.Y.Z)
FetchContent_MakeAvailable(izanagi)

target_link_libraries(your_app PRIVATE Izanagi::izanagi)
```

### Install and find_package

```sh
cmake --preset dev-windows-msvc
cmake --build --preset dev-windows-msvc
cmake --install build --config Debug --prefix <install-prefix>
```

```cmake
find_package(Izanagi CONFIG REQUIRED)
target_link_libraries(your_app PRIVATE Izanagi::izanagi)
```

The exported target carries the include directory and C++20 requirement. The
public header has no Vulkan-header dependency. CI builds an external
`find_package` consumer against the installed package.

## Verification coverage

The maintained matrix is encoded in
[`.github/workflows/ci.yml`](../.github/workflows/ci.yml): Windows Debug and
Release tests, Linux Clang/GCC and ARM64 builds, a Bindless Lavapipe run,
Android cross-compilation, macOS host compilation, XCB/Wayland compilation,
and installed-package consumer integration. Named physical-device evidence is
tracked in [HardwareSupport.md](HardwareSupport.md), not inferred from a green
compile job.
