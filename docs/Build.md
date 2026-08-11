# Building Izanagi

## CMake options

| Option | Values | Default | Meaning |
|---|---|---|---|
| `IZANAGI_BACKEND` | `VULKAN_NATIVE` (others declared) | `VULKAN_NATIVE` | Private backend implementation. One backend per compiled library. `VULKAN_COMPAT` and `METAL` are declared but rejected at configure with a clear message until their phases land. |
| `IZANAGI_WSI` | `AUTO`, `HEADLESS`, `WIN32`, `XCB`, `WAYLAND`, `ANDROID`, `METAL` | `AUTO` | Window-system integration. `HEADLESS` requires no desktop window system. `AUTO` resolves per host (Windows→`WIN32`, everything else→`HEADLESS` until those hosts land). Unsupported host/WSI combinations fail configure — never silently remapped. |
| `IZANAGI_BUILD_TESTS` | ON/OFF | top-level: ON | API test suite + GPU-independent common tests. |
| `IZANAGI_BUILD_EXAMPLES` | ON/OFF | top-level: ON | Example programs. Built only when a WSI host exists for the current platform. |
| `IZANAGI_BUILD_CAPABILITY_REPORT` | ON/OFF | top-level: ON | `izanagi_capability_report` tool (JSON capability output). |
| `IZANAGI_SLANGC` | path | empty | Explicit host `slangc` path; wins over PATH and the downloaded package. |
| `IZANAGI_PROFILE` | string | `IZANAGI_VK_NATIVE_1` | Capability profile name reported by the report tool. |

## Presets

- `dev-windows-msvc` — Windows MSVC, WIN32 WSI (existing default).
- `windows-headless` — Windows, no WSI (surface APIs fail cleanly).
- `linux-clang-debug`, `linux-gcc-release`, `linux-clang-asan` — Linux headless; ASan/UBSan preset targets the GPU-independent common tests.

## Host shader tool

`slangc` is a **build-host** executable. Discovery (in order): explicit `IZANAGI_SLANGC` → `slangc` on PATH → a repository-managed host package downloaded for `CMAKE_HOST_SYSTEM_NAME`/`CMAKE_HOST_SYSTEM_PROCESSOR`. Cross-compiling for Android/iOS never executes a target binary — the host package is selected by host, not target, properties. A missing tool is a clear configure error.

## Examples

```sh
# Windows (default)
cmake -S . -B build --preset dev-windows-msvc
cmake --build build --config Debug
ctest --test-dir build -C Debug --output-on-failure

# Windows headless (no WSI; surface APIs report errors)
cmake -S . -B build-headless --preset windows-headless
cmake --build build-headless --config Debug

# Linux headless (Clang Debug)
cmake -S . -B out/linux-native-debug -G Ninja -DCMAKE_BUILD_TYPE=Debug \
  -DIZANAGI_BACKEND=VULKAN_NATIVE -DIZANAGI_WSI=HEADLESS \
  -DIZANAGI_BUILD_TESTS=ON -DIZANAGI_BUILD_EXAMPLES=OFF
cmake --build out/linux-native-debug
ctest --test-dir out/linux-native-debug --output-on-failure
```

## Consumer target

`Izanagi::Izanagi` (and the existing `Izanagi::izanagi`) resolve to the selected backend implementation.

## Platform status

See `docs/PlatformSupport.md`. Only Windows is certified today; other platforms are in incremental phases and must not be treated as supported merely because they compile.
