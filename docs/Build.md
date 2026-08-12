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

## Android CI

The normal CI workflow cross-compiles the library for `arm64-v8a` with the
Android NDK (`android-26`), `IZANAGI_WSI=ANDROID`, and the bindless Vulkan
profile. This is a compile check, not a support claim; Android physical-device
conformance is still experimental.

The manually triggered `android-wireless-device` workflow runs on a Linux
self-hosted runner tagged `android-device`. The runner must have CMake 3.28 or
newer, Ninja, Git, a JDK, and `adb` available, and must be able to reach the
phone over the same LAN or VPN. On the phone, open **Developer options →
Wireless debugging → Pair device with pairing code**, then provide all three
values to the workflow:

- `adb_pair_address`: the temporary pairing `IP address & Port`;
- `adb_pairing_code`: the six-digit Wi-Fi pairing code;
- `adb_device_address`: the debugging `IP address & Port` used by `adb connect`.
The phone workflow builds and runs the GPU-independent common tests, verifies
the pinned Mesa Turnip Vulkan driver, and runs the Vulkan API suite on every
`main` push. Manual dispatch can also opt into the API suite. The stock
Qualcomm Vulkan driver is not used for conformance because this phone reports
an older Vulkan profile than Izanagi's bindless requirements.

## Android phone as a GitHub runner

The `android-phone-runner` workflow is for the phone itself acting as a
GitHub Actions ARM64 runner. It runs inside a Debian userspace provided by
Termux/proot and uses the labels `self-hosted`, `linux`, and `android-phone`.
On each job it installs the pinned Debian ARM64 Mesa Turnip package and
selects the KGSL backend. The phone must remain awake, charging, online, and
exempted from Samsung battery optimization for Termux.

Register the runner with the ARM64 Linux runner package and a repository runner
token generated in **Settings → Actions → Runners → New self-hosted runner**.

When running the official ARM64 runner inside `proot`, set
`DOTNET_GCHeapHardLimit=1C0000000` before `config.sh` and `run.sh`; otherwise
CoreCLR may fail its default heap reservation with error `0x8007000E`.

### Self-hosted runner security

- Keep the phone workflow limited to reviewed `main` pushes and manual runs.
  Do not route `pull_request` or unreviewed branch jobs to the phone.
- Use the unique `android-phone` label and a repository-scoped runner group;
  never use a generic self-hosted label by itself.
- Do not place repository secrets, signing keys, personal SSH keys, or cloud
  credentials on the phone. A job runs as the same unprivileged user that owns
  the runner credentials.
- Actions are pinned to immutable commit SHAs in the workflows. The phone job
  removes its build directory after every run.
- The runner process uses the unprivileged `runner` account inside Debian;
  registration must never be performed as root.

## Consumer target

`Izanagi::Izanagi` (and the existing `Izanagi::izanagi`) resolve to the selected backend implementation.

## Platform status

See `docs/PlatformSupport.md`. Only Windows is certified today; other platforms are in incremental phases and must not be treated as supported merely because they compile.
