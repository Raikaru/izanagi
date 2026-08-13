# Stability Policy

Izanagi is pre-1.0. This document states exactly what stability means
here, so depending on it is a decision, not a gamble.

## Versioning

- Version scheme: `0.MINOR.PATCH`, semver-compatible.
- **Breaking changes are allowed in every 0.x release.** They are always
  listed under a `### Breaking` heading in [CHANGELOG.md](../CHANGELOG.md)
  with the old and new shape.
- `PATCH` releases contain only fixes (bugs, crashes, validation issues).
  They never change the public header.
- `1.0.0` will be the first release with a no-breakage guarantee for the
  public API. No date is promised (see [ROADMAP.md](../ROADMAP.md)).

## How to depend on Izanagi

**Pin a tag. Never `main`.** `main` is where breaking changes land first.

```cmake
include(FetchContent)
FetchContent_Declare(izanagi
    GIT_REPOSITORY https://github.com/Raikaru/izanagi.git
    GIT_TAG        v0.1.0)
FetchContent_MakeAvailable(izanagi)
target_link_libraries(your_app PRIVATE Izanagi::izanagi)
```

When you upgrade to a new tag, read the `Breaking` sections between your
current tag and the new one. A breaking change is usually a 5-minute
rename; undeclared instability is what actually hurts.

## What "public API" means

- `include/izanagi/gpu.h` — the entire public surface. It is Vulkan-free;
  no `Vk*` types appear in it. Anything not declared there is internal and
  may change without a changelog entry.
- `shaders/izanagi.slang` — the shader-side prelude. Changes here are
  listed in the changelog too, because shaders and CPU code must upgrade
  together.
- The **compiled profile** (`IZANAGI_PROFILE`, e.g. `IZANAGI_VK_NATIVE_1`)
  is part of the API contract. Both profiles implement the same programming
  model; a shader compiled for one profile is not usable with a library
  built for the other. Select it at configure time:

  ```sh
  cmake -S . -B build -DIZANAGI_PROFILE=IZANAGI_VK_BINDLESS_1
  ```

  Defaults and per-platform behavior are documented in
  [VulkanProfiles.md](VulkanProfiles.md) and
  [HardwareSupport.md](HardwareSupport.md).

## Semantic guarantees (stable across 0.x)

These are the rules the tests enforce and the docs assume:

- **Handles are generation-checked at runtime.** A stale or double-freed
  handle is rejected deterministically (logged, no-op) — never undefined
  behavior. This is a guarantee, not a debugging aid.
- **Command recording fails deterministically.** A command that cannot be
  recorded correctly (invalid pointer, stale handle, unavailable pipeline
  variant) marks the command buffer failed; `queue_submit` rejects it
  without submitting or advancing the timeline.
- **A failed submit never publishes a timeline value.** `Submission.status`
  reflects the failure; resource retirement conservatively falls back to
  the latest successful submit.
- **Ranges are validated.** Buffer pointers are checked against the
  allocation they came from, and copy/flush ranges are bounds-checked with
  overflow-safe arithmetic.
- **Device addresses are stable** for the lifetime of an allocation.
  `GpuPtr` values do not move.

## Thread-safety

- One device, one queue: queue submission and presentation are serialized
  per queue; multiple threads may record **different** command buffers
  concurrently.
- Freeing a resource concurrently with a submit that uses it is an
  application error (the API provides `free_after` for exactly this case).
- The compiler worker is device-owned; `request_*_pipeline` never blocks.

## Error reporting contract

Diagnostics are delivered through the `ProcLogCallback` supplied in
`DeviceDesc` at `create_device` (level-gated by `log_level`). Messages
carry `file:line` and, for handle failures, the handle value and the
generation mismatch. There is no exception-based or return-code-based error
channel beyond what individual functions return; when a function cannot
report through its return type, the log callback is the authoritative
record. See [GettingStarted.md](GettingStarted.md#diagnostics).

## Platform support definition

A platform is "supported" only when the qualification evidence for it is
published (see [PlatformSupport.md](PlatformSupport.md)). Compiling on a
platform never counts. Unqualified platforms may work; they may also break
in a `PATCH` release without warning. **The only guaranteed behavior on an
unqualified platform is a clean capability rejection** (device creation
fails with a log message naming the missing requirements).
