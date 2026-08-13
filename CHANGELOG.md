# Changelog

All notable changes to Izanagi are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project adheres to
the versioning policy in [docs/Stability.md](docs/Stability.md): breaking
changes are permitted in every 0.x release and are always listed under
`### Breaking`.

## [Unreleased]

### Breaking
- `DrawIndexedIndirectGpuArgs` now has Vulkan's required tightly packed
  20-byte layout; arrays created with the former 24-byte C++ stride must be
  rebuilt.
- `DeviceLimits` gained `gpu_timestamps`, `dedicated_transfer_queue`, and
  `max_draw_indirect_count`.
- `SurfaceCapabilities` now reports supported presentation modes;
  `SurfaceConfiguration` gained `present_mode` and `frame_latency`.
- `get_queue` accepts a `QueueType`, and `queue_submit` accepts
  `SubmissionWait` dependencies for GPU-side cross-queue handoff.

### Added
- FIFO/mailbox/immediate/FIFO-relaxed presentation selection,
  `choose_present_mode` for player-facing vsync toggles, and configurable
  one-to-`kMaxFramesInFlight` frame latency.
- Public buffer, texture, and pipeline names plus command-buffer debug groups.
  Names are retained for deterministic diagnostics and forwarded through
  `VK_EXT_debug_utils` when the loader exposes it.
- A dedicated transfer queue when the device exposes a transfer-only family,
  with a graphics-queue fallback and submission-to-submission timeline waits.
- Submission-safe GPU duration queries through `GpuTimer`,
  `cmd_begin_gpu_timer`/`cmd_end_gpu_timer`, and non-blocking
  `get_gpu_timer_result`.
- Installed-package consumer CI now builds a standalone `find_package`
  application, including enforcement of the exported C++20 requirement.
- The Linux software-device lane now runs the complete Native profile suite
  on current Mesa llvmpipe and archives its capability report.

### Fixed
- Multi-draw indirect count now validates device limits, count/argument/index
  alignment, complete buffer ranges, and zero draw limits before recording.
- Swapchain reconfiguration creates a fresh frame-pacing timeline, avoiding
  non-monotonic timeline values after `frame_idx` resets.

### Documentation
- Added a dated GPUInfo feature matrix for Native-profile candidates, with
  per-driver caveats and a strict separation from qualified support.

## [0.2.0] — 2026-08-13

### Breaking (from 0.1.0)
- `SamplerDesc`: `filter` split into `min_filter`/`mag_filter`/`mip_filter`;
  added `mip_lod_bias`.
- Added `PolygonMode` (Fill/Line) and `RasterDesc::polygon_mode`.
- `Topology` and `Factor` enum value sets expanded.
- `DeviceLimits` gained `framebuffer_sample_counts`, `non_solid_fill`,
  `min_uniform_alignment`, `min_storage_alignment`, and
  `non_coherent_atom_size`.
- `queue_submit` now returns a `Submission` token (`queue`, timeline
  `value`, `status`); added `submission_complete`/`wait_submission` and the
  `PipelineStatus` request lifecycle.
- Added `BackendProfile`/`device_backend_profile()`.

### Security & robustness
- Runtime-validated handles everywhere: `SlotMap::try_get`/`erase`/
  `invalidate` verify index bounds, generation, and slot allocation before
  touching the pool. Stale or garbage handles now fail deterministically
  instead of relying on debug-only asserts.
- Fixed a stack buffer overflow in validation-layer enumeration when more
  than 32 instance layers are installed.
- New ref-taking, overflow-safe buffer-range resolver: `cmd_memcpy`,
  buffer↔texture copies, and `flush`/`invalidate_host_memory` validate the
  full range (not just the base pointer) under the allocation lock, closing
  a concurrent-free race.
- Commands that reference invalid pointers or stale handles now mark the
  command buffer failed deterministically (rejected at submit) instead of
  recording commands with null or garbage Vulkan objects.
- Checked arithmetic in `malloc` (size+alignment overflow, device-address
  align-up) and format-aware byte accounting for texture copies, including
  block-compressed formats.
- `create_device` failure path runs complete partial teardown (no leaked
  pools, heaps, or images).

### Diagnostics
- Handle failures log the handle value and the rejection reason
  ("generation mismatch", "slot not allocated", "out of range").
- Deterministic command-recording failures report the failing command's
  ordinal and the call-site file:line through the device log callback.

### Documentation
- New [Stability policy](docs/Stability.md) (pin a tag, never `main`),
  [GettingStarted.md](docs/GettingStarted.md),
  [PortingFromVulkan.md](docs/PortingFromVulkan.md) mental-model map,
  [ROADMAP.md](ROADMAP.md), [CONTRIBUTING.md](CONTRIBUTING.md), and a
  changelog. `CHANGELOG.md` documents every release from here on.

## [0.1.0] — 2026-08-10

First tagged snapshot. Tagged after the Turnip/Adreno 650 phone CI went
green; the exact commit is the `ci: pin runner to windows-2022` commit.

### Added
- Vulkan backend with two capability profiles: `IZANAGI_VK_NATIVE_1`
  (Vulkan 1.4 + `VK_EXT_descriptor_heap` + untyped pointers) and
  `IZANAGI_VK_BINDLESS_1` (1.2/1.3 + descriptor-indexing arrays).
- `malloc`-style GPU memory with 64-bit device addresses (`GpuPtr`),
  host mapping, flush/invalidate, and `free_after` timeline retirement.
- Global sampled/storage/sampler heaps with generation-checked handles.
- Graphics and compute pipelines with async compilation on a device-owned
  worker, transparent dedup, persistent native cache callbacks, and the
  private static-graphics-state fallback for devices without extended
  dynamic state.
- Dynamic rendering, stage-mask-only barriers, timeline-semaphore sync
  with `Submission` tokens, indirect draws/dispatch, MSAA resolve,
  `cmd_generate_mipmaps`, specialization constants.
- Headless API test suite (26 GPU tests + container/arena unit tests) and
  three examples (hello_triangle, compute_texture, textured_cube).
- Hardware qualification: Turnip/Adreno 650 (phone CI), WSL dzn
  (43/43 with the readback matrix), Lavapipe CI probe; capability report
  tool and qualification bundle for volunteered reports.
