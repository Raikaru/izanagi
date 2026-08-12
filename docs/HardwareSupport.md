# Hardware support

Support policy: **feature-based, never generation-name or Vulkan-version
based**. A device is supported when the exact capability profile's feature
bits, limits, shader ABI, and conformance tests pass. A GPU generation name is
only a *testing* label, used to organize which devices were actually run.

## Tested hardware

| Device | OS / driver | API | Profile | Result |
|---|---|---|---|---|
| NVIDIA GeForce RTX 4080 Laptop GPU | Windows 11, NVIDIA driver 610.88 | 1.4.341 | Native + Bindless | Full suite passes (both profiles) |
| NVIDIA GeForce RTX 4080 via D3D12 (**dzn**) | WSL Ubuntu / mesa 26.2.0 dzn | 1.2.354 | Bindless | Full suite passes (43/43, Debug + Release) |
| Lavapipe (mesa, CPU) | CI runner (ubuntu-latest, headless) | 1.x (CPU) | Bindless | Probe job added; qualification pending first successful CI run (non-authoritative) |
| Qualcomm Adreno 650 (Galaxy S20+ 5G) via Mesa **Turnip** | Debian arm64 (Termux/proot), mesa 26.2.0-devel Turnip/KGSL | 1.3.354 | Bindless | Full suite passes |

**Physical hardware tested: the RTX 4080 rig and the Adreno 650 phone.** No Maxwell, Polaris,
Skylake, GCN, or Intel device has been qualified. The bindless profile's
legacy fallbacks are now **runtime-exercised**: the legacy copy/blit path and
the private static graphics-state fallback run the full suite on the RTX 4080
Windows rig (both profiles, forced configurations) and on the WSL dzn rig
(the dzn configuration selects exactly those fallbacks). RADV/NVK/ANV remain
**unqualified** (no hardware; the `tools/run_hardware_qualification.sh`
bundle + hardware-qualification issue template exist for volunteered
reports).

### dzn (Vulkan-on-D3D12 in WSL) — qualified on the test rig — 2026-08-12

mesa 26.2.0's dzn (RTX 4080 through D3D12, api 1.2.354): **every
bindless-required capability is present** (BDA, int64, scalar block layout,
non-uniform indexing, sampled/storage update-after-bind, partially-bound,
runtime + variable-count arrays, update-unused-while-pending, timeline
semaphores, draw-indirect-count; 1M descriptor capacities), plus dynamic
rendering + synchronization2 (KHR forms). Missing: `VK_KHR_copy_commands2`
and `VK_EXT_extended_dynamic_state` — the dispatch route selects the private
legacy-copy fallback and the private static graphics-state fallback, and the
**full 43-test suite passes** (compute, copies, graphics draws with private
static variants, variant Pending/recovery, and the baked-state readback
matrix — cull, front-face, depth-compare, stencil-compare, viewport —
verified through pixel readbacks). The stencil pipeline uses a single
combined Depth24PlusStencil8 format (dzn has no standalone S8 image format
and no separate depth/stencil DSV formats).
Caveats: the experimental dzn driver traps the process on any malformed
SPIR-V (the bad-shader failure-injection test is gated off on the
limited-1.2 dispatch signature), and dzn's `WARNING: dzn is not a conformant
Vulkan implementation` banner is expected. "Qualified" here means
suite-green on this test rig — not a driver-conformance claim.

## Tiers

- **Supported**: a family is listed only after its oldest claimed
  representative passes the complete profile + conformance + stress suites on
  physical hardware. None besides the RTX 4080 (Windows) are claimed.
- **Experimental** (best effort, profile must still pass, no performance
  promise): no devices yet.
- **Unsupported / rejected**: devices lacking real shader-addressable GPU
  pointers, usable non-uniform global indexing, the shader ABI, sufficient
  descriptor capacity, or reliable submission/lifetime semantics. Rejection is
  feature-based with a complete reason list — no hardcoded blacklist.

## Descriptor capacities

The bindless profile allocates the global arrays from the update-after-bind
ceilings (per-stage and per-set) and the shared combined budget
(`maxPerStageUpdateAfterBindResources` /
`maxUpdateAfterBindDescriptorsInAllPools`). On the RTX 4080 the practical
capacities are the public constants 65536 sampled / 65536 storage / 4096
samplers; the exact numbers per device are reported by
`izanagi_capability_report` and `device_limits`.

## Known driver limitations

- The initial bindless slice requires dynamic rendering + synchronization2 +
  update-unused-while-pending (gated with a logged reason). Devices without
  `VK_KHR_copy_commands2` get the legacy copy/blit fallback; devices without
  `VK_EXT_extended_dynamic_state` get the private static graphics-state
  fallback (static pipeline variants + the `request_graphics_state` prewarm).
  Both are selected by capability, never by vendor or generation name.
- Validation-layer runs require the Vulkan SDK's `VK_LAYER_KHRONOS_validation`
  (test 33 reports when the layer is absent).
- No guarantee is based on the Vulkan version alone; no guarantee is based on
  a GPU generation name. A family is advertised only after the exact test
  matrix above passes on physical hardware.
