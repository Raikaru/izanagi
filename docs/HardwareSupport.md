# Hardware support

Support policy: **feature-based, never generation-name or Vulkan-version
based**. A device is supported when the exact capability profile's feature
bits, limits, shader ABI, and conformance tests pass. A GPU generation name is
only a *testing* label, used to organize which devices were actually run.

## Tested hardware

| Device | OS / driver | API | Profile | Result |
|---|---|---|---|---|
| NVIDIA GeForce RTX 4080 Laptop GPU | Windows 11, NVIDIA driver 610.88 | 1.4.341 | Native + Bindless | Full suite passes (both profiles) |

**This is the only hardware tested so far.** No Maxwell, Polaris, Skylake,
GCN, or Intel device has been qualified. The bindless profile's legacy
fallbacks (snapshot descriptor sets, render-pass fallback) are gated behind
explicit device-creation gates and are not runtime-exercised on this machine
(its driver has all the modern features); the legacy barrier path is
force-tested on this device.

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

- The initial bindless slice requires Vulkan 1.3 + dynamic rendering +
  synchronization2 + update-unused-while-pending (each gated with a logged
  reason in `create_device`; the fallbacks land in later phases).
- Validation-layer runs require the Vulkan SDK's `VK_LAYER_KHRONOS_validation`
  (test 33 reports when the layer is absent).
- No guarantee is based on the Vulkan version alone; no guarantee is based on
  a GPU generation name. A family is advertised only after the exact test
  matrix above passes on physical hardware.
