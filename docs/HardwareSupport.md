# Hardware support

Izanagi support is feature-based and evidence-based. A Vulkan version or GPU
generation is useful context, but it is never a support guarantee. The
selected profile's feature bits, limits, shader ABI, and complete test suite
decide.

See [PlatformSupport.md](PlatformSupport.md) for the meaning of certified,
qualified, CI-exercised, and compile-only.

## Evidence matrix

| Configuration | Profile | Evidence | Scope |
|---|---|---|---|
| NVIDIA GeForce RTX 4080 Laptop GPU; Windows 11; NVIDIA driver 610.88; Vulkan 1.4.341 | Native and Bindless | Complete suite passes on both profiles, including forced legacy-copy and static-graphics-state routes. | Certified Windows baseline. |
| NVIDIA GeForce RTX 4080 exposed through D3D12; WSL Ubuntu; Mesa dzn 26.2.0; Vulkan 1.2.354 | Bindless | Dated qualification run passed the then-current complete suite and archived capability output. | Qualified dzn configuration; not native Linux GPU-driver coverage. |
| Qualcomm Adreno 650; Android/bionic; pinned Turnip replacement driver loaded through adrenotools | Bindless | The physical phone CI runner executes the capability report and headless API suite on `main`. | Qualified physical-GPU API configuration; Android presentation remains unqualified. |
| llvmpipe (LLVM 22.1.8, 256 bits); openSUSE Tumbleweed; Mesa 26.2.0; Vulkan 1.4.354 | Native | Complete-suite CI run and archived capability report (`profile_supported: true`, no missing features). | Software-device CI coverage only; not physical-hardware qualification. |

The active phone and llvmpipe jobs are the current evidence for their rows.
The dzn record is dated; it should not be read as proof for public contracts
added after that qualification run.

No RADV, NVK, ANV, Maxwell, Polaris, Skylake, Vega, or other GCN configuration
has been qualified. Use `tools/run_hardware_qualification.sh` and the
hardware-qualification issue template to contribute a report.

## dzn qualification record

The 2026-08-12 WSL run used Mesa dzn 26.2.0 on the RTX 4080 through D3D12. It
reported the complete Bindless requirement set: buffer device address,
shader int64, scalar block layout, required non-uniform and update-after-bind
features, timeline semaphores, draw-indirect-count, and sufficient descriptor
capacities. Dynamic rendering and synchronization2 were available through
KHR routes.

`VK_KHR_copy_commands2` and `VK_EXT_extended_dynamic_state` were absent, so
the run exercised both private fallbacks: legacy copy/blit commands and
asynchronously compiled static graphics-state variants. Pixel readbacks
covered culling, front face, depth, stencil, and viewport behavior.

dzn identifies itself as non-conformant Vulkan and can terminate on malformed
SPIR-V. “Qualified” here means the Izanagi suite passed on this exact rig; it
is not a Vulkan conformance claim.

## Descriptor capacities

The Bindless profile sizes its global sampled-image, storage-image, and
sampler arrays from the device's update-after-bind limits and shared combined
budget. The profile minimums are documented in
[VulkanProfiles.md](VulkanProfiles.md). Exact capacities for a device come
from `izanagi_capability_report` and `device_limits`; do not copy numbers from
another GPU.

## Known limits

- Bindless requires dynamic rendering, synchronization2, and
  update-unused-while-pending. A missing required capability rejects device
  creation with a complete reason list.
- Devices without `VK_KHR_copy_commands2` use the legacy copy/blit route.
- Devices without `VK_EXT_extended_dynamic_state` use private static pipeline
  variants; applications can prewarm them with `request_graphics_state`.
- Validation runs require `VK_LAYER_KHRONOS_validation`.
- The stock Qualcomm driver on the qualified phone does not meet the Bindless
  profile. The passing Android row uses the pinned Turnip replacement driver.
