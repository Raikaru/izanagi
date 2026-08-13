# Hardware support

Izanagi support is feature-based and evidence-based. A Vulkan version or GPU
generation is useful context, but it is never a support guarantee. The
selected profile's feature bits, limits, shader ABI, and complete test suite
decide.

See [PlatformSupport.md](PlatformSupport.md) for the meaning of certified,
qualified, CI-exercised, and compile-only.

## GPUInfo extension observations

The Native profile's distinguishing feature bits can be checked against
GPUInfo's live extension database:

- [`descriptorHeap`](https://vulkan.gpuinfo.org/listdevicescoverage.php?extensionname=VK_EXT_descriptor_heap&extensionfeature=descriptorHeap&platform=all)
- [`shaderUntypedPointers`](https://vulkan.gpuinfo.org/listdevicescoverage.php?extensionname=VK_KHR_shader_untyped_pointers&extensionfeature=shaderUntypedPointers&platform=all)
- [`unifiedImageLayouts`](https://vulkan.gpuinfo.org/listdevicescoverage.php?extensionname=VK_KHR_unified_image_layouts&extensionfeature=unifiedImageLayouts&platform=all)

The table below is a 2026-08-13 snapshot of positive feature reports. “Yes”
means GPUInfo contains a report with that feature set to true for the named
device or group; “partial” means only some named members have a positive
report; “none” means the positive-results page contains no member of that
group. Absence is not proof that a newer driver cannot support the feature.

GPUInfo reports are user-submitted: stale drivers coexist with current ones,
and one GPU can appear under several names. The grouping below is only a
readable index into those reports, never a family-wide guarantee.

| GPUInfo device or driver group | Descriptor heap | Untyped pointers | Unified layouts | Native-profile reading |
|---|---:|---:|---:|---|
| NVIDIA proprietary: listed GTX 1650/1660, RTX 20/30/40/50, MX450/550, and Quadro/RTX workstation entries | Yes | Yes | Yes | Native candidates. Same-report examples include [GTX 1650](https://vulkan.gpuinfo.org/displayreport.php?id=50905) and [RTX 4080](https://vulkan.gpuinfo.org/displayreport.php?id=50636). |
| NVIDIA proprietary Maxwell/Pascal, including GTX 10 series | None | Yes | Yes | Current reports fail the Native trio at descriptor heap. This does not rule out Bindless. |
| NVK | Partial: [RTX 4070 / AD104](https://vulkan.gpuinfo.org/displayreport.php?id=49497) | Yes on AD104 | Yes on AD104 and some older GPUs | The AD104 report is a Native candidate; older NVK reports lack descriptor heap. |
| AMD RADV RDNA 3/4 and recent APUs | Partial | Yes | Partial | Same-report Native candidates exist for [Phoenix 780M](https://vulkan.gpuinfo.org/displayreport.php?id=50941), Strix1 890M, Navi33 RX 7600, Navi31 RX 7900 GRE/XT, and [GFX1201 RX 9070 XT](https://vulkan.gpuinfo.org/displayreport.php?id=50696). Do not generalize to every RDNA 3/4 name. |
| AMD RADV GCN, Vega, RDNA 1, and RDNA 2 | Partial | Yes | None | Current reports fail the Native trio at unified layouts. Examples with descriptor heap include Renoir, Polaris 12, Navi 21/22/24; this does not establish Bindless support or capacity. |
| Intel ANV: CFL, TGL, DG2, MTL, ARL, and BMG reports | Yes | Yes | None | Current reports fail the Native trio at unified layouts. |
| Qualcomm proprietary 512.863 reports represented by [Xiaomi 23049RAD8C](https://vulkan.gpuinfo.org/displayreport.php?id=50034) | Yes | Yes | Yes | The same report has all three bits but exposes Vulkan 1.3.295, below Izanagi Native's Vulkan 1.4 gate. |
| Apple/MoltenVK M1 through M5 reports | None | Yes | Yes | Current reports fail the Native trio at descriptor heap. |
| Mesa llvmpipe 26.2.0 reports | Yes | Yes | Yes | Native candidate and CI-exercised; see the evidence matrix. [Representative report](https://vulkan.gpuinfo.org/displayreport.php?id=50906). |

Three green feature cells still do **not** mean “supported by Izanagi.” They
must occur on the same current device/driver report, and Native additionally
requires Vulkan 1.4, maintenance5/6, the required commands, limits, and shader
ABI. Run `izanagi_capability_report`, then the complete suite. The three
GPUInfo pages also cannot determine Bindless support: that profile has a
larger feature-and-capacity requirement set documented in
[VulkanProfiles.md](VulkanProfiles.md).

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
