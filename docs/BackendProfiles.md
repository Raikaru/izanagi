# Backend capability profiles

Machine-readable profiles describe what a backend implementation requires and
advertises. The profile name is compiled into the library (`IZANAGI_PROFILE`)
and reported by `izanagi_capability_report`.

## IZANAGI_VK_NATIVE_1

The current high-end Vulkan model (the only implemented profile today).

Required extensions/features (derived from `src/vk/device.cpp`):

- Vulkan 1.4 (`VK_API_VERSION_1_4`)
- `VK_EXT_descriptor_heap` (global texture/sampler heaps, shader `Handle()` access)
- `VK_KHR_shader_untyped_pointers` (root-argument pointer model)
- `VK_KHR_unified_image_layouts` (GENERAL-only image layouts)
- `VK_KHR_swapchain` (only when a surface-capable WSI is selected)
- dynamic rendering, synchronization2, maintenance4/5/6, buffer device
  address, timeline semaphores, scalar block layout, draw-indirect count,
  shader int8/16 + float16 (as enabled in the device feature chain)

Shader model: Slang prelude `izanagi.slang` with `spvDescriptorHeapEXT` and
raw 64-bit pointers in root data (`vkCmdPushDataEXT`). Handles are 64-bit
descriptor indices; `GpuPtr` values are real device addresses.

## IZANAGI_VK_COMPAT_1

Declared (configure currently rejects it until the profile implementation
lands). Broad Vulkan 1.3-oriented compatibility path intended for Android and
optional Linux testing. It may use descriptor indexing + update-after-bind,
backend-private descriptor sets/pipeline layouts, and small root-data
transport privately, preserving the public `GpuPtr`/`TextureView`/`SamplerId`/
root-data model. If the root-pointer model cannot be preserved on a device,
profile/device creation must fail with a precise missing-capability report —
never reinterpret a `GpuPtr` as an unrelated table index.

## IZANAGI_METAL_1

Declared (configure rejects it until the Metal backend lands). Native Metal
shared by macOS and iOS; MoltenVK is for differential testing only. Minimum
OS/SDK/Metal-family requirements will be documented from the actual API calls
once implemented.

## Rejection behavior

- An unimplemented backend/WSI fails **configure** with a clear message.
- A device that cannot satisfy the selected profile must fail device
  creation with a precise missing-capability report (planned for profiles
  beyond the native one).
- A platform must reject an unsupported profile cleanly — never silently
  change public API semantics to claim support.

## Capability report

`izanagi_capability_report` prints JSON:

```json
{
  "backend": "VulkanNative",
  "profile": "IZANAGI_VK_NATIVE_1",
  "profile_supported": true,
  "platform": "Windows",
  "device_name": "...",
  "api_version": "1.4.341",
  "driver_version": 2559967232,
  "vendor_id": 4318,
  "device_id": 10208,
  "missing_features": [],
  "descriptor_capacity": { "sampled_textures": 65536, "storage_textures": 65536, "samplers": 4080 },
  "pipeline_cache_control": true
}
```

CI archives this report for every certified configuration.
