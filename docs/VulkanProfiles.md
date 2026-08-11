# Vulkan capability profiles

Izanagi's public model — real GPU pointers, raw-pointer root arguments, a
persistent GPU-indexed resource namespace, minimal public pipeline state, and
backend-private synchronization — is implemented by *named capability
profiles*. Each profile is an explicit, compiled-in choice; the profile never
degrades the public semantics, and devices that cannot preserve them are
rejected with a complete missing-requirement list.

## The profiles

### IZANAGI_VK_NATIVE_1 (BackendProfile::VulkanNative)

The modern path. Private mechanisms:

- Vulkan 1.4 instance/device gate
- `VK_EXT_descriptor_heap` global heap (sampled + storage images in one
  resource heap, samplers in a sampler heap; heaps bound per command buffer)
- `VK_KHR_shader_untyped_pointers` (enabled; the shader corpus uses typed
  pointers only — untyped support is a capability floor, not a dependency)
- `VK_KHR_unified_image_layouts` (images stay GENERAL; the public API has no
  layout state)
- `vkCmdPushDataEXT` root arguments (part of the descriptor-heap extension)
- SPIR-V 1.6 artifacts with the `spvDescriptorHeapEXT` capability
- maintenance5/6 (VMA allocator flags, `VkPipelineCreateFlags2CreateInfo`
  chained at pipeline creation, `vkCmdBindIndexBuffer2`)

Required extensions/features: see `src/vk/device.cpp`
(`kRequiredDeviceExtensions`, the feature chain, and the post-creation
verification).

### IZANAGI_VK_BINDLESS_1 (BackendProfile::VulkanBindless)

The compatibility path. Same public model, different private mechanisms:

- Vulkan 1.2+ via the KHR/EXT dispatch route (dynamic rendering, sync2,
  copy-commands2, extended-dynamic-state are enabled as extensions on 1.2
  devices and the core-1.3 command names are aliased to their KHR/EXT entry
  points; on 1.3+ the extension entry points are the same code). Devices
  lacking the copy/state extensions are rejected with the exact list until
  the legacy-copy and static-dynamic-state fallbacks land
- one long-lived update-after-bind descriptor set holding the global arrays
  (binding 0 = sampled images, 1 = storage images, 2 = samplers with variable
  descriptor count), sized from the update-after-bind ceilings and the shared
  combined budget; slot 0 is reserved as null and backed by a valid dummy
  resource (null-handle reads never fault)
- descriptor writes via `vkUpdateDescriptorSets` (host-synchronized)
- typed physical storage buffer pointers; `NonUniformResourceIndex`-wrapped
  array lookups (verified to emit SPIR-V `NonUniform` + `ShaderNonUniform`)
- root arguments via ordinary `vkCmdPushConstants` (compute 8 B, graphics
  16 B) with one private pipeline layout shared by every pipeline
- legacy pipeline path: real `VkShaderModule`s, legacy create flags,
  `vkCmdBindIndexBuffer`
- a `vkCmdPipelineBarrier` translation of the PUBLIC `cmd_barrier` (conservative
  memory-read/write access masks at mapped 1.0-era stage bits). It is
  force-testable on any device via a test hook, but devices without
  synchronization2 are still rejected by the initial-slice gate until every
  internal barrier site is converted — the translation is not yet
  sync2-less-device support
- SPIR-V 1.5 artifacts, no descriptor-heap capability

Required capabilities (the evaluator in `src/common/profile_report.cpp`):
buffer device address, shader int64, scalar block layout, non-uniform
sampled/storage indexing, sampler array indexing (derived — the Vulkan spec
has no separate sampler feature), sampled/storage update-after-bind,
partially-bound, runtime descriptor arrays, variable descriptor count, draw
indirect count, native timeline semaphores, and capacity floors
(1024/1024/256) whose sum fits the shared combined update-after-bind budget.
Version and GPU generation are never gates; exact feature bits and limits
are.

Optional conveniences (recorded in the report; only some are un-gated):
dynamic rendering (gated — the private render-pass fallback is planned),
synchronization2 (gated — the legacy `cmd_barrier` translation exists and is
force-tested, but internal sites are not converted yet), update-unused-while-
pending (gated — the descriptor-set snapshot path is planned), shader
int8/16 + storage access bits.

## Shared across profiles

The slot allocators / generations / state machines, the submission-token +
retirement machinery, the async compiler worker + pipeline dedup, the
persistent cache (identity includes the profile), the format bridging, the
VMA layer, the surface code, and the public ABI (handle encoding
`slot | gen<<32 | type`, 8/16-byte root shapes) are shared. Only the
descriptor *write target*, the root-argument *command*, and the pipeline
*creation* path differ.

## Capability detection and failure behavior

`evaluate_vulkan_bindless_profile` (pure, GPU-independent, unit-tested with
injectable feature data) decides support from a feature/limit snapshot
(`query_vulkan_profile_features` fills it from
`vkGetPhysicalDeviceFeatures2` + properties). Unsupported devices fail
device creation with the complete missing-requirement list logged; no silent
degradation (no pointer→ID substitution, no per-draw rebinding, no layout
changes).

## Shader artifact identity

Artifacts land in `shaders/<tag>/` where the tag encodes backend profile +
profile version + SPIR-V version (e.g. `vk_native_1_spv16`,
`vk_bindless_1_spv15`). The extracted-manifest test verifies the SPIR-V
header version matches the directory, enumerates the entry points, and checks
the compile-time profile-name version matches the tag version. Native and
Bindless blobs are never interchangeable (CacheIdentity carries the profile).

## Failure reporting

The capability-report tool (`izanagi_capability_report`) emits the selected
profile, the bindless evaluation (supported + missing requirements +
capacities + fallbacks), and the device identity — archive it with every
conformance run.
