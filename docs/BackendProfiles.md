# Backend capability profiles

The capability-profile documentation lives in
[docs/VulkanProfiles.md](VulkanProfiles.md) — the authoritative reference for
`IZANAGI_VK_NATIVE_1` and `IZANAGI_VK_BINDLESS_1`, their private mechanisms,
required/optional features, artifact identity, and failure behavior.

This page is retained for the backend-selection layer that predates the
Vulkan profile split:

## Backend selection (build configuration)

One private backend per compiled library: `IZANAGI_BACKEND` selects the
backend implementation (`VULKAN_NATIVE` implemented; `VULKAN_COMPAT` and
`METAL` declared and rejected at configure until their phases land), and
`IZANAGI_VK_PROFILE` selects the Vulkan capability profile within the Vulkan
backend (`NATIVE` or `BINDLESS`; `IZANAGI_PROFILE` advertises the profile
name, derived from the selection and self-healing stale caches). The
`VULKAN_COMPAT_1` profile name used in earlier planning was superseded by
`IZANAGI_VK_BINDLESS_1`.

`izanagi_capability_report` reports the selected backend, profile, device
identity, descriptor capacities, and the bindless-profile evaluation. Never
treat a Vulkan version or a GPU generation name as a support guarantee —
feature bits, limits, shader ABI, and conformance tests decide.
