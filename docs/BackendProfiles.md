# Backend and capability profiles

Izanagi makes two independent compile-time choices:

- `IZANAGI_BACKEND` selects the private implementation. The only implemented
  backend is currently `VULKAN_NATIVE`.
- `IZANAGI_VK_PROFILE` selects `NATIVE` or `BINDLESS` within that Vulkan
  backend. CMake derives the public artifact identity
  (`IZANAGI_VK_NATIVE_1` or `IZANAGI_VK_BINDLESS_1`).

Use [Build.md](Build.md) for configuration syntax and
[VulkanProfiles.md](VulkanProfiles.md) for the authoritative feature,
mechanism, shader-artifact, and rejection contracts. Support is determined by
those exact capabilities and qualification evidence, never by Vulkan version
or GPU generation alone.
