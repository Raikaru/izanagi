# Platform support

Status is reported honestly: **builds**, **runs basic tests**, **passes shared
conformance**, **passes WSI/lifecycle tests**, and **certified on named
physical hardware** are distinct claims. A platform is listed as supported
only when the required conformance passes on physical hardware.

| Platform | Backend | VK profile | Status |
|---|---|---|---|
| Windows | Vulkan Native | `IZANAGI_VK_NATIVE_1` + `IZANAGI_VK_BINDLESS_1` | Certified baseline on the RTX 4080: full suite (37 tests) passes on both profiles; CI. |
| Linux | Vulkan Native | — | Builds + common tests on CI; headless Vulkan conformance pending a GPU runner. |
| macOS | Metal (planned) | — | Build boundary in place; backend in a later phase. |
| Android | Vulkan Compat (planned) | — | Declared; profile experiment + NDK build in a later phase. |
| iOS | Metal (planned) | — | Declared; simulator/device phases planned. |

## Profile matrix

| Profile | Platforms | Status |
|---|---|---|
| `IZANAGI_VK_NATIVE_1` | Windows, Linux | Implemented; Windows certified (RTX 4080); Linux headless build green on CI. |
| `IZANAGI_VK_BINDLESS_1` | Windows (targets Linux/Android) | Implemented; Windows verified (RTX 4080, full suite). Legacy fallbacks (snapshot descriptor sets, private render pass) are gated behind device-creation gates until their phases land; hardware qualification (Maxwell/Polaris/Skylake/GCN) is pending physical hardware. |
| `IZANAGI_VK_COMPAT_1` (superseded name) | — | Replaced by `IZANAGI_VK_BINDLESS_1` (see docs/VulkanProfiles.md). |
| `IZANAGI_METAL_1` | macOS, iOS | Declared; rejected at configure until implemented. |

## Certified device policy

"Supported" requires a declared profile passing shared conformance on named
physical hardware, with the capability report and validation logs archived.
See docs/HardwareSupport.md for the exact tested matrix (currently the
Windows RTX 4080 only).
