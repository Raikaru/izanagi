# Platform support

Status is reported honestly: **builds**, **runs basic tests**, **passes shared
conformance**, **passes WSI/lifecycle tests**, and **certified on named
physical hardware** are distinct claims. A platform is listed as supported
only when the required conformance passes on physical hardware.

| Platform | Backend | WSI | Status |
|---|---|---|---|
| Windows | Vulkan Native (`IZANAGI_VK_NATIVE_1`) | WIN32 | Certified baseline: full conformance suite, examples, CI. |
| Linux | Vulkan Native | HEADLESS | Builds + common tests on CI; headless Vulkan conformance pending a GPU runner. |
| macOS | Metal (planned) | HEADLESS/METAL | Build boundary in place; backend in a later phase. |
| Android | Vulkan Compat (planned) | ANDROID | Declared; profile experiment + NDK build in a later phase. |
| iOS | Metal (planned) | METAL | Declared; simulator/device phases planned. |

## Profile matrix

| Profile | Platforms | Status |
|---|---|---|
| `IZANAGI_VK_NATIVE_1` | Windows, Linux | Implemented on Windows; Linux headless build green on CI. |
| `IZANAGI_VK_COMPAT_1` | Android, Linux | Declared; rejected at configure until implemented. |
| `IZANAGI_METAL_1` | macOS, iOS | Declared; rejected at configure until implemented. |

## Certified device policy

"Supported" requires a declared profile passing shared conformance on named
physical hardware, with the capability report and validation logs archived.
Until then a platform is listed as "builds" or "in progress".
