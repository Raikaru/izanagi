# Platform support

Status is reported honestly: **builds**, **runs basic tests**, **passes shared
conformance**, **passes WSI/lifecycle tests**, and **certified on named
physical hardware** are distinct claims. A platform is listed as supported
only when the required conformance passes on physical hardware.

| Platform | Backend | VK profile | Status |
|---|---|---|---|
| Windows | Vulkan Native | `IZANAGI_VK_NATIVE_1` + `IZANAGI_VK_BINDLESS_1` | Certified baseline on the RTX 4080: full suite (43 tests) passes on both profiles, including the forced legacy-copy and forced static-graphics-state configurations; CI. |
| Linux | Vulkan Bindless | `IZANAGI_VK_BINDLESS_1` | WSL dzn rig (mesa 26.2.0 dzn, RTX 4080 via D3D12): full suite (43/43, Debug + Release) passes with the private copy + static-state fallbacks; capability JSON + suite logs archived. Headless CI: builds + common tests; Lavapipe probe job added (qualification pending first green run). |
| Android | Vulkan Bindless via Mesa Turnip | `IZANAGI_VK_BINDLESS_1` | Qualified on the Adreno 650 phone rig two ways: (1) phone-hosted CI runs the full Vulkan suite on pinned Turnip/KGSL (Linux userland) on every `main` push; (2) a native NDK (bionic) test host passes the full suite on an adrenotools-loaded Turnip driver, and the capability-rejection path is verified against the stock Qualcomm Vulkan 1.1 driver and Qualcomm's v615.77 blob (also 1.1 on this GPU). Replacement drivers select at runtime via `IZANAGI_ADRENOTOOLS_*` env (build with `IZANAGI_ADRENOTOOLS_ROOT`). |
| macOS | Metal (planned) | — | Build boundary in place; backend in a later phase. |
| iOS | Metal (planned) | — | Declared; simulator/device phases planned. |

## Profile matrix

| Profile | Platforms | Status |
|---|---|---|
| `IZANAGI_VK_NATIVE_1` | Windows, Linux | Implemented; Windows certified (RTX 4080); Linux headless build green on CI. |
| `IZANAGI_VK_BINDLESS_1` | Windows, Linux (WSL dzn), Android | Implemented; Windows verified (RTX 4080, full suite) and Linux verified (WSL dzn, full suite with the private copy + static-state fallbacks). Remaining gated fallbacks: the snapshot descriptor-set path (update-unused-while-pending-less devices) and the private render-pass fallback (dynamic-rendering-less devices) are not yet provided. Hardware qualification (Maxwell/Polaris/Skylake/GCN, RADV/NVK/ANV) is pending physical hardware. |
| `IZANAGI_VK_COMPAT_1` (superseded name) | — | Replaced by `IZANAGI_VK_BINDLESS_1` (see docs/VulkanProfiles.md). |
| `IZANAGI_METAL_1` | macOS, iOS | Declared; rejected at configure until implemented. |

## Certified device policy

"Supported" requires a declared profile passing shared conformance on named
physical hardware, with the capability report and validation logs archived.
See docs/HardwareSupport.md for the exact tested matrix (Windows RTX 4080,
WSL dzn rig; Lavapipe CI probe pending its first green run).
