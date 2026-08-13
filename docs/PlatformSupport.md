# Platform support

Support claims use five distinct levels:

1. **Certified baseline**: the current public contract passes the complete
   suite on a named physical configuration used as a release baseline.
2. **Qualified configuration**: a dated device/OS/driver combination passed
   the complete suite; the claim does not generalize to every device on that
   platform.
3. **CI exercised**: the suite runs on a software or hosted CI device.
4. **Compile-only**: the target builds, but runtime behavior is not claimed.
5. **Planned**: no usable backend exists.

The exact evidence belongs in [HardwareSupport.md](HardwareSupport.md).
Vulkan API version or GPU family alone never establishes support.

## Current matrix

| Platform | Implementation | Status |
|---|---|---|
| Windows | Vulkan Native and Bindless; Win32 WSI | **Certified baseline** on the named RTX 4080 configuration. Both profiles pass the complete suite locally; Windows Debug and Release builds run in CI. |
| Linux | Vulkan Bindless; headless, XCB, Wayland | The dated WSL dzn configuration is qualified. Bindless runs on Lavapipe in CI when the capability probe accepts the runner. XCB and Wayland compile in CI, but no native Linux RADV/NVK/ANV hardware configuration is qualified. |
| Android | Vulkan Bindless; Android WSI builds | The Adreno 650 + pinned Turnip replacement-driver configuration passes the headless API suite on the physical phone runner. Android WSI presentation is not qualified, and the stock Qualcomm driver on that phone does not meet the profile. |
| macOS | Headless Vulkan host build | Compile-only. The Metal backend and Metal WSI are not implemented. |
| iOS | None | Planned; no runtime or build support claim. |

## What CI proves

The active matrix is defined in
[`.github/workflows/ci.yml`](../.github/workflows/ci.yml) and
[`.github/workflows/android-phone-runner.yml`](../.github/workflows/android-phone-runner.yml).
A green run proves only the jobs that actually executed:

- hosted compiler and package-consumer coverage;
- Bindless behavior on the selected Lavapipe software device when its profile
  probe succeeds;
- XCB and Wayland surface compilation, not presentation behavior;
- the headless Bindless suite on the named physical Android/Turnip runner.

It does not turn an untested GPU family or driver into a supported target.

## Qualification policy

A new hardware claim requires:

1. the capability report;
2. the complete API suite with the selected profile;
3. driver identity and validation output;
4. archived evidence tied to a commit.

Use `tools/run_hardware_qualification.sh` and the hardware-qualification issue
template. Claims are configuration-specific until enough evidence justifies a
broader platform statement.
