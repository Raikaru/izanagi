# Contributing

Izanagi is a small project with a strict correctness bar. Contributions
are welcome; this file is the bar.

## First: read the policy

- [docs/Stability.md](docs/Stability.md) — what may break, what may not.
- [CHANGELOG.md](CHANGELOG.md) — how changes are recorded.
- [docs/GettingStarted.md](docs/GettingStarted.md) — the working setup.

## The correctness bar

1. **Deterministic failure.** Every error path either returns a failure
   value the caller can act on, or marks the command buffer failed and is
   rejected at submit, or logs through the device log callback. No silent
   fallbacks, no assert-only validation on public-handle paths (runtime
   checks in release builds).
2. **Range-checked memory.** Anything derived from an app-supplied pointer,
   handle, size, or count is validated with overflow-safe arithmetic before
   it reaches a Vulkan call.
3. **Tests defend behavior.** GPU tests for GPU behavior, container/unit
   tests for CPU behavior. A fix for a bug ships with a test that fails
   before the fix.
4. **Both profiles.** Changes to the backend must build and pass on
   `IZANAGI_VK_NATIVE_1` and `IZANAGI_VK_BINDLESS_1`. The CI does this; run
   both locally before opening a PR.

## Local development

```sh
cmake -S . -B build --preset dev-windows-msvc
cmake --build build --config Debug
ctest --test-dir build -C Debug
```

Run with validation on during development of anything touching Vulkan
usage: pass `enable_validation = true` in `DeviceDesc` (requires the
validation layer from the Vulkan SDK installed).

## Reporting bugs

Use the bug template. A good report contains:

- `izanagi_capability_report` output (device, profile, limits) — non-negotiable
  for anything GPU-related.
- The deterministic error message from the log callback, if one fired.
- A minimal reproduction (example-sized, not app-sized).

## Hardware qualification reports

See the hardware-qualification issue template and
[tools/run_hardware_qualification.sh](tools/run_hardware_qualification.sh).
A qualification report is: capability report output + full test-suite log
+ driver/OS version, on hardware not already in
[docs/HardwareSupport.md](docs/HardwareSupport.md). These are the
highest-value contributions available, because the project's support matrix
is evidence-gated.

## PR conventions

- Small, single-purpose commits; reference the issue.
- Changelog entry under `[Unreleased]`, `### Breaking` when the public
  header changes.
- No formatting churn mixed with logic changes.
- CI must be green on the PR head before review.

## Non-goals for contributions

- New backends (Metal is v2, per ROADMAP).
- "While I'm in here" refactors.
- Vendor-specific fast paths that cannot be tested on CI.
