# Roadmap

Public, honest, no dates. Order within a section is roughly execution
order. Items move; the changelog and [Stability.md](docs/Stability.md) are
the source of truth for what *has* shipped.

## 0.2 — make first contact survivable

- [x] Runtime-validated handles, range-checked commands, checked memory
      arithmetic (the hardening pass).
- [x] Changelog + stability policy + pinned-tag consumption guidance.
- [ ] Enriched diagnostics: handle values and generation details in every
      failure message.
- [ ] Consumer-integration CI job (FetchContent/find_package a tagged
      release, build an example against it).
- [ ] Forward-renderer demo: material switching, streaming uploads,
      resize, a few thousand draws. This is the API-ergonomics test.

## Linux desktop

- [ ] Wayland + X11 (XCB or Xlib) surface support — the WSI path does not
      exist yet; today Linux builds are headless.
- [ ] RADV qualification run (native + bindless profiles) with published
      evidence. The bindless profile is the expected RADV path.
- [ ] Linux in the certified-baseline table, if the evidence holds.

## 1.0 criteria

A 1.0 tag will be cut when:

- The public API has survived the forward-renderer demo without
  structural changes.
- At least one qualified device per profile class (descriptor-heap native,
  bindless compatibility) is published in HardwareSupport.md.
- Linux desktop WSI is implemented and qualified.
- A second real application (not an example) has shipped on Izanagi, or
  the project has explicitly decided it doesn't need that evidence.

Anything not listed here is not planned. In particular: **the Metal
backend is v2**, per [Metal4Mapping.md](docs/Metal4Mapping.md) — the
mapping document exists so the public API stays portable, not as a near-term
commitment. Compute-only paths (no graphics) are possible today via the
headless build and are exercised by the test suite.

## How to influence this list

File an issue with a use case. The highest-signal issues are: a real
workload that hits an API gap, a qualification report from hardware in the
unqualified table, or a reproducible failure on an unqualified platform.
"Please add feature X" issues are read, but ranked below evidence.
