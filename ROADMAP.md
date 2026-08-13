# Roadmap

Public, honest, no dates. Order within a section is roughly execution
order. Items move; the changelog and [Stability.md](docs/Stability.md) are
the source of truth for what *has* shipped.

## Shipped in 0.2

- [x] Runtime-validated handles, range-checked commands, checked memory
      arithmetic.
- [x] Changelog, stability policy, pinned-release consumption guidance, and
      enriched deterministic diagnostics.
- [x] Consumer-integration CI using the installed CMake package.
- [x] Wayland and XCB surface creation paths. Linux `AUTO` remains headless;
      applications select either WSI explicitly.

## Next priorities

Items are ordered by measured cost and expected consumer payoff.

### 1. Presentation control

The current Arx profile attributes about **0.2 ms/frame** to presentation,
the largest measured Izanagi-side frame cost.

- [ ] Select FIFO, mailbox, or immediate presentation, with deterministic
      fallback when the requested mode is unavailable.
- [ ] Expose a vsync control an application can put directly in player-facing
      settings.
- [ ] Expose frame-latency control rather than fixing the number of frames in
      flight inside the backend.
- [ ] Benchmark CPU presentation cost and end-to-end frame latency with the
      existing Arx harness when this lands.

### 2. Debug labels and object names

- [ ] Public command-buffer label regions and resource/pipeline names.
- [ ] Forward names to `VK_EXT_debug_utils` when available, with zero feature
      dependency when it is absent.
- [ ] Include known object names in deterministic-failure diagnostics.
- [ ] Verify in a RenderDoc capture that renderer passes and batches are named,
      not anonymous Vulkan handles.

### 3. Dedicated transfer queue

- [ ] Use a dedicated transfer-capable queue for uploads when hardware exposes
      one; retain the graphics-queue fallback.
- [ ] Keep the initial scope to copies/uploads and the ownership/synchronization
      needed to consume them on graphics.
- [ ] Prove that streaming submissions no longer ride the graphics timeline.
      Full async compute remains deferred until a real consumer requires it.

### 4. Multi-draw indirect completion

`MultiDrawIndirectInfo` already exposes a GPU count buffer and the Vulkan
backend already records `vkCmdDrawIndexedIndirectCount`. This item is a
correctness and validation pass, not a second primitive.

- [ ] Fix the current multi-draw-indirect failure.
- [ ] Validate argument- and count-buffer synchronization, lifetime, alignment,
      offsets, bounds, and zero/max draw counts in the same pass.
- [ ] Exercise GPU-written count buffers so the GPU genuinely decides the draw
      count.

## Linux desktop qualification

- [x] Wayland + XCB surface support.
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
