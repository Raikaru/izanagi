# Pipeline compilation (async model)

Izanagi compiles pipelines on a device-owned background thread. Requesting a
pipeline never blocks on native compilation; binding a pipeline never triggers
compilation.

## Blocking vs non-blocking creation

- `create_compute_pipeline` / `create_graphics_pipeline` are **blocking**
  convenience wrappers: `request_*` + `wait_pipeline`. They may perform
  expensive native compilation and are intended for loading screens, tools,
  tests, and applications with a small known pipeline set. On failure they
  return a null handle.
- `request_compute_pipeline` / `request_graphics_pipeline` are **non-blocking**:
  they deep-copy the description (SPIR-V bytes, entry points, specialization
  values, and all raster state) into an owned record, deduplicate against live
  records, enqueue the record, and return immediately. All Vulkan pipeline
  creation runs on the single device-owned compiler worker (FIFO, low
  priority). Identical descriptions share one compiled pipeline.

Every input is deep-copied by the request: caller storage may be freed or
mutated immediately after the request returns.

## Status transitions

A pipeline handle starts `Pending` and moves monotonically to `Ready` or
`Failed`:

```
Pending -> Ready        (compiled, or served from the persistent cache)
Pending -> Failed       (native compilation failed — logged once)
```

`get_pipeline_status` polls; `wait_pipeline` blocks until Ready or Failed and
returns `true` only for Ready. A Ready pipeline is immutable: its native
pipeline is never replaced or patched.

## Fallback responsibility

Izanagi never substitutes one pipeline for another and never turns a failed
pipeline into a no-op. `cmd_set_pipeline` returns `bool`:

- **Ready** — records `vkCmdBindPipeline` and returns `true`.
- **Pending / Failed** — records nothing and returns `false` (no wait, no
  compile, no disk I/O).

The application decides what to do — bind an explicit fallback, skip the draw,
skip or reschedule a dispatch, request earlier, or block during a loading
phase:

```cpp
if (!cmd_set_pipeline(cmd, material_pipeline)) {
    if (cmd_set_pipeline(cmd, error_material_pipeline)) {
        cmd_draw(...);
    }
} else {
    cmd_draw(...);
}
```

```cpp
if (cmd_set_pipeline(cmd, compute_pipeline)) {
    cmd_dispatch(...);
} else {
    // Application explicitly skips, reschedules, or uses another algorithm.
}
```

Ignoring the return value is valid source-level usage for code that only binds
blocking-created pipelines.

## Prewarming

There is no "compile every permutation" operation. A prewarm set is just a set
of ordinary `request_*` calls made earlier:

- request menu-critical pipelines before showing the menu;
- request current-scene pipelines while loading the scene;
- request likely near-future pipelines ahead of use;
- let rare pipelines compile behind an explicit fallback.

An application or engine may record the pipeline descriptions it encounters
during development and request only the relevant set for a level, scene, or
content bundle.

## Persistent cache

- One device-level native cache (`VkPipelineCache`) is created per device,
  seeded from the application's `PipelineCacheCallbacks::load` at
  `create_device` and persisted via `store` at `destroy_device` (and on
  explicit `flush_pipeline_cache`).
- Cache data is optional: invalid or rejected blobs (driver/GPU mismatch) fall
  back to an empty cache. `CacheIdentity` (backend, vendor/device IDs, driver
  UUID) lets the application key storage; the blob itself is not transferable
  across drivers or GPUs.
- When the driver supports `pipelineCreationCacheControl`, the worker first
  probes with `VK_PIPELINE_CREATE_2_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT` and
  only runs a full compile when the driver reports compilation is required
  (`VK_PIPELINE_COMPILE_REQUIRED` — expected control flow, not an error).
- `flush_pipeline_cache` blocks (it waits for queued compilation to drain) and
  is for loading screens, checkpoints, or shutdown — not frame recording.
- No storage callback ever runs from `request_*` or `cmd_set_pipeline`.

## Lifetime rules

A compiled pipeline lives until **every** reference is released:

- one reference per live `Handle<Pipeline>` (released by `free`);
- one reference held by the compiler worker while the request is queued or
  compiling;
- one reference per command buffer that bound it (kept at record time);
- at submission, command-buffer references move to an in-flight batch released
  when the queue timeline passes the submission's value (`queue_process_events`).

`free` only drops the user reference: destroying the native pipeline is deferred
until no user, worker, command-buffer, or in-flight reference remains. If the
last user reference is dropped while compilation is queued, the worker finishes
and immediately retires the result. Device destruction joins the compiler
worker before tearing down Vulkan state.

## Cache misses and first-use latency

A genuinely new native shader — never compiled on this driver/GPU and absent
from the persistent cache — cannot reach Ready with zero latency: the driver
must compile it. That compilation runs on the background worker, so it never
blocks frame-critical threads, but until it finishes `cmd_set_pipeline`
returns false and the application uses its fallback, skip, or wait path.
Zero first-use latency for an unseen shader is only achievable by compiling
earlier (prewarming), omitting the work, or explicitly falling back — never by
silent substitution.

## Diagnostics

Best-effort internal counters track requests, dedup hits, cache-only probe
successes, `VK_PIPELINE_COMPILE_REQUIRED` results, full compilations, failures,
and peak queue depth. Compile and wait times are logged at Info level when a
log callback is configured.
