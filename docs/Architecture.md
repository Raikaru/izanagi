# Architecture

Izanagi is a minimal bindless GPU API: the application supplies **data, GPU
pointers, heap indices, shader code, minimal immutable pipeline state, dynamic
commands, and explicit asynchronous execution**. There are no descriptor sets,
render passes, root signatures, or resource-state objects in the public API.

## Public mental model

- **Memory** is `malloc`-style: `gpu::malloc` returns a 64-bit device address
  (`GpuPtr`). Pointers are valid for the GPU and the CPU (`get_host_pointer`).
- **Textures and samplers** live in device-global heaps. `TextureView` /
  `SamplerId` handles encode a descriptor index (low 32 bits, shader-visible)
  plus CPU-side generation/type metadata; shaders receive only the index.
- **Pipelines** are immutable descriptions: shader bytes + entry points +
  specialization values + minimal raster state. They are requested
  asynchronously and become `Pending` → `Ready`/`Failed`.
- **Commands** are recorded into command buffers and submitted explicitly;
  submission returns a token identifying the work.
- **Everything completes asynchronously**; the application polls or blocks
  with the explicit APIs (`submission_complete`, `wait_submission`,
  `get_pipeline_status`, `wait_pipeline`).

## Threading rules

- **Device creation/destruction:** externally synchronized (one thread).
- **Resource creation and non-blocking pipeline requests:** thread-safe
  (`create_texture`, `malloc`, `create_*_view`, `request_*_pipeline`, ...).
- **Command-buffer recording:** one recording thread per command buffer;
  multiple command buffers may be recorded concurrently (pool selection and
  checkout are serialized per queue).
- **Queue submit and present:** internally serialized per queue.
- **Queue submission, status, completion, and retirement queries**
  (`queue_submit`, `submission_complete`, `wait_submission`,
  `queue_process_events`): thread-safe (drained under the queue lock, callbacks
  and native destruction run outside it).
- **Handle destruction and retirement requests:** thread-safe, subject to the
  GPU lifetime contract below.
- **Scratch memory:** per-thread arenas; backend operations rewind them at
  exit (see ScratchProvider below).

## Submission tokens

`queue_submit` returns a `Submission { queue, value, status }`. The value is
the queue's timeline value for that work; it is **published only after the
native submit succeeds** — a failed submit never advances the logical timeline,
so no value is ever issued that the GPU will not signal. Submit failures reach
the caller (`DeviceLost` / `OutOfMemory` / `Error`) instead of being logged and
dropped.

`submission_complete(s)` polls the timeline; `wait_submission(s)` blocks
(loading screens and shutdown — not frame recording).

## Immediate vs deferred destruction

- **Immediate `free`** is valid only when the application guarantees no
  recorded, pending, in-flight, or future GPU access. It never calls
  `vkDeviceWaitIdle`; it simply destroys at the API layer.
- **`free_after(device, resource, submission)`** is the normal path for
  resources reachable through raw GPU pointers or descriptor indices stored in
  user GPU data (the backend cannot infer such usage). The CPU handle is
  invalidated immediately; native destruction happens when the target
  submission completes. An invalid/failed submission retires conservatively
  after the latest submitted work completes.
- Objects **explicitly named in recorded commands** (bound pipelines, render
  attachments, copy sources/destinations, memory-copy/index/indirect/draw-count
  buffers, root-argument buffers) are automatically retained by the command
  buffer, transferred to the queue's retirement queue at submit, and released
  only when the submission completes. Freeing the user handle cannot destroy a
  native object the recorded commands reference.
- Native destruction happens only after **all** references are gone: user
  handles, compiler-worker references, command-buffer references, and
  in-flight references.

## One queue retirement service

One queue-owned, timeline-keyed retirement queue handles command-pool reuse,
pipeline destruction, texture destruction, buffer destruction, and descriptor
slot recycling. `queue_process_events` drains batches whose timeline value has
completed, in ascending order, outside queue locks (the drain itself is
serialized); device shutdown drains everything after joining the compiler
worker. Texture initialization submits an internal transition command buffer
that participates in the same pool accounting, retirement, and failure
rollback as application command buffers; a texture freed while other
references keep it alive stays drainable (marked Released, removed from the
init list only at the final reference).

Descriptor slots follow an explicit state machine — `Free -> Allocated ->
Retiring -> Free` — guarded by a descriptor allocator lock: an accepted free
immediately moves the slot to Retiring and bumps the generation (the old
handle is stale at once); a second free while Retiring is rejected; the slot
is not reusable until the retirement submission completes (delayed duplicate
retirements are verified by state and generation). A sampled/storage
descriptor retains its owner texture record until the slot retires, so the
texture survives its public free while a view references it. A stale
GPU-side descriptor index cannot be made safe by generation bits — retirement
remains required.

## Command-pool retirement

Command pools are retired by **queue timeline**, never by presentation frame
counters. A pool is reusable only when the queue's completed timeline is at
least its retire value (the last submission using a command buffer from it).
Finalizing a command buffer does not make its pool reusable; headless
workloads work without presentation; abandoned command buffers release their
references at pool reset.

## Memory

- Every allocation records its user-visible range; any `GpuPtr` lookup
  validates that the pointer is inside the range (gaps and past-end pointers
  are rejected, never resolved to a plausible buffer/offset). Offsets are
  64-bit.
- `malloc(..., align)` honors power-of-two alignment by overallocating and
  aligning the user address inside the backing allocation; invalid alignment
  is rejected.
- `CacheIdentity` carries both `cache_uuid` (the Vulkan `pipelineCacheUUID`,
  the primary compatibility key, read from the cache blob header) and
  `driver_uuid` (fallback when the driver exposes no header for an empty
  cache), plus the backend **profile** that produced the blob — Native and
  Bindless cache blobs are never interchangeable, even for the same driver and
  GPU.
- `flush_host_memory` / `invalidate_host_memory` synchronize non-coherent
  allocations (aligned to the non-coherent atom size, validated against the
  range). Coherent memory is a successful no-op. Call flush after writing an
  upload buffer before submitting; call invalidate after GPU work completes
  before reading a readback buffer.

## Pipeline request states

`request_*_pipeline` deep-copies its inputs and returns immediately; a
device-owned compiler worker compiles in the background (never on the
requesting or recording thread). State transitions are monotonic:

```
Pending -> Ready   (compiled, or served from the persistent native cache)
Pending -> Failed  (compilation failed; logged once)
```

`cmd_set_pipeline` returns `bool`: Ready binds and returns true; Pending/Failed
record nothing and return false. **Fallback selection is application-controlled**
— Izanagi never substitutes a pipeline and never turns a compute dispatch into
a no-op:

```cpp
if (!cmd_set_pipeline(cmd, material_pipeline)) {
    if (cmd_set_pipeline(cmd, error_material_pipeline)) {
        cmd_draw(...);
    }
} else {
    cmd_draw(...);
}
```

See `docs/PipelineCompilation.md` for the compilation model, persistent cache,
and prewarming.

## Backend selection

One backend per compiled library: build configuration selects the backend
(currently Vulkan 1.4 on Windows); `device_backend()` reports it. There is no
runtime backend switching.

## Backend profiles

A backend may implement the public model through different private mechanisms.
Each compiled profile is named and reported (`device_backend_profile()`);
`IZANAGI_VK_PROFILE` selects the Vulkan profile at configure time:

- **Native** (`BackendProfile::VulkanNative`) — the modern path: Vulkan 1.4,
  `VK_EXT_descriptor_heap` global heap, untyped pointers, unified image
  layouts, `vkCmdPushDataEXT` root args.
- **Bindless** (`BackendProfile::VulkanBindless`) — the compatibility path:
  descriptor-indexing arrays in a backend-private set, typed physical storage
  buffer pointers, ordinary push constants, private image-layout handling.
  Same public semantics: real GPU pointers + a persistent GPU-indexed
  resource namespace. (Declared; its backend lands in the compatibility
  phases. `IZANAGI_VK_PROFILE=BINDLESS` fails configure until then.)
- **Metal** (`BackendProfile::Metal`) — future Apple backend.

The profile never degrades semantics: a device that cannot preserve the
pointer or global-resource model must be rejected with a complete capability
report (see the profile evaluator, `src/common/profile_report.cpp` and
`src/vk/profile.cpp`).

## Shutdown order

Device destruction is deterministic: stop accepting new work → join the
compiler worker → drain retirement batches → flush the persistent pipeline
cache → destroy remaining pipelines/textures/buffers/views/descriptor
state/pools/swapchain → destroy the pipeline cache → destroy the Vulkan device
and allocator. No detached thread or callback outlives `DeviceImpl`.
