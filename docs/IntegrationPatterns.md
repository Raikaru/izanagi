# Integration Patterns

This guide covers the application-side contracts that become important when
Izanagi is integrated into an existing renderer rather than used by a small
example. It uses the Arx Libertatis renderer integration as a worked example.
The rules are public-API rules; no Vulkan objects or backend-private state are
required.

## One owner, one frame transaction

Choose one renderer-level owner for surface acquisition, command recording,
submission, presentation, and resize. Subsystems record into that owner's
current command buffer and render pass. They must not acquire, submit, present,
or wait for the device independently.

A surface frame follows this state machine:

```text
idle -> acquired -> recording -> render pass active -> finalized
     -> submitted -> presented -> idle
```

A minimal checked frame loop is:

```cpp
SurfaceTextureInfo acquired = get_current_texture(device);
if (acquired.status == SurfaceStatus::OutOfDate) {
    configure_surface(device, surface_config);
    return;
}
if (acquired.status != SurfaceStatus::Success &&
    acquired.status != SurfaceStatus::Suboptimal) {
    // No texture was acquired. Do not record or present.
    return;
}

CommandBuffer cmd = queue_start_command_recording(queue);
cmd_wait_for_surface_texture(cmd);

RenderAttachment color{
    .texture = acquired.texture,
    .load_op = LoadOp::Clear,
    .store_op = StoreOp::Store,
    .clear_color = Color{0, 0, 0, 255},
};
RenderPassDesc pass{
    .color_attachments = Span<const RenderAttachment>(&color, 1),
    .render_area = Rect2D{.width = width, .height = height},
};
cmd_begin_render_pass(cmd, pass);

// Subsystems record into cmd and this pass.

cmd_end_render_pass(cmd);
cmd_signal_surface_texture(cmd);
cmd_finalize(cmd);

Submission submitted = queue_submit(queue, {&cmd, 1});
if (!submitted) {
    // Nothing was committed to the queue. Do not publish resource last-use
    // values and do not present. End or rebuild the configured surface before
    // acquiring again; the acquired image was not released by presentation.
    return;
}

// Publish submitted as the exact last use of indirectly reachable resources.
SurfaceStatus presented = present(device, queue);
if (presented == SurfaceStatus::OutOfDate ||
    presented == SurfaceStatus::Suboptimal) {
    // Reconfigure before the next acquisition.
}
queue_process_events(queue);
```

`queue_submit` is the transaction's commit point. Its logical timeline value is
published only after native submission succeeds. A failed submission does not
identify GPU work and must not be recorded as a resource's last use. Never call
`present` for a command buffer that was rejected or failed submission.

`cmd_wait_for_surface_texture` and `cmd_signal_surface_texture` belong in the
same command buffer as the rendering of that acquired image. Screenshot or
readback copies also belong there, before the surface signal. Waiting for the
readback submission is valid for a capture path, not as routine frame pacing.

## Explicitly named and indirectly reachable resources

Izanagi can retain resources named directly by commands, but it cannot walk
application data structures stored behind GPU pointers.

| Resource use | Backend retention | Application responsibility |
|---|---|---|
| Render attachment, bound pipeline, copy source/destination | Automatically retained by the command buffer | Keep the public handle valid while recording; use normal deferred destruction for replacement |
| Root argument, index, indirect-argument, draw-count buffer | The allocation named by the command is automatically retained | Track every allocation or descriptor reachable through pointers stored inside it |
| Pointer nested in a root structure | Not discoverable | Keep it alive through recording; retire it against the successful submission that consumed it |
| Texture view or sampler ID stored in GPU data | Not discoverable | Keep both descriptor and owning texture alive; retire by exact submission |
| CPU-side material or ownership record | Outside Izanagi | Keep it alive until submission succeeds or recording is abandoned |

This creates two separate lifetimes:

1. A CPU owner keeps integration records valid while commands are assembled.
2. A successful `Submission` protects GPU allocations and descriptor indices
   after recording ends.

Neither substitutes for the other. A common pattern is a frame-local,
deduplicated read set:

```cpp
struct FrameReads {
    std::vector<GpuPtr> allocations;
    std::vector<TextureView> views;
    std::vector<SamplerId> samplers;
    std::vector<std::shared_ptr<void>> cpu_owners;
};
```

Draw code registers indirect dependencies in this set. After successful
`queue_submit`, the renderer stamps each resource with that exact submission.
On failure it clears the set without publishing a last use. When an allocation
or descriptor is replaced, use `free_after` (or the corresponding
`free_*_after`) with its last successful submission. Immediate `free` is valid
only when no recorded, pending, in-flight, or future access exists.

An invalid or failed token passed to `free_after` retires conservatively after
the queue's latest successful work. This is useful during error cleanup, but it
does not make a failed submission a valid last-use value.

## Submission-owned transient uploads

Do not overwrite a transient upload range until its previous submission has
completed. Presentation frame numbers alone are insufficient for headless
work, extra submissions, failed submissions, or frames that do not present.

A boring ring allocator stores one `Submission` per slice:

```cpp
struct UploadSlice {
    GpuPtr base = 0;
    size_t offset = 0;
    Submission last_use{};
};

void begin_slice(UploadSlice & slice) {
    if (slice.last_use && !submission_complete(slice.last_use)) {
        wait_submission(slice.last_use);
    }
    slice.offset = 0;
}

void commit_slice(UploadSlice & slice, Submission submitted) {
    if (submitted) {
        slice.last_use = submitted;
    }
}
```

Use at least `kMaxFramesInFlight` slices for a surface-driven renderer.
`get_current_texture` paces Izanagi's corresponding surface slot, which is why
the example `GpuArgs` ring can reuse `frame_index % kMaxFramesInFlight` with
one surface submission per frame. An integration with extra queue submissions,
headless work, or independent upload streams should use exact per-slice
submission tracking instead of assuming the surface cadence protects it.

Flush only the bytes written in the selected slice with
`flush_host_memory`. Keep persistent meshes and materials out of the transient
ring; give them stable allocations and retire replacements by their exact last
use.

Partial updates to rotating persistent generations need special care. A newly
selected generation must contain a complete valid copy before a dirty range is
applied. Copying only changed bytes into an uninitialized generation produces
valid synchronization and corrupt data.

## Persistent data and copy-on-write

For long-lived geometry or material tables:

- allocate once and store stable `GpuPtr` values;
- keep an independent CPU mirror when detecting dirty ranges;
- consider both commands recorded in the current frame and incomplete prior
  submissions to be read hazards;
- update in place only when there is no hazard;
- otherwise allocate a replacement, copy the complete previous contents,
  apply the update, flush it, and retire the old allocation after its exact
  last use.

Do not compare desired values against mapped GPU memory that may still be read
by the device. The CPU mirror is the authoritative comparison source.

## Pipelines in frame code

Request pipelines outside the hot draw loop when possible. A request returns a
valid handle in `Pending`; it does not promise that the native pipeline is
ready. Frame code must branch on `get_pipeline_status` or the return value of
`cmd_set_pipeline`:

```cpp
if (!cmd_set_pipeline(cmd, material_pipeline)) {
    // Bind a known-ready fallback or skip and increment a diagnostic counter.
    return;
}
cmd_draw(cmd, vertex_args, fragment_args, vertex_count, 1);
```

Do not wait for compilation while recording a frame. Use `wait_pipeline` and
`wait_graphics_state` during loading or explicit prewarming.

## Resize, capture, and shutdown

Reconfigure the surface and replace size-dependent attachments only outside an
active frame transaction. Retire old application-owned attachments against
their exact last use. Surface-image ownership stays inside Izanagi.

For readback, record the barrier and copy before
`cmd_signal_surface_texture`, submit, wait for that submission, then call
`invalidate_host_memory` before reading non-coherent memory. Do not insert a
routine `device_wait_for_idle` into the frame loop.

At shutdown, stop producing frames, wait once for outstanding work, destroy
subsystems and application-owned resources, unconfigure the surface, then
destroy the device. A subsystem should not add its own device-idle wait.

## Worked integration: Arx Libertalis

Arx has a legacy renderer interface with many producers: opaque rooms,
transparent rooms, meshes, effects, UI, screenshots, and compatibility draws.
The Izanagi backend made the renderer the sole frame owner:

- `beginFrame` acquires one surface image, starts one command buffer, records
  the surface wait, and selects one upload slice;
- `clearFrame` opens the single renderer-owned pass;
- migrated subsystems record into that command buffer and pass while consuming
  the renderer's current transforms, viewport, scissor, depth, blend, cull,
  target formats, and sample count;
- every subsystem registers allocations and descriptor-backed resources that
  are reachable only through its GPU data;
- `present` ends the pass, signals the surface, finalizes, submits, publishes
  resource last-use values only on success, and then presents;
- resize happens while idle; shutdown performs one device wait rather than one
  wait per subsystem.

Persistent room geometry uses stable allocations. Dynamic room colors use
submission-safe generations with CPU mirrors and dirty-range flushes.
Compatibility vertices and per-draw arguments use submission-owned transient
slices. This lets native and compatibility paths share a frame without either
path owning presentation or guessing when another path's resources are safe to
reuse.

A subsystem migration is complete only when it:

- records into the owner's current command buffer and render pass;
- consumes every renderer state value that affects its legacy path;
- registers all indirectly reachable GPU resources before returning;
- publishes last use only from a successful submission;
- retires replaced resources by exact submission;
- performs no independent acquire, submit, present, or device-idle wait; and
- reports pipeline, draw, upload, allocation, and failure counters through the
  renderer's existing diagnostics.
