# Izanagi → Metal 4 API Mapping

Every public Izanagi API entry, its Vulkan 1.4 implementation (v1, this repo),
and the designed Metal 4 mapping. The Metal backend is explicitly **out of scope
for v1** — this document proves the API ports 1:1 and locks the design
constraints that Metal imposes on the public surface.

References: [Understanding the Metal 4 core API](https://developer.apple.com/documentation/metal/understanding-the-metal-4-core-api),
[MTL4ArgumentTable](https://developer.apple.com/documentation/metal/mtl4argumenttable),
[MTLResidencySet](https://developer.apple.com/documentation/metal/mtlresidencyset),
[Resource synchronization in Metal](https://developer.apple.com/documentation/metal/resource-synchronization),
plus rkevingibson/loon_gpu's Metal backend (loon proves the argument-table +
residency-set pattern against Metal 3; Metal 4's `MTL4ArgumentTable` and
`MTL4CommandAllocator` replace the descriptor-set / command-buffer equivalents
1:1).

## Why the handles are 64-bit

`GpuPtr`, `TextureView`, and `SamplerId` are all `uint64_t`. This is not an
accident:

- **Vulkan**: device addresses (`vkGetBufferDeviceAddress`) for `GpuPtr`;
  packed heap index for `TextureView`/`SamplerId` (low 32 bits = heap index).
- **Metal 4**: `MTLBuffer.gpuAddress` for `GpuPtr`; `MTLResourceID` (a 64-bit
  opaque handle) from `MTLTexture.gpuResourceID` /
  `MTLSamplerState.gpuResourceID` for views/samplers. The 64-bit width is what
  makes the Metal mapping a type-level identity: the same struct layouts and
  shader-side unpacking work on both backends.

## Core type mappings

| Izanagi | Vulkan 1.4 impl (v1) | Metal 4 mapping |
|---|---|---|
| `GpuPtr` (uint64) | `vkGetBufferDeviceAddress` of a VMA allocation (heap or dedicated) | `MTLBuffer.gpuAddress`; allocations from one `MTLHeap`, registered in a device `MTLResidencySet`, committed once; every command buffer calls `useResidencySet` |
| `TextureView` (uint64) | low 32 = descriptor-heap index, high 32 = 0; unpacked with `uint2(v, v>>32)` | `MTLTexture.gpuResourceID` (from the texture-view pool) — a 64-bit value, stored as-is |
| `SamplerId` (uint64) | low 32 = sampler-heap index, high 32 = 0 | `MTLSamplerState.gpuResourceID` |
| `Handle<Texture>` | `VkImage` + VMA allocation, tracked in device pool | `MTLTexture` in the residency set |
| `Handle<Pipeline>` | `VkPipeline` (compute/graphics) with `VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT`, `.layout = VK_NULL_HANDLE` | `MTL4Pipeline` produced by `MTL4Compiler` |
| `Handle<Semaphore>` | timeline `VkSemaphore` | `MTLSharedEvent` (timeline semantics match 1:1) |
| `Queue` | `VkQueue` + a submit counter | `MTL4CommandQueue` |
| `CommandBuffer` | `VkCommandBuffer` (vkCmd* recorded, `vkCmdPushDataEXT` for root args) | `MTL4CommandBuffer` from `MTL4CommandAllocator` |
| `DepthStencilState` | dynamic-state `VkPipeline` flags (`vkCmdSetDepth*`, `vkCmdSetStencil*`) | `MTLDepthStencilState` |

## Root arguments

| Izanagi | Vulkan 1.4 impl (v1) | Metal 4 mapping |
|---|---|---|
| Root args = raw pointers | `vkCmdPushDataEXT` writes `VkDeviceAddress` pairs (offset 0 = vertex/compute data pointer, offset 8 = fragment data pointer) | `MTL4ArgumentTable` with 2 buffer slots (0 = compute/vertex, 1 = fragment) — loon-proven on Metal 3, unchanged in Metal 4 |
| Push-constant `Args` struct | `uniform Args { T* vert; T* frag; }` via `[[vk::push_constant]]` | `mtl4argumenttable.setBuffer(slot, ptr, 0)` with the same struct layout |

Shaders never see bindings: they receive one pointer to a user struct, which
may itself contain pointers to resources. The 2-slot argument table is the
entire binding interface on both backends.

## Global resource heap

| Izanagi | Vulkan 1.4 impl (v1) | Metal 4 mapping |
|---|---|---|
| One device-global indexable heap for textures + samplers | `VK_EXT_descriptor_heap`: device buffers backed by `VkBindHeapInfoEXT`, written via `vkWriteDescriptorSet2`/`vkWriteSamplerDescriptorsEXT`; shaders index via handles from the `getTexture2D`/`getSampler` prelude helpers | Metal manages resource identity natively: `MTLResourceID` values are indexable from shaders without a CPU-side heap object; the residency set is the device-global "heap" |

**Design constraint Metal imposes (why the API is shaped this way):** Metal
does not guarantee contiguous range allocation — `MTLResourceID`s are opaque
and the compiler assigns them. Therefore the **public API never promises
adjacent indices for separately created views**, and never exposes heap-base
arithmetic. Callers must store the handles they create; there is no
`heap_base + i` pattern. (This is already true of the v1 Vulkan API surface:
`create_texture_view` returns an opaque `TextureView`, not an index.)

## Sync

| Izanagi | Vulkan 1.4 impl (v1) | Metal 4 mapping |
|---|---|---|
| `cmd_barrier(before, after)` | `vkCmdPipelineBarrier2` with stage masks bridged from `StageFlags` (memory barrier; images stay GENERAL) | MTL4 barrier scopes (`MTL4BarrierScope`) — stage-granular only, which is exactly what the API exposes; no per-resource hazard tracking |
| `create_semaphore(init)` | timeline `VkSemaphore` | `MTLSharedEvent` + `newSharedEventWithOptions` |
| `wait_semaphore(dev, sem, value)` | `vkWaitSemaphores` on the timeline value | `MTLSharedEvent.notify` / CPU wait |
| `queue_submit(..., wait, signal)` | `VkSubmitInfo2` with `VkSemaphoreSubmitInfo` (timeline waits/signals) | `MTL4CommandQueue.commit`; `MTLSharedEvent` wait/signal commands encoded into the command buffer |
| `queue_on_submitted_work_completed` | completion tracking via timeline counter (polled in `queue_process_events`) | `MTLSharedEvent` notify handler |
| `queue_process_events` | drain completed submissions, run callbacks, recycle heap slots by timeline value | `MTLSharedEvent` notifications (equivalent hook point) |

**Design constraint Metal imposes:** barrier granularity is stage-pair only —
Metal has no per-resource layout transitions. The API's `StageFlags`-only
barrier (`cmd_barrier`) is the exact shape Metal supports; there is no
`cmd_image_transition` in the public surface, and v1's Vulkan impl keeps images
in GENERAL layout for the same reason.

## Memory

| Izanagi | Vulkan 1.4 impl (v1) | Metal 4 mapping |
|---|---|---|
| `malloc(dev, bytes, Memory)` | VMA: `Memory::Gpu` = device-local, `Memory::Default` = host-visible + device-address, `Memory::Readback` = host-visible readback | `MTLHeap.newBufferWithLength` (Gpu); `MTLBuffer` with storage mode `Shared` (Default) / `Managed` or explicit readback staging (Readback) |
| `get_host_pointer` | VMA mapped pointer | `MTLBuffer.contents` (Shared/Managed) |
| `free(dev, GpuPtr)` | VMA free (deferred to GPU idle via queue completion) | release from residency set; deferred release after the last command buffer referencing it completes |
| `get_texture_size_align` | `vkGetImageSubresourceLayout` + required alignment | `MTLTexture` size/alignment query (heap allocation helpers) |

## Textures, views, samplers

| Izanagi | Vulkan 1.4 impl (v1) | Metal 4 mapping |
|---|---|---|
| `create_texture` | `VkImage` + VMA, `UNDEFINED→GENERAL` on first use | `MTLHeap.newTextureWithDescriptor` |
| `create_texture_view` | descriptor-heap write (`imageDescriptorSize` layout) → indexable handle | view pool returning `MTLResourceID`; `MTL4ArgumentTable` slot write |
| `create_rw_texture_view` | storage-heap descriptor write | same, storage access tier via `MTLTexture` usage |
| `create_sampler` | `vkWriteSamplerDescriptorsEXT` into the sampler heap | `MTLSamplerState.gpuResourceID` |
| `free_texture_view` / `free_sampler` | heap-slot recycling (deferred until safe by timeline value) | release `MTLResourceID` from pool (no CPU heap to recycle — Metal does this internally) |
| `getTexture2D(handle)` / `getSampler(handle)` (shader prelude) | unpack uint64 → heap index → read through heap buffer | unpack uint64 → `MTLResourceID` → direct resource access (Metal's native indexing) |

**Format note:** the `Format` enum keeps ETC2/ASTC entries for the future Metal
backend; the v1 Vulkan backend logs an error and returns a null handle for
unsupported formats at `create_texture`. Metal's `MTLPixelFormat` covers the
full enum including ETC2/ASTC, so the enum set ports 1:1.

## Pipelines & draw

| Izanagi | Vulkan 1.4 impl (v1) | Metal 4 mapping |
|---|---|---|
| `create_compute_pipeline` | `vkCreateComputePipelines2` + `VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT`, no layout object | `MTL4Compiler` compile from MSL (Slang cross-compiles via slangc `-target metal`) |
| `create_graphics_pipeline` | `vkCreateGraphicsPipelines2`, `VkPipelineRenderingCreateInfo` (dynamic rendering — no render pass objects) | `MTL4Compiler`; Metal 4's dynamic rendering model (`MTL4RenderPassDescriptor` built per frame) |
| `cmd_begin_render_pass` / `cmd_end_render_pass` | `vkCmdBeginRendering` / `vkCmdEndRendering` with `VkRenderingAttachmentInfo` | `MTL4RenderPassDescriptor` + `setRenderPassDescriptor` |
| `cmd_draw*` / `cmd_dispatch*` | `vkCmdDraw*` / `vkCmdDispatch*` (root args via `vkCmdPushDataEXT`) | `MTL4RenderCommandEncoder` / `MTL4ComputeCommandEncoder` draw/dispatch; argument table already set |
| `cmd_set_depth_stencil_state` | `vkCmdSetDepth*` / `vkCmdSetStencil*` dynamic state | `setDepthStencilState` |
| `cmd_set_front_face` / `cmd_set_cull_mode` | `vkCmdSetFrontFace` / `vkCmdSetCullMode` | `setFrontFacingWinding` / `setCullMode` |

## Surface / present

| Izanagi | Vulkan 1.4 impl (v1) | Metal 4 mapping |
|---|---|---|
| `configure_surface` | `VkSwapchainKHR` (images owned by the device, used as render targets, one transition GENERAL→PRESENT_SRC at frame end) | `CAMetalLayer` drawable pool; `MTLTexture` from `nextDrawable` |
| `get_current_texture` | acquire image + wait on frame-slot timeline value before `vkAcquireNextImage2KHR` | `CAMetalLayer.nextDrawable` — the drawable IS the texture; no acquire needed |
| `present` | `vkQueuePresentKHR` waiting on the per-image present semaphore | `drawable.present()` |
| `cmd_wait_for_surface_texture` / `cmd_signal_surface_texture` | wait on acquire semaphore / signal present semaphore inside the submission | drawable availability is implicit in Metal; signal becomes `MTLSharedEvent` for frame pacing |
| `get_surface_capabilities` | `vkGetPhysicalDeviceSurfaceCapabilities2KHR` + format/present-mode queries | `CAMetalLayer` supported pixel formats + display link refresh rate |

Frame pacing (2 frames in flight, per-slot timeline wait before acquire) is
identical in concept on Metal: wait the previous frame's shared event before
requesting the next drawable.

## Explicit design constraints Metal imposes (summary)

1. **No contiguous heap-range allocation.** Metal manages `MTLResourceID`
   identity internally; the API never promises adjacent indices and never
   exposes heap-base arithmetic. Handles are opaque 64-bit values.
2. **Stage-granular barriers only.** No per-resource layout transitions in the
   public API; `cmd_barrier(StageFlags, StageFlags)` is the whole sync model.
3. **64-bit handles everywhere.** `GpuPtr`/`TextureView`/`SamplerId` are
   `uint64_t` because Metal's `gpuAddress`/`gpuResourceID` are 64-bit.
4. **Root arguments are 2 pointers.** The argument table has exactly two buffer
   slots; shader `Args` structs are pointers into user memory.
