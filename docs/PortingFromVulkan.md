# Porting from Raw Vulkan

Izanagi is not a Vulkan wrapper with fewer letters. It is a different
programming model — the "No Graphics API" (NGA) model — that happens to have
a Vulkan backend. If you port by translating Vulkan calls 1:1 you will
fight the API. This page maps the concepts so you can *think* in the model
first and translate second.

## The one-sentence summary

You get `malloc`-style GPU memory that returns real 64-bit device
addresses, a single global heap of textures/samplers indexed by opaque
handles, and pipelines that take no binding layout. Everything else
(descriptors, binding tables, layouts, render passes) is gone.

## Descriptor sets → a global heap

| Vulkan | Izanagi |
|---|---|
| `VkDescriptorPool` / `vkAllocateDescriptorSets` | none — slots in a device-global heap |
| `VkDescriptorSetLayout` + binding tables | none — the shader gets a heap index |
| `vkUpdateDescriptorSets` per material | `create_texture_view` / `create_rw_texture_view` / `create_sampler` |
| descriptor write updates | handles are created once, referenced forever |
| `vkCmdBindDescriptorSets` | nothing — the heap is bound at recording start |

```cpp
// Vulkan: create image, image view, descriptor pool, set layout, update set...
// Izanagi:
Handle<Texture> tex = create_texture(device, TextureDesc{ .format = Format::RGBA8Unorm,
                                                           .dimensions = {512, 512, 1},
                                                           .usage = UsageFlags::Sampled });
TextureView view = create_texture_view(device, TextureViewDesc{ .texture = tex });
SamplerId sampler = create_sampler(device, SamplerDesc{ .min_filter = SamplerFilter::Linear });
```

The shader receives the handles as plain integers and resolves them through
the `izanagi.slang` prelude:

```hlsl
Texture2D g_albedo = getTexture2D(albedo_handle);
float4 c = g_albedo.Sample(getSampler(sampler_handle), uv);
```

Handles are generation-checked 64-bit values. Freeing one makes the old
handle permanently stale; reusing the slot yields a new handle. Copy them,
store them in GPU memory, pass them through root structs — any 64-bit
storage works. See [ShaderABI.md](ShaderABI.md) for the exact encoding.

## Vertex/index buffers and uniforms → root pointers

| Vulkan | Izanagi |
|---|---|
| `vkCmdBindVertexBuffers` / `vkCmdBindIndexBuffer` | `GpuPtr` arguments to draw calls |
| `vkCmdPushConstants` / UBO descriptors | one or two root pointers per pipeline |
| `vkCmdUpdateBuffer` | `cmd_memcpy` + barrier |

```cpp
// Vulkan: bind vertex buffer at slot 0, uniform buffer at set 0...
// Izanagi:
cmd_draw(cmd, vertexDataGpu, fragmentDataGpu, vertexCount, instanceCount);
```

```hlsl
struct Args { Vertex* verts; PerObject* frag; };
[shader("compute")] // or the vs/ps entry with both pointers
void main(uint3 tid : SV_DispatchThreadID, uniform Args args) { ... }
```

The shader **dereferences the pointers itself**. There are no attribute
bindings, no `VkVertexInputState` — vertex fetch is pointer math in the
vertex shader. This is the biggest mental shift; it is also the point: the
backend never has to guess how you use memory, so lifetime is yours.

## Render passes → dynamic rendering, always

| Vulkan | Izanagi |
|---|---|
| `VkRenderPass` + `VkFramebuffer` objects | none |
| `vkCmdBeginRenderPass` | `cmd_begin_render_pass(RenderPassDesc)` |
| subpasses | none — one attachment set, load/store ops |

Attachments are textures you created (any mip/layer), including MSAA with
an optional resolve target:

```cpp
RenderPassDesc pass{};
pass.color_attachments = Span<const RenderAttachment>(&attachment, 1);
pass.depth_attachment  = depthAttachment;
pass.render_area       = {0, 0, width, height};
cmd_begin_render_pass(cmd, pass);
```

## Pipeline barriers → stage masks

| Vulkan | Izanagi |
|---|---|
| `VkImageMemoryBarrier` + layout transitions | nothing public |
| `vkCmdPipelineBarrier(srcStage, dstStage, accessMasks...)` | `cmd_barrier(before, after)` |

```cpp
cmd_barrier(cmd, StageFlags::Compute, StageFlags::Compute);  // dispatch -> dispatch
cmd_barrier(cmd, StageFlags::RasterColorOut, StageFlags::Transfer);
```

There are no image layouts and no access masks in the public API. The
backend derives them from the stage pair (and uses `VK_KHR_unified_image
layouts` / `GENERAL` underneath). Texture copy commands take their own
barriers around them via the stage arguments where needed — read the
`cmd_copy_to_texture` docs in `gpu.h` for the exact contracts.

## Fences and binary semaphores → Submission tokens

| Vulkan | Izanagi |
|---|---|
| `VkFence` + `vkWaitForFences` | `wait_submission(Submission)` |
| `vkGetFenceStatus` | `submission_complete(Submission)` |
| binary semaphore pairs | timeline `SemaphoreInfo` wait/signal values in `queue_submit` |

```cpp
Submission s = queue_submit(queue, cbs, waitInfos, signalInfos);
// now: continue recording the next frame — do not block
// when the next frame needs it:
if (!submission_complete(prevFrame)) { /* skip update, present late, etc. */ }
```

`queue_submit` is non-blocking. Frame pacing is
`submission_complete`/`wait_submission` against your own timeline values —
the submission token is the fence.

## Resource lifetime → explicit, timeline-retired

| Vulkan | Izanagi |
|---|---|
| implicit lifetime (driver tracks usage) | **you** own lifetime |
| `vkDestroyBuffer` when "done" | `free` (safe only when the GPU cannot touch it) or `free_after(resource, submission)` |
| `vkDeviceWaitIdle` before teardown | `device_wait_for_idle` — same instinct, same cost |

```cpp
free_after(device, vertexBuffer, s);        // retire when s completes
free_after(device, texture, s);
// or, for the whole device at shutdown:
device_wait_for_idle(device);
```

The rule: if a resource is reachable through a GPU pointer or a heap handle
stored in GPU data, the backend cannot infer its lifetime — use
`free_after`. Textures named by command arguments are auto-retained by the
command buffer; that does **not** extend to GPU-pointer reachability.

## Pipelines → async requests with dedup

| Vulkan | Izanagi |
|---|---|
| `vkCreateGraphicsPipelines` (blocking, on your thread) | `request_graphics_pipeline` (returns immediately) |
| you manage pipeline caching | optional persistent cache via `PipelineCacheCallbacks` |
| you deduplicate | identical descriptions share one compiled pipeline |

```cpp
Handle<Pipeline> p = request_graphics_pipeline(device, vs, fs, raster);
if (!cmd_set_pipeline(cmd, p)) {
    // Pending or Failed: bind a fallback or skip the draw — never block here.
}
```

Blocking creators (`create_*_pipeline`) exist for loading screens. See
[PipelineCompilation.md](PipelineCompilation.md).

## The things that look similar (and are)

- `cmd_memcpy` ≈ `vkCmdCopyBuffer` — but takes `GpuPtr`s, not buffer
  objects, and validates the range for you.
- `cmd_copy_to_texture` / `cmd_copy_from_texture` ≈ `vkCmdCopyBufferToImage`
  and back, with the layout work hidden.
- `cmd_draw_indexed_instanced` ≈ `vkCmdDrawIndexed` — the index buffer is a
  `GpuPtr`, not a bound object.
- `wait_semaphore` ≈ `vkWaitSemaphores` on your own timeline values.

## Porting recipe

1. Replace your allocator with `malloc`/`free_after` (memory becomes
   addressable immediately — same upload flow, `get_host_pointer` +
   `flush_host_memory`).
2. Replace material descriptors with heap handles stored in your per-object
   structs (they are already in GPU memory — stop uploading binding tables).
3. Rewrite vertex shaders to fetch from the vertex root pointer instead of
   bound attribute buffers.
4. Replace render passes with `cmd_begin_render_pass`; replace barriers
   with stage pairs.
5. Replace fences with `Submission` tokens; replace pipeline creation with
   `request_*` + a fallback path.
6. Delete your `VkDescriptorSetLayout` / `VkPipelineLayout` /
   `VkRenderPass` plumbing entirely. It has no counterpart. That deletion
   is the point.
