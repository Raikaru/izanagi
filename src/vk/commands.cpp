// commands.cpp — all cmd_* recording, command pool management, queue submission.

#include <algorithm>
#include <new>

#include "internal.h"

namespace gpu {

// Vulkan 1.3 commands are available on 1.2 devices only through their KHR/EXT
// extensions. There are NO compile-time aliases: dispatch is route-aware
// (core symbols on 1.3+, extension symbols on 1.2), because promoted-but-
// unadvertised extension names must never be enabled and the core symbols do
// not resolve on 1.2 devices.
static inline bool route_is_core13(const DeviceImpl* d) {
    return d->dispatch.effective_api_version >= VK_API_VERSION_1_3;
}

// --- Route-aware dispatch helpers ----------------------------------------------------
// Core-1.3 symbols on 1.3+, KHR/EXT symbols on the 1.2 extension route.

static void backend_pipeline_barrier2(DeviceImpl* d, VkCommandBuffer cmd, const VkDependencyInfo* info) {
    if (route_is_core13(d)) { vkCmdPipelineBarrier2(cmd, info); } else { vkCmdPipelineBarrier2KHR(cmd, info); }
}

static VkResult backend_queue_submit2(DeviceImpl* d, VkQueue q, uint32_t count,
                                      const VkSubmitInfo2* info, VkFence fence) {
    if (route_is_core13(d)) { return vkQueueSubmit2(q, count, info, fence); }
    return vkQueueSubmit2KHR(q, count, info, fence);
}

static void backend_begin_rendering(DeviceImpl* d, VkCommandBuffer cmd, const VkRenderingInfo* info) {
    if (route_is_core13(d)) { vkCmdBeginRendering(cmd, info); } else { vkCmdBeginRenderingKHR(cmd, info); }
}

static void backend_end_rendering(DeviceImpl* d, VkCommandBuffer cmd) {
    if (route_is_core13(d)) { vkCmdEndRendering(cmd); } else { vkCmdEndRenderingKHR(cmd); }
}

// --- Descriptor heap binding ---------------------------------------------------------

void cmd_bind_descriptor_heaps(DeviceImpl* d, VkCommandBuffer cmd) {
#if defined(IZ_VK_PROFILE_BINDLESS)
    // One global descriptor set bound to both bind points at recording start;
    // every pipeline shares the same private pipeline layout.
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, d->bindless_pipeline_layout,
                            0, 1, &d->bindless_set, 0, nullptr);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, d->bindless_pipeline_layout,
                            0, 1, &d->bindless_set, 0, nullptr);
#else
    const VkBindHeapInfoEXT sampler_bind{
        .sType               = VK_STRUCTURE_TYPE_BIND_HEAP_INFO_EXT,
        .pNext               = nullptr,
        .heapRange           = {d->heap.sampler_device_ptr, d->heap.sampler_heap_size},
        .reservedRangeOffset = d->heap.sampler_reserved_offset,
        .reservedRangeSize   = d->heap.sampler_reserved_size,
    };
    const VkBindHeapInfoEXT resource_bind{
        .sType               = VK_STRUCTURE_TYPE_BIND_HEAP_INFO_EXT,
        .pNext               = nullptr,
        .heapRange           = {d->heap.resource_device_ptr, d->heap.resource_heap_size},
        .reservedRangeOffset = d->heap.resource_reserved_offset,
        .reservedRangeSize   = d->heap.resource_reserved_size,
    };
    vkCmdBindSamplerHeapEXT(cmd, &sampler_bind);
    vkCmdBindResourceHeapEXT(cmd, &resource_bind);
#endif
}

// --- Command pool management -----------------------------------------------------------

// Clears the command buffer's static-variant binding state (borrows — the
// variants live for the base pipeline's lifetime, so nothing is released).
static void release_cb_variant(CommandBufferImpl* cb) {
    cb->bound_variant       = nullptr;
    cb->bound_base_pipeline = nullptr;
}

static void reset_command_pool(DeviceImpl* d, CommandPool* pool) {
    // Release references retained by recorded-but-never-submitted command
    // buffers (their commands are discarded; the GPU never executed them,
    // and pool reuse is gated on queue timeline completion).
    for (uint32_t i = 0; i < pool->command_buffers.size(); ++i) {
        auto& cb = pool->command_buffers[i];
        if (cb.device == nullptr) { continue; }
        release_cb_variant(&cb);
        for (PipelineRecord* rec : cb.retained_pipelines) {
            release_pipeline_ref(cb.device, rec);
        }
        cb.retained_pipelines.clear();
        for (TextureImpl* rec : cb.retained_textures) {
            Handle<Texture> h = handle_cast<Texture>(cb.device->texture_pool.find_handle(rec));
            if (h.h != 0) { release_texture_ref(cb.device, h); }
        }
        cb.retained_textures.clear();
        for (Handle<Buffer> b : cb.retained_buffers) {
            release_buffer_ref(cb.device, b);
        }
        cb.retained_buffers.clear();
    }
    vkResetCommandPool(d->device, pool->command_pool, 0);
    pool->buffer_free_idx = 0;
    atomic_fetch_add(&d->stat_pool_resets, 1);
}

// Retains a texture explicitly named by a command until the command buffer is
// submitted (or abandoned), so freeing the user handle cannot destroy a native
// image the recorded commands still reference. Retains the stable record, not
// the public handle.
static void retain_texture(CommandBufferImpl* cmd, Handle<Texture> tex) {
    TextureImpl* rec = &cmd->device->texture_pool[handle_cast<TextureImpl>(tex)];
    for (TextureImpl* r : cmd->retained_textures) {
        if (r == rec) { return; }   // already retained
    }
    atomic_fetch_add(&rec->refs, 1);
    cmd->retained_textures.push_back(rec);
}

// Retains a buffer explicitly named by a command (memory copies, index and
// indirect operands). Resolves the allocation; invalid pointers are not
// retained (the command will fail at submit).
static void retain_buffer(CommandBufferImpl* cmd, GpuPtr ptr) {
    if (ptr == 0) { return; }
    VkDeviceSize offset = 0;
    Buffer* buf = find_buffer_for_ptr(cmd->device, ptr, &offset);
    if (buf == nullptr) { return; }
    Handle<Buffer> h = cmd->device->buffer_pool.find_handle(buf);
    if (h.h == 0) { return; }
    for (Handle<Buffer> b : cmd->retained_buffers) {
        if (b.h == h.h) { return; }   // already retained
    }
    atomic_fetch_add(&buf->refs, 1);
    cmd->retained_buffers.push_back(h);
}

CommandPool* get_command_pool(QueueImpl* queue) {
    // record_lock is held by the caller (queue_start_command_recording) across
    // pool selection AND command-buffer checkout.
    auto* d = queue->device;

    // Completed timeline: a pool is reusable only once all submitted command
    // buffers allocated from it have completed (retire_value <= completed).
    uint64_t completed = 0;
    VkSemaphore timeline_sem =
        d->semaphore_pool[handle_cast<SemaphoreImpl>(queue->timeline)].vk_semaphore;
    vkGetSemaphoreCounterValue(d->device, timeline_sem, &completed);

    for (auto& pool : queue->command_superpool.pools) {
        if (pool.outstanding != 0) { continue; }
        if (pool.command_pool == VK_NULL_HANDLE) {
            const VkCommandPoolCreateInfo pool_info{
                .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                .pNext            = nullptr,
                .flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
                .queueFamilyIndex = queue->queue_family,
            };
            VkCommandPool command_pool = VK_NULL_HANDLE;
            if (!IZ_CHK(d, vkCreateCommandPool(d->device, &pool_info, nullptr, &command_pool),
                        "get_command_pool failed")) {
                return nullptr;
            }
            pool.command_pool      = command_pool;
            pool.command_buffers   = SegmentArray<CommandBufferImpl>(d->allocator);
            pool.buffer_free_idx   = 0;
            pool.retire_value      = 0;
            pool.outstanding       = 0;
            return &pool;
        }
        if (pool.retire_value <= completed) {
            reset_command_pool(d, &pool);
            return &pool;
        }
    }
    IZ_LOG(d, LogLevel::Error, "Too many command buffers in flight");
    return nullptr;
}

CommandBuffer get_command_buffer(QueueImpl* q, CommandPool* pool) {
    auto* d = q->device;
    // record_lock is held by the caller (queue_start_command_recording).
    if (pool->command_buffers.size() <= pool->buffer_free_idx) {
        const VkCommandBufferAllocateInfo info{
            .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .pNext              = nullptr,
            .commandPool        = pool->command_pool,
            .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };
        VkCommandBuffer buf;
        if (!IZ_CHK(d, vkAllocateCommandBuffers(d->device, &info, &buf),
                    "get_command_buffer failed")) {
            return nullptr;
        }
        pool->command_buffers.emplace_back(CommandBufferImpl{
            .device             = d,
            .queue              = q,
            .pool               = pool,
            .buffer             = buf,
            .current_idx_buffer = 0,
        });
        pool->command_buffers[pool->command_buffers.size() - 1].retained_pipelines =
            Vector<PipelineRecord*>(d->allocator);
        pool->command_buffers[pool->command_buffers.size() - 1].retained_textures =
            Vector<TextureImpl*>(d->allocator);
        pool->command_buffers[pool->command_buffers.size() - 1].retained_buffers =
            Vector<Handle<Buffer>>(d->allocator);
    }
    CommandBufferImpl* result        = &pool->command_buffers[pool->buffer_free_idx];
    result->wait_for_surface_texture = false;
    result->signal_surface_texture   = false;
    pool->buffer_free_idx++;
    pool->outstanding++;   // the pool stays checked out until this cb is submitted
    return result;
}

CommandBuffer queue_start_command_recording(Queue q) {
    auto* d = q->device;
    mutex_lock(&q->record_lock);
    CommandPool* pool = get_command_pool(q);
    CommandBuffer buffer = (pool != nullptr) ? get_command_buffer(q, pool) : nullptr;
    mutex_unlock(&q->record_lock);
    if (pool == nullptr) { return nullptr; }

    if (buffer) {
        buffer->current_idx_buffer = 0;
        const VkCommandBufferBeginInfo begin_info{
            .sType            = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .pNext            = nullptr,
            .flags            = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            .pInheritanceInfo = nullptr,
        };
        if (!IZ_CHK(d, vkBeginCommandBuffer(buffer->buffer, &begin_info),
                    "queue_start_command_recording failed")) {
            return nullptr;
        }
        reset_logical_graphics_state(buffer);
        release_cb_variant(buffer);
        buffer->recording_failed = false;
        buffer->fail_reason      = nullptr;
        // Bind descriptor heaps once at recording start
        cmd_bind_descriptor_heaps(d, buffer->buffer);
    }
    return buffer;
}

void cmd_finalize(CommandBuffer cmd) {
    // Preserve the recording-failed state (the application can inspect the
    // error via submission rejection / debug logs; the buffer must not be
    // submitted with partial or invalid commands).
    if (cmd->recording_failed) {
        IZ_LOG(cmd->device, LogLevel::Error,
               cmd->fail_reason != nullptr ? cmd->fail_reason : "command buffer recording failed");
        return;
    }

    auto* d = cmd->device;
    IZ_CHK(d, vkEndCommandBuffer(cmd->buffer), "cmd_finalize failed");
    // The command pool stays checked out until this command buffer is
    // submitted; finalizing does not make the pool GPU-safe for reuse.
}

// --- Unified retirement queue --------------------------------------------------------------
// Caller holds q->submit_lock.
// Finds or creates the retire batch for `value` and reserves item capacity.
// Returns nullptr on allocation failure (the caller must abort the
// submission before GPU work is enqueued). Caller holds q->submit_lock.
static RetireBatch* prepare_retire_batch(DeviceImpl* d, QueueImpl* q, uint64_t value, uint32_t item_capacity) {
    for (uint32_t i = 0; i < q->retire_queue.size(); ++i) {
        RetireBatch* b = q->retire_queue[i];
        if (b->value == value) {
            if (!b->items.reserve(b->items.size() + item_capacity)) { return nullptr; }
            return b;
        }
        if (b->value > value) {
            // Reserve the queue slot BEFORE allocating the batch so the
            // insertion below cannot fail and orphan the batch.
            if (!q->retire_queue.reserve(q->retire_queue.size() + 1)) { return nullptr; }
            MemoryBlock blk = d->allocator.alloc(sizeof(RetireBatch));
            if (blk.ptr == nullptr) { return nullptr; }
            auto* nb = ::new (blk.ptr) RetireBatch{
                .value = value,
                .items = Vector<RetireItem>(d->allocator),
            };
            if (!nb->items.reserve(item_capacity)) {
                nb->~RetireBatch();
                d->allocator.free({.ptr = nb, .len = sizeof(RetireBatch)});
                return nullptr;
            }
            q->retire_queue.insert(q->retire_queue.begin() + i, nb);
            return nb;
        }
    }
    if (!q->retire_queue.reserve(q->retire_queue.size() + 1)) { return nullptr; }
    MemoryBlock blk = d->allocator.alloc(sizeof(RetireBatch));
    if (blk.ptr == nullptr) { return nullptr; }
    auto* nb = ::new (blk.ptr) RetireBatch{
        .value = value,
        .items = Vector<RetireItem>(d->allocator),
    };
    if (!nb->items.reserve(item_capacity)) {
        nb->~RetireBatch();
        d->allocator.free({.ptr = nb, .len = sizeof(RetireBatch)});
        return nullptr;
    }
    q->retire_queue.push_back(nb);
    return nb;
}

// Fallible retirement insertion. Returns false on allocation failure: the
// caller must keep ownership (conservative retention until device shutdown)
// rather than silently dropping the reference.
bool enqueue_retire(QueueImpl* q, uint64_t value, const RetireItem& item) {
    auto* d = q->device;
    mutex_lock(&q->submit_lock);
    RetireBatch* batch = prepare_retire_batch(d, q, value, 1);
    bool ok = batch != nullptr;
    if (ok) { batch->items.push_back(item); }
    mutex_unlock(&q->submit_lock);
    if (!ok) {
        IZ_LOG(d, LogLevel::Error, "enqueue_retire: allocation failure — resource retained until shutdown");
    }
    return ok;
}

// Destroys/retires the batch's resources. Called without queue locks held.
void process_retire_batch(DeviceImpl* d, RetireBatch* batch) {
    for (const RetireItem& item : batch->items) {
        switch (item.kind) {
            case RetireKind::Buffer:
                release_buffer_ref(d, Handle<Buffer>{.h = item.a});
                break;
            case RetireKind::Texture: {
                auto* rec = reinterpret_cast<TextureImpl*>(item.a);
                Handle<Texture> h = handle_cast<Texture>(d->texture_pool.find_handle(rec));
                if (h.h != 0) {
                    remove_uninitialized_texture(d, rec);
                    release_texture_ref(d, h);
                }
                break;
            }
            case RetireKind::PipelineRef:
                release_pipeline_ref(d, reinterpret_cast<PipelineRecord*>(item.a));
                break;
            case RetireKind::SampledSlot:
                desc_retire_slot(d, RetireKind::SampledSlot, static_cast<uint32_t>(item.a),
                                 static_cast<uint16_t>(item.b));
                break;
            case RetireKind::StorageSlot:
                desc_retire_slot(d, RetireKind::StorageSlot, static_cast<uint32_t>(item.a),
                                 static_cast<uint16_t>(item.b));
                break;
            case RetireKind::SamplerSlot:
                desc_retire_slot(d, RetireKind::SamplerSlot, static_cast<uint32_t>(item.a),
                                 static_cast<uint16_t>(item.b));
                break;
        }
    }
    batch->items.clear();
    batch->~RetireBatch();
    d->allocator.free({.ptr = batch, .len = sizeof(RetireBatch)});
}

// --- Queue submission -----------------------------------------------------------------

Submission queue_submit(Queue                     q,
                        Span<const CommandBuffer> command_buffers,
                        Span<const SemaphoreInfo> wait_semaphores,
                        Span<const SemaphoreInfo> signal_semaphores) {
    auto* d = q->device;
    Arena*  arena = get_thread_local_arena(d);
    if (arena == nullptr) { return {}; }
    ScratchScope scope(*arena);

    Span<VkSemaphoreSubmitInfo>     wait_info{};
    Span<VkCommandBufferSubmitInfo> command_info{};
    Span<VkSemaphoreSubmitInfo>     signal_info{};

    // Reject command buffers with a recording error deterministically:
    // nothing is submitted, no timeline advances. The never-submitted
    // checkouts and retained references MUST be released here — otherwise
    // every rejected buffer leaks its command pool (outstanding never
    // returns to 0) and the retained buffers/textures/pipelines never drain,
    // starving later submissions (pool exhaustion / in-flight hangs).
    for (uint32_t i = 0; i < command_buffers.size(); ++i) {
        if (command_buffers[i]->recording_failed) {
            IZ_LOG(d, LogLevel::Error,
                   command_buffers[i]->fail_reason != nullptr
                       ? command_buffers[i]->fail_reason
                       : "queue_submit: command buffer recording failed");
            mutex_lock(&q->record_lock);
            for (uint32_t j = 0; j < command_buffers.size(); ++j) {
                CommandBufferImpl* cb = command_buffers[j];
                if (cb->pool != nullptr && cb->pool->outstanding > 0) { cb->pool->outstanding--; }
            }
            mutex_unlock(&q->record_lock);
            for (uint32_t j = 0; j < command_buffers.size(); ++j) {
                CommandBufferImpl* cb = command_buffers[j];
                release_cb_variant(cb);
                for (PipelineRecord* r : cb->retained_pipelines) { release_pipeline_ref(d, r); }
                cb->retained_pipelines.clear();
                for (TextureImpl* rec : cb->retained_textures) {
                    Handle<Texture> h = handle_cast<Texture>(d->texture_pool.find_handle(rec));
                    if (h.h != 0) { release_texture_ref(d, h); }
                }
                cb->retained_textures.clear();
                for (Handle<Buffer> b : cb->retained_buffers) { release_buffer_ref(d, b); }
                cb->retained_buffers.clear();
            }
            return Submission{.queue = q, .value = q->timeline_value + 1, .status = SubmitStatus::Error};
        }
    }

    // Serialize submission (and presentation) per queue. The serialized region
    // covers the entire stateful sequence: claim texture initialization
    // records, record the optional internal transition command buffer, reserve
    // retirement storage, submit, and commit/roll back every participant.
    mutex_lock(&q->submit_lock);
    const uint64_t submit_value = q->timeline_value + 1;

    // Test hook: injected failure — nothing is claimed, no timeline advance.
    if (atomic_load(&d->force_submit_failure)) {
        atomic_fetch_add(&d->stat_failed_submits, 1);
        mutex_unlock(&q->submit_lock);
        return Submission{.queue = q, .value = submit_value, .status = SubmitStatus::Error};
    }

    // Claim uninitialized textures — one-time UNDEFINED->GENERAL transition.
    // The list holds stable records, and each is temporarily referenced so a
    // concurrent free/free_after cannot destroy the record mid-preparation.
    mutex_lock(&d->texture_init_lock);
    Vector<TextureImpl*> texture_list(d->allocator);
    swap(d->uninitialized_textures, texture_list);
    for (TextureImpl* rec : texture_list) { atomic_fetch_add(&rec->refs, 1); }
    mutex_unlock(&d->texture_init_lock);

    CommandBufferImpl* internal_cmd = nullptr;
    if (texture_list.size() > 0) {
        internal_cmd = queue_start_command_recording(q);
        if (internal_cmd == nullptr) {
            mutex_lock(&d->texture_init_lock);
            for (TextureImpl* rec : texture_list) {
                if (rec->init_state == TextureInitState::NeedsTransition) {
                    d->uninitialized_textures.push_back(rec);
                }
            }
            mutex_unlock(&d->texture_init_lock);
            for (TextureImpl* rec : texture_list) {
                Handle<Texture> h = handle_cast<Texture>(d->texture_pool.find_handle(rec));
                if (h.h != 0) { release_texture_ref(d, h); }
            }
            mutex_unlock(&q->submit_lock);
            return Submission{.queue = q, .value = submit_value, .status = SubmitStatus::Error};
        }
        Span<VkImageMemoryBarrier2> image_barriers{};
        for (TextureImpl* rec : texture_list) {
            // Claim under the lock: NeedsTransition -> TransitionQueued only;
            // never clobber a concurrent Released (a public free racing the
            // drain). Records not claimed get no barrier.
            mutex_lock(&d->texture_init_lock);
            // Never clobber a concurrent Released: only NeedsTransition
            // advances to TransitionQueued. The barrier is still emitted for
            // every drained record (a Released record may be retained by
            // application command buffers in this very submit and its GPU use
            // still needs the UNDEFINED->GENERAL transition).
            if (rec->init_state == TextureInitState::NeedsTransition) {
                rec->init_state = TextureInitState::TransitionQueued;
            }
            mutex_unlock(&d->texture_init_lock);
            image_barriers = concat(
                arena, image_barriers,
                VkImageMemoryBarrier2{
                    .sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                    .pNext         = nullptr,
                    .srcStageMask  = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
                    .srcAccessMask = 0,
                    .dstStageMask  = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                    .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                    .oldLayout     = VK_IMAGE_LAYOUT_UNDEFINED,
                    .newLayout     = VK_IMAGE_LAYOUT_GENERAL,
                    .image         = rec->vk_image,
                    .subresourceRange =
                        VkImageSubresourceRange{
                            .aspectMask     = aspects_for_format(rec->format),
                            .baseMipLevel   = 0,
                            .levelCount     = VK_REMAINING_MIP_LEVELS,
                            .baseArrayLayer = 0,
                            .layerCount     = VK_REMAINING_ARRAY_LAYERS,
                        },
                });
        }
        const VkDependencyInfo dependency_info{
            .sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .pNext                    = nullptr,
            .dependencyFlags          = 0,
            .memoryBarrierCount       = 0,
            .pMemoryBarriers          = nullptr,
            .bufferMemoryBarrierCount = 0,
            .pBufferMemoryBarriers    = nullptr,
            .imageMemoryBarrierCount  = static_cast<uint32_t>(image_barriers.size()),
            .pImageMemoryBarriers     = image_barriers.data(),
        };
                backend_pipeline_barrier2(d, internal_cmd->buffer, &dependency_info);
        cmd_finalize(internal_cmd);
        command_info = concat(arena, command_info,
                              VkCommandBufferSubmitInfo{
                                  .sType         = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
                                  .pNext         = nullptr,
                                  .commandBuffer = internal_cmd->buffer,
                                  .deviceMask    = 1,
                              });
    }

    // Wait semaphores
    for (uint32_t i = 0; i < wait_semaphores.size(); ++i) {
        wait_info = concat(arena, wait_info,
                           VkSemaphoreSubmitInfo{
                               .sType       = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                               .pNext       = nullptr,
                               .semaphore   = d->semaphore_pool[handle_cast<SemaphoreImpl>(wait_semaphores[i].semaphore)].vk_semaphore,
                               .value       = wait_semaphores[i].value,
                               .stageMask   = bridge_pipeline_stage(wait_semaphores[i].stage),
                               .deviceIndex = 0,
                           });
    }

    // Command buffers + surface semaphore handling
    for (uint32_t i = 0; i < command_buffers.size(); i++) {
        auto buf = command_buffers[i]->buffer;
        command_info = concat(arena, command_info,
                              VkCommandBufferSubmitInfo{
                                  .sType         = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
                                  .pNext         = nullptr,
                                  .commandBuffer = buf,
                                  .deviceMask    = 1,
                              });

        if (command_buffers[i]->wait_for_surface_texture) {
            const auto acquire_sem =
                d->semaphore_pool[handle_cast<SemaphoreImpl>(d->surface.acquire_semaphores[d->surface.frame_idx % kMaxFramesInFlight])]
                    .vk_semaphore;
            wait_info = concat(arena, wait_info,
                               VkSemaphoreSubmitInfo{
                                   .sType       = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                                   .pNext       = nullptr,
                                   .semaphore   = acquire_sem,
                                   .value       = 0,
                                   .stageMask   = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                                   .deviceIndex = 0,
                               });
        }
        if (command_buffers[i]->signal_surface_texture) {
            const VkSemaphore present_sem =
                d->semaphore_pool[handle_cast<SemaphoreImpl>(d->surface.present_semaphores[d->surface.current_image_idx])]
                    .vk_semaphore;
            signal_info = concat(arena, signal_info,
                                 VkSemaphoreSubmitInfo{
                                     .sType       = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                                     .pNext       = nullptr,
                                     .semaphore   = present_sem,
                                     .value       = 0,
                                     .stageMask   = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
                                     .deviceIndex = 0,
                                 });
            const VkSemaphore frame_sem =
                d->semaphore_pool[handle_cast<SemaphoreImpl>(d->surface.frame_semaphore)].vk_semaphore;
            signal_info = concat(arena, signal_info,
                                 VkSemaphoreSubmitInfo{
                                     .sType       = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                                     .pNext       = nullptr,
                                     .semaphore   = frame_sem,
                                     .value       = d->surface.frame_idx + 1,
                                     .stageMask   = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
                                     .deviceIndex = 0,
                                 });
        }
    }

    // Signal semaphores
    for (uint32_t i = 0; i < signal_semaphores.size(); ++i) {
        signal_info = concat(arena, signal_info,
                             VkSemaphoreSubmitInfo{
                                 .sType     = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                                 .pNext     = nullptr,
                                 .semaphore = d->semaphore_pool[handle_cast<SemaphoreImpl>(signal_semaphores[i].semaphore)].vk_semaphore,
                                 .value     = signal_semaphores[i].value,
                                 .stageMask = bridge_pipeline_stage(signal_semaphores[i].stage),
                                 .deviceIndex = 0,
                             });
    }

    // Advance queue timeline with the prospective value; published only on a
    // successful submit so the GPU never receives an unsignalable value.
    signal_info = concat(arena, signal_info,
                         VkSemaphoreSubmitInfo{
                             .sType       = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                             .pNext       = nullptr,
                             .semaphore   = d->semaphore_pool[handle_cast<SemaphoreImpl>(q->timeline)].vk_semaphore,
                             .value       = submit_value,
                             .stageMask   = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
                             .deviceIndex = 0,
                         });

    // Rollback: restore texture transitions for a later retry, return every
    // participating command buffer (internal + application) to a reusable or
    // abandoned state, and release never-submitted retained references.
    auto rollback = [&]() {
        mutex_lock(&d->texture_init_lock);
        for (TextureImpl* rec : texture_list) {
            // Requeue only records still awaiting their transition; records a
            // concurrent free/free_after released must not be resurrected.
            if (rec->init_state == TextureInitState::TransitionQueued) {
                rec->init_state = TextureInitState::NeedsTransition;
                d->uninitialized_textures.push_back(rec);
            }
        }
        mutex_unlock(&d->texture_init_lock);
        for (TextureImpl* rec : texture_list) {
            Handle<Texture> h = handle_cast<Texture>(d->texture_pool.find_handle(rec));
            if (h.h != 0) { release_texture_ref(d, h); }   // the preparation ref
        }
        mutex_lock(&q->record_lock);   // pool accounting is record_lock-guarded
        if (internal_cmd != nullptr && internal_cmd->pool != nullptr &&
            internal_cmd->pool->outstanding > 0) {
            internal_cmd->pool->outstanding--;   // never submitted: release the checkout
        }
        for (uint32_t i = 0; i < command_buffers.size(); ++i) {
            auto* cb = command_buffers[i];
            if (cb->pool != nullptr && cb->pool->outstanding > 0) { cb->pool->outstanding--; }
        }
        mutex_unlock(&q->record_lock);
        for (uint32_t i = 0; i < command_buffers.size(); ++i) {
            auto* cb = command_buffers[i];
            release_cb_variant(cb);
            for (PipelineRecord* rec : cb->retained_pipelines) { release_pipeline_ref(d, rec); }
            cb->retained_pipelines.clear();
            for (TextureImpl* rec : cb->retained_textures) {
                Handle<Texture> h = handle_cast<Texture>(d->texture_pool.find_handle(rec));
                if (h.h != 0) { release_texture_ref(d, h); }
            }
            cb->retained_textures.clear();
            for (Handle<Buffer> b : cb->retained_buffers) { release_buffer_ref(d, b); }
            cb->retained_buffers.clear();
        }
    };

    if (arena->overflowed()) {
        atomic_fetch_add(&d->stat_failed_submits, 1);
        rollback();
        mutex_unlock(&q->submit_lock);
        IZ_LOG(d, LogLevel::Error, "queue_submit: scratch arena overflow, submission aborted");
        return Submission{.queue = q, .value = submit_value, .status = SubmitStatus::Error};
    }

    // Reserve retirement storage BEFORE the native submit so the post-submit
    // reference transfer cannot fail (ownership must never be dropped).
    uint32_t retain_count = static_cast<uint32_t>(texture_list.size());   // prep refs
    for (uint32_t i = 0; i < command_buffers.size(); ++i) {
        retain_count += static_cast<uint32_t>(command_buffers[i]->retained_pipelines.size());
        retain_count += static_cast<uint32_t>(command_buffers[i]->retained_textures.size());
        retain_count += static_cast<uint32_t>(command_buffers[i]->retained_buffers.size());
    }
    RetireBatch* retire_batch = prepare_retire_batch(d, q, submit_value, retain_count);
    if (retire_batch == nullptr) {
        atomic_fetch_add(&d->stat_failed_submits, 1);
        rollback();
        mutex_unlock(&q->submit_lock);
        IZ_LOG(d, LogLevel::Error, "queue_submit: failed to reserve retirement storage");
        return Submission{.queue = q, .value = submit_value, .status = SubmitStatus::Error};
    }

    const VkSubmitInfo2 submit_info{
        .sType                    = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
        .pNext                    = nullptr,
        .flags                    = 0,
        .waitSemaphoreInfoCount   = static_cast<uint32_t>(wait_info.size()),
        .pWaitSemaphoreInfos      = wait_info.data(),
        .commandBufferInfoCount   = static_cast<uint32_t>(command_info.size()),
        .pCommandBufferInfos      = command_info.data(),
        .signalSemaphoreInfoCount = static_cast<uint32_t>(signal_info.size()),
        .pSignalSemaphoreInfos    = signal_info.data(),
    };

    const VkResult result =         backend_queue_submit2(d, q->queue, 1, &submit_info, VK_NULL_HANDLE);
    if (result != VK_SUCCESS) {
        atomic_fetch_add(&d->stat_failed_submits, 1);
        log_vk_impl(d, result, "queue_submit: vkQueueSubmit2 failed", __LINE__, "commands.cpp"_sv);
        rollback();
        mutex_unlock(&q->submit_lock);
        Submission s{.queue = q, .value = submit_value, .status = SubmitStatus::Error};
        if (result == VK_ERROR_DEVICE_LOST) { s.status = SubmitStatus::DeviceLost; }
        if (result == VK_ERROR_OUT_OF_DEVICE_MEMORY || result == VK_ERROR_OUT_OF_HOST_MEMORY) {
            s.status = SubmitStatus::OutOfMemory;
        }
        return s;
    }

    // Commit: publish the logical timeline, mark transitions initialized, and
    // account every participating command buffer (internal + application) with
    // one path — pool outstanding/retire value and retained-reference transfer.
    q->timeline_value = submit_value;
    atomic_fetch_add(&d->stat_submissions, 1);
    for (TextureImpl* rec : texture_list) {
        mutex_lock(&d->texture_init_lock);
        if (rec->init_state == TextureInitState::TransitionQueued) {
            rec->init_state = TextureInitState::Initialized;   // never clobber Released
        }
        mutex_unlock(&d->texture_init_lock);
        // Release the preparation reference when THIS submission completes
        // (the transition barrier runs in it; the record must survive until
        // then even if a concurrent free dropped the user reference).
        retire_batch->items.push_back(
            RetireItem{RetireKind::Texture, reinterpret_cast<uint64_t>(rec), 0});
    }
    auto commit_cb = [&](CommandBufferImpl* cb) {
        if (cb->pool != nullptr) {
            mutex_lock(&q->record_lock);   // pool accounting is record_lock-guarded
            if (cb->pool->outstanding > 0) {
                cb->pool->outstanding--;
                if (cb->pool->retire_value < submit_value) { cb->pool->retire_value = submit_value; }
            }
            mutex_unlock(&q->record_lock);
        }
        for (PipelineRecord* rec : cb->retained_pipelines) {
            retire_batch->items.push_back(
                RetireItem{RetireKind::PipelineRef, reinterpret_cast<uint64_t>(rec), 0});
        }
        cb->retained_pipelines.clear();
        for (TextureImpl* rec : cb->retained_textures) {
            retire_batch->items.push_back(
                RetireItem{RetireKind::Texture, reinterpret_cast<uint64_t>(rec), 0});
        }
        cb->retained_textures.clear();
        for (Handle<Buffer> b : cb->retained_buffers) {
            retire_batch->items.push_back(RetireItem{RetireKind::Buffer, b.h, 0});
        }
        cb->retained_buffers.clear();
    };
    if (internal_cmd != nullptr) { commit_cb(internal_cmd); }
    for (uint32_t i = 0; i < command_buffers.size(); ++i) { commit_cb(command_buffers[i]); }
    mutex_unlock(&q->submit_lock);
    return Submission{.queue = q, .value = submit_value, .status = SubmitStatus::Success};
}

bool submission_complete(Submission s) {
    if (s.status != SubmitStatus::Success || s.queue == nullptr) { return false; }
    auto* q = reinterpret_cast<QueueImpl*>(s.queue);
    auto* d = q->device;
    uint64_t counter = 0;
    VkSemaphore sem = d->semaphore_pool[handle_cast<SemaphoreImpl>(q->timeline)].vk_semaphore;
    if (vkGetSemaphoreCounterValue(d->device, sem, &counter) != VK_SUCCESS) { return false; }
    return counter >= s.value;
}

bool wait_submission(Submission s) {
    if (s.status != SubmitStatus::Success || s.queue == nullptr) { return false; }
    auto* q = reinterpret_cast<QueueImpl*>(s.queue);
    auto* d = q->device;
    VkSemaphore sem = d->semaphore_pool[handle_cast<SemaphoreImpl>(q->timeline)].vk_semaphore;
    const VkSemaphoreWaitInfo wait_info{
        .sType          = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
        .pNext          = nullptr,
        .flags          = 0,
        .semaphoreCount = 1,
        .pSemaphores    = &sem,
        .pValues        = &s.value,
    };
    return vkWaitSemaphores(d->device, &wait_info, UINT64_MAX) == VK_SUCCESS;
}

// White-box test hooks
void debug_force_submit_failure(DeviceImpl* d, bool force) {
    atomic_exchange(&d->force_submit_failure, force ? 1 : 0);
}

uint64_t debug_queue_timeline(DeviceImpl* d) {
    return d->default_queue ? d->default_queue->timeline_value : 0;
}

bool debug_validation_active(DeviceImpl* d) {
    return d->enable_validation && d->debug_messenger != VK_NULL_HANDLE;
}

void debug_force_legacy_copy(DeviceImpl* d, bool force) {
    d->force_legacy_copy = force ? 1 : 0;
    capture_device_capabilities(d);
}

void debug_force_static_state(DeviceImpl* d, bool force) {
    d->force_static_state = force ? 1 : 0;
    capture_device_capabilities(d);
}

void debug_derive_dispatch(DeviceImpl* d) {
    capture_device_capabilities(d);
}

bool debug_using_static_state(DeviceImpl* d) {
    return d->dispatch.use_static_graphics_state;
}

uint32_t debug_effective_api_version(DeviceImpl* d) {
    return d->dispatch.effective_api_version;
}

// True for the observable "limited 1.2" dispatch signature: 1.2 device with
// neither copy-commands2 nor extended-dynamic-state (Mesa dzn). Behavior-
// based; never vendor-ID-based.
bool debug_limited_1_2(DeviceImpl* d) {
    return d->dispatch.effective_api_version < VK_API_VERSION_1_3 &&
           d->dispatch.use_legacy_copy_commands && d->dispatch.use_static_graphics_state;
}

int64_t debug_stat(DeviceImpl* d, int which) {
    switch (which) {
        case 0: return atomic_load(&d->stat_copy2_calls);
        case 1: return atomic_load(&d->stat_legacy_copy_calls);
        case 2: return atomic_load(&d->stat_ext_dyn_state_calls);
        case 3: return atomic_load(&d->stat_static_variant_lookups);
        case 4: return atomic_load(&d->stat_static_variant_hits);
        case 5: return atomic_load(&d->stat_static_variant_misses);
        case 6: return atomic_load(&d->stat_static_variant_pending);
        case 7: return atomic_load(&d->stat_static_variant_compilations);
        default: return 0;
    }
}

void debug_force_legacy_barriers(DeviceImpl* d, bool force) {
    d->force_legacy_barriers = force ? 1 : 0;
}

void reset_logical_graphics_state(CommandBufferImpl* cb) {
    cb->graphics_state = LogicalGraphicsState{};
    cb->bound_variant       = nullptr;
    cb->bound_base_pipeline = nullptr;
}

const LogicalGraphicsState* debug_cb_graphics_state(CommandBufferImpl* cb) {
    return &cb->graphics_state;
}

uint32_t debug_legacy_stage_mask(StageFlags stage) {
    return static_cast<uint32_t>(bridge_pipeline_stage_legacy(stage));
}

bool debug_bindless_null_slot_written(DeviceImpl* d) {
#if defined(IZ_VK_PROFILE_BINDLESS)
    return d->bindless_sampled_views.size() > 0 && d->bindless_sampled_views[0] != VK_NULL_HANDLE &&
           d->bindless_sampler_handles.size() > 0 && d->bindless_sampler_handles[0] != VK_NULL_HANDLE;
#else
    (void)d;
    return false;
#endif
}

int64_t debug_pool_resets(DeviceImpl* d) {
    return atomic_load(&d->stat_pool_resets);
}

// Extended-dynamic-state setters: core-1.3 symbols on 1.3+, EXT symbols on
// the 1.2 route. Only called when the dynamic-state path is selected (not on
// the static-variant fallback).
#define IZ_BACKEND_EDS_HELPER(name)                                                        \
    static void backend_##name(DeviceImpl* d, VkCommandBuffer cmd, auto... args) {         \
        if (route_is_core13(d)) { vk##name(cmd, args...); } else { vk##name##EXT(cmd, args...); } \
    }

static void backend_set_front_face(DeviceImpl* d, VkCommandBuffer cmd, VkFrontFace face) {
    atomic_fetch_add(&d->stat_ext_dyn_state_calls, 1);
    if (route_is_core13(d)) { vkCmdSetFrontFace(cmd, face); } else { vkCmdSetFrontFaceEXT(cmd, face); }
}
static void backend_set_cull_mode(DeviceImpl* d, VkCommandBuffer cmd, VkCullModeFlags mode) {
    atomic_fetch_add(&d->stat_ext_dyn_state_calls, 1);
    if (route_is_core13(d)) { vkCmdSetCullMode(cmd, mode); } else { vkCmdSetCullModeEXT(cmd, mode); }
}
static void backend_set_depth_write_enable(DeviceImpl* d, VkCommandBuffer cmd, VkBool32 b) {
    atomic_fetch_add(&d->stat_ext_dyn_state_calls, 1);
    if (route_is_core13(d)) { vkCmdSetDepthWriteEnable(cmd, b); } else { vkCmdSetDepthWriteEnableEXT(cmd, b); }
}
static void backend_set_depth_test_enable(DeviceImpl* d, VkCommandBuffer cmd, VkBool32 b) {
    atomic_fetch_add(&d->stat_ext_dyn_state_calls, 1);
    if (route_is_core13(d)) { vkCmdSetDepthTestEnable(cmd, b); } else { vkCmdSetDepthTestEnableEXT(cmd, b); }
}
static void backend_set_depth_compare_op(DeviceImpl* d, VkCommandBuffer cmd, VkCompareOp op) {
    atomic_fetch_add(&d->stat_ext_dyn_state_calls, 1);
    if (route_is_core13(d)) { vkCmdSetDepthCompareOp(cmd, op); } else { vkCmdSetDepthCompareOpEXT(cmd, op); }
}
static void backend_set_depth_bounds_test_enable(DeviceImpl* d, VkCommandBuffer cmd, VkBool32 b) {
    atomic_fetch_add(&d->stat_ext_dyn_state_calls, 1);
    if (route_is_core13(d)) { vkCmdSetDepthBoundsTestEnable(cmd, b); } else { vkCmdSetDepthBoundsTestEnableEXT(cmd, b); }
}
static void backend_set_stencil_test_enable(DeviceImpl* d, VkCommandBuffer cmd, VkBool32 b) {
    atomic_fetch_add(&d->stat_ext_dyn_state_calls, 1);
    if (route_is_core13(d)) { vkCmdSetStencilTestEnable(cmd, b); } else { vkCmdSetStencilTestEnableEXT(cmd, b); }
}
static void backend_set_stencil_op(DeviceImpl* d, VkCommandBuffer cmd, VkStencilFaceFlags face,
                                   VkStencilOp fail, VkStencilOp pass, VkStencilOp depth_fail,
                                   VkCompareOp compare) {
    atomic_fetch_add(&d->stat_ext_dyn_state_calls, 1);
    if (route_is_core13(d)) { vkCmdSetStencilOp(cmd, face, fail, pass, depth_fail, compare); }
    else { vkCmdSetStencilOpEXT(cmd, face, fail, pass, depth_fail, compare); }
}
static void backend_set_viewport_with_count(DeviceImpl* d, VkCommandBuffer cmd, uint32_t n, const VkViewport* v) {
    atomic_fetch_add(&d->stat_ext_dyn_state_calls, 1);
    if (route_is_core13(d)) { vkCmdSetViewportWithCount(cmd, n, v); } else { vkCmdSetViewportWithCountEXT(cmd, n, v); }
}
static void backend_set_scissor_with_count(DeviceImpl* d, VkCommandBuffer cmd, uint32_t n, const VkRect2D* r) {
    atomic_fetch_add(&d->stat_ext_dyn_state_calls, 1);
    if (route_is_core13(d)) { vkCmdSetScissorWithCount(cmd, n, r); } else { vkCmdSetScissorWithCountEXT(cmd, n, r); }
}

// --- Backend copy dispatch -----------------------------------------------------------
// Selects the modern (VK_KHR_copy_commands2) or legacy copy/blit command from
// the device's dispatch capabilities. Region conversions use stack storage for
// the common single-region case and the thread-local scratch arena for many
// regions; any failure marks the command buffer failed deterministically
// (no Vulkan command is issued with mismatched data).

static void cmd_record_fail(CommandBufferImpl* cb, const char* reason) {
    cb->recording_failed = true;
    cb->fail_reason      = reason;
    IZ_LOG(cb->device, LogLevel::Error, reason);
}

void convert_buffer_copy_regions(const VkBufferCopy2* src, uint32_t count, VkBufferCopy* dst) {
    for (uint32_t i = 0; i < count; ++i) {
        dst[i] = VkBufferCopy{
            .srcOffset = src[i].srcOffset,
            .dstOffset = src[i].dstOffset,
            .size      = src[i].size,
        };
    }
}

void convert_buffer_image_copy_regions(const VkBufferImageCopy2* src, uint32_t count, VkBufferImageCopy* dst) {
    for (uint32_t i = 0; i < count; ++i) {
        dst[i] = VkBufferImageCopy{
            .bufferOffset      = src[i].bufferOffset,
            .bufferRowLength   = src[i].bufferRowLength,
            .bufferImageHeight = src[i].bufferImageHeight,
            .imageSubresource  = src[i].imageSubresource,
            .imageOffset       = src[i].imageOffset,
            .imageExtent       = src[i].imageExtent,
        };
    }
}

void convert_image_blit_regions(const VkImageBlit2* src, uint32_t count, VkImageBlit* dst) {
    for (uint32_t i = 0; i < count; ++i) {
        dst[i] = VkImageBlit{
            .srcSubresource = src[i].srcSubresource,
            .srcOffsets     = {src[i].srcOffsets[0], src[i].srcOffsets[1]},
            .dstSubresource = src[i].dstSubresource,
            .dstOffsets     = {src[i].dstOffsets[0], src[i].dstOffsets[1]},
        };
    }
}

// Rejects non-null pNext in the *2 info and its typed regions (Izanagi never
// uses extension-specific copy payloads; silently discarding them would be
// wrong). Typed access — a region array is an array of structs, not pointers.
template <class T>
static bool copy_info_has_pnext(const void* info_pnext, const T* regions, uint32_t region_count) {
    if (info_pnext != nullptr) { return true; }
    for (uint32_t i = 0; i < region_count; ++i) {
        if (regions[i].pNext != nullptr) { return true; }
    }
    return false;
}

void backend_copy_buffer(CommandBufferImpl* cb, const VkCopyBufferInfo2& info) {
    auto* d = cb->device;
    if (!d->dispatch.use_legacy_copy_commands) {
        atomic_fetch_add(&d->stat_copy2_calls, 1);
        if (route_is_core13(d)) { vkCmdCopyBuffer2(cb->buffer, &info); } else { vkCmdCopyBuffer2KHR(cb->buffer, &info); }
        return;
    }
    atomic_fetch_add(&d->stat_legacy_copy_calls, 1);
    if (copy_info_has_pnext(info.pNext, info.pRegions, info.regionCount)) {
        cmd_record_fail(cb, "copy: unsupported non-null pNext in copy info");
        return;
    }
    if (info.regionCount == 0) { return; }   // legal no-op
    VkBufferCopy stack_region{};
    VkBufferCopy* regions = &stack_region;
    if (info.regionCount > 1) {
        Arena* arena = get_thread_local_arena(d);
        if (arena == nullptr) { cmd_record_fail(cb, "copy: no scratch arena"); return; }
        ScratchScope scope(*arena);
        regions = static_cast<VkBufferCopy*>(arena->alloc(sizeof(VkBufferCopy) * info.regionCount));
        if (regions == nullptr) { cmd_record_fail(cb, "copy: region conversion allocation failed"); return; }
    }
    convert_buffer_copy_regions(info.pRegions, info.regionCount, regions);
    vkCmdCopyBuffer(cb->buffer, info.srcBuffer, info.dstBuffer, info.regionCount, regions);
}

void backend_copy_buffer_to_image(CommandBufferImpl* cb, const VkCopyBufferToImageInfo2& info) {
    auto* d = cb->device;
    if (!d->dispatch.use_legacy_copy_commands) {
        atomic_fetch_add(&d->stat_copy2_calls, 1);
        if (route_is_core13(d)) { vkCmdCopyBufferToImage2(cb->buffer, &info); } else { vkCmdCopyBufferToImage2KHR(cb->buffer, &info); }
        return;
    }
    atomic_fetch_add(&d->stat_legacy_copy_calls, 1);
    if (copy_info_has_pnext(info.pNext, info.pRegions, info.regionCount)) {
        cmd_record_fail(cb, "copy: unsupported non-null pNext in copy info");
        return;
    }
    if (info.regionCount == 0) { return; }
    VkBufferImageCopy stack_region{};
    VkBufferImageCopy* regions = &stack_region;
    if (info.regionCount > 1) {
        Arena* arena = get_thread_local_arena(d);
        if (arena == nullptr) { cmd_record_fail(cb, "copy: no scratch arena"); return; }
        ScratchScope scope(*arena);
        regions = static_cast<VkBufferImageCopy*>(arena->alloc(sizeof(VkBufferImageCopy) * info.regionCount));
        if (regions == nullptr) { cmd_record_fail(cb, "copy: region conversion allocation failed"); return; }
    }
    convert_buffer_image_copy_regions(info.pRegions, info.regionCount, regions);
    vkCmdCopyBufferToImage(cb->buffer, info.srcBuffer, info.dstImage, info.dstImageLayout,
                           info.regionCount, regions);
}

void backend_copy_image_to_buffer(CommandBufferImpl* cb, const VkCopyImageToBufferInfo2& info) {
    auto* d = cb->device;
    if (!d->dispatch.use_legacy_copy_commands) {
        atomic_fetch_add(&d->stat_copy2_calls, 1);
        if (route_is_core13(d)) { vkCmdCopyImageToBuffer2(cb->buffer, &info); } else { vkCmdCopyImageToBuffer2KHR(cb->buffer, &info); }
        return;
    }
    atomic_fetch_add(&d->stat_legacy_copy_calls, 1);
    if (copy_info_has_pnext(info.pNext, info.pRegions, info.regionCount)) {
        cmd_record_fail(cb, "copy: unsupported non-null pNext in copy info");
        return;
    }
    if (info.regionCount == 0) { return; }
    VkBufferImageCopy stack_region{};
    VkBufferImageCopy* regions = &stack_region;
    if (info.regionCount > 1) {
        Arena* arena = get_thread_local_arena(d);
        if (arena == nullptr) { cmd_record_fail(cb, "copy: no scratch arena"); return; }
        ScratchScope scope(*arena);
        regions = static_cast<VkBufferImageCopy*>(arena->alloc(sizeof(VkBufferImageCopy) * info.regionCount));
        if (regions == nullptr) { cmd_record_fail(cb, "copy: region conversion allocation failed"); return; }
    }
    convert_buffer_image_copy_regions(info.pRegions, info.regionCount, regions);
    vkCmdCopyImageToBuffer(cb->buffer, info.srcImage, info.srcImageLayout, info.dstBuffer,
                           info.regionCount, regions);
}

void backend_blit_image(CommandBufferImpl* cb, const VkBlitImageInfo2& info) {
    auto* d = cb->device;
    if (!d->dispatch.use_legacy_copy_commands) {
        atomic_fetch_add(&d->stat_copy2_calls, 1);
        if (route_is_core13(d)) { vkCmdBlitImage2(cb->buffer, &info); } else { vkCmdBlitImage2KHR(cb->buffer, &info); }
        return;
    }
    atomic_fetch_add(&d->stat_legacy_copy_calls, 1);
    if (copy_info_has_pnext(info.pNext, info.pRegions, info.regionCount)) {
        cmd_record_fail(cb, "blit: unsupported non-null pNext in blit info");
        return;
    }
    if (info.regionCount == 0) { return; }
    VkImageBlit stack_region{};
    VkImageBlit* regions = &stack_region;
    if (info.regionCount > 1) {
        Arena* arena = get_thread_local_arena(d);
        if (arena == nullptr) { cmd_record_fail(cb, "blit: no scratch arena"); return; }
        ScratchScope scope(*arena);
        regions = static_cast<VkImageBlit*>(arena->alloc(sizeof(VkImageBlit) * info.regionCount));
        if (regions == nullptr) { cmd_record_fail(cb, "blit: region conversion allocation failed"); return; }
    }
    convert_image_blit_regions(info.pRegions, info.regionCount, regions);
    vkCmdBlitImage(cb->buffer, info.srcImage, info.srcImageLayout, info.dstImage, info.dstImageLayout,
                   info.regionCount, regions, info.filter);
}

// --- Commands -------------------------------------------------------------------------

void cmd_memcpy(CommandBuffer cmd, GpuPtr destGpu, GpuPtr srcGpu, size_t size) {
    auto* d = cmd->device;
    retain_buffer(cmd, destGpu);
    retain_buffer(cmd, srcGpu);
    auto src = buffer_and_offset_from_ptr(d, srcGpu);
    auto dst = buffer_and_offset_from_ptr(d, destGpu);
    VkBufferCopy2 region{
        .sType     = VK_STRUCTURE_TYPE_BUFFER_COPY_2,
        .srcOffset = src.offset,
        .dstOffset = dst.offset,
        .size      = size,
    };
    VkCopyBufferInfo2 copy_info{
        .sType       = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
        .srcBuffer   = src.buffer,
        .dstBuffer   = dst.buffer,
        .regionCount = 1,
        .pRegions    = &region,
    };
    backend_copy_buffer(cmd, copy_info);
}

void cmd_copy_to_texture(CommandBuffer                cmd,
                         GpuPtr                       srcPtr,
                         Handle<Texture>              texture,
                         const BufferTextureCopyInfo& info) {
    auto* d = cmd->device;
    retain_buffer(cmd, srcPtr);
    retain_texture(cmd, texture);
    auto src = buffer_and_offset_from_ptr(d, srcPtr);
    auto& tex = d->texture_pool[handle_cast<TextureImpl>(texture)];

    VkBufferImageCopy2 region{
        .sType             = VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2,
        .pNext             = nullptr,
        .bufferOffset      = src.offset,
        .bufferRowLength   = info.buffer_row_pixels_stride,
        .bufferImageHeight = info.buffer_plane_rows_stride,
        .imageSubresource  = {
             .aspectMask     = aspects_for_format(tex.format),
             .mipLevel       = info.base_mip,
             .baseArrayLayer = info.base_layer,
             .layerCount     = 1,
        },
        .imageOffset = {
            .x = static_cast<int32_t>(info.texture_image_offset.x),
            .y = static_cast<int32_t>(info.texture_image_offset.y),
            .z = static_cast<int32_t>(info.texture_image_offset.z),
        },
        .imageExtent = {
            .width  = info.image_extent.x,
            .height = info.image_extent.y,
            .depth  = info.image_extent.z,
        },
    };
    VkCopyBufferToImageInfo2 copy_info{
        .sType          = VK_STRUCTURE_TYPE_COPY_BUFFER_TO_IMAGE_INFO_2,
        .pNext          = nullptr,
        .srcBuffer      = src.buffer,
        .dstImage       = tex.vk_image,
        .dstImageLayout = VK_IMAGE_LAYOUT_GENERAL,
        .regionCount    = 1,
        .pRegions       = &region,
    };
    backend_copy_buffer_to_image(cmd, copy_info);
}

void cmd_copy_from_texture(CommandBuffer                cmd,
                           Handle<Texture>              texture,
                           GpuPtr                       destGpu,
                           const BufferTextureCopyInfo& info) {
    auto* d = cmd->device;
    retain_buffer(cmd, destGpu);
    retain_texture(cmd, texture);
    auto dst = buffer_and_offset_from_ptr(d, destGpu);
    auto& tex = d->texture_pool[handle_cast<TextureImpl>(texture)];

    VkBufferImageCopy2 region{
        .sType             = VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2,
        .pNext             = nullptr,
        .bufferOffset      = dst.offset,
        .bufferRowLength   = info.buffer_row_pixels_stride,
        .bufferImageHeight = info.buffer_plane_rows_stride,
        .imageSubresource  = {
             .aspectMask     = aspects_for_format(tex.format),
             .mipLevel       = info.base_mip,
             .baseArrayLayer = info.base_layer,
             .layerCount     = 1,
        },
        .imageOffset = {
            .x = static_cast<int32_t>(info.texture_image_offset.x),
            .y = static_cast<int32_t>(info.texture_image_offset.y),
            .z = static_cast<int32_t>(info.texture_image_offset.z),
        },
        .imageExtent = {
            .width  = info.image_extent.x,
            .height = info.image_extent.y,
            .depth  = info.image_extent.z,
        },
    };
    VkCopyImageToBufferInfo2 copy_info{
        .sType          = VK_STRUCTURE_TYPE_COPY_IMAGE_TO_BUFFER_INFO_2,
        .pNext          = nullptr,
        .srcImage       = tex.vk_image,
        .srcImageLayout = VK_IMAGE_LAYOUT_GENERAL,
        .dstBuffer      = dst.buffer,
        .regionCount    = 1,
        .pRegions       = &region,
    };
    backend_copy_image_to_buffer(cmd, copy_info);
}

void cmd_barrier(CommandBuffer cmd, StageFlags before, StageFlags after) {
    auto* d = cmd->device;
    if (d->force_legacy_barriers != 0 || !d->use_synchronization2) {
        // Legacy fallback (devices without synchronization2, or forced for
        // testing): conservative memory-read/write access at the mapped
        // 1.0-era stage bits. Preserves the public stage-only barrier model.
        const VkPipelineStageFlags src_stage = bridge_pipeline_stage_legacy(before);
        const VkPipelineStageFlags dst_stage = bridge_pipeline_stage_legacy(after);
        constexpr VkAccessFlags access = VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT;
        const VkMemoryBarrier barrier{
            .sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext         = nullptr,
            .srcAccessMask = access,
            .dstAccessMask = access,
        };
        vkCmdPipelineBarrier(cmd->buffer, src_stage, dst_stage, 0, 1, &barrier, 0, nullptr, 0, nullptr);
        return;
    }
    const auto src_stage = bridge_pipeline_stage(before);
    const auto dst_stage = bridge_pipeline_stage(after);
    constexpr auto access = VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT;

    const VkMemoryBarrier2 barrier_info{
        .sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .pNext         = nullptr,
        .srcStageMask  = src_stage,
        .srcAccessMask = access,
        .dstStageMask  = dst_stage,
        .dstAccessMask = access,
    };
    const VkDependencyInfo info{
        .sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .pNext                    = nullptr,
        .dependencyFlags          = 0,
        .memoryBarrierCount       = 1,
        .pMemoryBarriers          = &barrier_info,
        .bufferMemoryBarrierCount = 0,
        .pBufferMemoryBarriers    = nullptr,
        .imageMemoryBarrierCount  = 0,
        .pImageMemoryBarriers     = nullptr,
    };
            backend_pipeline_barrier2(d, cmd->buffer, &info);
}

void cmd_generate_mipmaps(CommandBuffer cmd, Handle<Texture> texture) {
    auto* d   = cmd->device;
    retain_texture(cmd, texture);
    auto& tex = d->texture_pool[handle_cast<TextureImpl>(texture)];

    const uint32_t aspect = aspects_for_format(tex.format);
    const uint32_t w      = tex.dimensions.x;
    const uint32_t h      = tex.dimensions.y;

    // Successive linear blits: mip i-1 -> mip i, layer 0 only.
    for (uint32_t i = 1; i < tex.mip_count; ++i) {
        const VkImageBlit2 region{
            .sType          = VK_STRUCTURE_TYPE_IMAGE_BLIT_2,
            .pNext          = nullptr,
            .srcSubresource = {
                .aspectMask     = aspect,
                .mipLevel       = i - 1,
                .baseArrayLayer = 0,
                .layerCount     = 1,
            },
            .srcOffsets = {
                {0, 0, 0},
                {static_cast<int32_t>(std::max(w >> (i - 1), 1u)),
                 static_cast<int32_t>(std::max(h >> (i - 1), 1u)), 1},
            },
            .dstSubresource = {
                .aspectMask     = aspect,
                .mipLevel       = i,
                .baseArrayLayer = 0,
                .layerCount     = 1,
            },
            .dstOffsets = {
                {0, 0, 0},
                {static_cast<int32_t>(std::max(w >> i, 1u)),
                 static_cast<int32_t>(std::max(h >> i, 1u)), 1},
            },
        };
        const VkBlitImageInfo2 info{
            .sType          = VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2,
            .pNext          = nullptr,
            .srcImage       = tex.vk_image,
            .srcImageLayout = VK_IMAGE_LAYOUT_GENERAL,
            .dstImage       = tex.vk_image,
            .dstImageLayout = VK_IMAGE_LAYOUT_GENERAL,
            .regionCount    = 1,
            .pRegions       = &region,
            .filter         = VK_FILTER_LINEAR,
        };
        backend_blit_image(cmd, info);

        // Serialize blit write (mip i) vs next blit read (mip i).
        if (i + 1 < tex.mip_count) {
            cmd_barrier(cmd, StageFlags::Transfer, StageFlags::Transfer);
        }
    }
}

bool cmd_set_pipeline(CommandBuffer cmd, Handle<Pipeline> pipeline) {
    auto* d = cmd->device;
    PipelineRecord* rec = d->pipeline_pool[handle_cast<PipelineImpl>(pipeline)].record;
    // Pending or Failed: record nothing so the application can explicitly
    // bind a fallback or skip the operation. Never blocks on compilation.
    if (rec->state.load(std::memory_order_acquire) != InternalPipelineState::Ready) {
        return false;
    }
    vkCmdBindPipeline(cmd->buffer, rec->bind_point, rec->vk_pipeline);
    // Track the base pipeline for static-variant resolution at draw time.
    cmd->bound_base_pipeline = rec;
    cmd->graphics_state.dirty_static_state = true;
    // Retain the native pipeline for the command buffer's lifetime (one
    // reference per distinct pipeline; repeated binds reuse it).
    for (PipelineRecord* r : cmd->retained_pipelines) {
        if (r == rec) { return true; }
    }
    rec->refs.fetch_add(1, std::memory_order_relaxed);
    cmd->retained_pipelines.push_back(rec);
    return true;
}

// Draw-time variant resolution on the static-graphics-state fallback. On the
// dynamic path this is a no-op. On the static path it resolves the private
// variant for (bound base pipeline, baked shadow):
//   - default baked state -> the base pipeline itself is the variant
//   - Ready variant       -> bound (deduped; map-owned)
//   - Pending/Failed      -> deterministic recording failure (submit rejects)
bool prepare_static_graphics(CommandBufferImpl* cb) {
    DeviceImpl* d = cb->device;
    if (!d->dispatch.use_static_graphics_state) { return true; }
    PipelineRecord* base = cb->bound_base_pipeline;
    if (base == nullptr) {
        cmd_record_fail(cb, "draw without a bound graphics pipeline");
        return false;
    }
    if (!cb->graphics_state.dirty_static_state && cb->bound_variant != nullptr &&
        cb->bound_variant->base == base) {
        return true;   // the currently bound variant still applies
    }
    if (is_default_baked_state(cb->graphics_state)) {
        if (cb->bound_variant != nullptr) {
            vkCmdBindPipeline(cb->buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, base->vk_pipeline);
            cb->bound_variant = nullptr;
        }
        cb->graphics_state.dirty_static_state = false;
        return true;
    }
    StaticVariantRecord* v = find_or_request_static_variant(d, base, cb->graphics_state);
    if (v == nullptr) {
        cmd_record_fail(cb, "static graphics variant allocation failed");
        return false;
    }
    const InternalPipelineState st = v->state.load(std::memory_order_acquire);
    if (st != InternalPipelineState::Ready) {
        if (st == InternalPipelineState::Queued || st == InternalPipelineState::Compiling ||
            st == InternalPipelineState::ProbingCache) {
            atomic_fetch_add(&d->stat_static_variant_pending, 1);
        }
        cmd_record_fail(cb, st == InternalPipelineState::Failed
                                ? "static graphics variant compile failed"
                                : "static graphics variant pending");
        return false;
    }
    vkCmdBindPipeline(cb->buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, v->vk_pipeline);
    cb->bound_variant = v;
    cb->graphics_state.dirty_static_state = false;
    return true;
}

void cmd_set_depth_stencil_state(CommandBuffer cmd, Handle<DepthStencilState> state) {
    auto* d = cmd->device;
    auto& desc = d->depth_stencil_pool[state].desc;

    // Update the logical shadow (baked members -> variant key on the static
    // fallback; core-dynamic members are applied on every path).
    LogicalGraphicsState& gs = cmd->graphics_state;
    apply_depth_stencil_to_shadow(desc, gs);
    gs.dirty_static_state     = true;

    // Core-dynamic (applied on every path): depth bias, stencil references,
    // stencil write/compare masks.
    vkCmdSetDepthBias(cmd->buffer, desc.depth_bias, desc.depth_bias_clamp,
                      desc.depth_bias_slope_factor);
    vkCmdSetStencilReference(cmd->buffer, VK_STENCIL_FACE_FRONT_BIT, desc.stencil_front.reference);
    vkCmdSetStencilReference(cmd->buffer, VK_STENCIL_FACE_BACK_BIT, desc.stencil_back.reference);
    vkCmdSetStencilWriteMask(cmd->buffer, VK_STENCIL_FACE_FRONT_AND_BACK, desc.stencil_write_mask);
    vkCmdSetStencilCompareMask(cmd->buffer, VK_STENCIL_FACE_FRONT_AND_BACK, desc.stencil_read_mask);

    if (d->dispatch.use_static_graphics_state) {
        // Baked members: shadow only (the variant carries them). The
        // extended-dynamic-state commands are not available.
        return;
    }
    backend_set_depth_write_enable(d, cmd->buffer, bool(desc.depth_mode & DepthFlags::Write));
    backend_set_depth_test_enable(d, cmd->buffer, bool(desc.depth_mode & DepthFlags::Read));
    backend_set_depth_compare_op(d, cmd->buffer, bridge(desc.depth_test));
    backend_set_stencil_test_enable(d, cmd->buffer, true);
    backend_set_stencil_op(d, cmd->buffer, VK_STENCIL_FACE_FRONT_BIT,
                      bridge(desc.stencil_front.fail_op), bridge(desc.stencil_front.pass_op),
                      bridge(desc.stencil_front.depth_fail_op), bridge(desc.stencil_front.test));
    backend_set_stencil_op(d, cmd->buffer, VK_STENCIL_FACE_BACK_BIT,
                      bridge(desc.stencil_back.fail_op), bridge(desc.stencil_back.pass_op),
                      bridge(desc.stencil_back.depth_fail_op), bridge(desc.stencil_back.test));
}

void cmd_set_viewport(CommandBuffer cmd, const Rect2D& rect) {
    auto* d = cmd->device;
    cmd->graphics_state.viewport = rect;
    cmd->graphics_state.dirty_core_dynamic_state = true;
    VkViewport viewport{
        .x        = static_cast<float>(rect.offset_x),
        .y        = static_cast<float>(rect.offset_y + rect.height),
        .width    = static_cast<float>(rect.width),
        .height   = -static_cast<float>(rect.height),
        .minDepth = 0,
        .maxDepth = 1.0,
    };
    if (d->dispatch.use_static_graphics_state) {
        // Core-1.0 dynamic viewport (the WithCount variant requires the
        // extended-dynamic-state extension).
        vkCmdSetViewport(cmd->buffer, 0, 1, &viewport);
        return;
    }
    backend_set_viewport_with_count(d, cmd->buffer, 1, &viewport);
}

void cmd_set_scissor_rect(CommandBuffer cmd, const Rect2D& rect) {
    auto* d = cmd->device;
    cmd->graphics_state.scissor = rect;
    cmd->graphics_state.dirty_core_dynamic_state = true;
    const VkRect2D vk_rect{
        .offset = {.x = (int32_t)rect.offset_x, .y = (int32_t)rect.offset_y},
        .extent = {.width = rect.width, .height = rect.height},
    };
    if (d->dispatch.use_static_graphics_state) {
        vkCmdSetScissor(cmd->buffer, 0, 1, &vk_rect);
        return;
    }
    backend_set_scissor_with_count(d, cmd->buffer, 1, &vk_rect);
}

void cmd_set_front_face(CommandBuffer cmd, FrontFace front) {
    auto* d = cmd->device;
    cmd->graphics_state.front_face = front;
    cmd->graphics_state.dirty_static_state = true;
    if (d->dispatch.use_static_graphics_state) { return; }
    backend_set_front_face(d, cmd->buffer,
                           front == FrontFace::CCW ? VK_FRONT_FACE_COUNTER_CLOCKWISE
                                                   : VK_FRONT_FACE_CLOCKWISE);
}

void cmd_set_cull_mode(CommandBuffer cmd, Cull cull) {
    auto* d = cmd->device;
    cmd->graphics_state.cull = cull;
    cmd->graphics_state.dirty_static_state = true;
    if (d->dispatch.use_static_graphics_state) { return; }
    backend_set_cull_mode(d, cmd->buffer, bridge(cull));
}

// --- Push data (root arguments) ------------------------------------------------------

static void push_compute_ptr(DeviceImpl* d, VkCommandBuffer buf, GpuPtr dataGpu) {
    if (dataGpu != 0) {
#if defined(IZ_VK_PROFILE_BINDLESS)
        // Root pointer via ordinary push constants (private pipeline layout).
        vkCmdPushConstants(buf, d->bindless_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(GpuPtr), &dataGpu);
#else
        VkPushDataInfoEXT push_info{
            .sType  = VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT,
            .pNext  = nullptr,
            .offset = 0,
            .data   = {.address = &dataGpu, .size = sizeof(VkDeviceAddress)},
        };
        vkCmdPushDataEXT(buf, &push_info);
#endif
    }
}

static void push_graphics_ptrs(DeviceImpl* d, VkCommandBuffer buf, GpuPtr vertexDataGpu, GpuPtr fragmentDataGpu) {
    if (vertexDataGpu != 0 || fragmentDataGpu != 0) {
#if defined(IZ_VK_PROFILE_BINDLESS)
        VkDeviceAddress addresses[] = {vertexDataGpu, fragmentDataGpu};
        vkCmdPushConstants(buf, d->bindless_pipeline_layout,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, 2 * sizeof(VkDeviceAddress), addresses);
#else
        VkDeviceAddress addresses[] = {vertexDataGpu, fragmentDataGpu};
        VkPushDataInfoEXT push_info{
            .sType  = VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT,
            .pNext  = nullptr,
            .offset = 0,
            .data   = {.address = addresses, .size = 2 * sizeof(VkDeviceAddress)},
        };
        vkCmdPushDataEXT(buf, &push_info);
#endif
    }
}

void cmd_dispatch(CommandBuffer cmd, GpuPtr dataGpu, const Dimension3D& gridDimensions) {
    retain_buffer(cmd, dataGpu);
    push_compute_ptr(cmd->device, cmd->buffer, dataGpu);
    vkCmdDispatch(cmd->buffer, gridDimensions.x, gridDimensions.y, gridDimensions.z);
}

void cmd_dispatch_indirect(CommandBuffer cmd, GpuPtr dataGpu, GpuPtr gridDimensionsGpu) {
    auto* d = cmd->device;
    retain_buffer(cmd, dataGpu);
    retain_buffer(cmd, gridDimensionsGpu);
    auto dim = buffer_and_offset_from_ptr(d, gridDimensionsGpu);
    push_compute_ptr(cmd->device, cmd->buffer, dataGpu);
    vkCmdDispatchIndirect(cmd->buffer, dim.buffer, dim.offset);
}

// --- Render pass ----------------------------------------------------------------------

// Returns (creating on first use) a native image view for the attachment
// subresource (texture, mip, layer). Views are cached per texture and
// destroyed with it; the caller must have retained the texture.
static VkImageView get_attachment_view(DeviceImpl* d, Handle<Texture> tex, uint16_t mip, uint16_t layer) {
    auto& t = d->texture_pool[handle_cast<TextureImpl>(tex)];
    mutex_lock(&d->attachment_view_lock);
    for (auto& av : t.attachment_views) {
        if (av.mip == mip && av.layer == layer) {
            mutex_unlock(&d->attachment_view_lock);
            return av.view;
        }
    }
    // Cube faces use per-face 2D views (a cube view requires 6 layers).
    const bool is_cube = (t.vk_type == VK_IMAGE_VIEW_TYPE_CUBE ||
                          t.vk_type == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY);
    const VkImageViewCreateInfo view_info{
        .sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext    = nullptr,
        .flags    = 0,
        .image    = t.vk_image,
        .viewType = is_cube ? VK_IMAGE_VIEW_TYPE_2D : t.vk_type,
        .format   = bridge(t.format),
        .components = {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                       VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY},
        .subresourceRange = {
            .aspectMask     = aspects_for_format(t.format),
            .baseMipLevel   = mip,
            .levelCount     = 1,
            .baseArrayLayer = layer,
            .layerCount     = 1,
        },
    };
    VkImageView view = VK_NULL_HANDLE;
    if (!IZ_CHK(d, vkCreateImageView(d->device, &view_info, nullptr, &view),
                "get_attachment_view failed")) {
        mutex_unlock(&d->attachment_view_lock);
        return VK_NULL_HANDLE;
    }
    t.attachment_views.push_back(TextureImpl::AttachmentView{mip, layer, view});
    mutex_unlock(&d->attachment_view_lock);
    return view;
}

void cmd_begin_render_pass(CommandBuffer cmd, const RenderPassDesc& desc) {
    auto* d = cmd->device;
    // Retain every texture named by the pass so freeing the user handle
    // cannot destroy a native image the recorded commands reference.
    for (const auto& attachment : desc.color_attachments) {
        retain_texture(cmd, attachment.texture);
        if (attachment.resolve_texture.h != 0) { retain_texture(cmd, attachment.resolve_texture); }
    }
    if (desc.depth_attachment.texture.h != 0) { retain_texture(cmd, desc.depth_attachment.texture); }
    if (desc.stencil_attachment.texture.h != 0) { retain_texture(cmd, desc.stencil_attachment.texture); }
    Arena*  arena = get_thread_local_arena(d);
    ScratchScope scope(*arena);
    Span<VkRenderingAttachmentInfo> color_attachments{};

    for (const auto& attachment : desc.color_attachments) {
        VkRenderingAttachmentInfo info{
            .sType              = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .pNext              = nullptr,
            .imageView          = get_attachment_view(d, attachment.texture, attachment.mip, attachment.layer),
            .imageLayout        = VK_IMAGE_LAYOUT_GENERAL,
            .resolveMode        = VK_RESOLVE_MODE_NONE,
            .resolveImageView   = VK_NULL_HANDLE,
            .resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .loadOp             = bridge(attachment.load_op),
            .storeOp            = bridge(attachment.store_op),
            .clearValue         = {.color = {.float32 = {attachment.clear_color.r / 255.0f,
                                                          attachment.clear_color.g / 255.0f,
                                                          attachment.clear_color.b / 255.0f,
                                                          attachment.clear_color.a / 255.0f}}},
        };
        if (attachment.resolve_texture.h != 0) {
            // MSAA color resolve (sample_count 1 target); depth/stencil
            // attachments never resolve.
            info.resolveMode        = VK_RESOLVE_MODE_AVERAGE_BIT;
            info.resolveImageView   = get_attachment_view(d, attachment.resolve_texture, attachment.mip, attachment.layer);
            info.resolveImageLayout = VK_IMAGE_LAYOUT_GENERAL;
        }
        color_attachments = concat(arena, color_attachments, info);
    }

    const bool has_depth = desc.depth_attachment.texture.h != 0;
    VkRenderingAttachmentInfo depth_attachment{};
    if (has_depth) {
        depth_attachment = VkRenderingAttachmentInfo{
            .sType              = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .pNext              = nullptr,
            .imageView          = get_attachment_view(d, desc.depth_attachment.texture, desc.depth_attachment.mip, desc.depth_attachment.layer),
            .imageLayout        = VK_IMAGE_LAYOUT_GENERAL,
            .resolveMode        = VK_RESOLVE_MODE_NONE,
            .resolveImageView   = VK_NULL_HANDLE,
            .resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .loadOp             = bridge(desc.depth_attachment.load_op),
            .storeOp            = bridge(desc.depth_attachment.store_op),
            .clearValue         = {.depthStencil = {.depth = desc.depth_attachment.clear_color.r / 255.0f,
                                                      .stencil = desc.depth_attachment.clear_color.a}},
        };
    }

    const bool has_stencil = desc.stencil_attachment.texture.h != 0;
    VkRenderingAttachmentInfo stencil_attachment{};
    if (has_stencil) {
        stencil_attachment = VkRenderingAttachmentInfo{
            .sType              = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .pNext              = nullptr,
            .imageView          = get_attachment_view(d, desc.stencil_attachment.texture, desc.stencil_attachment.mip, desc.stencil_attachment.layer),
            .imageLayout        = VK_IMAGE_LAYOUT_GENERAL,
            .resolveMode        = VK_RESOLVE_MODE_NONE,
            .resolveImageView   = VK_NULL_HANDLE,
            .resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .loadOp             = bridge(desc.stencil_attachment.load_op),
            .storeOp            = bridge(desc.stencil_attachment.store_op),
            .clearValue         = {.color = {.uint32 = {desc.stencil_attachment.clear_color.r,
                                                          desc.stencil_attachment.clear_color.g,
                                                          desc.stencil_attachment.clear_color.b,
                                                          desc.stencil_attachment.clear_color.a}}},
        };
    }

    const VkRect2D render_rect = {
        .offset = {.x = (int32_t)desc.render_area.offset_x, .y = (int32_t)desc.render_area.offset_y},
        .extent = {.width = desc.render_area.width, .height = desc.render_area.height},
    };
    if (arena->overflowed()) {
        IZ_LOG(d, LogLevel::Error, "cmd_begin_render_pass: scratch arena overflow, pass skipped");
        return;
    }
    const VkRenderingInfo rendering_info{
        .sType                = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .pNext                = nullptr,
        .flags                = 0,
        .renderArea           = render_rect,
        .layerCount           = 1,
        .viewMask             = 0,
        .colorAttachmentCount = static_cast<uint32_t>(color_attachments.size()),
        .pColorAttachments    = color_attachments.data(),
        .pDepthAttachment     = has_depth ? &depth_attachment : nullptr,
        .pStencilAttachment   = has_stencil ? &stencil_attachment : nullptr,
    };

    auto buf = cmd->buffer;
            backend_begin_rendering(d, buf, &rendering_info);

    VkViewport viewport{
        .x        = 0,
        .y        = (float)desc.render_area.height,
        .width    = (float)desc.render_area.width,
        .height   = -(float)desc.render_area.height,
        .minDepth = 0,
        .maxDepth = 1.0,
    };
    // Default dynamic state. On the static fallback the baked depth/stencil
    // members live in the pipeline variant (the shadow was reset to defaults
    // at recording start) and only the core-dynamic viewport/scissor are
    // issued, with plain core-1.0 commands — never call the unavailable EDS
    // function pointers.
    if (d->dispatch.use_static_graphics_state) {
        vkCmdSetViewport(buf, 0, 1, &viewport);
        vkCmdSetScissor(buf, 0, 1, &render_rect);
    } else {
        backend_set_depth_write_enable(d, buf, false);
        backend_set_depth_test_enable(d, buf, false);
        backend_set_depth_compare_op(d, buf, VK_COMPARE_OP_ALWAYS);
        backend_set_depth_bounds_test_enable(d, buf, false);
        backend_set_stencil_test_enable(d, buf, false);
        vkCmdSetStencilOp(buf, VK_STENCIL_FACE_FRONT_AND_BACK,
                          VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP,
                          VK_COMPARE_OP_ALWAYS);
        backend_set_viewport_with_count(d, buf, 1, &viewport);
        backend_set_scissor_with_count(d, buf, 1, &render_rect);
    }
    cmd_set_front_face(cmd, FrontFace::CCW);
    cmd_set_cull_mode(cmd, Cull::None);
}

void cmd_end_render_pass(CommandBuffer cmd) {
    auto* d = cmd->device;
    backend_end_rendering(d, cmd->buffer);
}

void cmd_draw(CommandBuffer cmd, GpuPtr vertexDataGpu, GpuPtr fragmentDataGpu,
              uint32_t vertexCount, uint32_t instanceCount) {
    if (!prepare_static_graphics(cmd)) { return; }
    retain_buffer(cmd, vertexDataGpu);
    retain_buffer(cmd, fragmentDataGpu);
    push_graphics_ptrs(cmd->device, cmd->buffer, vertexDataGpu, fragmentDataGpu);
    vkCmdDraw(cmd->buffer, vertexCount, instanceCount, 0, 0);
}

void cmd_draw_indexed_instanced(CommandBuffer cmd, const DrawIndexedInstancedInfo& args) {
    auto* d = cmd->device;
    if (!prepare_static_graphics(cmd)) { return; }
    retain_buffer(cmd, args.vertexDataGpu);
    retain_buffer(cmd, args.fragmentDataGpu);
    retain_buffer(cmd, args.indicesGpu);
    push_graphics_ptrs(cmd->device, cmd->buffer, args.vertexDataGpu, args.fragmentDataGpu);
    if (args.indicesGpu != cmd->current_idx_buffer) {
        const auto indices = buffer_and_offset_from_ptr(d, args.indicesGpu);
#if defined(IZ_VK_PROFILE_BINDLESS)
        // 1.3/no maintenance5: legacy index buffer bind (64-bit offset).
        vkCmdBindIndexBuffer(cmd->buffer, indices.buffer, indices.offset, bridge(args.type));
#else
        vkCmdBindIndexBuffer2(cmd->buffer, indices.buffer, indices.offset,
                              VK_WHOLE_SIZE, bridge(args.type));
#endif
        cmd->current_idx_buffer = args.indicesGpu;
    }
    vkCmdDrawIndexed(cmd->buffer, args.indexCount, args.instanceCount, 0, 0, 0);
}

void cmd_draw_indexed_instanced_indirect(CommandBuffer cmd, const DrawIndexedIndirectInfo& args) {
    auto* d = cmd->device;
    if (!prepare_static_graphics(cmd)) { return; }
    retain_buffer(cmd, args.vertexDataGpu);
    retain_buffer(cmd, args.fragmentDataGpu);
    retain_buffer(cmd, args.indicesGpu);
    retain_buffer(cmd, args.argsGpu);
    push_graphics_ptrs(cmd->device, cmd->buffer, args.vertexDataGpu, args.fragmentDataGpu);
    if (args.indicesGpu != cmd->current_idx_buffer) {
        const auto indices = buffer_and_offset_from_ptr(d, args.indicesGpu);
#if defined(IZ_VK_PROFILE_BINDLESS)
        // 1.3/no maintenance5: legacy index buffer bind (64-bit offset).
        vkCmdBindIndexBuffer(cmd->buffer, indices.buffer, indices.offset, bridge(args.type));
#else
        vkCmdBindIndexBuffer2(cmd->buffer, indices.buffer, indices.offset,
                              VK_WHOLE_SIZE, bridge(args.type));
#endif
        cmd->current_idx_buffer = args.indicesGpu;
    }
    const auto gpu_args = buffer_and_offset_from_ptr(d, args.argsGpu);
    vkCmdDrawIndexedIndirect(cmd->buffer, gpu_args.buffer, gpu_args.offset, 1,
                             sizeof(VkDrawIndexedIndirectCommand));
}

void cmd_draw_indexed_instanced_indirect_multi(CommandBuffer cmd, const MultiDrawIndirectInfo& args) {
    auto* d = cmd->device;
    if (!prepare_static_graphics(cmd)) { return; }
    retain_buffer(cmd, args.vertexDataGpu);
    retain_buffer(cmd, args.pixelDataGpu);
    retain_buffer(cmd, args.indicesGpu);
    retain_buffer(cmd, args.argsGpu);
    retain_buffer(cmd, args.drawCountGpu);
    push_graphics_ptrs(cmd->device, cmd->buffer, args.vertexDataGpu, args.pixelDataGpu);
    if (args.indicesGpu != cmd->current_idx_buffer) {
        const auto indices = buffer_and_offset_from_ptr(d, args.indicesGpu);
#if defined(IZ_VK_PROFILE_BINDLESS)
        // 1.3/no maintenance5: legacy index buffer bind (64-bit offset).
        vkCmdBindIndexBuffer(cmd->buffer, indices.buffer, indices.offset, bridge(args.type));
#else
        vkCmdBindIndexBuffer2(cmd->buffer, indices.buffer, indices.offset,
                              VK_WHOLE_SIZE, bridge(args.type));
#endif
        cmd->current_idx_buffer = args.indicesGpu;
    }
    const auto gpu_args = buffer_and_offset_from_ptr(d, args.argsGpu);
    const auto count    = buffer_and_offset_from_ptr(d, args.drawCountGpu);
    vkCmdDrawIndexedIndirectCount(cmd->buffer, gpu_args.buffer, gpu_args.offset,
                                  count.buffer, count.offset, args.maxDraws,
                                  sizeof(VkDrawIndexedIndirectCommand));
}

// --- Surface texture commands ----------------------------------------------------------

void cmd_wait_for_surface_texture(CommandBuffer cmd) {
    cmd->wait_for_surface_texture = true;
    auto* d = cmd->device;
    const auto current_image = d->surface.swapchain_images[d->surface.current_image_idx];
    const VkImageMemoryBarrier2 image_barrier{
        .sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .pNext         = nullptr,
        .srcStageMask  = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
        .srcAccessMask = 0,
        .dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        .dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
        .oldLayout     = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout     = VK_IMAGE_LAYOUT_GENERAL,
        .image         = d->texture_pool[handle_cast<TextureImpl>(current_image)].vk_image,
        .subresourceRange = {
            .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel   = 0,
            .levelCount     = VK_REMAINING_MIP_LEVELS,
            .baseArrayLayer = 0,
            .layerCount     = VK_REMAINING_ARRAY_LAYERS,
        },
    };
    const VkDependencyInfo info{
        .sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .pNext                    = nullptr,
        .dependencyFlags          = 0,
        .memoryBarrierCount       = 0,
        .pMemoryBarriers          = nullptr,
        .bufferMemoryBarrierCount = 0,
        .pBufferMemoryBarriers    = nullptr,
        .imageMemoryBarrierCount  = 1,
        .pImageMemoryBarriers     = &image_barrier,
    };
            backend_pipeline_barrier2(d, cmd->buffer, &info);
}

void cmd_signal_surface_texture(CommandBuffer cmd) {
    cmd->signal_surface_texture = true;
    auto* d = cmd->device;
    const auto current_image = d->surface.swapchain_images[d->surface.current_image_idx];
    const VkImageMemoryBarrier2 image_barrier{
        .sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .pNext         = nullptr,
        .srcStageMask  = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT |
                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                        VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        .srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_SHADER_WRITE_BIT |
                        VK_ACCESS_2_TRANSFER_READ_BIT,
        .dstStageMask  = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
        .dstAccessMask = 0,
        .oldLayout     = VK_IMAGE_LAYOUT_GENERAL,
        .newLayout     = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        .image         = d->texture_pool[handle_cast<TextureImpl>(current_image)].vk_image,
        .subresourceRange = {
            .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel   = 0,
            .levelCount     = VK_REMAINING_MIP_LEVELS,
            .baseArrayLayer = 0,
            .layerCount     = VK_REMAINING_ARRAY_LAYERS,
        },
    };
    const VkDependencyInfo info{
        .sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .pNext                    = nullptr,
        .dependencyFlags          = 0,
        .memoryBarrierCount       = 0,
        .pMemoryBarriers          = nullptr,
        .bufferMemoryBarrierCount = 0,
        .pBufferMemoryBarriers    = nullptr,
        .imageMemoryBarrierCount  = 1,
        .pImageMemoryBarriers     = &image_barrier,
    };
            backend_pipeline_barrier2(d, cmd->buffer, &info);
}

// --- Debug groups ----------------------------------------------------------------------

void cmd_push_debug_group(CommandBuffer cmd, Span<const char> label) {
    auto* d = cmd->device;
    if (d->has_debug_markers) {
        Arena* a = get_thread_local_arena(d);
        const VkDebugUtilsLabelEXT info{
            .sType       = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT,
            .pNext       = nullptr,
            .pLabelName  = make_null_terminated(a, label),
            .color       = {0, 0, 0, 0},
        };
        vkCmdBeginDebugUtilsLabelEXT(cmd->buffer, &info);
    }
}

void cmd_pop_debug_group(CommandBuffer cmd) {
    auto* d = cmd->device;
    if (d->has_debug_markers) {
        vkCmdEndDebugUtilsLabelEXT(cmd->buffer);
    }
}

}  // namespace gpu
