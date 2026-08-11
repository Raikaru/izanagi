// commands.cpp — all cmd_* recording, command pool management, queue submission.

#include <algorithm>
#include <new>

#include "internal.h"

namespace gpu {

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

static void reset_command_pool(DeviceImpl* d, CommandPool* pool) {
    // Release references retained by recorded-but-never-submitted command
    // buffers (their commands are discarded; the GPU never executed them,
    // and pool reuse is gated on queue timeline completion).
    for (uint32_t i = 0; i < pool->command_buffers.size(); ++i) {
        auto& cb = pool->command_buffers[i];
        if (cb.device == nullptr) { continue; }
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
        // Bind descriptor heaps once at recording start
        cmd_bind_descriptor_heaps(d, buffer->buffer);
    }
    return buffer;
}

void cmd_finalize(CommandBuffer cmd) {
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
        vkCmdPipelineBarrier2(internal_cmd->buffer, &dependency_info);
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

    const VkResult result = vkQueueSubmit2(q->queue, 1, &submit_info, VK_NULL_HANDLE);
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

int64_t debug_pool_resets(DeviceImpl* d) {
    return atomic_load(&d->stat_pool_resets);
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
    vkCmdCopyBuffer2(cmd->buffer, &copy_info);
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
    vkCmdCopyBufferToImage2(cmd->buffer, &copy_info);
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
    vkCmdCopyImageToBuffer2(cmd->buffer, &copy_info);
}

void cmd_barrier(CommandBuffer cmd, StageFlags before, StageFlags after) {
    auto* d = cmd->device;
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
    vkCmdPipelineBarrier2(cmd->buffer, &info);
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
        vkCmdBlitImage2(cmd->buffer, &info);

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
    // Retain the native pipeline for the command buffer's lifetime (one
    // reference per distinct pipeline; repeated binds reuse it).
    for (PipelineRecord* r : cmd->retained_pipelines) {
        if (r == rec) { return true; }
    }
    rec->refs.fetch_add(1, std::memory_order_relaxed);
    cmd->retained_pipelines.push_back(rec);
    return true;
}

void cmd_set_depth_stencil_state(CommandBuffer cmd, Handle<DepthStencilState> state) {
    auto* d = cmd->device;
    auto& desc = d->depth_stencil_pool[state].desc;

    vkCmdSetDepthWriteEnable(cmd->buffer, bool(desc.depth_mode & DepthFlags::Write));
    vkCmdSetDepthTestEnable(cmd->buffer, bool(desc.depth_mode & DepthFlags::Read));
    vkCmdSetDepthCompareOp(cmd->buffer, bridge(desc.depth_test));
    vkCmdSetDepthBias(cmd->buffer, desc.depth_bias, desc.depth_bias_clamp,
                      desc.depth_bias_slope_factor);

    vkCmdSetStencilTestEnable(cmd->buffer, true);
    vkCmdSetStencilOp(cmd->buffer, VK_STENCIL_FACE_FRONT_BIT,
                      bridge(desc.stencil_front.fail_op), bridge(desc.stencil_front.pass_op),
                      bridge(desc.stencil_front.depth_fail_op), bridge(desc.stencil_front.test));
    vkCmdSetStencilReference(cmd->buffer, VK_STENCIL_FACE_FRONT_BIT, desc.stencil_front.reference);
    vkCmdSetStencilOp(cmd->buffer, VK_STENCIL_FACE_BACK_BIT,
                      bridge(desc.stencil_back.fail_op), bridge(desc.stencil_back.pass_op),
                      bridge(desc.stencil_back.depth_fail_op), bridge(desc.stencil_back.test));
    vkCmdSetStencilReference(cmd->buffer, VK_STENCIL_FACE_BACK_BIT, desc.stencil_back.reference);
    vkCmdSetStencilWriteMask(cmd->buffer, VK_STENCIL_FACE_FRONT_AND_BACK, desc.stencil_write_mask);
    vkCmdSetStencilCompareMask(cmd->buffer, VK_STENCIL_FACE_FRONT_AND_BACK, desc.stencil_read_mask);
}

void cmd_set_viewport(CommandBuffer cmd, const Rect2D& rect) {
    VkViewport viewport{
        .x        = static_cast<float>(rect.offset_x),
        .y        = static_cast<float>(rect.offset_y + rect.height),
        .width    = static_cast<float>(rect.width),
        .height   = -static_cast<float>(rect.height),
        .minDepth = 0,
        .maxDepth = 1.0,
    };
    vkCmdSetViewportWithCount(cmd->buffer, 1, &viewport);
}

void cmd_set_scissor_rect(CommandBuffer cmd, const Rect2D& rect) {
    const VkRect2D vk_rect{
        .offset = {.x = (int32_t)rect.offset_x, .y = (int32_t)rect.offset_y},
        .extent = {.width = rect.width, .height = rect.height},
    };
    vkCmdSetScissorWithCount(cmd->buffer, 1, &vk_rect);
}

void cmd_set_front_face(CommandBuffer cmd, FrontFace front) {
    vkCmdSetFrontFace(cmd->buffer,
                      front == FrontFace::CCW ? VK_FRONT_FACE_COUNTER_CLOCKWISE
                                              : VK_FRONT_FACE_CLOCKWISE);
}

void cmd_set_cull_mode(CommandBuffer cmd, Cull cull) {
    vkCmdSetCullMode(cmd->buffer, bridge(cull));
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
    vkCmdBeginRendering(buf, &rendering_info);

    // Default dynamic state
    vkCmdSetDepthWriteEnable(buf, false);
    vkCmdSetDepthTestEnable(buf, false);
    vkCmdSetDepthCompareOp(buf, VK_COMPARE_OP_ALWAYS);
    vkCmdSetDepthBoundsTestEnable(buf, false);
    vkCmdSetStencilTestEnable(buf, false);
    vkCmdSetStencilOp(buf, VK_STENCIL_FACE_FRONT_AND_BACK,
                      VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP,
                      VK_COMPARE_OP_ALWAYS);

    VkViewport viewport{
        .x        = 0,
        .y        = (float)desc.render_area.height,
        .width    = (float)desc.render_area.width,
        .height   = -(float)desc.render_area.height,
        .minDepth = 0,
        .maxDepth = 1.0,
    };
    vkCmdSetViewportWithCount(buf, 1, &viewport);
    vkCmdSetScissorWithCount(buf, 1, &render_rect);
    cmd_set_front_face(cmd, FrontFace::CCW);
    cmd_set_cull_mode(cmd, Cull::None);
}

void cmd_end_render_pass(CommandBuffer cmd) {
    vkCmdEndRendering(cmd->buffer);
}

void cmd_draw(CommandBuffer cmd, GpuPtr vertexDataGpu, GpuPtr fragmentDataGpu,
              uint32_t vertexCount, uint32_t instanceCount) {
    retain_buffer(cmd, vertexDataGpu);
    retain_buffer(cmd, fragmentDataGpu);
    push_graphics_ptrs(cmd->device, cmd->buffer, vertexDataGpu, fragmentDataGpu);
    vkCmdDraw(cmd->buffer, vertexCount, instanceCount, 0, 0);
}

void cmd_draw_indexed_instanced(CommandBuffer cmd, const DrawIndexedInstancedInfo& args) {
    auto* d = cmd->device;
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
    vkCmdPipelineBarrier2(cmd->buffer, &info);
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
    vkCmdPipelineBarrier2(cmd->buffer, &info);
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
