// commands.cpp — all cmd_* recording, command pool management, queue submission.

#include <algorithm>
#include <new>

#include "internal.h"

namespace gpu {

// --- Descriptor heap binding ---------------------------------------------------------

void cmd_bind_descriptor_heaps(DeviceImpl* d, VkCommandBuffer cmd) {
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
}

// --- Command pool management -----------------------------------------------------------

static void reset_command_pool(VkDevice device, CommandPool* pool) {
    // Release pipeline references retained by recorded-but-never-submitted
    // command buffers (their commands are discarded; the GPU never executed
    // them, and the pool reuse is frame-paced past any prior completion).
    for (uint32_t i = 0; i < pool->command_buffers.size(); ++i) {
        auto& cb = pool->command_buffers[i];
        if (cb.device == nullptr) { continue; }
        for (PipelineRecord* rec : cb.retained_pipelines) {
            release_pipeline_ref(cb.device, rec);
        }
        cb.retained_pipelines.clear();
    }
    vkResetCommandPool(device, pool->command_pool, 0);
    pool->buffer_free_idx = 0;
}

CommandPool* get_command_pool(QueueImpl* queue, uint64_t frame_idx) {
    CommandSuperpool& superpool       = queue->command_superpool;
    CommandPool*      pool            = nullptr;
    int64_t           available_pools = atomic_load(&superpool.available_pools);
    bool              index_good      = false;
    uint64_t          idx;
    while (!index_good && available_pools != 0) {
        idx                    = count_trailing_zeros(available_pools);
        const int64_t  mask    = ~(1ll << idx);
        const int64_t  desired = static_cast<int64_t>(available_pools & mask);
        index_good = atomic_compare_exchange(&superpool.available_pools, &available_pools, desired);
    };

    if (index_good) {
        pool = &superpool.pools[CommandSuperpool::kPoolsPerGroup * idx +
                                (frame_idx % CommandSuperpool::kPoolsPerGroup)];

        if (pool->command_pool == VK_NULL_HANDLE) {
            VkCommandPoolCreateInfo pool_info{
                .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                .pNext            = nullptr,
                .flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
                .queueFamilyIndex = queue->queue_family,
            };
            VkCommandPool command_pool = VK_NULL_HANDLE;
            if (!IZ_CHK(queue->device,
                        vkCreateCommandPool(queue->device->device, &pool_info, nullptr, &command_pool),
                        "get_command_pool failed")) {
                return nullptr;
            }
            *pool = CommandPool{
                .command_pool    = command_pool,
                .command_buffers = SegmentArray<CommandBufferImpl>(queue->device->allocator),
                .buffer_free_idx = 0,
                .frame_idx       = frame_idx,
            };
        } else if (pool->frame_idx != frame_idx) {
            reset_command_pool(queue->device->device, pool);
            pool->frame_idx = frame_idx;
        }
    } else {
        IZ_LOG(queue->device, LogLevel::Error, "Too many command buffers in flight");
    }
    return pool;
}

void release_command_pool(QueueImpl* q, CommandPool* pool) {
    auto&         superpool = q->command_superpool;
    const int64_t idx       = (pool - superpool.pools) / CommandSuperpool::kPoolsPerGroup;
    int64_t previous = atomic_load(&superpool.available_pools);
    int64_t desired  = previous | (1ll << idx);
    while (!atomic_compare_exchange(&superpool.available_pools, &previous, desired)) {
        desired = previous | (1ll << idx);
    }
}

CommandBuffer get_command_buffer(QueueImpl* q, CommandPool* pool) {
    auto* d = q->device;
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
    }
    CommandBufferImpl* result        = &pool->command_buffers[pool->buffer_free_idx];
    result->wait_for_surface_texture = false;
    result->signal_surface_texture   = false;
    pool->buffer_free_idx++;
    return result;
}

CommandBuffer queue_start_command_recording(Queue q) {
    auto* d = q->device;
    CommandPool* pool = get_command_pool(q, d->surface.frame_idx);
    if (pool == nullptr) { return nullptr; }

    CommandBuffer buffer = get_command_buffer(q, pool);
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
    auto* q = cmd->queue;
    IZ_CHK(d, vkEndCommandBuffer(cmd->buffer), "cmd_finalize failed");
    release_command_pool(q, cmd->pool);
}

// --- Queue submission -----------------------------------------------------------------

void queue_submit(Queue                     q,
                  Span<const CommandBuffer> command_buffers,
                  Span<const SemaphoreInfo> wait_semaphores,
                  Span<const SemaphoreInfo> signal_semaphores) {
    auto* d = q->device;
    Arena*  arena = get_thread_local_arena(d);

    Span<VkSemaphoreSubmitInfo>     wait_info{};
    Span<VkCommandBufferSubmitInfo> command_info{};
    Span<VkSemaphoreSubmitInfo>     signal_info{};

    // Drain uninitialized textures — one-time UNDEFINED->GENERAL transition
    mutex_lock(&d->texture_init_lock);
    Vector<Handle<Texture>> texture_list(d->allocator);
    swap(d->uninitialized_textures, texture_list);
    mutex_unlock(&d->texture_init_lock);

    if (texture_list.size() > 0) {
        CommandBuffer               cmd = queue_start_command_recording(q);
        Span<VkImageMemoryBarrier2> image_barriers{};
        for (auto tex : texture_list) {
            auto& image    = d->texture_pool[handle_cast<TextureImpl>(tex)];
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
                    .image         = image.vk_image,
                    .subresourceRange =
                        VkImageSubresourceRange{
                            .aspectMask     = aspects_for_format(image.format),
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
        vkCmdPipelineBarrier2(cmd->buffer, &dependency_info);
        cmd_finalize(cmd);
        command_info = concat(arena, command_info,
                              VkCommandBufferSubmitInfo{
                                  .sType         = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
                                  .pNext         = nullptr,
                                  .commandBuffer = cmd->buffer,
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

    // Advance queue timeline
    signal_info = concat(arena, signal_info,
                         VkSemaphoreSubmitInfo{
                             .sType       = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                             .pNext       = nullptr,
                             .semaphore   = d->semaphore_pool[handle_cast<SemaphoreImpl>(q->timeline)].vk_semaphore,
                             .value       = ++(q->timeline_value),
                             .stageMask   = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
                             .deviceIndex = 0,
                         });

    // Retain pipelines bound in these command buffers until this submit's
    // timeline value completes (released by queue_process_events).
    for (uint32_t i = 0; i < command_buffers.size(); ++i) {
        auto& retained = command_buffers[i]->retained_pipelines;
        if (retained.is_empty()) { continue; }
        MemoryBlock blk = d->allocator.alloc(sizeof(PipelineRefBatch));
        if (blk.ptr == nullptr) { continue; }   // refs stay in the cb; released at pool reset
        auto* batch = ::new (blk.ptr) PipelineRefBatch{
            .device         = d,
            .timeline_value = q->timeline_value,
            .records        = Vector<PipelineRecord*>(d->allocator),
        };
        for (PipelineRecord* rec : retained) { batch->records.push_back(rec); }
        retained.clear();
        q->in_flight_batches.push_back(batch);
    }

    VkSubmitInfo2 submit_info{
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
    vkQueueSubmit2(q->queue, 1, &submit_info, VK_NULL_HANDLE);
}

// Releases the pipeline references retained for a completed submission.
void release_inflight_batch(PipelineRefBatch* batch) {
    auto* d = batch->device;
    for (PipelineRecord* rec : batch->records) { release_pipeline_ref(d, rec); }
    batch->records.clear();
    batch->~PipelineRefBatch();
    d->allocator.free({.ptr = batch, .len = sizeof(PipelineRefBatch)});
}

// --- Commands -------------------------------------------------------------------------

void cmd_memcpy(CommandBuffer cmd, GpuPtr destGpu, GpuPtr srcGpu, size_t size) {
    auto* d = cmd->device;
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

static void push_compute_ptr(VkCommandBuffer buf, GpuPtr dataGpu) {
    if (dataGpu != 0) {
        VkPushDataInfoEXT push_info{
            .sType  = VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT,
            .pNext  = nullptr,
            .offset = 0,
            .data   = {.address = &dataGpu, .size = sizeof(VkDeviceAddress)},
        };
        vkCmdPushDataEXT(buf, &push_info);
    }
}

static void push_graphics_ptrs(VkCommandBuffer buf, GpuPtr vertexDataGpu, GpuPtr fragmentDataGpu) {
    if (vertexDataGpu != 0 || fragmentDataGpu != 0) {
        VkDeviceAddress addresses[] = {vertexDataGpu, fragmentDataGpu};
        VkPushDataInfoEXT push_info{
            .sType  = VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT,
            .pNext  = nullptr,
            .offset = 0,
            .data   = {.address = addresses, .size = 2 * sizeof(VkDeviceAddress)},
        };
        vkCmdPushDataEXT(buf, &push_info);
    }
}

void cmd_dispatch(CommandBuffer cmd, GpuPtr dataGpu, const Dimension3D& gridDimensions) {
    push_compute_ptr(cmd->buffer, dataGpu);
    vkCmdDispatch(cmd->buffer, gridDimensions.x, gridDimensions.y, gridDimensions.z);
}

void cmd_dispatch_indirect(CommandBuffer cmd, GpuPtr dataGpu, GpuPtr gridDimensionsGpu) {
    auto* d = cmd->device;
    auto dim = buffer_and_offset_from_ptr(d, gridDimensionsGpu);
    push_compute_ptr(cmd->buffer, dataGpu);
    vkCmdDispatchIndirect(cmd->buffer, dim.buffer, dim.offset);
}

// --- Render pass ----------------------------------------------------------------------

void cmd_begin_render_pass(CommandBuffer cmd, const RenderPassDesc& desc) {
    auto* d = cmd->device;
    Arena*  arena = get_thread_local_arena(d);
    Span<VkRenderingAttachmentInfo> color_attachments{};

    for (const auto& attachment : desc.color_attachments) {
        VkRenderingAttachmentInfo info{
            .sType              = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .pNext              = nullptr,
            .imageView          = d->texture_pool[handle_cast<TextureImpl>(attachment.texture)].default_image_view,
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
            info.resolveImageView   = d->texture_pool[handle_cast<TextureImpl>(attachment.resolve_texture)].default_image_view;
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
            .imageView          = d->texture_pool[handle_cast<TextureImpl>(desc.depth_attachment.texture)].default_image_view,
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
            .imageView          = d->texture_pool[handle_cast<TextureImpl>(desc.stencil_attachment.texture)].default_image_view,
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
    push_graphics_ptrs(cmd->buffer, vertexDataGpu, fragmentDataGpu);
    vkCmdDraw(cmd->buffer, vertexCount, instanceCount, 0, 0);
}

void cmd_draw_indexed_instanced(CommandBuffer cmd, const DrawIndexedInstancedInfo& args) {
    auto* d = cmd->device;
    push_graphics_ptrs(cmd->buffer, args.vertexDataGpu, args.fragmentDataGpu);
    if (args.indicesGpu != cmd->current_idx_buffer) {
        const auto indices = buffer_and_offset_from_ptr(d, args.indicesGpu);
        vkCmdBindIndexBuffer2(cmd->buffer, indices.buffer, indices.offset,
                              VK_WHOLE_SIZE, bridge(args.type));
        cmd->current_idx_buffer = args.indicesGpu;
    }
    vkCmdDrawIndexed(cmd->buffer, args.indexCount, args.instanceCount, 0, 0, 0);
}

void cmd_draw_indexed_instanced_indirect(CommandBuffer cmd, const DrawIndexedIndirectInfo& args) {
    auto* d = cmd->device;
    push_graphics_ptrs(cmd->buffer, args.vertexDataGpu, args.fragmentDataGpu);
    if (args.indicesGpu != cmd->current_idx_buffer) {
        const auto indices = buffer_and_offset_from_ptr(d, args.indicesGpu);
        vkCmdBindIndexBuffer2(cmd->buffer, indices.buffer, indices.offset,
                              VK_WHOLE_SIZE, bridge(args.type));
        cmd->current_idx_buffer = args.indicesGpu;
    }
    const auto gpu_args = buffer_and_offset_from_ptr(d, args.argsGpu);
    vkCmdDrawIndexedIndirect(cmd->buffer, gpu_args.buffer, gpu_args.offset, 1,
                             sizeof(VkDrawIndexedIndirectCommand));
}

void cmd_draw_indexed_instanced_indirect_multi(CommandBuffer cmd, const MultiDrawIndirectInfo& args) {
    auto* d = cmd->device;
    push_graphics_ptrs(cmd->buffer, args.vertexDataGpu, args.pixelDataGpu);
    if (args.indicesGpu != cmd->current_idx_buffer) {
        const auto indices = buffer_and_offset_from_ptr(d, args.indicesGpu);
        vkCmdBindIndexBuffer2(cmd->buffer, indices.buffer, indices.offset,
                              VK_WHOLE_SIZE, bridge(args.type));
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
