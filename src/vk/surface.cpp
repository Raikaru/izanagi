// surface.cpp — swapchain creation, acquire/present, frame pacing.

#include <algorithm>

#include "internal.h"

namespace gpu {
static bool has_present_mode(Span<const PresentMode> modes, PresentMode wanted) {
    for (size_t i = 0; i < modes.size(); ++i) {
        if (modes[i] == wanted) { return true; }
    }
    return false;
}

PresentMode choose_present_mode(Span<const PresentMode> supported, bool vsync) {
    if (vsync) {
        return has_present_mode(supported, PresentMode::Fifo)
                   ? PresentMode::Fifo
                   : (supported.size() > 0 ? supported[0] : PresentMode::Fifo);
    }
    constexpr PresentMode preference[] = {
        PresentMode::Immediate,
        PresentMode::Mailbox,
        PresentMode::FifoRelaxed,
        PresentMode::Fifo,
    };
    for (PresentMode mode : preference) {
        if (has_present_mode(supported, mode)) { return mode; }
    }
    return PresentMode::Fifo;
}


SurfaceCapabilities get_surface_capabilities(Device dev) {
    auto* d = reinterpret_cast<DeviceImpl*>(dev);
    auto& s = d->surface;

    if (s.surface == VK_NULL_HANDLE) {
        return SurfaceCapabilities{};
    }

    // Formats
    uint32_t format_count = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(d->physical_device, s.surface, &format_count, nullptr);
    Arena* arena = get_thread_local_arena(d);
    if (arena == nullptr) { return SurfaceCapabilities{}; }
    auto vk_formats = reinterpret_cast<VkSurfaceFormatKHR*>(
        arena->alloc(sizeof(VkSurfaceFormatKHR) * format_count));
    vkGetPhysicalDeviceSurfaceFormatsKHR(d->physical_device, s.surface, &format_count, vk_formats);

    s.num_supported_formats = 0;
    for (uint32_t i = 0; i < format_count; ++i) {
        Format fmt = bridge(vk_formats[i].format);
        if (fmt < Format::ValidCount) {
            s.supported_formats[s.num_supported_formats++] = fmt;
        }
    }

    // Present modes
    uint32_t present_mode_count = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(d->physical_device, s.surface, &present_mode_count, nullptr);
    auto vk_modes = reinterpret_cast<VkPresentModeKHR*>(
        arena->alloc(sizeof(VkPresentModeKHR) * present_mode_count));
    vkGetPhysicalDeviceSurfacePresentModesKHR(d->physical_device, s.surface, &present_mode_count, vk_modes);

    s.num_supported_present_modes = 0;
    for (uint32_t i = 0; i < present_mode_count; ++i) {
        PresentMode mode = bridge(vk_modes[i]);
        if (mode < PresentMode::ValidCount) {
            s.supported_present_modes[s.num_supported_present_modes++] = mode;
        }
    }

    // Capabilities
    VkSurfaceCapabilitiesKHR vk_caps;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(d->physical_device, s.surface, &vk_caps);

    return SurfaceCapabilities{
        .usages        = bridge_usage_flags(vk_caps.supportedUsageFlags),
        .formats       = Span<const Format>(s.supported_formats, s.num_supported_formats),
        .present_modes = Span<const PresentMode>(s.supported_present_modes, s.num_supported_present_modes),
    };
}

bool configure_surface(Device dev, const SurfaceConfiguration& config) {
    auto* d = reinterpret_cast<DeviceImpl*>(dev);
    auto& s = d->surface;

    if (s.surface == VK_NULL_HANDLE) {
        IZ_LOG(d, LogLevel::Error, "No surface — device was created without a window handle");
        return false;
    }
    if (config.frame_latency == 0 || config.frame_latency > kMaxFramesInFlight) {
        log_fmt(d, LogLevel::Error, __LINE__, __FILE__,
                "configure_surface: frame_latency must be in [1, %u]",
                kMaxFramesInFlight);
        return false;
    }

    const SurfaceCapabilities advertised = get_surface_capabilities(dev);
    PresentMode selected_present_mode = config.present_mode;
    if (!has_present_mode(advertised.present_modes, selected_present_mode)) {
        selected_present_mode = PresentMode::Fifo;
        if (!has_present_mode(advertised.present_modes, selected_present_mode)) {
            IZ_LOG(d, LogLevel::Error,
                   "configure_surface: requested present mode unsupported and FIFO unavailable");
            return false;
        }
        IZ_LOG(d, LogLevel::Warning,
               "configure_surface: requested present mode unsupported; falling back to FIFO");
    }


    // 1. Retire old swapchain use before touching anything.
    vkDeviceWaitIdle(d->device);

    // 2. Capabilities; clamp the requested extent.
    VkSurfaceCapabilitiesKHR vk_caps;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(d->physical_device, s.surface, &vk_caps);

    uint32_t image_count = std::max(vk_caps.minImageCount, config.frame_latency + 1);
    if (vk_caps.maxImageCount > 0 && image_count > vk_caps.maxImageCount) {
        image_count = vk_caps.maxImageCount;
    }

    VkExtent2D extent{.width = config.width, .height = config.height};
    if (vk_caps.currentExtent.width != 0xFFFFFFFFu) {
        extent = vk_caps.currentExtent;
    } else {
        extent.width  = std::clamp(extent.width, vk_caps.minImageExtent.width, vk_caps.maxImageExtent.width);
        extent.height = std::clamp(extent.height, vk_caps.minImageExtent.height, vk_caps.maxImageExtent.height);
    }

    // 3. Create the candidate swapchain (handoff from the old one).
    VkSwapchainKHR candidate = VK_NULL_HANDLE;
    const VkSwapchainCreateInfoKHR swapchain_info{
        .sType                 = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .pNext                 = nullptr,
        .flags                 = 0,
        .surface               = s.surface,
        .minImageCount         = image_count,
        .imageFormat           = bridge(config.format),
        .imageColorSpace       = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
        .imageExtent           = extent,
        .imageArrayLayers      = 1,
        .imageUsage            = bridge_usage_flags(config.usages),
        .imageSharingMode      = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices   = nullptr,
        .preTransform          = vk_caps.currentTransform,
        .compositeAlpha        = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        .presentMode           = bridge(selected_present_mode),
        .clipped               = true,
        .oldSwapchain          = s.swapchain,
    };
    if (!IZ_CHK(d, vkCreateSwapchainKHR(d->device, &swapchain_info, nullptr, &candidate),
                "configure_surface: vkCreateSwapchainKHR failed")) {
        return false;
    }

    // 4. Build candidate state (semaphores, views, texture records) fully
    //    before touching the installed state. Any failure rolls back.
    const VkSemaphoreCreateInfo sem_info{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
    };
    VkSemaphore candidate_acquire[kMaxFramesInFlight] = {};
    VkSemaphore candidate_present[Surface::kMaxSwapchainImages] = {};
    Handle<Texture> candidate_textures[Surface::kMaxSwapchainImages] = {};
    uint32_t created_textures = 0;
    VkImage vk_images[Surface::kMaxSwapchainImages];
    Handle<Semaphore> candidate_frame = create_semaphore_internal(d, 0);
    if (candidate_frame.h == 0) { goto fail_candidate; }

    // 5. Query candidate images.
    image_count = 0;
    if (!IZ_CHK(d, vkGetSwapchainImagesKHR(d->device, candidate, &image_count, nullptr),
                "configure_surface: vkGetSwapchainImagesKHR failed")) {
        goto fail_candidate;
    }
    if (image_count == 0 || image_count > Surface::kMaxSwapchainImages) {
        IZ_LOG(d, LogLevel::Error, "Swapchain image count out of range");
        goto fail_candidate;
    }
    if (!IZ_CHK(d, vkGetSwapchainImagesKHR(d->device, candidate, &image_count, vk_images),
                "configure_surface: vkGetSwapchainImagesKHR failed")) {
        goto fail_candidate;
    }

    for (uint32_t i = 0; i < kMaxFramesInFlight; ++i) {
        if (!IZ_CHK(d, vkCreateSemaphore(d->device, &sem_info, nullptr, &candidate_acquire[i]),
                    "configure_surface: failed to create acquire semaphore")) {
            goto fail_candidate;
        }
    }
    for (uint32_t i = 0; i < image_count; ++i) {
        if (!IZ_CHK(d, vkCreateSemaphore(d->device, &sem_info, nullptr, &candidate_present[i]),
                    "configure_surface: failed to create present semaphore")) {
            goto fail_candidate;
        }
    }
    for (uint32_t i = 0; i < image_count; ++i) {
        const VkImageViewCreateInfo view_info{
            .sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .pNext    = nullptr,
            .flags    = 0,
            .image    = vk_images[i],
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format   = bridge(config.format),
            .components = {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                           VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY},
            .subresourceRange = {
                .aspectMask     = aspects_for_format(config.format),
                .baseMipLevel   = 0,
                .levelCount     = 1,
                .baseArrayLayer = 0,
                .layerCount     = 1,
            },
        };
        VkImageView view = VK_NULL_HANDLE;
        if (!IZ_CHK(d, vkCreateImageView(d->device, &view_info, nullptr, &view),
                    "configure_surface: failed to create swapchain image view")) {
            goto fail_candidate;
        }
        candidate_textures[i] = handle_cast<Texture>(d->texture_pool.emplace(TextureImpl{
            .vk_image           = vk_images[i],
            .default_image_view = view,
            .vk_allocation      = VK_NULL_HANDLE,
            .vk_type            = VK_IMAGE_VIEW_TYPE_2D,
            .format             = config.format,
            .is_swapchain_image = true,
            .mip_count          = 1,
            .dimensions         = {extent.width, extent.height, 1},
        }));
        created_textures = i + 1;
        d->texture_pool[handle_cast<TextureImpl>(candidate_textures[i])].attachment_views =
            Vector<TextureImpl::AttachmentView>(d->allocator);
    }

    // 6. Commit: retire the old state, install the candidate atomically.
    for (uint32_t i = 0; i < Surface::kMaxSwapchainImages; ++i) {
        if (s.swapchain_images[i].h != 0) {
            d->texture_pool.erase(handle_cast<TextureImpl>(s.swapchain_images[i]));
            s.swapchain_images[i] = {};
        }
    }
    if (s.swapchain != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(d->device, s.swapchain, nullptr);
    }
    for (uint32_t i = 0; i < kMaxFramesInFlight; ++i) {
        if (s.acquire_semaphores[i].h != 0) {
            d->semaphore_pool.erase(handle_cast<SemaphoreImpl>(s.acquire_semaphores[i]));
            s.acquire_semaphores[i] = {};
        }
    }
    for (uint32_t i = 0; i < Surface::kMaxSwapchainImages; ++i) {
        if (s.present_semaphores[i].h != 0) {
            d->semaphore_pool.erase(handle_cast<SemaphoreImpl>(s.present_semaphores[i]));
            s.present_semaphores[i] = {};
        }
    }
    if (s.frame_semaphore.h != 0) {
        d->semaphore_pool.erase(handle_cast<SemaphoreImpl>(s.frame_semaphore));
    }

    s.swapchain        = candidate;
    s.image_count      = image_count;
    s.swapchain_format = bridge(config.format);
    s.swapchain_extent = extent;
    s.frame_latency    = std::min(config.frame_latency, image_count);
    s.frame_semaphore  = candidate_frame;
    for (uint32_t i = 0; i < image_count; ++i) {
        s.swapchain_images[i] = candidate_textures[i];
        s.present_semaphores[i] =
            handle_cast<Semaphore>(d->semaphore_pool.emplace(SemaphoreImpl{.vk_semaphore = candidate_present[i]}));
    }
    for (uint32_t i = 0; i < kMaxFramesInFlight; ++i) {
        s.acquire_semaphores[i] =
            handle_cast<Semaphore>(d->semaphore_pool.emplace(SemaphoreImpl{.vk_semaphore = candidate_acquire[i]}));
    }
    s.frame_idx = 0;
    s.current_image_idx = 0;
    return true;

fail_candidate:
    // Destroy every object created for the failed candidate; the installed
    // state is untouched.
    for (uint32_t i = 0; i < created_textures; ++i) {
        d->texture_pool.erase(handle_cast<TextureImpl>(candidate_textures[i]));
    }
    for (uint32_t i = 0; i < image_count; ++i) {
        if (candidate_present[i] != VK_NULL_HANDLE) {
            vkDestroySemaphore(d->device, candidate_present[i], nullptr);
        }
    }
    for (uint32_t i = 0; i < kMaxFramesInFlight; ++i) {
        if (candidate_acquire[i] != VK_NULL_HANDLE) {
            vkDestroySemaphore(d->device, candidate_acquire[i], nullptr);
        }
    }
    if (candidate_frame.h != 0) {
        d->semaphore_pool.erase(handle_cast<SemaphoreImpl>(candidate_frame));
    }
    vkDestroySwapchainKHR(d->device, candidate, nullptr);
    return false;
}

void unconfigure_surface(Device dev) {
    auto* d = reinterpret_cast<DeviceImpl*>(dev);
    auto& s = d->surface;
    if (s.swapchain != VK_NULL_HANDLE || s.frame_semaphore.h != 0) {
        vkDeviceWaitIdle(d->device);
    }

    if (s.swapchain != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(d->device, s.swapchain, nullptr);
        s.swapchain = VK_NULL_HANDLE;
    }
    s.current_image_idx = 0;
    s.frame_latency    = kDefaultFrameLatency;
    s.image_count       = 0;
    for (uint32_t i = 0; i < Surface::kMaxSwapchainImages; ++i) {
        if (s.swapchain_images[i].h != 0) {
            d->texture_pool.erase(handle_cast<TextureImpl>(s.swapchain_images[i]));
            s.swapchain_images[i] = {};
        }
        if (s.present_semaphores[i].h != 0) {
            d->semaphore_pool.erase(handle_cast<SemaphoreImpl>(s.present_semaphores[i]));
            s.present_semaphores[i] = {};
        }
        s.transitioning_command[i] = VK_NULL_HANDLE;
    }
    for (uint32_t i = 0; i < kMaxFramesInFlight; ++i) {
        if (s.acquire_semaphores[i].h != 0) {
            d->semaphore_pool.erase(handle_cast<SemaphoreImpl>(s.acquire_semaphores[i]));
            s.acquire_semaphores[i] = {};
        }
        s.first_use_command[i] = VK_NULL_HANDLE;
    }
    if (s.frame_semaphore.h != 0) {
        d->semaphore_pool.erase(handle_cast<SemaphoreImpl>(s.frame_semaphore));
        s.frame_semaphore = {};
    }
    s.frame_idx = 0;
}

SurfaceTextureInfo get_current_texture(Device dev) {
    auto* d = reinterpret_cast<DeviceImpl*>(dev);
    auto& s = d->surface;

    SurfaceTextureInfo info{.status = SurfaceStatus::Error, .texture = {}};

    if (s.swapchain == VK_NULL_HANDLE) { return info; }

    // Wait for the configured frame-latency slot to be free.
    const uint64_t wait_value = (s.frame_idx >= s.frame_latency)
                                    ? s.frame_idx + 1 - s.frame_latency
                                    : 0;
    if (wait_value > 0 && s.frame_semaphore.h != 0) {
        VkSemaphore frame_sem = d->semaphore_pool[handle_cast<SemaphoreImpl>(s.frame_semaphore)].vk_semaphore;
        VkSemaphoreWaitInfo sem_wait{
            .sType          = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
            .pNext          = nullptr,
            .flags          = 0,
            .semaphoreCount = 1,
            .pSemaphores    = &frame_sem,
            .pValues        = &wait_value,
        };
        vkWaitSemaphores(d->device, &sem_wait, UINT64_MAX);
    }

    // Acquire next image
    const uint32_t slot = static_cast<uint32_t>(s.frame_idx % s.frame_latency);
    VkSemaphore    acquire_sem = d->semaphore_pool[handle_cast<SemaphoreImpl>(s.acquire_semaphores[slot])].vk_semaphore;

    VkAcquireNextImageInfoKHR acquire_info{
        .sType      = VK_STRUCTURE_TYPE_ACQUIRE_NEXT_IMAGE_INFO_KHR,
        .pNext      = nullptr,
        .swapchain  = s.swapchain,
        .timeout    = 0,
        .semaphore  = acquire_sem,
        .fence      = VK_NULL_HANDLE,
        .deviceMask = 1,
    };
    uint32_t image_idx = 0;
    VkResult result = vkAcquireNextImage2KHR(d->device, &acquire_info, &image_idx);
    s.current_image_idx = image_idx;

    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        info.status = SurfaceStatus::OutOfDate;
    } else if (result == VK_SUBOPTIMAL_KHR) {
        info.status = SurfaceStatus::Suboptimal;
    } else if (result == VK_SUCCESS) {
        info.status = SurfaceStatus::Success;
    } else {
        log_vk_impl(d, result, "get_current_texture: acquire failed", __LINE__, "surface.cpp"_sv);
        info.status = SurfaceStatus::Error;
    }

    info.texture = s.swapchain_images[image_idx];
    return info;
}

SurfaceStatus present(Device dev, Queue q) {
    auto* d = reinterpret_cast<DeviceImpl*>(dev);
    auto& s = d->surface;
    if (s.swapchain == VK_NULL_HANDLE) {
        return SurfaceStatus::Error;   // no surface in this build/configuration
    }
    auto* queue_impl = reinterpret_cast<QueueImpl*>(q);

    const VkSemaphore present_sem =
        d->semaphore_pool[handle_cast<SemaphoreImpl>(s.present_semaphores[s.current_image_idx])].vk_semaphore;

    VkPresentInfoKHR present_info{
        .sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .pNext              = nullptr,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores    = &present_sem,
        .swapchainCount     = 1,
        .pSwapchains        = &s.swapchain,
        .pImageIndices      = &s.current_image_idx,
        .pResults           = nullptr,
    };

    // Presentation and submission are serialized per queue.
    mutex_lock(&queue_impl->submit_lock);
    VkResult result = vkQueuePresentKHR(q->queue, &present_info);
    mutex_unlock(&queue_impl->submit_lock);
    s.frame_idx++;

    switch (result) {
        case VK_SUCCESS: return SurfaceStatus::Success;
        case VK_SUBOPTIMAL_KHR: return SurfaceStatus::Suboptimal;
        case VK_ERROR_OUT_OF_DATE_KHR: return SurfaceStatus::OutOfDate;
        default: return SurfaceStatus::Error;
    }
}

}  // namespace gpu
