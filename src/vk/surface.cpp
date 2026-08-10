// surface.cpp — swapchain creation, acquire/present, frame pacing.

#include "internal.h"

namespace gpu {

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

    // Wait for idle before recreating swapchain
    vkDeviceWaitIdle(d->device);

    VkSurfaceCapabilitiesKHR vk_caps;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(d->physical_device, s.surface, &vk_caps);

    uint32_t image_count = vk_caps.minImageCount + 1;
    // Prefer 3 images for presentation, clamped to caps
    if (image_count < 3 && vk_caps.maxImageCount >= 3) { image_count = 3; }
    if (vk_caps.maxImageCount > 0 && image_count > vk_caps.maxImageCount) {
        image_count = vk_caps.maxImageCount;
    }

    const auto extent = VkExtent2D{.width = config.width, .height = config.height};

    VkSwapchainKHR old_swapchain = s.swapchain;

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
        .presentMode           = bridge(config.present_mode),
        .clipped               = true,
        .oldSwapchain          = old_swapchain,
    };

    if (!IZ_CHK(d, vkCreateSwapchainKHR(d->device, &swapchain_info, nullptr, &s.swapchain),
                "configure_surface: vkCreateSwapchainKHR failed")) {
        return false;
    }

    // Destroy old swapchain
    if (old_swapchain != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(d->device, old_swapchain, nullptr);
    }

    // Get swapchain images
    image_count = 0;
    if (!IZ_CHK(d, vkGetSwapchainImagesKHR(d->device, s.swapchain, &image_count, nullptr),
                "configure_surface: vkGetSwapchainImagesKHR failed")) {
        return false;
    }
    if (image_count > Surface::kMaxSwapchainImages) {
        IZ_LOG(d, LogLevel::Error, "Too many swapchain images");
        return false;
    }

    VkImage vk_images[Surface::kMaxSwapchainImages];
    if (!IZ_CHK(d, vkGetSwapchainImagesKHR(d->device, s.swapchain, &image_count, vk_images),
                "configure_surface: vkGetSwapchainImagesKHR failed")) {
        return false;
    }

    s.image_count       = image_count;
    s.swapchain_format  = bridge(config.format);
    s.swapchain_extent  = extent;

    // Create semaphores for acquire/present
    const VkSemaphoreCreateInfo sem_info{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
    };

    // Free old semaphores
    for (uint32_t i = 0; i < kMaxFramesInFlight; ++i) {
        if (s.acquire_semaphores[i].h != 0) {
            d->semaphore_pool.erase(handle_cast<SemaphoreImpl>(s.acquire_semaphores[i]));
        }
    }
    for (uint32_t i = 0; i < Surface::kMaxSwapchainImages; ++i) {
        if (s.present_semaphores[i].h != 0) {
            d->semaphore_pool.erase(handle_cast<SemaphoreImpl>(s.present_semaphores[i]));
        }
    }

    // Acquire semaphores: one per frame-in-flight
    for (uint32_t i = 0; i < kMaxFramesInFlight; ++i) {
        VkSemaphore sem;
        if (!IZ_CHK(d, vkCreateSemaphore(d->device, &sem_info, nullptr, &sem),
                    "configure_surface: failed to create acquire semaphore")) {
            return false;
        }
        s.acquire_semaphores[i] = handle_cast<Semaphore>(
            d->semaphore_pool.emplace(SemaphoreImpl{.vk_semaphore = sem}));
    }

    // Present semaphores: one per swapchain image
    for (uint32_t i = 0; i < image_count; ++i) {
        VkSemaphore sem;
        if (!IZ_CHK(d, vkCreateSemaphore(d->device, &sem_info, nullptr, &sem),
                    "configure_surface: failed to create present semaphore")) {
            return false;
        }
        s.present_semaphores[i] = handle_cast<Semaphore>(
            d->semaphore_pool.emplace(SemaphoreImpl{.vk_semaphore = sem}));
    }

    // Register swapchain images as textures
    for (uint32_t i = 0; i < image_count; ++i) {
        // Create default image view for each swapchain image
        VkImageViewCreateInfo view_info{
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
        VkImageView view;
        if (!IZ_CHK(d, vkCreateImageView(d->device, &view_info, nullptr, &view),
                    "configure_surface: failed to create swapchain image view")) {
            return false;
        }

        s.swapchain_images[i] = handle_cast<Texture>(d->texture_pool.emplace(TextureImpl{
            .vk_image           = vk_images[i],
            .default_image_view = view,
            .vk_allocation      = VK_NULL_HANDLE,
            .vk_type            = VK_IMAGE_VIEW_TYPE_2D,
            .format             = config.format,
            .is_swapchain_image = true,
        }));
    }

    // Frame semaphore (timeline for frame pacing)
    if (s.frame_semaphore.h == 0) {
        s.frame_semaphore = create_semaphore_internal(d, 0);
    }
    s.frame_idx = 0;

    return true;
}

void unconfigure_surface(Device dev) {
    auto* d = reinterpret_cast<DeviceImpl*>(dev);
    auto& s = d->surface;

    if (s.swapchain != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(d->device, s.swapchain, nullptr);
        s.swapchain = VK_NULL_HANDLE;
    }
    s.current_image_idx = 0;
    s.image_count       = 0;
    for (uint32_t i = 0; i < Surface::kMaxSwapchainImages; ++i) {
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
}

SurfaceTextureInfo get_current_texture(Device dev) {
    auto* d = reinterpret_cast<DeviceImpl*>(dev);
    auto& s = d->surface;

    SurfaceTextureInfo info{.status = SurfaceStatus::Error, .texture = {}};

    if (s.swapchain == VK_NULL_HANDLE) { return info; }

    // Wait for the frame slot to be free (timeline wait on the slot's last signal)
    const uint64_t wait_value = (s.frame_idx >= kMaxFramesInFlight)
                                    ? s.frame_idx + 1 - kMaxFramesInFlight
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
    const uint32_t slot = s.frame_idx % kMaxFramesInFlight;
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

    VkResult result = vkQueuePresentKHR(q->queue, &present_info);
    s.frame_idx++;

    switch (result) {
        case VK_SUCCESS: return SurfaceStatus::Success;
        case VK_SUBOPTIMAL_KHR: return SurfaceStatus::Suboptimal;
        case VK_ERROR_OUT_OF_DATE_KHR: return SurfaceStatus::OutOfDate;
        default: return SurfaceStatus::Error;
    }
}

}  // namespace gpu
