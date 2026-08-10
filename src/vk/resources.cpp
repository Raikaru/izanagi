// resources.cpp — textures, views, samplers, descriptor heap writes.

#include "internal.h"

namespace gpu {

// --- Descriptor heap write helpers ---------------------------------------------------

void write_sampled_descriptor(DeviceImpl* d, uint32_t slot, const VkImageViewCreateInfo& view_info) {
    VkImageDescriptorInfoEXT image_desc{
        .sType  = VK_STRUCTURE_TYPE_IMAGE_DESCRIPTOR_INFO_EXT,
        .pNext  = nullptr,
        .pView  = &view_info,
        .layout = VK_IMAGE_LAYOUT_GENERAL,
    };
    VkResourceDescriptorInfoEXT res_info{
        .sType = VK_STRUCTURE_TYPE_RESOURCE_DESCRIPTOR_INFO_EXT,
        .pNext = nullptr,
        .type  = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
        .data  = {.pImage = &image_desc},
    };
    VkHostAddressRangeEXT dst{
        .address = static_cast<uint8_t*>(d->heap.resource_host_ptr) +
                   (size_t)slot * d->heap.sampled_descriptor_size,
        .size = d->heap.sampled_descriptor_size,
    };
    VkResult result = vkWriteResourceDescriptorsEXT(d->device, 1, &res_info, &dst);
    if (result != VK_SUCCESS) {
        log_vk_impl(d, result, "write_sampled_descriptor failed", __LINE__, "resources.cpp"_sv);
    }
}

void write_storage_descriptor(DeviceImpl* d, uint32_t slot, const VkImageViewCreateInfo& view_info) {
    VkImageDescriptorInfoEXT image_desc{
        .sType  = VK_STRUCTURE_TYPE_IMAGE_DESCRIPTOR_INFO_EXT,
        .pNext  = nullptr,
        .pView  = &view_info,
        .layout = VK_IMAGE_LAYOUT_GENERAL,
    };
    VkResourceDescriptorInfoEXT res_info{
        .sType = VK_STRUCTURE_TYPE_RESOURCE_DESCRIPTOR_INFO_EXT,
        .pNext = nullptr,
        .type  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .data  = {.pImage = &image_desc},
    };
    VkHostAddressRangeEXT dst{
        .address = static_cast<uint8_t*>(d->heap.resource_host_ptr) +
                   d->heap.storage_region_offset +
                   (size_t)slot * d->heap.storage_descriptor_size,
        .size = d->heap.storage_descriptor_size,
    };
    VkResult result = vkWriteResourceDescriptorsEXT(d->device, 1, &res_info, &dst);
    if (result != VK_SUCCESS) {
        log_vk_impl(d, result, "write_storage_descriptor failed", __LINE__, "resources.cpp"_sv);
    }
}

void write_sampler_descriptor(DeviceImpl* d, uint32_t slot, const VkSamplerCreateInfo& sampler_info) {
    VkHostAddressRangeEXT dst{
        .address = static_cast<uint8_t*>(d->heap.sampler_host_ptr) +
                   (size_t)slot * d->heap.sampler_descriptor_size,
        .size = d->heap.sampler_descriptor_size,
    };
    VkResult result = vkWriteSamplerDescriptorsEXT(d->device, 1, &sampler_info, &dst);
    if (result != VK_SUCCESS) {
        log_vk_impl(d, result, "write_sampler_descriptor failed", __LINE__, "resources.cpp"_sv);
    }
}

// --- Texture creation -----------------------------------------------------------------

TextureSizeAlign get_texture_size_align(Device dev, const TextureDesc& desc) {
    auto* d = reinterpret_cast<DeviceImpl*>(dev);

    const bool is_cubemap = (desc.type == TextureType::TexCube || desc.type == TextureType::TexCubeArray);
    VkImageCreateInfo info{
        .sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext         = nullptr,
        .flags         = is_cubemap ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : VkImageCreateFlags(0),
        .imageType     = bridge(desc.type),
        .format        = bridge(desc.format),
        .extent        = {.width  = desc.dimensions.x,
                           .height = desc.dimensions.y,
                           .depth  = desc.dimensions.z},
        .mipLevels     = desc.mip_count,
        .arrayLayers   = is_cubemap ? 6 * desc.array_count : desc.array_count,
        .samples       = VK_SAMPLE_COUNT_1_BIT,
        .tiling        = VK_IMAGE_TILING_OPTIMAL,
        .usage         = bridge_usage_flags(desc.usage),
        .sharingMode   = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices   = nullptr,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };

    // Use maintenance4 to get size/align without creating the image
    VkDeviceImageMemoryRequirements mem_req{
        .sType = VK_STRUCTURE_TYPE_DEVICE_IMAGE_MEMORY_REQUIREMENTS,
        .pNext = nullptr,
        .pCreateInfo = &info,
        .planeAspect = VK_IMAGE_ASPECT_COLOR_BIT,
    };
    VkMemoryRequirements2 mem_reqs{
        .sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2,
        .pNext = nullptr,
    };
    vkGetDeviceImageMemoryRequirements(d->device, &mem_req, &mem_reqs);
    return TextureSizeAlign{.size = mem_reqs.memoryRequirements.size,
                            .align = mem_reqs.memoryRequirements.alignment};
}

Handle<Texture> create_texture(Device dev, const TextureDesc& desc, GpuPtr location) {
    auto* d = reinterpret_cast<DeviceImpl*>(dev);

    // Check for unsupported formats (ETC2/ASTC on desktop)
    if (desc.format >= Format::ETC2RGB8Unorm) {
        IZ_LOG(d, LogLevel::Error, "ETC2/ASTC formats not supported on desktop Vulkan");
        return {};
    }

    const bool is_cubemap = (desc.type == TextureType::TexCube || desc.type == TextureType::TexCubeArray);
    const VkImageCreateInfo info{
        .sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext         = nullptr,
        .flags         = is_cubemap ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : VkImageCreateFlags(0),
        .imageType     = bridge(desc.type),
        .format        = bridge(desc.format),
        .extent        = {.width  = desc.dimensions.x,
                           .height = desc.dimensions.y,
                           .depth  = desc.dimensions.z},
        .mipLevels     = desc.mip_count,
        .arrayLayers   = is_cubemap ? 6 * desc.array_count : desc.array_count,
        .samples       = VK_SAMPLE_COUNT_1_BIT,
        .tiling        = VK_IMAGE_TILING_OPTIMAL,
        .usage         = bridge_usage_flags(desc.usage),
        .sharingMode   = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices   = nullptr,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };

    VkImage       image      = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;

    if (location != 0) {
        // Placement: bind image into an existing allocation
        if (!IZ_CHK(d, vkCreateImage(d->device, &info, nullptr, &image),
                    "create_texture: vkCreateImage failed")) {
            return {};
        }
        auto mem = buffer_and_offset_from_ptr(d, location);
        if (!IZ_CHK(d, vmaBindImageMemory(d->vma, mem.alloc, image),
                    "create_texture: vmaBindImageMemory failed")) {
            vkDestroyImage(d->device, image, nullptr);
            return {};
        }
        allocation = mem.alloc;
    } else {
        VmaAllocationCreateInfo alloc_info{
            .flags          = 0,
            .usage          = VMA_MEMORY_USAGE_AUTO,
            .requiredFlags  = 0,
            .preferredFlags = 0,
            .memoryTypeBits = 0,
            .pool           = VK_NULL_HANDLE,
            .pUserData      = nullptr,
            .priority       = 0.0f,
        };
        if (!IZ_CHK(d, vmaCreateImage(d->vma, &info, &alloc_info, &image, &allocation, nullptr),
                    "create_texture: vmaCreateImage failed")) {
            return {};
        }
    }

    // Create default image view for attachment use
    VkImageView default_image_view = VK_NULL_HANDLE;
    if (any(desc.usage & UsageFlags::ColorAttachment) ||
        any(desc.usage & UsageFlags::DepthStencilAttachment)) {
        const VkImageViewCreateInfo view_info{
            .sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .pNext    = nullptr,
            .flags    = 0,
            .image    = image,
            .viewType = bridge_view_type(desc.type),
            .format   = bridge(desc.format),
            .components = {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                           VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY},
            .subresourceRange = {
                .aspectMask     = aspects_for_format(desc.format),
                .baseMipLevel   = 0,
                .levelCount     = 1,
                .baseArrayLayer = 0,
                .layerCount     = 1,
            },
        };
        if (!IZ_CHK(d, vkCreateImageView(d->device, &view_info, nullptr, &default_image_view),
                    "create_texture: vkCreateImageView failed")) {
            if (allocation != VK_NULL_HANDLE) {
                vmaDestroyImage(d->vma, image, allocation);
            } else {
                vkDestroyImage(d->device, image, nullptr);
            }
            return {};
        }
    }

    const auto handle = handle_cast<Texture>(d->texture_pool.emplace(TextureImpl{
        .vk_image           = image,
        .default_image_view = default_image_view,
        .vk_allocation      = allocation,
        .vk_type            = bridge_view_type(desc.type),
        .format             = desc.format,
        .is_swapchain_image = false,
    }));

    mutex_lock(&d->texture_init_lock);
    d->uninitialized_textures.push_back(handle);
    mutex_unlock(&d->texture_init_lock);

    return handle;
}

void free(Device dev, Handle<Texture> t) {
    auto* d = reinterpret_cast<DeviceImpl*>(dev);
    d->texture_pool.erase(handle_cast<TextureImpl>(t));
}

// --- Texture views (object-less, heap-descriptor-only) -----------------------------------

static VkImageViewCreateInfo make_view_info(DeviceImpl* d, const TextureViewDesc& desc) {
    auto& texture = d->texture_pool[handle_cast<TextureImpl>(desc.texture)];
    return VkImageViewCreateInfo{
        .sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext    = nullptr,
        .flags    = 0,
        .image    = texture.vk_image,
        .viewType = bridge_view_type(desc.type),
        .format   = bridge(desc.format != Format::None ? desc.format : texture.format),
        .components = {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                       VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY},
        .subresourceRange = {
            .aspectMask     = aspects_for_format(desc.format != Format::None ? desc.format : texture.format),
            .baseMipLevel   = desc.base_mip,
            .levelCount     = desc.mip_count,
            .baseArrayLayer = desc.base_layer,
            .layerCount     = desc.layer_count,
        },
    };
}

TextureView create_texture_view(Device dev, const TextureViewDesc& desc) {
    auto* d = reinterpret_cast<DeviceImpl*>(dev);
    uint32_t slot = d->sampled_bitset.set_leading_zero();
    if (slot == ~0u) {
        IZ_LOG(d, LogLevel::Error, "Sampled texture heap full");
        return ~0ull;
    }
    auto view_info = make_view_info(d, desc);
    write_sampled_descriptor(d, slot, view_info);
    return static_cast<TextureView>(slot);
}

TextureView create_rw_texture_view(Device dev, const TextureViewDesc& desc) {
    auto* d = reinterpret_cast<DeviceImpl*>(dev);
    uint32_t slot = d->storage_bitset.set_leading_zero();
    if (slot == ~0u) {
        IZ_LOG(d, LogLevel::Error, "Storage texture heap full");
        return ~0ull;
    }
    auto view_info = make_view_info(d, desc);
    write_storage_descriptor(d, slot, view_info);
    return static_cast<TextureView>(slot);
}

SamplerId create_sampler(Device dev, const SamplerDesc& desc) {
    auto* d = reinterpret_cast<DeviceImpl*>(dev);
    uint32_t slot = d->sampler_bitset.set_leading_zero();
    if (slot == ~0u) {
        IZ_LOG(d, LogLevel::Error, "Sampler heap full");
        return ~0ull;
    }

    const VkSamplerCreateInfo sampler_info{
        .sType            = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .pNext            = nullptr,
        .flags            = 0,
        .magFilter        = bridge(desc.filter),
        .minFilter        = bridge(desc.filter),
        .mipmapMode       = bridge_mip_mode(desc.filter),
        .addressModeU     = bridge(desc.address),
        .addressModeV     = bridge(desc.address),
        .addressModeW     = bridge(desc.address),
        .mipLodBias       = 0,
        .anisotropyEnable = (desc.max_anisotropy != 1.0f),
        .maxAnisotropy    = desc.max_anisotropy,
        .compareEnable    = VK_FALSE,
        .compareOp        = VK_COMPARE_OP_NEVER,
        .minLod           = 0,
        .maxLod           = VK_LOD_CLAMP_NONE,
        .borderColor      = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
        .unnormalizedCoordinates = (desc.coord == SamplerCoords::Pixel),
    };
    write_sampler_descriptor(d, slot, sampler_info);
    return static_cast<SamplerId>(slot);
}

// --- Free with deferred recycling --------------------------------------------------------

static void defer_free_slot(DeviceImpl* d, uint32_t slot, uint8_t region) {
    // Get the queue's current timeline value for deferred recycling
    uint64_t timeline_val = 0;
    if (d->default_queue) {
        timeline_val = d->default_queue->timeline_value;
    }
    mutex_lock(&d->deferred_lock);
    d->deferred_frees.push_back(DeviceImpl::DeferredFree{
        .slot           = slot,
        .timeline_value = timeline_val,
        .region         = region,
    });
    mutex_unlock(&d->deferred_lock);
}

void free_texture_view(Device dev, TextureView view) {
    auto* d = reinterpret_cast<DeviceImpl*>(dev);
    defer_free_slot(d, static_cast<uint32_t>(view), 0);
}

void free_rw_texture_view(Device dev, TextureView view) {
    auto* d = reinterpret_cast<DeviceImpl*>(dev);
    defer_free_slot(d, static_cast<uint32_t>(view), 1);
}

void free_sampler(Device dev, SamplerId sampler) {
    auto* d = reinterpret_cast<DeviceImpl*>(dev);
    defer_free_slot(d, static_cast<uint32_t>(sampler), 2);
}

}  // namespace gpu
