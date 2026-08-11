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

// --- Descriptor handles ------------------------------------------------------------------
// CPU handle encoding (must match the shader prelude's low-32 unpack):
//   bits  0..31  descriptor index (shader-visible)
//   bits 32..47  CPU generation (stale-handle detection)
//   bits 48..55  descriptor type (1=sampled, 2=storage, 3=sampler)
// Descriptor index 0 is reserved as null: handle 0 is never valid. Allocation
// failure returns 0; a bitwise-all-ones value is never produced.
static constexpr uint64_t kDescSlotMask = 0xFFFFFFFFull;
static constexpr uint64_t kDescGenShift  = 32;
static constexpr uint64_t kDescTypeShift = 48;
static constexpr uint64_t kDescTypeSampled = 1;
static constexpr uint64_t kDescTypeStorage = 2;
static constexpr uint64_t kDescTypeSampler = 3;

static uint64_t encode_desc_handle(uint32_t slot, uint16_t gen, uint64_t type) {
    return static_cast<uint64_t>(slot) | (static_cast<uint64_t>(gen) << kDescGenShift) |
           (type << kDescTypeShift);
}

// Validates the handle's type metadata and returns slot/gen. Returns false for
// a handle of the wrong descriptor type (or with garbage type bits).
static bool decode_desc_handle(uint64_t h, uint64_t expect_type, uint32_t* slot, uint16_t* gen) {
    if (((h >> kDescTypeShift) & 0xFF) != expect_type) { return false; }
    *slot = static_cast<uint32_t>(h & kDescSlotMask);
    *gen  = static_cast<uint16_t>((h >> kDescGenShift) & 0xFFFF);
    return true;
}

// Validates the handle's type metadata and CPU generation; rejects stale
// handles and double frees (logged, no-op). Returns the slot on success.
static bool validate_free_handle(DeviceImpl* d, uint64_t handle, uint64_t expect_type,
                                 Vector<uint16_t>& gen_table, uint32_t* slot) {
    uint16_t gen = 0;
    if (!decode_desc_handle(handle, expect_type, slot, &gen)) {
        IZ_LOG(d, LogLevel::Error, "free: handle has the wrong descriptor type (corrupt or double free)");
        return false;
    }
    if (*slot >= gen_table.size() || gen_table[*slot] != gen) {
        IZ_LOG(d, LogLevel::Error, "free: stale descriptor handle (generation mismatch)");
        return false;
    }
    return true;
}

// --- Texture creation -----------------------------------------------------------------
// Textures allocate their own GPU memory (VMA). There is no buffer-backed
// placement token: a shader-visible GpuPtr is not a texture-memory token.

Handle<Texture> create_texture(Device dev, const TextureDesc& desc) {
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
        .samples       = static_cast<VkSampleCountFlagBits>(desc.sample_count == 0 ? 1 : desc.sample_count),
        .tiling        = VK_IMAGE_TILING_OPTIMAL,
        .usage         = bridge_usage_flags(desc.usage),
        .sharingMode   = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices   = nullptr,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };

    VkImage       image      = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    {
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

    // Default image view for attachment use (mip 0 / layer 0); subresource
    // attachment views are cached lazily in cmd_begin_render_pass.
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
            vmaDestroyImage(d->vma, image, allocation);
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
        .mip_count          = desc.mip_count,
        .dimensions         = desc.dimensions,
    }));
    d->texture_pool[handle_cast<TextureImpl>(handle)].attachment_views =
        Vector<TextureImpl::AttachmentView>(d->allocator);

    mutex_lock(&d->texture_init_lock);
    d->uninitialized_textures.push_back(handle);
    mutex_unlock(&d->texture_init_lock);

    return handle;
}

void remove_uninitialized_texture(DeviceImpl* d, Handle<Texture> tex) {
    mutex_lock(&d->texture_init_lock);
    for (uint32_t i = 0; i < d->uninitialized_textures.size(); ++i) {
        if (d->uninitialized_textures[i].h == tex.h) {
            d->uninitialized_textures.erase(d->uninitialized_textures.begin() + i,
                                            d->uninitialized_textures.begin() + i + 1);
            break;
        }
    }
    mutex_unlock(&d->texture_init_lock);
}

void release_texture_ref(DeviceImpl* d, Handle<Texture> tex) {
    // 1 per public handle + 1 per command-buffer/in-flight retention; the
    // pool slot (and native image) is destroyed at zero.
    auto& t = d->texture_pool[handle_cast<TextureImpl>(tex)];
    if (atomic_fetch_add(&t.refs, -1) == 1) {
        d->texture_pool.erase(handle_cast<TextureImpl>(tex));
    }
}

void free(Device dev, Handle<Texture> t) {
    auto* d = reinterpret_cast<DeviceImpl*>(dev);
    // The init-transition list must never retain a handle whose slot is gone.
    remove_uninitialized_texture(d, t);
    release_texture_ref(d, t);
}

// --- Deferred destruction against a submission ------------------------------------------------

void free_after(Device dev, Handle<Texture> tex, Submission s) {
    auto* d = reinterpret_cast<DeviceImpl*>(dev);
    QueueImpl* q = s.queue ? reinterpret_cast<QueueImpl*>(s.queue) : d->default_queue;
    if (q == nullptr) { return; }
    const uint64_t value = s.status == SubmitStatus::Success ? s.value : q->timeline_value;
    enqueue_retire(q, value, RetireItem{RetireKind::Texture, tex.h, 0});
}

void free_texture_view_after(Device dev, TextureView view, Submission s) {
    auto* d = reinterpret_cast<DeviceImpl*>(dev);
    uint32_t slot = 0;
    if (!validate_free_handle(d, view, kDescTypeSampled, d->sampled_gen, &slot)) { return; }
    QueueImpl* q = s.queue ? reinterpret_cast<QueueImpl*>(s.queue) : d->default_queue;
    if (q == nullptr) { return; }
    const uint64_t value = s.status == SubmitStatus::Success ? s.value : q->timeline_value;
    enqueue_retire(q, value, RetireItem{RetireKind::SampledSlot, static_cast<uint64_t>(slot), 0});
}

void free_rw_texture_view_after(Device dev, TextureView view, Submission s) {
    auto* d = reinterpret_cast<DeviceImpl*>(dev);
    uint32_t slot = 0;
    if (!validate_free_handle(d, view, kDescTypeStorage, d->storage_gen, &slot)) { return; }
    QueueImpl* q = s.queue ? reinterpret_cast<QueueImpl*>(s.queue) : d->default_queue;
    if (q == nullptr) { return; }
    const uint64_t value = s.status == SubmitStatus::Success ? s.value : q->timeline_value;
    enqueue_retire(q, value, RetireItem{RetireKind::StorageSlot, static_cast<uint64_t>(slot), 0});
}

void free_sampler_after(Device dev, SamplerId sampler, Submission s) {
    auto* d = reinterpret_cast<DeviceImpl*>(dev);
    uint32_t slot = 0;
    if (!validate_free_handle(d, sampler, kDescTypeSampler, d->sampler_gen, &slot)) { return; }
    QueueImpl* q = s.queue ? reinterpret_cast<QueueImpl*>(s.queue) : d->default_queue;
    if (q == nullptr) { return; }
    const uint64_t value = s.status == SubmitStatus::Success ? s.value : q->timeline_value;
    enqueue_retire(q, value, RetireItem{RetireKind::SamplerSlot, static_cast<uint64_t>(slot), 0});
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
        return 0;   // null descriptor index — never a valid handle
    }
    auto view_info = make_view_info(d, desc);
    write_sampled_descriptor(d, slot, view_info);
    const uint16_t gen = static_cast<uint16_t>(d->sampled_gen[slot] + 1);
    d->sampled_gen[slot] = gen;
    return encode_desc_handle(slot, gen, kDescTypeSampled);
}

TextureView create_rw_texture_view(Device dev, const TextureViewDesc& desc) {
    auto* d = reinterpret_cast<DeviceImpl*>(dev);
    uint32_t slot = d->storage_bitset.set_leading_zero();
    if (slot == ~0u) {
        IZ_LOG(d, LogLevel::Error, "Storage texture heap full");
        return 0;
    }
    auto view_info = make_view_info(d, desc);
    write_storage_descriptor(d, slot, view_info);
    const uint16_t gen = static_cast<uint16_t>(d->storage_gen[slot] + 1);
    d->storage_gen[slot] = gen;
    return encode_desc_handle(slot, gen, kDescTypeStorage);
}

SamplerId create_sampler(Device dev, const SamplerDesc& desc) {
    auto* d = reinterpret_cast<DeviceImpl*>(dev);
    uint32_t slot = d->sampler_bitset.set_leading_zero();
    if (slot == ~0u) {
        IZ_LOG(d, LogLevel::Error, "Sampler heap full");
        return 0;
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
    const uint16_t gen = static_cast<uint16_t>(d->sampler_gen[slot] + 1);
    d->sampler_gen[slot] = gen;
    return encode_desc_handle(slot, gen, kDescTypeSampler);
}

// --- Free with deferred recycling (unified queue retirement) ---------------------

static void defer_free_slot(DeviceImpl* d, uint32_t slot, RetireKind kind) {
    QueueImpl* q = d->default_queue;
    if (q == nullptr) { return; }
    // Retire after the latest submitted work completes (conservative default;
    // explicit free_*_after takes an exact submission).
    enqueue_retire(q, q->timeline_value, RetireItem{kind, slot, 0});
}

void free_texture_view(Device dev, TextureView view) {
    auto* d = reinterpret_cast<DeviceImpl*>(dev);
    uint32_t slot = 0;
    if (!validate_free_handle(d, view, kDescTypeSampled, d->sampled_gen, &slot)) { return; }
    defer_free_slot(d, slot, RetireKind::SampledSlot);
}

void free_rw_texture_view(Device dev, TextureView view) {
    auto* d = reinterpret_cast<DeviceImpl*>(dev);
    uint32_t slot = 0;
    if (!validate_free_handle(d, view, kDescTypeStorage, d->storage_gen, &slot)) { return; }
    defer_free_slot(d, slot, RetireKind::StorageSlot);
}

void free_sampler(Device dev, SamplerId sampler) {
    auto* d = reinterpret_cast<DeviceImpl*>(dev);
    uint32_t slot = 0;
    if (!validate_free_handle(d, sampler, kDescTypeSampler, d->sampler_gen, &slot)) { return; }
    defer_free_slot(d, slot, RetireKind::SamplerSlot);
}

}  // namespace gpu
