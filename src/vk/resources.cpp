// resources.cpp — textures, views, samplers, descriptor heap writes.

#include "internal.h"

namespace gpu {

// --- Descriptor heap write helpers ---------------------------------------------------

bool write_sampled_descriptor(DeviceImpl* d, uint32_t slot, const VkImageViewCreateInfo& view_info) {
#if defined(IZ_VK_PROFILE_BINDLESS)
    // vkUpdateDescriptorSets requires host synchronization on the set: hold
    // desc_lock across the update and the sidecar assignment (callers release
    // desc_lock before calling, see desc_allocate_image_view).
    VkImageView view = VK_NULL_HANDLE;
    if (!IZ_CHK(d, vkCreateImageView(d->device, &view_info, nullptr, &view),
                "write_sampled_descriptor: vkCreateImageView failed")) {
        return false;
    }
    const VkDescriptorImageInfo image_info{
        .sampler     = VK_NULL_HANDLE,
        .imageView   = view,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };
    const VkWriteDescriptorSet write{
        .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .pNext           = nullptr,
        .dstSet          = d->bindless_set,
        .dstBinding      = 0,
        .dstArrayElement = slot,
        .descriptorCount = 1,
        .descriptorType  = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
        .pImageInfo      = &image_info,
    };
    // vkUpdateDescriptorSets returns void; validation errors surface through
    // the debug messenger. Store the view unconditionally (freed at slot
    // retirement or device destroy).
    mutex_lock(&d->desc_lock);
    vkUpdateDescriptorSets(d->device, 1, &write, 0, nullptr);
    if (slot < d->bindless_sampled_views.size()) { d->bindless_sampled_views[slot] = view; }
    mutex_unlock(&d->desc_lock);
    return true;
#else
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
        return false;
    }
    return true;
#endif
}

bool write_storage_descriptor(DeviceImpl* d, uint32_t slot, const VkImageViewCreateInfo& view_info) {
#if defined(IZ_VK_PROFILE_BINDLESS)
    VkImageView view = VK_NULL_HANDLE;
    if (!IZ_CHK(d, vkCreateImageView(d->device, &view_info, nullptr, &view),
                "write_storage_descriptor: vkCreateImageView failed")) {
        return false;
    }
    const VkDescriptorImageInfo image_info{
        .sampler     = VK_NULL_HANDLE,
        .imageView   = view,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };
    const VkWriteDescriptorSet write{
        .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .pNext           = nullptr,
        .dstSet          = d->bindless_set,
        .dstBinding      = 1,
        .dstArrayElement = slot,
        .descriptorCount = 1,
        .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .pImageInfo      = &image_info,
    };
    mutex_lock(&d->desc_lock);
    vkUpdateDescriptorSets(d->device, 1, &write, 0, nullptr);
    if (slot < d->bindless_storage_views.size()) { d->bindless_storage_views[slot] = view; }
    mutex_unlock(&d->desc_lock);
    return true;
#else
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
        return false;
    }
    return true;
#endif
}

bool write_sampler_descriptor(DeviceImpl* d, uint32_t slot, const VkSamplerCreateInfo& sampler_info) {
#if defined(IZ_VK_PROFILE_BINDLESS)
    VkSampler sampler = VK_NULL_HANDLE;
    if (!IZ_CHK(d, vkCreateSampler(d->device, &sampler_info, nullptr, &sampler),
                "write_sampler_descriptor: vkCreateSampler failed")) {
        return false;
    }
    const VkDescriptorImageInfo image_info{
        .sampler     = sampler,
        .imageView   = VK_NULL_HANDLE,
        .imageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    const VkWriteDescriptorSet write{
        .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .pNext           = nullptr,
        .dstSet          = d->bindless_set,
        .dstBinding      = 2,
        .dstArrayElement = slot,
        .descriptorCount = 1,
        .descriptorType  = VK_DESCRIPTOR_TYPE_SAMPLER,
        .pImageInfo      = &image_info,
    };
    mutex_lock(&d->desc_lock);
    vkUpdateDescriptorSets(d->device, 1, &write, 0, nullptr);
    if (slot < d->bindless_sampler_handles.size()) { d->bindless_sampler_handles[slot] = sampler; }
    mutex_unlock(&d->desc_lock);
    return true;
#else
    VkHostAddressRangeEXT dst{
        .address = static_cast<uint8_t*>(d->heap.sampler_host_ptr) +
                   (size_t)slot * d->heap.sampler_descriptor_size,
        .size = d->heap.sampler_descriptor_size,
    };
    VkResult result = vkWriteSamplerDescriptorsEXT(d->device, 1, &sampler_info, &dst);
    if (result != VK_SUCCESS) {
        log_vk_impl(d, result, "write_sampler_descriptor failed", __LINE__, "resources.cpp"_sv);
        return false;
    }
    return true;
#endif
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

// Region accessors for the descriptor allocator (desc_lock held by callers).
static TwoLevelBitset& desc_bitset(DeviceImpl* d, RetireKind kind) {
    switch (kind) {
        case RetireKind::SampledSlot: return d->sampled_bitset;
        case RetireKind::StorageSlot: return d->storage_bitset;
        default:                       return d->sampler_bitset;
    }
}
static Vector<uint16_t>& desc_gen(DeviceImpl* d, RetireKind kind) {
    switch (kind) {
        case RetireKind::SampledSlot: return d->sampled_gen;
        case RetireKind::StorageSlot: return d->storage_gen;
        default:                       return d->sampler_gen;
    }
}
static Vector<uint8_t>& desc_state(DeviceImpl* d, RetireKind kind) {
    switch (kind) {
        case RetireKind::SampledSlot: return d->sampled_state;
        case RetireKind::StorageSlot: return d->storage_state;
        default:                       return d->sampler_state;
    }
}

// Accepts a descriptor free atomically: validates type/index/generation and
// the Allocated state, moves the slot to Retiring, and bumps the generation so
// the old handle is stale immediately. Outputs the slot and the post-bump
// generation the retirement must verify. Returns false (no-op) for stale
// handles, double frees (including a second free while Retiring), and wrong
// descriptor types.
static bool desc_free_accept(DeviceImpl* d, uint64_t handle, uint64_t expect_type, RetireKind kind,
                             uint32_t* slot_out, uint16_t* retire_gen_out) {
    uint32_t slot = 0;
    uint16_t gen  = 0;
    if (!decode_desc_handle(handle, expect_type, &slot, &gen)) {
        IZ_LOG(d, LogLevel::Error, "descriptor free: wrong descriptor type (corrupt or double free)");
        return false;
    }
    mutex_lock(&d->desc_lock);
    auto& states = desc_state(d, kind);
    auto& gens   = desc_gen(d, kind);
    bool ok = false;
    if (slot < states.size() &&
        states[slot] == static_cast<uint8_t>(DeviceImpl::DescriptorSlotState::Allocated) &&
        gens[slot] == gen) {
        states[slot] = static_cast<uint8_t>(DeviceImpl::DescriptorSlotState::Retiring);
        gens[slot]   = static_cast<uint16_t>(gen + 1);   // old handle stale immediately
        *slot_out       = slot;
        *retire_gen_out = gens[slot];
        ok = true;
    } else {
        IZ_LOG(d, LogLevel::Error, "descriptor free: stale handle or double free rejected");
    }
    mutex_unlock(&d->desc_lock);
    return ok;
}

// Completes a slot's retirement: verifies the slot is still Retiring with the
// matching generation (a delayed duplicate retirement is a no-op), returns the
// slot to Free, and releases the owner texture reference (outside the lock).
void desc_retire_slot(DeviceImpl* d, RetireKind kind, uint32_t slot, uint16_t gen) {
    TextureImpl* owner = nullptr;
    mutex_lock(&d->desc_lock);
    auto& states = desc_state(d, kind);
    auto& gens   = desc_gen(d, kind);
    if (slot < states.size() &&
        states[slot] == static_cast<uint8_t>(DeviceImpl::DescriptorSlotState::Retiring) &&
        gens[slot] == gen) {
        states[slot] = static_cast<uint8_t>(DeviceImpl::DescriptorSlotState::Free);
        desc_bitset(d, kind).clear_bit(slot);
#if defined(IZ_VK_PROFILE_BINDLESS)
        // Destroy the per-slot native handle written into the global set.
        if (kind == RetireKind::SampledSlot && slot < d->bindless_sampled_views.size()) {
            VkImageView v = d->bindless_sampled_views[slot];
            d->bindless_sampled_views[slot] = VK_NULL_HANDLE;
            if (v != VK_NULL_HANDLE) { vkDestroyImageView(d->device, v, nullptr); }
        } else if (kind == RetireKind::StorageSlot && slot < d->bindless_storage_views.size()) {
            VkImageView v = d->bindless_storage_views[slot];
            d->bindless_storage_views[slot] = VK_NULL_HANDLE;
            if (v != VK_NULL_HANDLE) { vkDestroyImageView(d->device, v, nullptr); }
        } else if (kind == RetireKind::SamplerSlot && slot < d->bindless_sampler_handles.size()) {
            VkSampler s = d->bindless_sampler_handles[slot];
            d->bindless_sampler_handles[slot] = VK_NULL_HANDLE;
            if (s != VK_NULL_HANDLE) { vkDestroySampler(d->device, s, nullptr); }
        }
#endif
        if (kind == RetireKind::SampledSlot) {
            owner = d->sampled_owner[slot];
            d->sampled_owner[slot] = nullptr;
        } else if (kind == RetireKind::StorageSlot) {
            owner = d->storage_owner[slot];
            d->storage_owner[slot] = nullptr;
        }
    }
    mutex_unlock(&d->desc_lock);
    if (owner != nullptr) {
        Handle<Texture> h = handle_cast<Texture>(d->texture_pool.find_handle(owner));
        if (h.h != 0) { release_texture_ref(d, h); }
    }
}

// Enqueues an accepted descriptor free against the given retirement value.
static void desc_free_enqueue(DeviceImpl* d, RetireKind kind, uint32_t slot, uint16_t gen) {
    QueueImpl* q = d->default_queue;
    if (q == nullptr) { return; }
    enqueue_retire(q, q->timeline_value, RetireItem{kind, slot, gen});
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
    const uint32_t queue_families[2] = {
        d->graphics_queue_family,
        d->transfer_queue_family,
    };
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
        .sharingMode = d->dedicated_transfer_queue
                           ? VK_SHARING_MODE_CONCURRENT
                           : VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = d->dedicated_transfer_queue ? 2u : 0u,
        .pQueueFamilyIndices   = d->dedicated_transfer_queue ? queue_families : nullptr,
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
    d->uninitialized_textures.push_back(&d->texture_pool[handle_cast<TextureImpl>(handle)]);
    mutex_unlock(&d->texture_init_lock);

    return handle;
}

// Marks the record released (its public free/free_after was accepted). The
// record stays in the init-transition list while other references keep it
// alive, so a retained command buffer can still trigger the UNDEFINED->GENERAL
// barrier for it; it is removed from the list only when the final reference
// drops (release_texture_ref, before erase). The Released state prevents a
// drain rollback from requeueing it.
void remove_uninitialized_texture(DeviceImpl* d, TextureImpl* rec) {
    mutex_lock(&d->texture_init_lock);
    rec->init_state = TextureInitState::Released;
    mutex_unlock(&d->texture_init_lock);
}

void release_texture_ref(DeviceImpl* d, Handle<Texture> tex) {
    // Resolve the record first (pool mutex), then serialize the decrement and
    // the init-list removal under texture_init_lock: a concurrent submit can
    // claim a still-listed record in that gap, so the list must never hold a
    // pointer to a record whose refs reached zero.
    const char* why = nullptr;
    TextureImpl* rec = d->texture_pool.try_get_ex(handle_cast<TextureImpl>(tex), &why);
    if (rec == nullptr) {
        log_fmt(d, LogLevel::Error, __LINE__, __FILE__,
                "release_texture_ref: stale texture handle 0x%016llx (%s)",
                (unsigned long long)tex.h, why ? why : "rejected");
        return;
    }
    mutex_lock(&d->texture_init_lock);
    const bool last = (atomic_fetch_add(&rec->refs, -1) == 1);
    if (last) {
        for (uint32_t i = 0; i < d->uninitialized_textures.size(); ++i) {
            if (d->uninitialized_textures[i] == rec) {
                d->uninitialized_textures.erase(d->uninitialized_textures.begin() + i,
                                                d->uninitialized_textures.begin() + i + 1);
                break;
            }
        }
    }
    mutex_unlock(&d->texture_init_lock);
    if (last) {
        d->texture_pool.erase(handle_cast<TextureImpl>(tex));
    }
}
void free(Device dev, Handle<Texture> t) {
    auto* d = reinterpret_cast<DeviceImpl*>(dev);
    const char* why = nullptr;
    TextureImpl* rec = d->texture_pool.try_get_ex(handle_cast<TextureImpl>(t), &why);
    if (rec == nullptr) {
        log_fmt(d, LogLevel::Error, __LINE__, __FILE__,
                "free(texture): invalid handle 0x%016llx (%s)",
                (unsigned long long)t.h, why ? why : "rejected");
        return;
    }
    remove_uninitialized_texture(d, rec);
    // Invalidate the public handle immediately even when descriptor or
    // command-buffer retentions keep the record alive; the user reference is
    // released through the record's current-generation handle. A concurrent
    // double free loses the invalidate race and retires nothing.
    if (!d->texture_pool.invalidate(handle_cast<TextureImpl>(t))) { return; }
    Handle<Texture> h = handle_cast<Texture>(d->texture_pool.find_handle(rec));
    if (h.h != 0) { release_texture_ref(d, h); }
}

// --- Deferred destruction against a submission ------------------------------------------------

void free_after(Device dev, Handle<Texture> tex, Submission s) {
    auto* d = reinterpret_cast<DeviceImpl*>(dev);
    QueueImpl* q = s.queue ? reinterpret_cast<QueueImpl*>(s.queue) : d->default_queue;
    if (q == nullptr) { return; }
    const uint64_t value = s.status == SubmitStatus::Success ? s.value : q->timeline_value;
    // Invalidate the public handle immediately; the stable record survives
    // via the deferred user reference until the submission completes.
    const char* why = nullptr;
    TextureImpl* rec = d->texture_pool.try_get_ex(handle_cast<TextureImpl>(tex), &why);
    if (rec == nullptr) {
        log_fmt(d, LogLevel::Error, __LINE__, __FILE__,
                "free_after(texture): invalid handle 0x%016llx (%s)",
                (unsigned long long)tex.h, why ? why : "rejected");
        return;
    }
    if (!d->texture_pool.invalidate(handle_cast<TextureImpl>(tex))) { return; }
    remove_uninitialized_texture(d, rec);
    enqueue_retire(q, value, RetireItem{RetireKind::Texture, reinterpret_cast<uint64_t>(rec), 0});
}

static void desc_free_after_accept(DeviceImpl* d, uint64_t handle, uint64_t expect_type,
                                     RetireKind kind, Submission s) {
    uint32_t slot = 0;
    uint16_t gen  = 0;
    if (!desc_free_accept(d, handle, expect_type, kind, &slot, &gen)) { return; }
    QueueImpl* q = s.queue ? reinterpret_cast<QueueImpl*>(s.queue) : d->default_queue;
    if (q == nullptr) { return; }
    const uint64_t value = s.status == SubmitStatus::Success ? s.value : q->timeline_value;
    enqueue_retire(q, value, RetireItem{kind, slot, gen});
}

void free_texture_view_after(Device dev, TextureView view, Submission s) {
    desc_free_after_accept(reinterpret_cast<DeviceImpl*>(dev), view, kDescTypeSampled,
                           RetireKind::SampledSlot, s);
}

void free_rw_texture_view_after(Device dev, TextureView view, Submission s) {
    desc_free_after_accept(reinterpret_cast<DeviceImpl*>(dev), view, kDescTypeStorage,
                           RetireKind::StorageSlot, s);
}

void free_sampler_after(Device dev, SamplerId sampler, Submission s) {
    desc_free_after_accept(reinterpret_cast<DeviceImpl*>(dev), sampler, kDescTypeSampler,
                           RetireKind::SamplerSlot, s);
}

// --- Texture views (object-less, heap-descriptor-only) -----------------------------------

static VkImageViewCreateInfo make_view_info(const TextureImpl& texture, const TextureViewDesc& desc) {
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

// Allocates a sampled/storage descriptor slot, retaining the owner texture
// record (the descriptor references its image until the slot retires).
// Returns 0 on failure; the slot state and owner are rolled back if the
// native descriptor write fails.
static TextureView desc_allocate_image_view(DeviceImpl* d, RetireKind kind, uint64_t type,
                                            const TextureViewDesc& desc) {
    const char* why = nullptr;
    TextureImpl* owner = d->texture_pool.try_get_ex(handle_cast<TextureImpl>(desc.texture), &why);
    if (owner == nullptr) {
        log_fmt(d, LogLevel::Error, __LINE__, __FILE__,
                "create_texture_view: invalid texture handle 0x%016llx (%s)",
                (unsigned long long)desc.texture.h, why ? why : "rejected");
        return 0;
    }
    atomic_fetch_add(&owner->refs, 1);   // descriptor owns the texture until retirement
    mutex_lock(&d->desc_lock);
    uint32_t slot = desc_bitset(d, kind).set_leading_zero();
    if (slot == ~0u) {
        mutex_unlock(&d->desc_lock);
        atomic_fetch_add(&owner->refs, -1);
        IZ_LOG(d, LogLevel::Error, "Descriptor heap full");
        return 0;
    }
    auto& states = desc_state(d, kind);
    auto& gens   = desc_gen(d, kind);
    auto& owners = (kind == RetireKind::SampledSlot) ? d->sampled_owner : d->storage_owner;
    states[slot] = static_cast<uint8_t>(DeviceImpl::DescriptorSlotState::Allocated);
    const uint16_t gen = static_cast<uint16_t>(gens[slot] + 1);
    gens[slot] = gen;
    owners[slot] = owner;
    mutex_unlock(&d->desc_lock);

    auto view_info = make_view_info(*owner, desc);
    const bool written = (kind == RetireKind::SampledSlot)
                             ? write_sampled_descriptor(d, slot, view_info)
                             : write_storage_descriptor(d, slot, view_info);
    if (!written) {
        // Roll back the slot state and the owner reference exactly once.
        mutex_lock(&d->desc_lock);
        states[slot] = static_cast<uint8_t>(DeviceImpl::DescriptorSlotState::Free);
        desc_bitset(d, kind).clear_bit(slot);
        owners[slot] = nullptr;
        mutex_unlock(&d->desc_lock);
        atomic_fetch_add(&owner->refs, -1);
        IZ_LOG(d, LogLevel::Error, "create_texture_view: descriptor write failed");
        return 0;
    }
    return encode_desc_handle(slot, gen, type);
}

TextureView create_texture_view(Device dev, const TextureViewDesc& desc) {
    return desc_allocate_image_view(reinterpret_cast<DeviceImpl*>(dev), RetireKind::SampledSlot,
                                    kDescTypeSampled, desc);
}

TextureView create_rw_texture_view(Device dev, const TextureViewDesc& desc) {
    return desc_allocate_image_view(reinterpret_cast<DeviceImpl*>(dev), RetireKind::StorageSlot,
                                    kDescTypeStorage, desc);
}

SamplerId create_sampler(Device dev, const SamplerDesc& desc) {
    auto* d = reinterpret_cast<DeviceImpl*>(dev);
    mutex_lock(&d->desc_lock);
    uint32_t slot = d->sampler_bitset.set_leading_zero();
    if (slot == ~0u) {
        mutex_unlock(&d->desc_lock);
        IZ_LOG(d, LogLevel::Error, "Sampler heap full");
        return 0;
    }
    d->sampler_state[slot] = static_cast<uint8_t>(DeviceImpl::DescriptorSlotState::Allocated);
    const uint16_t gen = static_cast<uint16_t>(d->sampler_gen[slot] + 1);
    d->sampler_gen[slot] = gen;
    mutex_unlock(&d->desc_lock);

    const VkSamplerCreateInfo sampler_info{
        .sType            = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .pNext            = nullptr,
        .flags            = 0,
        .magFilter        = bridge(desc.mag_filter),
        .minFilter        = bridge(desc.min_filter),
        .mipmapMode       = bridge_mip_mode(desc.mip_filter),
        .addressModeU     = bridge(desc.address),
        .addressModeV     = bridge(desc.address),
        .addressModeW     = bridge(desc.address),
        .mipLodBias       = desc.mip_lod_bias,
        .anisotropyEnable = (desc.max_anisotropy != 1.0f),
        .maxAnisotropy    = desc.max_anisotropy,
        .compareEnable    = VK_FALSE,
        .compareOp        = VK_COMPARE_OP_NEVER,
        .minLod           = 0,
        .maxLod           = VK_LOD_CLAMP_NONE,
        .borderColor      = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
        .unnormalizedCoordinates = (desc.coord == SamplerCoords::Pixel),
    };
    if (!write_sampler_descriptor(d, slot, sampler_info)) {
        mutex_lock(&d->desc_lock);
        d->sampler_state[slot] = static_cast<uint8_t>(DeviceImpl::DescriptorSlotState::Free);
        d->sampler_bitset.clear_bit(slot);
        mutex_unlock(&d->desc_lock);
        IZ_LOG(d, LogLevel::Error, "create_sampler: descriptor write failed");
        return 0;
    }
    return encode_desc_handle(slot, gen, kDescTypeSampler);
}

// --- Free with deferred recycling (unified queue retirement) ---------------------

void free_texture_view(Device dev, TextureView view) {
    auto* d = reinterpret_cast<DeviceImpl*>(dev);
    uint32_t slot = 0;
    uint16_t gen  = 0;
    if (!desc_free_accept(d, view, kDescTypeSampled, RetireKind::SampledSlot, &slot, &gen)) { return; }
    desc_free_enqueue(d, RetireKind::SampledSlot, slot, gen);
}

void free_rw_texture_view(Device dev, TextureView view) {
    auto* d = reinterpret_cast<DeviceImpl*>(dev);
    uint32_t slot = 0;
    uint16_t gen  = 0;
    if (!desc_free_accept(d, view, kDescTypeStorage, RetireKind::StorageSlot, &slot, &gen)) { return; }
    desc_free_enqueue(d, RetireKind::StorageSlot, slot, gen);
}

void free_sampler(Device dev, SamplerId sampler) {
    auto* d = reinterpret_cast<DeviceImpl*>(dev);
    uint32_t slot = 0;
    uint16_t gen  = 0;
    if (!desc_free_accept(d, sampler, kDescTypeSampler, RetireKind::SamplerSlot, &slot, &gen)) { return; }
    desc_free_enqueue(d, RetireKind::SamplerSlot, slot, gen);
}

}  // namespace gpu
