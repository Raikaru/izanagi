#include "common/profile_report.h"

namespace gpu {

const char* vulkan_requirement_name(VulkanProfileRequirement r) {
    switch (r) {
        case VulkanProfileRequirement::BufferDeviceAddress:
            return "buffer_device_address";
        case VulkanProfileRequirement::ShaderInt64:
            return "shader_int64";
        case VulkanProfileRequirement::ScalarBlockLayout:
            return "scalar_block_layout";
        case VulkanProfileRequirement::SampledNonUniformIndexing:
            return "sampled_image_nonuniform_indexing";
        case VulkanProfileRequirement::StorageNonUniformIndexing:
            return "storage_image_nonuniform_indexing";
        case VulkanProfileRequirement::SamplerArrayIndexing:
            return "sampler_array_indexing";
        case VulkanProfileRequirement::SampledUpdateAfterBind:
            return "sampled_image_update_after_bind";
        case VulkanProfileRequirement::StorageUpdateAfterBind:
            return "storage_image_update_after_bind";

        case VulkanProfileRequirement::PartiallyBound:
            return "descriptor_binding_partially_bound";
        case VulkanProfileRequirement::RuntimeDescriptorArray:
            return "runtime_descriptor_array";
        case VulkanProfileRequirement::DrawIndirectCount:
            return "draw_indirect_count";
        case VulkanProfileRequirement::TimelineSemaphore:
            return "timeline_semaphore";
        case VulkanProfileRequirement::SampledImageCapacity:
            return "sampled_image_capacity";
        case VulkanProfileRequirement::StorageImageCapacity:
            return "storage_image_capacity";
        case VulkanProfileRequirement::SamplerCapacity:
            return "sampler_capacity";
        default:
            return "unknown";
    }
}

VulkanProfileReport evaluate_vulkan_bindless_profile(const VulkanProfileFeatures& f) {
    VulkanProfileReport r;
    r.api_version          = f.api_version;
    r.dynamic_rendering    = f.dynamic_rendering;
    r.synchronization2     = f.synchronization2;
    r.descriptor_snapshots = f.descriptor_binding_update_unused_while_pending ? 1u : 2u;

    // Capacities: clamp to the profile floor; insufficient capacity is a
    // rejection, but the (possibly larger) device limit is reported.
    r.sampled_image_capacity = f.max_sampled_descriptors < kMinBindlessSampledImages
                                   ? f.max_sampled_descriptors
                                   : kMinBindlessSampledImages;
    r.storage_image_capacity = f.max_storage_descriptors < kMinBindlessStorageImages
                                   ? f.max_storage_descriptors
                                   : kMinBindlessStorageImages;
    r.sampler_capacity       = f.max_samplers < kMinBindlessSamplers ? f.max_samplers
                                                                     : kMinBindlessSamplers;

    auto add_missing = [&r](bool ok, VulkanProfileRequirement req) {
        if (!ok && r.missing_count < 16) { r.missing[r.missing_count++] = req; }
    };

    // Real pointers + shader ABI
    add_missing(f.buffer_device_address, VulkanProfileRequirement::BufferDeviceAddress);
    add_missing(f.shader_int64, VulkanProfileRequirement::ShaderInt64);
    add_missing(f.scalar_block_layout, VulkanProfileRequirement::ScalarBlockLayout);

    // Global non-uniform resource indexing
    add_missing(f.sampled_image_nonuniform_indexing, VulkanProfileRequirement::SampledNonUniformIndexing);
    add_missing(f.storage_image_nonuniform_indexing, VulkanProfileRequirement::StorageNonUniformIndexing);
    // Sampler arrays follow the sampled-image features (no separate Vulkan
    // sampler feature exists); the snapshot derives sampler_array_indexing.
    add_missing(f.sampler_array_indexing, VulkanProfileRequirement::SamplerArrayIndexing);

    // Descriptor indexing / update-after-bind model
    add_missing(f.sampled_image_update_after_bind, VulkanProfileRequirement::SampledUpdateAfterBind);
    add_missing(f.storage_image_update_after_bind, VulkanProfileRequirement::StorageUpdateAfterBind);

    add_missing(f.descriptor_binding_partially_bound, VulkanProfileRequirement::PartiallyBound);
    add_missing(f.runtime_descriptor_array, VulkanProfileRequirement::RuntimeDescriptorArray);

    // Mandatory public-command feature (multi-draw indirect count)
    add_missing(f.draw_indirect_count, VulkanProfileRequirement::DrawIndirectCount);

    // Public Submission model needs native timeline semaphores
    add_missing(f.timeline_semaphore, VulkanProfileRequirement::TimelineSemaphore);

    // Capacity floors
    add_missing(r.sampled_image_capacity >= kMinBindlessSampledImages, VulkanProfileRequirement::SampledImageCapacity);
    add_missing(r.storage_image_capacity >= kMinBindlessStorageImages, VulkanProfileRequirement::StorageImageCapacity);
    add_missing(r.sampler_capacity >= kMinBindlessSamplers, VulkanProfileRequirement::SamplerCapacity);

    r.supported = r.missing_count == 0;
    return r;
}

}  // namespace gpu
