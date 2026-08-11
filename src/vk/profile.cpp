// profile.cpp — fills a VulkanProfileFeatures snapshot from a physical
// device's actual feature set and limits (vkGetPhysicalDeviceFeatures2 +
// properties). The evaluation itself lives in common/profile_report.cpp so it
// is unit-testable without a device.

#include "internal.h"

namespace gpu {

VulkanProfileFeatures query_vulkan_profile_features(DeviceImpl* d) {
    VulkanProfileFeatures f;

    VkPhysicalDeviceDescriptorIndexingFeatures indexing{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES,
        .pNext = nullptr,
    };
    VkPhysicalDeviceVulkan13Features vulkan13{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
        .pNext = &indexing,
    };
    VkPhysicalDeviceVulkan12Features vulkan12{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        .pNext = &vulkan13,
    };
    VkPhysicalDeviceFeatures2 features2{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext = &vulkan12,
    };
    vkGetPhysicalDeviceFeatures2(d->physical_device, &features2);

    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(d->physical_device, &props);
    f.api_version = props.apiVersion;

    f.buffer_device_address = vulkan12.bufferDeviceAddress == VK_TRUE;
    f.shader_int64          = features2.features.shaderInt64 == VK_TRUE;
    f.scalar_block_layout   = vulkan12.scalarBlockLayout == VK_TRUE;

    f.sampled_image_nonuniform_indexing = indexing.shaderSampledImageArrayNonUniformIndexing == VK_TRUE;
    f.storage_image_nonuniform_indexing = indexing.shaderStorageImageArrayNonUniformIndexing == VK_TRUE;
    // No separate sampler non-uniform/update-after-bind feature exists in the
    // Vulkan spec; sampler arrays follow the sampled-image features.
    f.sampler_array_indexing = indexing.shaderSampledImageArrayNonUniformIndexing == VK_TRUE &&
                               indexing.descriptorBindingSampledImageUpdateAfterBind == VK_TRUE;

    f.sampled_image_update_after_bind = indexing.descriptorBindingSampledImageUpdateAfterBind == VK_TRUE;
    f.storage_image_update_after_bind = indexing.descriptorBindingStorageImageUpdateAfterBind == VK_TRUE;
    f.descriptor_binding_partially_bound               = indexing.descriptorBindingPartiallyBound == VK_TRUE;
    f.runtime_descriptor_array                         = indexing.runtimeDescriptorArray == VK_TRUE;
    f.descriptor_binding_update_unused_while_pending   = indexing.descriptorBindingUpdateUnusedWhilePending == VK_TRUE;

    f.draw_indirect_count = vulkan12.drawIndirectCount == VK_TRUE;
    f.timeline_semaphore  = vulkan12.timelineSemaphore == VK_TRUE;

    f.dynamic_rendering = vulkan13.dynamicRendering == VK_TRUE;
    f.synchronization2  = vulkan13.synchronization2 == VK_TRUE;

    f.max_sampled_descriptors = props.limits.maxDescriptorSetSampledImages;
    f.max_storage_descriptors = props.limits.maxDescriptorSetStorageImages;
    f.max_samplers            = props.limits.maxDescriptorSetSamplers;

    return f;
}

}  // namespace gpu
