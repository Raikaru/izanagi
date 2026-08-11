// profile.cpp — fills a VulkanProfileFeatures snapshot from a physical
// device's actual feature set and limits (vkGetPhysicalDeviceFeatures2 +
// properties). The evaluation itself lives in common/profile_report.cpp so it
// is unit-testable without a device.

#include <algorithm>

#include "internal.h"

namespace gpu {

VulkanProfileFeatures query_vulkan_profile_features(DeviceImpl* d) {
    VulkanProfileFeatures f;

    // Descriptor-indexing booleans come consistently from the Vulkan 1.2
    // core struct (the same struct the native backend enables features on).
    VkPhysicalDeviceVulkan13Features vulkan13{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
        .pNext = nullptr,
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

    // The compatibility profile binds one update-after-bind set holding the
    // global descriptor arrays, so the usable capacity is the update-after-
    // bind set ceiling AND the per-stage ceiling (a runtime array's
    // maxDescriptorCount is capped by both). The exact limit is the minimum.
    VkPhysicalDeviceDescriptorIndexingProperties indexing_props{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_PROPERTIES,
        .pNext = nullptr,
    };
    VkPhysicalDeviceProperties2 props2{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        .pNext = &indexing_props,
    };
    vkGetPhysicalDeviceProperties2(d->physical_device, &props2);

    f.api_version = props2.properties.apiVersion;

    f.buffer_device_address = vulkan12.bufferDeviceAddress == VK_TRUE;
    f.shader_int64          = features2.features.shaderInt64 == VK_TRUE;
    f.scalar_block_layout   = vulkan12.scalarBlockLayout == VK_TRUE;

    f.sampled_image_nonuniform_indexing = vulkan12.shaderSampledImageArrayNonUniformIndexing == VK_TRUE;
    f.storage_image_nonuniform_indexing = vulkan12.shaderStorageImageArrayNonUniformIndexing == VK_TRUE;
    // No separate sampler non-uniform/update-after-bind feature exists in the
    // Vulkan spec; sampler arrays follow the sampled-image features.
    f.sampler_array_indexing = vulkan12.shaderSampledImageArrayNonUniformIndexing == VK_TRUE &&
                               vulkan12.descriptorBindingSampledImageUpdateAfterBind == VK_TRUE;

    f.sampled_image_update_after_bind = vulkan12.descriptorBindingSampledImageUpdateAfterBind == VK_TRUE;
    f.storage_image_update_after_bind = vulkan12.descriptorBindingStorageImageUpdateAfterBind == VK_TRUE;
    f.descriptor_binding_partially_bound               = vulkan12.descriptorBindingPartiallyBound == VK_TRUE;
    f.runtime_descriptor_array                         = vulkan12.runtimeDescriptorArray == VK_TRUE;
    f.descriptor_binding_update_unused_while_pending   = vulkan12.descriptorBindingUpdateUnusedWhilePending == VK_TRUE;

    f.draw_indirect_count = vulkan12.drawIndirectCount == VK_TRUE;
    f.timeline_semaphore  = vulkan12.timelineSemaphore == VK_TRUE;

    f.dynamic_rendering = vulkan13.dynamicRendering == VK_TRUE;
    f.synchronization2  = vulkan13.synchronization2 == VK_TRUE;

    // Per-type ceilings: the update-after-bind variants (min of the per-stage
    // and per-set limits for each type). The shared combined budget is
    // carried separately — the evaluator enforces that the arrays TOGETHER
    // fit min(maxPerStageUpdateAfterBindResources,
    //          maxUpdateAfterBindDescriptorsInAllPools).
    f.max_sampled_descriptors = std::min(indexing_props.maxPerStageDescriptorUpdateAfterBindSampledImages,
                                         indexing_props.maxDescriptorSetUpdateAfterBindSampledImages);
    f.max_storage_descriptors = std::min(indexing_props.maxPerStageDescriptorUpdateAfterBindStorageImages,
                                         indexing_props.maxDescriptorSetUpdateAfterBindStorageImages);
    f.max_samplers            = std::min(indexing_props.maxPerStageDescriptorUpdateAfterBindSamplers,
                                         indexing_props.maxDescriptorSetUpdateAfterBindSamplers);
    f.combined_descriptor_budget = std::min(indexing_props.maxPerStageUpdateAfterBindResources,
                                            indexing_props.maxUpdateAfterBindDescriptorsInAllPools);

    return f;
}

}  // namespace gpu
