#pragma once
// Vulkan capability-profile evaluation — pure logic, no Vulkan types.
//
// A compatibility (Bindless) profile must be decided by exact feature bits
// and limits, never by Vulkan version or GPU generation name. This module
// owns that decision: it takes a plain snapshot of the features/limits a
// physical device actually supports and answers "does the IZANAGI_VK_BINDLESS_1
// profile hold?" with a complete, human-readable missing-requirement list.
//
// The snapshot is injectable, so the evaluator is unit-testable without a
// device (negative tests construct snapshots missing each requirement).
// The Vulkan backend fills a snapshot from vkGetPhysicalDeviceFeatures2
// (see src/vk/profile.cpp).

#include <cstdint>

#include "izanagi/gpu.h"

namespace gpu {

// --- Bindless-profile requirements (mandatory unless noted) -------------------
// Real pointers:      bufferDeviceAddress (+ shader pointer support implied
//                     by Vulkan 1.2+ drivers that expose it).
// Pointer ABI:        shaderInt64 (GpuPtr/handles are 64-bit), scalarBlockLayout.
// Global indexing:    non-uniform array indexing for sampled/storage/sampler.
// Descriptor model:   update-after-bind + partially bound + runtime arrays
//                     (the strongly preferred set; snapshot path substitutes
//                     for update-unused-while-pending, which stays optional).
// Mandatory commands: drawIndirectCount (public multi-draw-indirect count API).
// Lifetime:           native timeline semaphores (public Submission model).
//
// Optional (recorded, never gating): dynamic rendering (else private render
// pass), synchronization2 (else legacy barriers), update-unused-while-pending
// (else private descriptor-set snapshots).

enum class VulkanProfileRequirement : uint8_t {
    BufferDeviceAddress,
    ShaderInt64,
    ScalarBlockLayout,
    SampledNonUniformIndexing,
    StorageNonUniformIndexing,
    SamplerArrayIndexing,
    SampledUpdateAfterBind,
    StorageUpdateAfterBind,
    PartiallyBound,
    RuntimeDescriptorArray,
    DrawIndirectCount,
    TimelineSemaphore,
    SampledImageCapacity,
    StorageImageCapacity,
    SamplerCapacity,
    CombinedDescriptorBudget,
    ValidCount,
};

const char* vulkan_requirement_name(VulkanProfileRequirement r);

// Floor capacities for the compatibility profile's global descriptor arrays.
// A smaller heap is acceptable per the philosophy (still one persistent
// GPU-indexed namespace), but these are the minimum usable sizes.
inline constexpr uint32_t kMinBindlessSampledImages = 1024;
inline constexpr uint32_t kMinBindlessStorageImages = 1024;
inline constexpr uint32_t kMinBindlessSamplers     = 256;

// Plain snapshot of what a physical device supports. api_version is reported
// for context only — it never gates the decision.
struct VulkanProfileFeatures {
    uint32_t api_version = 0;   // raw VK_MAKE_API_VERSION value (0 = unknown)

    // Core pointer/ABI features
    bool buffer_device_address = false;
    bool shader_int64          = false;
    bool scalar_block_layout   = false;

    // Non-uniform array indexing (global resource namespace). The Vulkan
    // spec has no separate sampler non-uniform-indexing/update-after-bind
    // feature (VkPhysicalDeviceDescriptorIndexingFeatures has no sampler
    // members); sampler arrays follow the sampled-image features. The
    // snapshot's sampler_array_indexing is derived from those by the fill
    // layer (src/vk/profile.cpp).
    bool sampled_image_nonuniform_indexing = false;
    bool storage_image_nonuniform_indexing = false;
    bool sampler_array_indexing            = false;

    // Descriptor indexing / update-after-bind
    bool sampled_image_update_after_bind   = false;
    bool storage_image_update_after_bind   = false;
    bool descriptor_binding_partially_bound                = false;
    bool runtime_descriptor_array                          = false;
    bool descriptor_binding_update_unused_while_pending    = false;

    // Mandatory public-command features
    bool draw_indirect_count = false;

    // Lifetime
    bool timeline_semaphore = false;

    // Optional rendering/sync conveniences (recorded, not required)
    bool dynamic_rendering = false;
    bool synchronization2  = false;

    // Per-type descriptor-array ceilings: the update-after-bind variants —
    // min(per-stage UAB, set UAB) for each type.
    uint32_t max_sampled_descriptors = 0;
    uint32_t max_storage_descriptors = 0;
    uint32_t max_samplers            = 0;
    // Shared combined budget for the one global update-after-bind set/pool:
    // min(maxPerStageUpdateAfterBindResources,
    //     maxUpdateAfterBindDescriptorsInAllPools). The three arrays TOGETHER
    // must fit it (the evaluator checks floors' sum against it).
    uint32_t combined_descriptor_budget = 0;
};

struct VulkanProfileReport {
    bool      supported     = false;
    uint32_t  api_version   = 0;

    // Selected fallbacks (echo of the feature snapshot)
    bool dynamic_rendering = false;
    bool synchronization2  = false;
    // 1 = direct update-after-bind path; 2+ = private descriptor-set snapshot
    // path (used when update-unused-while-pending is absent).
    uint32_t descriptor_snapshots = 0;

    uint32_t sampled_image_capacity = 0;   // clamped to the profile floor
    uint32_t storage_image_capacity = 0;
    uint32_t sampler_capacity       = 0;
    uint32_t combined_descriptor_budget = 0;   // echoed from the snapshot

    uint32_t missing_count = 0;
    VulkanProfileRequirement missing[16];

    // True when the given requirement is listed as missing.
    bool missing_has(VulkanProfileRequirement r) const {
        for (uint32_t i = 0; i < missing_count; ++i) {
            if (missing[i] == r) { return true; }
        }
        return false;
    }
};

// Evaluates the IZANAGI_VK_BINDLESS_1 requirements against a feature snapshot.
// Pure: no device, no allocation (fixed-size arrays), deterministic — safe for
// common/container tests.
VulkanProfileReport evaluate_vulkan_bindless_profile(const VulkanProfileFeatures& f);

}  // namespace gpu
