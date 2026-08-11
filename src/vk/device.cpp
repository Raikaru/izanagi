// device.cpp — instance/device creation, features, descriptor heap, memory, queues.

#include "internal.h"

#include <cstdio>
#include <cstring>

// Template instantiations for the public header's extern declarations
namespace gpu {
template class Span<const char>;
template class Span<uint8_t>;
template class Span<const ColorTarget>;
template class Span<const RenderAttachment>;
template class Span<const Format>;
template class Span<const PresentMode>;
template class Span<const CommandBuffer>;
template class Span<const SemaphoreInfo>;
template class Span<const SpecializationConstant>;

// --- Required device extensions ---------------------------------------------------
static const char* kRequiredDeviceExtensions[] = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME,
    VK_KHR_SHADER_UNTYPED_POINTERS_EXTENSION_NAME,
    VK_KHR_UNIFIED_IMAGE_LAYOUTS_EXTENSION_NAME,
};
static constexpr size_t kRequiredDeviceExtensionsCount =
    sizeof(kRequiredDeviceExtensions) / sizeof(kRequiredDeviceExtensions[0]);

// --- Logging -------------------------------------------------------------------------
static void null_log_callback(LogLevel, Span<const char>, uint32_t, Span<const char>, void*) {}

void log_impl(DeviceImpl* d, LogLevel lvl, Span<const char> msg, uint32_t line, Span<const char> file) {
    if (d->log_callback && lvl <= d->log_level) {
        d->log_callback(lvl, msg, line, file, d->log_userdata);
    }
}

void log_vk_impl(DeviceImpl* d, VkResult res, Span<const char> msg, uint32_t line, Span<const char> file) {
    if (d->log_callback) {
        char buf[512];
        auto result_str = string_from_result(res);
        int  len        = snprintf(buf, sizeof(buf), "%.*s [%.*s]",
                                    (int)msg.size(), msg.data(),
                                    (int)result_str.size(), result_str.data());
        d->log_callback(LogLevel::Error,
                        Span<const char>(buf, len > 0 ? (size_t)len : 0),
                        line, file, d->log_userdata);
    }
}

// --- Thread-local arena ----------------------------------------------------------------
Arena* get_thread_local_arena(DeviceImpl* d) {
    auto state = reinterpret_cast<ThreadLocalState*>(tls_get_data(d->thread_local_key));
    if (state == nullptr) {
        auto tls_block = d->allocator.alloc(sizeof(ThreadLocalState));
        if (tls_block.ptr == nullptr) {
            IZ_LOG(d, LogLevel::Error, "Allocator out of memory");
            return nullptr;
        }
        state = ::new (tls_block.ptr) ThreadLocalState(d->allocator, d->log_callback, d->log_userdata);
        tls_set_data(d->thread_local_key, state);
    }
    return &state->arena;
}

// --- Buffer/ptr map ----------------------------------------------------------------------
// Finds the live allocation whose user-visible range contains `ptr` (largest
// base <= ptr, then bounds check against the user size). Rejects pointers in
// gaps between allocations and past the end. Returns the buffer record and the
// VkBuffer-relative offset (64-bit, includes the interior alignment).
static Buffer* find_buffer_for_ptr(DeviceImpl* d, GpuPtr ptr, VkDeviceSize* out_offset) {
    rwlock_lock_read(&d->ptr_map_lock);
    uint32_t lo = 0, hi = d->ptr_map.size();
    uint32_t best = ~0u;
    while (lo < hi) {
        uint32_t mid = (lo + hi) / 2;
        if (d->ptr_map[mid].ptr <= ptr) {
            best = mid;
            lo   = mid + 1;
        } else {
            hi = mid;
        }
    }
    if (best == ~0u) {
        rwlock_unlock_read(&d->ptr_map_lock);
        return nullptr;
    }
    auto& entry = d->ptr_map[best];
    auto& buf   = d->buffer_pool[entry.buffer];
    // Overflow-safe bounds check (exclusive end).
    if (ptr - buf.user_address >= buf.user_size) {
        rwlock_unlock_read(&d->ptr_map_lock);
        return nullptr;
    }
    *out_offset = buf.user_offset + (ptr - buf.user_address);
    rwlock_unlock_read(&d->ptr_map_lock);
    return &buf;
}

BufferAndOffset buffer_and_offset_from_ptr(DeviceImpl* d, GpuPtr ptr) {
    VkDeviceSize offset = 0;
    Buffer* buf = find_buffer_for_ptr(d, ptr, &offset);
    if (buf == nullptr) {
        return {.buffer = VK_NULL_HANDLE, .offset = 0, .alloc = VK_NULL_HANDLE};
    }
    return {
        .buffer = buf->vk_buffer,
        .offset = offset,
        .alloc  = buf->vk_allocation,
    };
}

// --- Instance creation -------------------------------------------------------------------
static VkResult create_instance(DeviceImpl* d, const DeviceDesc& desc) {
    VkResult result = volkInitialize();
    if (result != VK_SUCCESS) { return result; }

    VkApplicationInfo app_info{
        .sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pNext              = nullptr,
        .pApplicationName   = "izanagi",
        .applicationVersion = 0,
        .pEngineName        = "izanagi",
        .engineVersion      = 0,
        .apiVersion         = VK_API_VERSION_1_4,
    };

    const char* instance_extensions[] = {
        VK_KHR_SURFACE_EXTENSION_NAME,
        VK_KHR_WIN32_SURFACE_EXTENSION_NAME,
    };
    constexpr uint32_t num_instance_extensions =
        sizeof(instance_extensions) / sizeof(instance_extensions[0]);

    // Optional validation layer
    const char* layers[1]     = {"VK_LAYER_KHRONOS_validation"};
    uint32_t    layer_count   = 0;
    const char* ext_ptrs[8];
    uint32_t    ext_count     = 0;
    for (uint32_t i = 0; i < num_instance_extensions; ++i) { ext_ptrs[ext_count++] = instance_extensions[i]; }

    if (desc.enable_validation) {
        uint32_t avail_layer_count = 0;
        vkEnumerateInstanceLayerProperties(&avail_layer_count, nullptr);
        if (avail_layer_count > 0) {
            VkLayerProperties avail[32];
            vkEnumerateInstanceLayerProperties(&avail_layer_count, avail);
            for (uint32_t i = 0; i < avail_layer_count; ++i) {
                if (strcmp(avail[i].layerName, "VK_LAYER_KHRONOS_validation") == 0) {
                    layer_count = 1;
                    ext_ptrs[ext_count++] = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
                    break;
                }
            }
        }
    }

    const VkInstanceCreateInfo instance_info{
        .sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pNext                   = nullptr,
        .flags                   = 0,
        .pApplicationInfo        = &app_info,
        .enabledLayerCount       = layer_count,
        .ppEnabledLayerNames     = layer_count > 0 ? layers : nullptr,
        .enabledExtensionCount   = ext_count,
        .ppEnabledExtensionNames = ext_ptrs,
    };

    result = vkCreateInstance(&instance_info, nullptr, &d->instance);
    if (result != VK_SUCCESS) { return result; }

    volkLoadInstance(d->instance);
    return VK_SUCCESS;
}

// --- Physical device selection ------------------------------------------------------------
static VkResult select_physical_device(DeviceImpl* d, GpuPreference preference) {
    uint32_t device_count = 0;
    VkResult result = vkEnumeratePhysicalDevices(d->instance, &device_count, nullptr);
    if (result != VK_SUCCESS || device_count == 0) {
        return result != VK_SUCCESS ? result : VK_ERROR_INITIALIZATION_FAILED;
    }

    VkPhysicalDevice devices[16];
    if (device_count > 16) device_count = 16;
    result = vkEnumeratePhysicalDevices(d->instance, &device_count, devices);
    if (result != VK_SUCCESS) { return result; }

    VkPhysicalDevice best         = VK_NULL_HANDLE;
    uint32_t         best_family  = 0;
    bool             prefer_integrated = (preference == GpuPreference::Integrated);

    for (uint32_t i = 0; i < device_count; ++i) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(devices[i], &props);

        // Require Vulkan 1.4
        if (props.apiVersion < VK_API_VERSION_1_4) { continue; }

        // Find graphics+compute queue family
        uint32_t family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(devices[i], &family_count, nullptr);
        VkQueueFamilyProperties families[16];
        if (family_count > 16) family_count = 16;
        vkGetPhysicalDeviceQueueFamilyProperties(devices[i], &family_count, families);

        int32_t graphics_family = -1;
        for (uint32_t f = 0; f < family_count; ++f) {
            if ((families[f].queueFlags & VK_QUEUE_GRAPHICS_BIT) &&
                (families[f].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
                graphics_family = f;
                break;
            }
        }
        if (graphics_family < 0) { continue; }

        // Check required extensions
        uint32_t ext_count = 0;
        vkEnumerateDeviceExtensionProperties(devices[i], nullptr, &ext_count, nullptr);
        Vector<VkExtensionProperties> exts(d->allocator, {}, ext_count);
        vkEnumerateDeviceExtensionProperties(devices[i], nullptr, &ext_count, exts.data());

        bool all_supported = true;
        for (size_t req = 0; req < kRequiredDeviceExtensionsCount; ++req) {
            bool found = false;
            for (uint32_t e = 0; e < ext_count; ++e) {
                if (strcmp(exts[e].extensionName, kRequiredDeviceExtensions[req]) == 0) {
                    found = true;
                    break;
                }
            }
            if (!found) { all_supported = false; break; }
        }
        if (!all_supported) { continue; }

        // Select based on preference
        if (best == VK_NULL_HANDLE) {
            best        = devices[i];
            best_family = graphics_family;
        } else {
            const bool is_integrated = props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
            const bool is_dedicated  = props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
            if ((prefer_integrated && is_integrated) || (!prefer_integrated && is_dedicated)) {
                best        = devices[i];
                best_family = graphics_family;
            }
        }
    }

    if (best == VK_NULL_HANDLE) { return VK_ERROR_INITIALIZATION_FAILED; }
    d->physical_device      = best;
    d->graphics_queue_family = best_family;
    return VK_SUCCESS;
}

// --- Logical device creation ---------------------------------------------------------------
static VkResult create_logical_device(DeviceImpl* d) {
    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_create_info{
        .sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .pNext            = nullptr,
        .flags            = 0,
        .queueFamilyIndex = d->graphics_queue_family,
        .queueCount       = 1,
        .pQueuePriorities = &queue_priority,
    };

    // Feature chain — query first, then enable
    VkPhysicalDeviceDescriptorHeapFeaturesEXT desc_heap_features{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_FEATURES_EXT,
    };
    VkPhysicalDeviceShaderUntypedPointersFeaturesKHR untyped_ptr_features{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_UNTYPED_POINTERS_FEATURES_KHR,
        .pNext = &desc_heap_features,
    };
    VkPhysicalDeviceUnifiedImageLayoutsFeaturesKHR unified_layouts_features{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_UNIFIED_IMAGE_LAYOUTS_FEATURES_KHR,
        .pNext = &untyped_ptr_features,
    };
    VkPhysicalDeviceVulkan14Features vulkan14_features{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES,
        .pNext = &unified_layouts_features,
    };
    VkPhysicalDeviceVulkan13Features vulkan13_features{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
        .pNext = &vulkan14_features,
    };
    VkPhysicalDeviceVulkan12Features vulkan12_features{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        .pNext = &vulkan13_features,
    };
    VkPhysicalDeviceVulkan11Features vulkan11_features{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
        .pNext = &vulkan12_features,
    };
    VkPhysicalDeviceFeatures2 device_features{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext = &vulkan11_features,
    };

    // Query supported features
    vkGetPhysicalDeviceFeatures2(d->physical_device, &device_features);

    // Enable only what we need — dual-source blending is optional, so query
    // it first and enable only when supported (pipeline creation rejects
    // Src1* factors on devices without it).
    device_features.features.samplerAnisotropy = VK_TRUE;
    device_features.features.shaderInt16       = VK_TRUE;
    device_features.features.multiDrawIndirect = VK_TRUE;
    d->dual_src_blend = device_features.features.dualSrcBlend == VK_TRUE;
    if (d->dual_src_blend) {
        device_features.features.dualSrcBlend = VK_TRUE;
    }

    vulkan11_features.shaderDrawParameters = VK_TRUE;

    vulkan12_features.timelineSemaphore       = VK_TRUE;
    vulkan12_features.bufferDeviceAddress     = VK_TRUE;
    vulkan12_features.shaderInt8              = VK_TRUE;
    vulkan12_features.shaderFloat16           = VK_TRUE;
    vulkan12_features.scalarBlockLayout       = VK_TRUE;
    vulkan12_features.drawIndirectCount       = VK_TRUE;
    vulkan12_features.storagePushConstant8    = VK_TRUE;
    // Clear descriptor-indexing bits — descriptor_heap replaces them
    vulkan12_features.descriptorIndexing      = VK_FALSE;
    vulkan12_features.runtimeDescriptorArray  = VK_FALSE;
    vulkan12_features.descriptorBindingPartiallyBound = VK_FALSE;
    vulkan12_features.descriptorBindingVariableDescriptorCount = VK_FALSE;
    vulkan12_features.descriptorBindingSampledImageUpdateAfterBind = VK_FALSE;
    vulkan12_features.descriptorBindingStorageImageUpdateAfterBind = VK_FALSE;
    vulkan12_features.descriptorBindingStorageBufferUpdateAfterBind = VK_FALSE;
    vulkan12_features.descriptorBindingUniformBufferUpdateAfterBind = VK_FALSE;
    vulkan12_features.descriptorBindingUpdateUnusedWhilePending = VK_FALSE;

    vulkan13_features.dynamicRendering = VK_TRUE;
    vulkan13_features.synchronization2 = VK_TRUE;
    vulkan13_features.maintenance4     = VK_TRUE;
    // Optional: lets the compiler worker probe the cache with
    // FAIL_ON_PIPELINE_COMPILE_REQUIRED instead of compiling blind.
    d->pipeline_cache_control = vulkan13_features.pipelineCreationCacheControl == VK_TRUE;
    if (d->pipeline_cache_control) {
        vulkan13_features.pipelineCreationCacheControl = VK_TRUE;
    }

    vulkan14_features.maintenance5 = VK_TRUE;
    vulkan14_features.maintenance6 = VK_TRUE;

    desc_heap_features.descriptorHeap             = VK_TRUE;
    untyped_ptr_features.shaderUntypedPointers    = VK_TRUE;
    unified_layouts_features.unifiedImageLayouts  = VK_TRUE;

    const VkDeviceCreateInfo create_info{
        .sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext                   = &device_features,
        .flags                   = 0,
        .queueCreateInfoCount    = 1,
        .pQueueCreateInfos       = &queue_create_info,
        .enabledLayerCount       = 0,
        .ppEnabledLayerNames     = nullptr,
        .enabledExtensionCount   = kRequiredDeviceExtensionsCount,
        .ppEnabledExtensionNames = kRequiredDeviceExtensions,
        .pEnabledFeatures        = nullptr, // Using pNext chain instead
    };

    VkResult result = vkCreateDevice(d->physical_device, &create_info, nullptr, &d->device);
    if (result != VK_SUCCESS) { return result; }

    volkLoadDevice(d->device);

    // Verify critical features were enabled
    if (!desc_heap_features.descriptorHeap) {
        IZ_LOG(d, LogLevel::Error, "VK_EXT_descriptor_heap feature not enabled");
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }
    if (!untyped_ptr_features.shaderUntypedPointers) {
        IZ_LOG(d, LogLevel::Error, "VK_KHR_shader_untyped_pointers feature not enabled");
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }
    if (!unified_layouts_features.unifiedImageLayouts) {
        IZ_LOG(d, LogLevel::Error, "VK_KHR_unified_image_layouts feature not enabled");
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }

    return VK_SUCCESS;
}

// --- VMA allocator ------------------------------------------------------------------------
static VkResult create_vma_allocator(DeviceImpl* d) {
    VmaVulkanFunctions vulkan_functions{};
    vulkan_functions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    vulkan_functions.vkGetDeviceProcAddr   = vkGetDeviceProcAddr;

    VmaAllocatorCreateInfo create_info{
        .flags                       = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT |
                                        VMA_ALLOCATOR_CREATE_KHR_MAINTENANCE5_BIT,
        .physicalDevice              = d->physical_device,
        .device                      = d->device,
        .preferredLargeHeapBlockSize = 0,
        .pAllocationCallbacks        = nullptr,
        .pDeviceMemoryCallbacks      = nullptr,
        .pHeapSizeLimit              = nullptr,
        .pVulkanFunctions            = &vulkan_functions,
        .instance                    = d->instance,
        .vulkanApiVersion            = VK_API_VERSION_1_4,
        .pTypeExternalMemoryHandleTypes = nullptr,
    };

    return vmaCreateAllocator(&create_info, &d->vma);
}

// --- Descriptor heap creation ---------------------------------------------------------------
static VkResult create_descriptor_heap(DeviceImpl* d) {
    // Query heap properties
    VkPhysicalDeviceDescriptorHeapPropertiesEXT heap_props{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_PROPERTIES_EXT,
    };
    VkPhysicalDeviceProperties2 props2{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        .pNext = &heap_props,
    };
    vkGetPhysicalDeviceProperties2(d->physical_device, &props2);

    // Per-type descriptor sizes
    d->heap.sampler_descriptor_size = static_cast<uint32_t>(
        vkGetPhysicalDeviceDescriptorSizeEXT(d->physical_device, VK_DESCRIPTOR_TYPE_SAMPLER));
    d->heap.sampled_descriptor_size = static_cast<uint32_t>(
        vkGetPhysicalDeviceDescriptorSizeEXT(d->physical_device, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE));
    d->heap.storage_descriptor_size = static_cast<uint32_t>(
        vkGetPhysicalDeviceDescriptorSizeEXT(d->physical_device, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE));

    // Clamp capacities
    const uint32_t max_sampler_capacity = static_cast<uint32_t>(
        (heap_props.maxSamplerHeapSize - heap_props.minSamplerHeapReservedRange) /
        d->heap.sampler_descriptor_size);
    const uint32_t max_image_capacity = static_cast<uint32_t>(
        (heap_props.maxResourceHeapSize - heap_props.minResourceHeapReservedRange) /
        d->heap.sampled_descriptor_size);

    d->heap.sampler_capacity = kMaxSamplers < max_sampler_capacity ? kMaxSamplers : max_sampler_capacity;
    d->heap.sampled_capacity = kMaxSampledTextures < max_image_capacity ? kMaxSampledTextures : max_image_capacity;
    d->heap.storage_capacity = kMaxStorageTextures < max_image_capacity ? kMaxStorageTextures : max_image_capacity;

    if (d->heap.sampler_capacity < kMaxSamplers) {
        IZ_LOG(d, LogLevel::Warning, "Sampler heap capacity clamped by driver");
    }
    if (d->heap.sampled_capacity < kMaxSampledTextures) {
        IZ_LOG(d, LogLevel::Warning, "Sampled texture heap capacity clamped by driver");
    }

    // Heap buffer usage flags
    VkBufferUsageFlags2 heap_usage = VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT |
                                     VK_BUFFER_USAGE_2_DESCRIPTOR_HEAP_BIT_EXT;

    // --- Sampler heap ---
    {
        VkDeviceSize data_size = (VkDeviceSize)d->heap.sampler_capacity * d->heap.sampler_descriptor_size;
        VkDeviceSize total_size = data_size + heap_props.minSamplerHeapReservedRange;
        // Align up to heap alignment
        total_size = (total_size + heap_props.samplerHeapAlignment - 1) & ~(heap_props.samplerHeapAlignment - 1);

        VkBufferUsageFlags2CreateInfo usage_info{
            .sType = VK_STRUCTURE_TYPE_BUFFER_USAGE_FLAGS_2_CREATE_INFO,
            .usage = heap_usage,
        };
        VkBufferCreateInfo buffer_info{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = &usage_info,
            .size  = total_size,
        };

        VmaAllocationCreateInfo alloc_info{
            .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                     VMA_ALLOCATION_CREATE_MAPPED_BIT,
            .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        };

        VmaAllocationInfo alloc_result{};
        VkResult result = vmaCreateBuffer(d->vma, &buffer_info, &alloc_info,
                                           &d->heap.sampler_buffer, &d->heap.sampler_allocation,
                                           &alloc_result);
        if (result != VK_SUCCESS) { return result; }

        d->heap.sampler_host_ptr = alloc_result.pMappedData;

        VkBufferDeviceAddressInfo addr_info{
            .sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
            .buffer = d->heap.sampler_buffer,
        };
        d->heap.sampler_device_ptr = vkGetBufferDeviceAddress(d->device, &addr_info);

        d->heap.sampler_heap_size       = total_size;
        d->heap.sampler_reserved_offset = data_size;
        d->heap.sampler_reserved_size   = heap_props.minSamplerHeapReservedRange;
    }

    // --- Resource heap (sampled + storage in one buffer) ---
    {
        // Sampled region
        VkDeviceSize sampled_size = (VkDeviceSize)d->heap.sampled_capacity * d->heap.sampled_descriptor_size;
        // Storage region begins after sampled, aligned to storage descriptor size
        VkDeviceSize storage_offset = (sampled_size + d->heap.storage_descriptor_size - 1) &
                                      ~((VkDeviceSize)d->heap.storage_descriptor_size - 1);
        VkDeviceSize storage_size = (VkDeviceSize)d->heap.storage_capacity * d->heap.storage_descriptor_size;

        VkDeviceSize data_size = storage_offset + storage_size;
        VkDeviceSize total_size = data_size + heap_props.minResourceHeapReservedRange;
        total_size = (total_size + heap_props.resourceHeapAlignment - 1) & ~(heap_props.resourceHeapAlignment - 1);

        VkBufferUsageFlags2CreateInfo usage_info{
            .sType = VK_STRUCTURE_TYPE_BUFFER_USAGE_FLAGS_2_CREATE_INFO,
            .usage = heap_usage,
        };
        VkBufferCreateInfo buffer_info{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = &usage_info,
            .size  = total_size,
        };

        VmaAllocationCreateInfo alloc_info{
            .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                     VMA_ALLOCATION_CREATE_MAPPED_BIT,
            .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        };

        VmaAllocationInfo alloc_result{};
        VkResult result = vmaCreateBuffer(d->vma, &buffer_info, &alloc_info,
                                           &d->heap.resource_buffer, &d->heap.resource_allocation,
                                           &alloc_result);
        if (result != VK_SUCCESS) { return result; }

        d->heap.resource_host_ptr = alloc_result.pMappedData;

        VkBufferDeviceAddressInfo addr_info{
            .sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
            .buffer = d->heap.resource_buffer,
        };
        d->heap.resource_device_ptr = vkGetBufferDeviceAddress(d->device, &addr_info);

        d->heap.resource_heap_size       = total_size;
        d->heap.resource_reserved_offset = data_size;
        d->heap.resource_reserved_size   = heap_props.minResourceHeapReservedRange;
        d->heap.storage_region_offset    = storage_offset;
    }

    // Initialize free-slot bitsets
    d->sampled_bitset = TwoLevelBitset(d->allocator, d->heap.sampled_capacity);
    d->storage_bitset = TwoLevelBitset(d->allocator, d->heap.storage_capacity);
    d->sampler_bitset = TwoLevelBitset(d->allocator, d->heap.sampler_capacity);

    return VK_SUCCESS;
}

// --- Debug messenger -------------------------------------------------------------------------
static VkBool32 debug_messenger_callback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                          VkDebugUtilsMessageTypeFlagsEXT,
                                          const VkDebugUtilsMessengerCallbackDataEXT* data,
                                          void* userdata) {
    auto* d = reinterpret_cast<DeviceImpl*>(userdata);
    LogLevel lvl = LogLevel::Info;
    if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) { lvl = LogLevel::Error; }
    else if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) { lvl = LogLevel::Warning; }
    log_impl(d, lvl, Span<const char>(data->pMessage, strlen(data->pMessage)), 0, "vulkan"_sv);
    return VK_FALSE;
}

// --- Public API -------------------------------------------------------------------------------

Device create_device(const DeviceDesc& desc) {
    Allocator alloc = desc.alloc_callback ? Allocator(desc.alloc_callback, desc.alloc_userdata)
                                           : Allocator();
    auto blk = alloc.alloc(sizeof(DeviceImpl));
    if (blk.ptr == nullptr) { return nullptr; }

    auto* d = reinterpret_cast<DeviceImpl*>(blk.ptr);
    ::new (d) DeviceImpl();
    d->allocator     = alloc;
    d->log_callback  = desc.log_callback ? desc.log_callback : null_log_callback;
    d->log_userdata  = desc.log_userdata;
    d->log_level     = desc.log_level;
    d->enable_validation = desc.enable_validation;

    d->thread_local_key = tls_alloc([](void* data) {
        auto* state = reinterpret_cast<ThreadLocalState*>(data);
        state->~ThreadLocalState();
    });

    VkResult result = create_instance(d, desc);
    if (result != VK_SUCCESS) {
        log_vk_impl(d, result, "Failed to create Vulkan instance", __LINE__, "device.cpp"_sv);
        goto fail;
    }

    // Create surface if window handle provided
    if (desc.native_window_handle != 0) {
        VkWin32SurfaceCreateInfoKHR surface_info{
            .sType     = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR,
            .pNext     = nullptr,
            .flags     = 0,
            .hinstance = (HINSTANCE)desc.native_instance_handle,
            .hwnd      = (HWND)desc.native_window_handle,
        };
        result = vkCreateWin32SurfaceKHR(d->instance, &surface_info, nullptr, &d->surface.surface);
        if (result != VK_SUCCESS) {
            log_vk_impl(d, result, "Failed to create Win32 surface", __LINE__, "device.cpp"_sv);
            goto fail;
        }
    }

    result = select_physical_device(d, desc.gpu_preference);
    if (result != VK_SUCCESS) {
        log_vk_impl(d, result, "Failed to select physical device (need Vulkan 1.4 + descriptor_heap)", __LINE__, "device.cpp"_sv);
        goto fail;
    }

    result = create_logical_device(d);
    if (result != VK_SUCCESS) {
        log_vk_impl(d, result, "Failed to create logical device", __LINE__, "device.cpp"_sv);
        goto fail;
    }

    // Debug messenger
    if (desc.enable_validation) {
        VkDebugUtilsMessengerCreateInfoEXT debug_info{
            .sType           = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
            .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
            .messageType     = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
            .pfnUserCallback = debug_messenger_callback,
            .pUserData       = d,
        };
        if (vkCreateDebugUtilsMessengerEXT) {
            vkCreateDebugUtilsMessengerEXT(d->instance, &debug_info, nullptr, &d->debug_messenger);
        }
    }

    result = create_vma_allocator(d);
    if (result != VK_SUCCESS) {
        log_vk_impl(d, result, "Failed to create VMA allocator", __LINE__, "device.cpp"_sv);
        goto fail;
    }

    result = create_descriptor_heap(d);
    if (result != VK_SUCCESS) {
        log_vk_impl(d, result, "Failed to create descriptor heap", __LINE__, "device.cpp"_sv);
        goto fail;
    }

    // Persistent native pipeline cache (opt-in via DeviceDesc callbacks).
    {
        VkPhysicalDeviceProperties2 props2{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
            .pNext = nullptr,
        };
        VkPhysicalDeviceIDProperties id_props{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES,
            .pNext = nullptr,
        };
        props2.pNext = &id_props;
        vkGetPhysicalDeviceProperties2(d->physical_device, &props2);
        d->cache_identity.backend   = Backend::Vulkan;
        d->cache_identity.vendor_id = props2.properties.vendorID;
        d->cache_identity.device_id = props2.properties.deviceID;
        memcpy(d->cache_identity.driver_uuid, id_props.driverUUID,
               sizeof(d->cache_identity.driver_uuid));
        d->non_coherent_atom_size = props2.properties.limits.nonCoherentAtomSize;

        d->cache_callbacks = desc.pipeline_cache_callbacks;
        if (d->cache_callbacks.load || d->cache_callbacks.store) {
            VkPipelineCacheCreateInfo cache_info{
                .sType           = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
                .pNext           = nullptr,
                .flags           = 0,
                .initialDataSize = 0,
                .pInitialData    = nullptr,
            };
            MemoryBlock blob{};
            if (d->cache_callbacks.load &&
                d->cache_callbacks.load(d->cache_identity, d->cache_callbacks.user, &blob) &&
                blob.ptr != nullptr && blob.len > 0) {
                cache_info.initialDataSize = blob.len;
                cache_info.pInitialData    = blob.ptr;
            }
            if (!IZ_CHK(d, vkCreatePipelineCache(d->device, &cache_info, nullptr, &d->vk_pipeline_cache),
                        "create_device: vkCreatePipelineCache failed")) {
                // Invalid or rejected blob (driver/device mismatch): fall back
                // to an empty cache; the app's load hook is advisory only.
                cache_info.initialDataSize = 0;
                cache_info.pInitialData    = nullptr;
                IZ_CHK(d, vkCreatePipelineCache(d->device, &cache_info, nullptr, &d->vk_pipeline_cache),
                       "create_device: vkCreatePipelineCache (empty) failed");
            }
        }
    }

    // Initialize pools
    d->buffer_pool = SlotMap<Buffer>(d->allocator, [](Buffer* b, void* ud) {
        auto* dd = reinterpret_cast<DeviceImpl*>(ud);
        vmaDestroyBuffer(dd->vma, b->vk_buffer, b->vk_allocation);
    }, d);
    d->texture_pool = SlotMap<TextureImpl>(d->allocator, [](TextureImpl* t, void* ud) {
        auto* dd = reinterpret_cast<DeviceImpl*>(ud);
        if (t->default_image_view != VK_NULL_HANDLE) {
            vkDestroyImageView(dd->device, t->default_image_view, nullptr);
        }
        if (!t->is_swapchain_image) {
            if (t->vk_allocation != VK_NULL_HANDLE) {
                vmaDestroyImage(dd->vma, t->vk_image, t->vk_allocation);
            } else {
                vkDestroyImage(dd->device, t->vk_image, nullptr);
            }
        }
    }, d);
    d->semaphore_pool = SlotMap<SemaphoreImpl>(d->allocator, [](SemaphoreImpl* s, void* ud) {
        auto* dd = reinterpret_cast<DeviceImpl*>(ud);
        vkDestroySemaphore(dd->device, s->vk_semaphore, nullptr);
    }, d);
    d->pipeline_pool = SlotMap<PipelineImpl>(d->allocator, [](PipelineImpl*, void*) {
        // Shared pipelines are refcounted in pipeline.cpp; free() destroys the
        // last reference and destroy_device cleans up leaked handles.
    }, d);
    d->depth_stencil_pool = SlotMap<DepthStencilState>(d->allocator, [](DepthStencilState*, void*) {});

    // Initialize ptr_map and other vectors
    d->ptr_map                = Vector<GpuPtrMap>(d->allocator);
    d->uninitialized_textures = Vector<Handle<Texture>>(d->allocator);
    d->pipeline_records       = Vector<PipelineRecord*>(d->allocator);
    d->compiler_queue         = Vector<PipelineRecord*>(d->allocator);

    // Start the async compiler worker (single thread, FIFO, low priority).
    condvar_init(&d->compiler_cv);
    if (!thread_create(&d->compiler_thread, &compiler_worker_main, d)) {
        IZ_LOG(d, LogLevel::Error, "create_device: failed to start compiler worker");
        condvar_destroy(&d->compiler_cv);
        goto fail;
    }
    thread_set_low_priority(d->compiler_thread);

    return d;

fail:
    if (d->surface.surface != VK_NULL_HANDLE) {
        vkDestroySurfaceKHR(d->instance, d->surface.surface, nullptr);
    }
    if (d->vk_pipeline_cache != VK_NULL_HANDLE) {
        vkDestroyPipelineCache(d->device, d->vk_pipeline_cache, nullptr);
    }
    if (d->device != VK_NULL_HANDLE) { vkDestroyDevice(d->device, nullptr); }
    if (d->instance != VK_NULL_HANDLE) { vkDestroyInstance(d->instance, nullptr); }
    tls_free(d->thread_local_key);
    alloc.free({.ptr = d, .len = sizeof(DeviceImpl)});
    return nullptr;
}

void destroy_device(Device dev) {
    auto* d = reinterpret_cast<DeviceImpl*>(dev);
    device_wait_for_idle(d);
    unconfigure_surface(d);

    // Stop accepting new pipeline requests.
    atomic_exchange(&d->device_destroying, 1);

    // Signal the compiler worker to drain its queue and exit; join it so no
    // work touches Vulkan state after this point.
    mutex_lock(&d->compiler_lock);
    atomic_exchange(&d->compiler_shutdown, 1);
    condvar_broadcast(&d->compiler_cv);
    mutex_unlock(&d->compiler_lock);
    thread_join(d->compiler_thread);
    d->compiler_thread = 0;
    condvar_destroy(&d->compiler_cv);

    // Drain all retirement batches (GPU work is complete).
    if (d->default_queue) {
        Vector<RetireBatch*> pending(d->allocator);
        mutex_lock(&d->default_queue->submit_lock);
        for (RetireBatch* b : d->default_queue->retire_queue) { pending.push_back(b); }
        d->default_queue->retire_queue.clear();
        mutex_unlock(&d->default_queue->submit_lock);
        for (RetireBatch* b : pending) { process_retire_batch(d, b); }
    }

    // Destroy every remaining record (leaked handles, failed compiles, etc.).
    mutex_lock(&d->pipeline_lock);
    for (PipelineRecord* rec : d->pipeline_records) {
        if (rec->vk_pipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(d->device, rec->vk_pipeline, nullptr);
        }
        if (rec->key_block.ptr != nullptr) { d->allocator.free(rec->key_block); }
        free_record(d, rec);
    }
    d->pipeline_records.clear();
    mutex_unlock(&d->pipeline_lock);

    // Persist the native pipeline cache, then destroy it.
    store_pipeline_cache(d);
    if (d->vk_pipeline_cache != VK_NULL_HANDLE) {
        vkDestroyPipelineCache(d->device, d->vk_pipeline_cache, nullptr);
        d->vk_pipeline_cache = VK_NULL_HANDLE;
    }

    // Destroy queues
    if (d->default_queue) {
        for (auto& p : d->default_queue->command_superpool.pools) {
            if (p.command_pool) {
                vkDestroyCommandPool(d->device, p.command_pool, nullptr);
            }
        }
        d->default_queue->pending_events.clear();
        d->allocator.free({.ptr = d->default_queue, .len = sizeof(QueueImpl)});
    }

    // Destroy pools (destructors call the registered destructors)
    d->buffer_pool.clear();
    d->texture_pool.clear();
    d->semaphore_pool.clear();
    d->pipeline_pool.clear();
    d->depth_stencil_pool.clear();

    // Destroy descriptor heap buffers
    if (d->heap.sampler_buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(d->vma, d->heap.sampler_buffer, d->heap.sampler_allocation);
    }
    if (d->heap.resource_buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(d->vma, d->heap.resource_buffer, d->heap.resource_allocation);
    }

    if (d->vma != VK_NULL_HANDLE) { vmaDestroyAllocator(d->vma); }
    // Device before messenger: vkDestroyDevice's own lifetime VUIDs still
    // reach the debug messenger (canonical Khronos teardown order; the
    // messenger is instance-level and must die before vkDestroyInstance).
    if (d->device != VK_NULL_HANDLE) { vkDestroyDevice(d->device, nullptr); }
    if (d->debug_messenger != VK_NULL_HANDLE && vkDestroyDebugUtilsMessengerEXT) {
        vkDestroyDebugUtilsMessengerEXT(d->instance, d->debug_messenger, nullptr);
    }
    if (d->surface.surface != VK_NULL_HANDLE) {
        vkDestroySurfaceKHR(d->instance, d->surface.surface, nullptr);
    }
    vkDestroyInstance(d->instance, nullptr);
    volkFinalize();

    tls_free(d->thread_local_key);
    auto alloc = d->allocator;
    d->~DeviceImpl();
    alloc.free({.ptr = d, .len = sizeof(DeviceImpl)});
}

Backend device_backend() {
    return Backend::Vulkan;
}

bool device_supports_dual_source_blend(Device dev) {
    auto* d = reinterpret_cast<DeviceImpl*>(dev);
    return d != nullptr && d->dual_src_blend;
}

void device_wait_for_idle(Device dev) {
    auto* d = reinterpret_cast<DeviceImpl*>(dev);
    IZ_CHK(d, vkDeviceWaitIdle(d->device), "device_wait_for_idle failed");
}

// --- Memory ------------------------------------------------------------------------------

GpuPtr malloc(Device dev, size_t bytes, Memory memory) {
    return malloc(dev, bytes, 0, memory);
}

GpuPtr malloc(Device dev, size_t bytes, size_t align, Memory memory) {
    auto* d = reinterpret_cast<DeviceImpl*>(dev);

    // Validate alignment: must be a power of two (0 = no specific alignment).
    if (align == 0) { align = 1; }
    if ((align & (align - 1)) != 0) {
        IZ_LOG(d, LogLevel::Error, "malloc: alignment must be a power of two");
        return 0;
    }

    constexpr VkBufferUsageFlags kDefaultUsages =
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
        VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;

    // Overallocate by (align - 1) so the user address can be aligned inside.
    const VkDeviceSize backing_size =
        static_cast<VkDeviceSize>(bytes) + static_cast<VkDeviceSize>(align - 1);

    VkBufferCreateInfo create_info{
        .sType                 = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext                 = nullptr,
        .flags                 = 0,
        .size                  = backing_size,
        .usage                 = kDefaultUsages,
        .sharingMode           = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices   = nullptr,
    };

    VmaAllocationCreateFlags vma_flags = 0;
    VkMemoryPropertyFlags    vk_flags  = 0;
    switch (memory) {
        case Memory::Default:
            vma_flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                        VMA_ALLOCATION_CREATE_MAPPED_BIT;
            vk_flags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
            break;
        case Memory::Gpu:
            vma_flags = 0;
            vk_flags  = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
            break;
        case Memory::Readback:
            vma_flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                        VMA_ALLOCATION_CREATE_MAPPED_BIT;
            vk_flags = VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
            break;
    }

    VmaAllocationCreateInfo alloc_info{
        .flags          = vma_flags,
        .usage          = VMA_MEMORY_USAGE_AUTO,
        .requiredFlags  = vk_flags,
        .preferredFlags = 0,
        .memoryTypeBits = 0,
        .pool           = VK_NULL_HANDLE,
        .pUserData      = nullptr,
        .priority       = 0.0f,
    };

    VkBuffer       vk_buffer = VK_NULL_HANDLE;
    VmaAllocation  vk_allocation = VK_NULL_HANDLE;
    VmaAllocationInfo alloc_result{};

    VkResult result = vmaCreateBuffer(d->vma, &create_info, &alloc_info,
                                       &vk_buffer, &vk_allocation, &alloc_result);
    if (result != VK_SUCCESS) {
        log_vk_impl(d, result, "gpu::malloc: vkCreateBuffer failed", __LINE__, "device.cpp"_sv);
        return 0;
    }

    VkBufferDeviceAddressInfo addr_info{
        .sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
        .pNext  = nullptr,
        .buffer = vk_buffer,
    };
    const VkDeviceAddress backing_address = vkGetBufferDeviceAddress(d->device, &addr_info);

    // Align the user address inside the backing allocation.
    const GpuPtr user_address =
        (backing_address + align - 1) & ~(static_cast<VkDeviceAddress>(align) - 1);
    const VkDeviceSize user_offset = user_address - backing_address;

    VkMemoryPropertyFlags props = 0;
    vmaGetMemoryTypeProperties(d->vma, alloc_result.memoryType, &props);
    const bool coherent = (props & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;

    auto handle = d->buffer_pool.emplace(Buffer{
        .vk_buffer       = vk_buffer,
        .vk_allocation   = vk_allocation,
        .host_ptr        = alloc_result.pMappedData,
        .memory          = alloc_result.deviceMemory,
        .memory_offset   = alloc_result.offset,
        .backing_address = backing_address,
        .backing_size    = backing_size,
        .user_address    = user_address,
        .user_size       = static_cast<VkDeviceSize>(bytes),
        .user_offset     = user_offset,
        .coherent        = coherent,
    });

    // Insert into sorted ptr_map (by user address; align_up is monotonic).
    rwlock_lock_write(&d->ptr_map_lock);
    uint32_t lo = 0, hi = d->ptr_map.size();
    while (lo < hi) {
        uint32_t mid = (lo + hi) / 2;
        if (d->ptr_map[mid].ptr < user_address) { lo = mid + 1; } else { hi = mid; }
    }
    d->ptr_map.insert(d->ptr_map.begin() + lo, GpuPtrMap{.ptr = user_address, .buffer = handle});
    rwlock_unlock_write(&d->ptr_map_lock);

    return user_address;
}

void free(Device dev, GpuPtr ptr) {
    auto* d = reinterpret_cast<DeviceImpl*>(dev);
    rwlock_lock_write(&d->ptr_map_lock);
    // Binary search for exact match
    uint32_t lo = 0, hi = d->ptr_map.size();
    while (lo < hi) {
        uint32_t mid = (lo + hi) / 2;
        if (d->ptr_map[mid].ptr == ptr) {
            d->buffer_pool.erase(d->ptr_map[mid].buffer);
            d->ptr_map.erase(d->ptr_map.begin() + mid, d->ptr_map.begin() + mid + 1);
            rwlock_unlock_write(&d->ptr_map_lock);
            return;
        }
        if (d->ptr_map[mid].ptr < ptr) { lo = mid + 1; } else { hi = mid; }
    }
    rwlock_unlock_write(&d->ptr_map_lock);
}

void free_after(Device dev, GpuPtr ptr, Submission s) {
    auto* d = reinterpret_cast<DeviceImpl*>(dev);
    QueueImpl* q = s.queue ? reinterpret_cast<QueueImpl*>(s.queue) : d->default_queue;
    if (q == nullptr) { return; }
    // Conservative retirement for invalid/failed submissions: after the
    // latest successfully submitted work completes.
    const uint64_t value = s.status == SubmitStatus::Success ? s.value : q->timeline_value;

    // Invalidate the pointer now; capture the buffer handle for destruction
    // once the target submission completes.
    rwlock_lock_write(&d->ptr_map_lock);
    uint32_t lo = 0, hi = d->ptr_map.size();
    uint32_t idx = ~0u;
    while (lo < hi) {
        const uint32_t mid = (lo + hi) / 2;
        if (d->ptr_map[mid].ptr == ptr) { idx = mid; break; }
        if (d->ptr_map[mid].ptr < ptr) { lo = mid + 1; } else { hi = mid; }
    }
    if (idx == ~0u) {
        rwlock_unlock_write(&d->ptr_map_lock);
        return;   // not a live allocation (already freed)
    }
    const Handle<Buffer> h = d->ptr_map[idx].buffer;
    d->ptr_map.erase(d->ptr_map.begin() + idx, d->ptr_map.begin() + idx + 1);
    rwlock_unlock_write(&d->ptr_map_lock);

    enqueue_retire(q, value, RetireItem{RetireKind::Buffer, h.h, 0});
}

void* get_host_pointer(Device dev, GpuPtr ptr) {
    auto* d = reinterpret_cast<DeviceImpl*>(dev);
    VkDeviceSize offset = 0;
    Buffer* buf = find_buffer_for_ptr(d, ptr, &offset);
    if (buf == nullptr || buf->host_ptr == nullptr) { return nullptr; }
    return static_cast<char*>(buf->host_ptr) + buf->user_offset + (ptr - buf->user_address);
}

bool flush_host_memory(Device dev, GpuPtr ptr, size_t size) {
    auto* d = reinterpret_cast<DeviceImpl*>(dev);
    if (size == 0) { return true; }
    VkDeviceSize offset = 0;
    Buffer* buf = find_buffer_for_ptr(d, ptr, &offset);
    if (buf == nullptr) {
        IZ_LOG(d, LogLevel::Error, "flush_host_memory: pointer is not in a live allocation");
        return false;
    }
    if (offset + size > buf->user_offset + buf->user_size) {
        IZ_LOG(d, LogLevel::Error, "flush_host_memory: range exceeds allocation bounds");
        return false;
    }
    if (buf->coherent || buf->memory == VK_NULL_HANDLE) { return true; }

    const VkDeviceSize atom       = d->non_coherent_atom_size;
    const VkDeviceSize range_begin = buf->memory_offset + offset;
    const VkDeviceSize range_end   = range_begin + size;
    const VkDeviceSize aligned_begin = range_begin & ~(atom - 1);
    const VkDeviceSize aligned_end   = (range_end + atom - 1) & ~(atom - 1);
    const VkMappedMemoryRange range{
        .sType  = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
        .pNext  = nullptr,
        .memory = buf->memory,
        .offset = aligned_begin,
        .size   = aligned_end - aligned_begin,
    };
    return vkFlushMappedMemoryRanges(d->device, 1, &range) == VK_SUCCESS;
}

bool invalidate_host_memory(Device dev, GpuPtr ptr, size_t size) {
    auto* d = reinterpret_cast<DeviceImpl*>(dev);
    if (size == 0) { return true; }
    VkDeviceSize offset = 0;
    Buffer* buf = find_buffer_for_ptr(d, ptr, &offset);
    if (buf == nullptr) {
        IZ_LOG(d, LogLevel::Error, "invalidate_host_memory: pointer is not in a live allocation");
        return false;
    }
    if (offset + size > buf->user_offset + buf->user_size) {
        IZ_LOG(d, LogLevel::Error, "invalidate_host_memory: range exceeds allocation bounds");
        return false;
    }
    if (buf->coherent || buf->memory == VK_NULL_HANDLE) { return true; }

    const VkDeviceSize atom       = d->non_coherent_atom_size;
    const VkDeviceSize range_begin = buf->memory_offset + offset;
    const VkDeviceSize range_end   = range_begin + size;
    const VkDeviceSize aligned_begin = range_begin & ~(atom - 1);
    const VkDeviceSize aligned_end   = (range_end + atom - 1) & ~(atom - 1);
    const VkMappedMemoryRange range{
        .sType  = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
        .pNext  = nullptr,
        .memory = buf->memory,
        .offset = aligned_begin,
        .size   = aligned_end - aligned_begin,
    };
    return vkInvalidateMappedMemoryRanges(d->device, 1, &range) == VK_SUCCESS;
}

// --- Semaphores ---------------------------------------------------------------------------

Handle<Semaphore> create_semaphore_internal(DeviceImpl* d, uint64_t init_value) {
    VkSemaphoreTypeCreateInfo semaphore_type{
        .sType         = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
        .pNext         = nullptr,
        .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
        .initialValue  = init_value,
    };
    VkSemaphoreCreateInfo create_info{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        .pNext = &semaphore_type,
        .flags = 0,
    };
    VkSemaphore s = VK_NULL_HANDLE;
    if (!IZ_CHK(d, vkCreateSemaphore(d->device, &create_info, nullptr, &s),
                "create_semaphore failed")) {
        return {};
    }
    return handle_cast<Semaphore>(d->semaphore_pool.emplace(SemaphoreImpl{.vk_semaphore = s}));
}

Handle<Semaphore> create_semaphore(Device dev, uint64_t init_value) {
    return create_semaphore_internal(reinterpret_cast<DeviceImpl*>(dev), init_value);
}

void wait_semaphore(Device dev, Handle<Semaphore> sema, uint64_t value) {
    auto* d = reinterpret_cast<DeviceImpl*>(dev);
    VkSemaphore s = d->semaphore_pool[handle_cast<SemaphoreImpl>(sema)].vk_semaphore;
    VkSemaphoreWaitInfo wait_info{
        .sType          = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
        .pNext          = nullptr,
        .flags          = 0,
        .semaphoreCount = 1,
        .pSemaphores    = &s,
        .pValues        = &value,
    };
    IZ_CHK(d, vkWaitSemaphores(d->device, &wait_info, UINT64_MAX), "wait_semaphore failed");
}

void free(Device dev, Handle<Semaphore> sema) {
    auto* d = reinterpret_cast<DeviceImpl*>(dev);
    d->semaphore_pool.erase(handle_cast<SemaphoreImpl>(sema));
}

// --- Queue -------------------------------------------------------------------------------

Queue get_queue(Device dev, QueueType type) {
    auto* d = reinterpret_cast<DeviceImpl*>(dev);
    (void)type; // Only Default queue in v1

    if (d->default_queue == nullptr) {
        VkQueue queue;
        vkGetDeviceQueue(d->device, d->graphics_queue_family, 0, &queue);
        auto timeline = create_semaphore_internal(d, 0);

        auto blk = d->allocator.alloc(sizeof(QueueImpl));
        auto* q  = ::new (blk.ptr) QueueImpl();
        q->device         = d;
        q->queue          = queue;
        q->timeline       = timeline;
        q->queue_family   = d->graphics_queue_family;
        q->timeline_value = 0;
        q->pending_events = Vector<CompletionEvent>(d->allocator);
        q->retire_queue    = Vector<RetireBatch*>(d->allocator);
        d->default_queue  = q;
    }
    return d->default_queue;
}

void queue_on_submitted_work_completed(Queue q, void (*fn)(void*), void* userdata) {
    q->pending_events.emplace_back(CompletionEvent{
        .completed_time = q->timeline_value,
        .callback       = fn,
        .userdata       = userdata,
    });
}

void queue_process_events(Queue q) {
    auto* d = q->device;
    uint64_t current_time = 0;
    VkSemaphore timeline_sem = d->semaphore_pool[handle_cast<SemaphoreImpl>(q->timeline)].vk_semaphore;
    IZ_CHK(d, vkGetSemaphoreCounterValue(d->device, timeline_sem, &current_time),
           "queue_process_events failed");

    uint32_t i = 0;
    while (i < q->pending_events.size() && q->pending_events[i].completed_time <= current_time) {
        q->pending_events[i].callback(q->pending_events[i].userdata);
        i++;
    }
    if (i != 0) {
        q->pending_events.erase(q->pending_events.begin(), q->pending_events.begin() + i);
    }

    // Drain the retirement queue (batches sorted by value, ascending).
    mutex_lock(&q->submit_lock);
    uint32_t j = 0;
    while (j < q->retire_queue.size() && q->retire_queue[j]->value <= current_time) { j++; }
    Vector<RetireBatch*> ready(d->allocator);
    for (uint32_t k = 0; k < j; ++k) { ready.push_back(q->retire_queue[k]); }
    if (j != 0) {
        q->retire_queue.erase(q->retire_queue.begin(), q->retire_queue.begin() + j);
    }
    mutex_unlock(&q->submit_lock);

    // Process outside queue locks (may take the pipeline/map locks and
    // destroy native objects).
    for (RetireBatch* batch : ready) { process_retire_batch(d, batch); }
}


}  // namespace gpu
