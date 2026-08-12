// device.cpp — instance/device creation, features, descriptor heap, memory, queues.

#include <algorithm>

#include "internal.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace gpu {

// --- Required device extensions ---------------------------------------------------
// VK_KHR_SWAPCHAIN is required only when a surface-capable WSI is selected;
// headless builds never request it.
#if defined(IZ_WSI_WIN32) || defined(IZ_WSI_ANDROID)
#    define IZ_REQUIRE_SWAPCHAIN 1
#else
#    define IZ_REQUIRE_SWAPCHAIN 0
#endif
#if defined(IZ_VK_PROFILE_BINDLESS)
#    define IZ_REQUIRE_HEAP_TRIO 0
#else
#    define IZ_REQUIRE_HEAP_TRIO 1
#endif
// Candidate extension list, ordered [required prefix ..., optional ...].
// Under bindless the REQUIRED prefix is swapchain + (api < 1.3 ? dynamic
// rendering + sync2 : 0); copy_commands2 and extended_dynamic_state are
// OPTIONAL — their presence selects the modern path, their absence selects
// the private fallbacks (legacy copy / static graphics state). 1.3+ devices
// never require the promoted extension names.
static const char* kRequiredDeviceExtensions[IZ_REQUIRE_SWAPCHAIN + IZ_REQUIRE_HEAP_TRIO * 3 + 4 + 1] = {
#if defined(IZ_WSI_WIN32) || defined(IZ_WSI_ANDROID)
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
#endif
    // The descriptor-heap trio is native-profile-only; the bindless profile
    // uses descriptor-indexing arrays (core 1.2+).
#if !defined(IZ_VK_PROFILE_BINDLESS)
    VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME,
    VK_KHR_SHADER_UNTYPED_POINTERS_EXTENSION_NAME,
    VK_KHR_UNIFIED_IMAGE_LAYOUTS_EXTENSION_NAME,
#endif
#if defined(IZ_VK_PROFILE_BINDLESS)
    VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME,
    VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME,
    VK_KHR_COPY_COMMANDS_2_EXTENSION_NAME,       // optional
    VK_EXT_EXTENDED_DYNAMIC_STATE_EXTENSION_NAME, // optional
#endif
};
// Number of extensions to ENABLE at device creation: the required prefix
// plus the optional families when present on the device. On 1.3+ the
// promoted KHR names are legal no-ops but still needed so the extension
// entry points (used by the bindless command aliases) resolve in volk.
// enabled extensions = required prefix + optional families present on the
// device. The optional entries sit at fixed offsets after the required
// prefix; count them only when the device exports them (their absence must
// not enable an invalid extension name).
static size_t enabled_extension_count(DeviceImpl* d) {
    size_t n = IZ_REQUIRE_SWAPCHAIN;
#if defined(IZ_VK_PROFILE_BINDLESS)
    n += 2;                                        // dynamic rendering + sync2 (alias entry points need them)
    if (d->has_copy2_ext)  { n += 1; }
    if (d->has_extdyn_ext) { n += 1; }
#else
    n += IZ_REQUIRE_HEAP_TRIO * 3;
#endif
    return n;
}

// Per-candidate required count: the required prefix of kRequiredDeviceExtensions.
static size_t required_extension_count(uint32_t device_api_version) {
    size_t n = IZ_REQUIRE_SWAPCHAIN;
#if defined(IZ_VK_PROFILE_BINDLESS)
    if (device_api_version < VK_API_VERSION_1_3) { n += 2; }   // dyn rendering + sync2
    // copy_commands2 / extended_dynamic_state: never required
#else
    n += IZ_REQUIRE_HEAP_TRIO * 3;
#endif
    return n;
}

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
Buffer* find_buffer_for_ptr(DeviceImpl* d, GpuPtr ptr, VkDeviceSize* out_offset) {
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
#if defined(IZ_VK_PROFILE_BINDLESS)
        // Bindless: Vulkan 1.2 instance; dynamic rendering + synchronization2
        // are enabled as KHR extensions on 1.2 devices (or left as 1.3 cores
        // when available — the promoted entry points resolve identically).
        .apiVersion         = VK_API_VERSION_1_2,
#else
        .apiVersion         = VK_API_VERSION_1_4,
#endif
    };

    // WSI extensions are enabled only when a surface-capable WSI is selected;
    // a HEADLESS build never requests surface/swapchain extensions.
    const char* wsi_extensions[2] = {};
    uint32_t    num_wsi_extensions = 0;
#if defined(IZ_WSI_WIN32)
    wsi_extensions[num_wsi_extensions++] = VK_KHR_SURFACE_EXTENSION_NAME;
    wsi_extensions[num_wsi_extensions++] = VK_KHR_WIN32_SURFACE_EXTENSION_NAME;
#endif
#if defined(IZ_WSI_ANDROID)
    wsi_extensions[num_wsi_extensions++] = VK_KHR_SURFACE_EXTENSION_NAME;
    wsi_extensions[num_wsi_extensions++] = VK_KHR_ANDROID_SURFACE_EXTENSION_NAME;
#endif

    // Optional validation layer
    const char* layers[1]     = {"VK_LAYER_KHRONOS_validation"};
    uint32_t    layer_count   = 0;
    const char* ext_ptrs[8];
    uint32_t    ext_count     = 0;
    for (uint32_t i = 0; i < num_wsi_extensions; ++i) { ext_ptrs[ext_count++] = wsi_extensions[i]; }

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

#if defined(IZ_VK_PROFILE_BINDLESS)
        // Bindless: Vulkan 1.2+ (dynamic rendering + synchronization2 are
        // required and enabled either as 1.3 cores or via their KHR
        // extensions on 1.2 devices — see kRequiredDeviceExtensions). The
        // exact feature gate is in create_device, below.
        if (props.apiVersion < VK_API_VERSION_1_2) { continue; }
#else
        // Require Vulkan 1.4
        if (props.apiVersion < VK_API_VERSION_1_4) { continue; }
#endif

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
        const size_t required_count = required_extension_count(props.apiVersion);
        for (size_t req = 0; req < required_count; ++req) {
            bool found = false;
            for (uint32_t e = 0; e < ext_count; ++e) {
                if (strcmp(exts[e].extensionName, kRequiredDeviceExtensions[req]) == 0) {
                    found = true;
                    break;
                }
            }
            if (!found) {
#if defined(IZ_VK_PROFILE_BINDLESS)
                // Precise diagnostics: which required extension is missing
                // (e.g. dzn lacks copy_commands2 + extended_dynamic_state,
                // which the legacy-copy / static-dynamic-state fallbacks
                // will add in the command phase).
                IZ_LOG(d, LogLevel::Error, "bindless: required device extension missing:");
                log_impl(d, LogLevel::Error,
                         Span<const char>(kRequiredDeviceExtensions[req], strlen(kRequiredDeviceExtensions[req])),
                         __LINE__, "device.cpp"_sv);
#endif
                all_supported = false;
                break;
            }
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

// Captures effective API version + extension presence + derived dispatch on
// the SELECTED device. Must run BEFORE logical-device feature routing, VMA
// creation, and pipeline-cache identity construction.
void capture_device_capabilities(DeviceImpl* d) {
    uint32_t loader_version = VK_API_VERSION_1_0;
    if (vkEnumerateInstanceVersion != nullptr) { vkEnumerateInstanceVersion(&loader_version); }
    d->instance_api_version = loader_version;

    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(d->physical_device, &props);
    d->device_api_version = props.apiVersion;

    uint32_t requested = VK_API_VERSION_1_4;
#if defined(IZ_VK_PROFILE_BINDLESS)
    requested = VK_API_VERSION_1_2;
#endif
    uint32_t effective = props.apiVersion;
    if (loader_version < effective) { effective = loader_version; }
    if (requested < effective) { effective = requested; }
    d->dispatch.effective_api_version = effective;

    // Extension presence on the selected device.
    uint32_t count = 0;
    vkEnumerateDeviceExtensionProperties(d->physical_device, nullptr, &count, nullptr);
    Vector<VkExtensionProperties> exts(d->allocator, {}, count);
    vkEnumerateDeviceExtensionProperties(d->physical_device, nullptr, &count, exts.data());
    auto has_ext = [&](const char* name) -> bool {
        for (uint32_t i = 0; i < count; ++i) {
            if (strcmp(exts[i].extensionName, name) == 0) { return true; }
        }
        return false;
    };
    d->has_copy2_ext  = has_ext(VK_KHR_COPY_COMMANDS_2_EXTENSION_NAME);
    d->has_extdyn_ext = has_ext(VK_EXT_EXTENDED_DYNAMIC_STATE_EXTENSION_NAME);

    d->dispatch = derive_dispatch_capabilities(
        effective,
        has_ext(VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME),
        has_ext(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME),
        d->has_copy2_ext,
        d->has_extdyn_ext,
        atomic_load(&d->force_legacy_copy) != 0,
        atomic_load(&d->force_static_state) != 0);
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
#if !defined(IZ_VK_PROFILE_BINDLESS)
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
#endif
#if defined(IZ_VK_PROFILE_BINDLESS)
    // 1.2 route: the promoted families are enabled through their EXTENSION
    // feature structs (VkPhysicalDeviceDynamicRenderingFeaturesKHR /
    // VkPhysicalDeviceSynchronization2FeaturesKHR /
    // VkPhysicalDeviceExtendedDynamicStateFeaturesEXT). 1.3+ devices use the
    // core aggregate instead — never both in one chain.
    VkPhysicalDeviceExtendedDynamicStateFeaturesEXT ext_dyn_state_features{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_FEATURES_EXT,
        .pNext = nullptr,
    };
    VkPhysicalDeviceSynchronization2FeaturesKHR sync2_features{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES_KHR,
        .pNext = nullptr,
    };
    VkPhysicalDeviceDynamicRenderingFeaturesKHR dyn_rendering_features{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES_KHR,
        .pNext = nullptr,
    };
    VkPhysicalDeviceVulkan13Features vulkan13_features{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
        .pNext = nullptr,
    };
    // Pointers to the booleans that gate dynamic rendering / sync2 / ext
    // dynamic state for the selected route (used for enable + verification).
    // VK_EXT_extended_dynamic_state was promoted to 1.3 as MANDATORY — the
    // 1.3 aggregate has no bit (verified: Vulkan-Headers 1.4.341), so the
    // EXT feature struct is only chained/enabled on the 1.2 extension route
    // when the extension is advertised.
    VkBool32* dyn_rendering_out = nullptr;
    VkBool32* sync2_out         = nullptr;
    VkBool32* ext_dyn_state_out = nullptr;
    void*     render_features_chain = nullptr;
    if (d->dispatch.effective_api_version >= VK_API_VERSION_1_3) {
        vulkan13_features.pNext = nullptr;
        render_features_chain   = &vulkan13_features;
        dyn_rendering_out       = &vulkan13_features.dynamicRendering;
        sync2_out               = &vulkan13_features.synchronization2;
    } else {
        ext_dyn_state_features.pNext = nullptr;
        sync2_features.pNext         = &ext_dyn_state_features;
        dyn_rendering_features.pNext = &sync2_features;
        render_features_chain        = &dyn_rendering_features;
        dyn_rendering_out            = &dyn_rendering_features.dynamicRendering;
        sync2_out                    = &sync2_features.synchronization2;
        ext_dyn_state_out            = &ext_dyn_state_features.extendedDynamicState;
    }
    VkPhysicalDeviceVulkan12Features vulkan12_features{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        .pNext = render_features_chain,
    };
#else
    VkPhysicalDeviceVulkan13Features vulkan13_features{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
        .pNext = &vulkan14_features,
    };
    VkPhysicalDeviceVulkan12Features vulkan12_features{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        .pNext = &vulkan13_features,
    };
#endif
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
#if defined(IZ_VK_PROFILE_BINDLESS)
    // OPTIONAL features are enabled only when the device supports them —
    // forcing them would fail vkCreateDevice on otherwise-valid bindless
    // devices (e.g. dzn lacks shaderInt8/storage8 access).
    auto enable_if_supported = [](VkBool32 supported) { return supported == VK_TRUE ? VK_TRUE : VK_FALSE; };
    device_features.features.shaderInt16      = enable_if_supported(device_features.features.shaderInt16);
#else
    device_features.features.shaderInt16       = VK_TRUE;
#endif
    device_features.features.multiDrawIndirect = VK_TRUE;
    d->dual_src_blend = device_features.features.dualSrcBlend == VK_TRUE;
    if (d->dual_src_blend) {
        device_features.features.dualSrcBlend = VK_TRUE;
    }

    vulkan11_features.shaderDrawParameters = VK_TRUE;
#if !defined(IZ_VK_PROFILE_BINDLESS)
    // 8/16-bit storage access through physical-storage-buffer pointers
    // (needed by the capability-gated ABI test; universal on 1.1+ drivers).
    vulkan11_features.storageBuffer16BitAccess = VK_TRUE;

    vulkan12_features.storageBuffer8BitAccess = VK_TRUE;
#endif

    vulkan12_features.timelineSemaphore       = VK_TRUE;
    vulkan12_features.bufferDeviceAddress     = VK_TRUE;
    vulkan12_features.scalarBlockLayout       = VK_TRUE;
    vulkan12_features.drawIndirectCount       = VK_TRUE;
#if defined(IZ_VK_PROFILE_BINDLESS)
    // Bindless: descriptor-indexing arrays ARE the global heap. Enable the
    // exact bits the global-set model needs; unsupported bits were already
    // rejected by the bindless gate in create_device.
    vulkan12_features.shaderInt8              = enable_if_supported(vulkan12_features.shaderInt8);
    vulkan12_features.shaderFloat16           = enable_if_supported(vulkan12_features.shaderFloat16);
    vulkan12_features.storagePushConstant8    = enable_if_supported(vulkan12_features.storagePushConstant8);
    vulkan12_features.storageBuffer8BitAccess = enable_if_supported(vulkan12_features.storageBuffer8BitAccess);
    vulkan11_features.storageBuffer16BitAccess = enable_if_supported(vulkan11_features.storageBuffer16BitAccess);
    vulkan12_features.descriptorIndexing = enable_if_supported(vulkan12_features.descriptorIndexing);
    vulkan12_features.shaderSampledImageArrayNonUniformIndexing =
        enable_if_supported(vulkan12_features.shaderSampledImageArrayNonUniformIndexing);
    vulkan12_features.shaderStorageImageArrayNonUniformIndexing =
        enable_if_supported(vulkan12_features.shaderStorageImageArrayNonUniformIndexing);
    vulkan12_features.runtimeDescriptorArray =
        enable_if_supported(vulkan12_features.runtimeDescriptorArray);
    vulkan12_features.descriptorBindingPartiallyBound =
        enable_if_supported(vulkan12_features.descriptorBindingPartiallyBound);
    vulkan12_features.descriptorBindingSampledImageUpdateAfterBind =
        enable_if_supported(vulkan12_features.descriptorBindingSampledImageUpdateAfterBind);
    vulkan12_features.descriptorBindingStorageImageUpdateAfterBind =
        enable_if_supported(vulkan12_features.descriptorBindingStorageImageUpdateAfterBind);
    // REQUIRED by the initial one-set implementation (no snapshot path yet):
    // updates while submissions are pending must be legal.
    vulkan12_features.descriptorBindingUpdateUnusedWhilePending = VK_TRUE;
    vulkan12_features.descriptorBindingVariableDescriptorCount = VK_TRUE;
#else
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
#endif

#if defined(IZ_VK_PROFILE_BINDLESS)
    // Required on the bindless route (either core-1.3 or the KHR/EXT structs).
    *dyn_rendering_out = VK_TRUE;
    *sync2_out         = VK_TRUE;
    if (ext_dyn_state_out != nullptr) { *ext_dyn_state_out = VK_TRUE; }
    d->use_synchronization2 = *sync2_out == VK_TRUE;
#else
    vulkan13_features.dynamicRendering = VK_TRUE;
    vulkan13_features.synchronization2 = VK_TRUE;
    vulkan13_features.maintenance4     = VK_TRUE;
    d->use_synchronization2 = vulkan13_features.synchronization2 == VK_TRUE;
#endif
    // Optional: lets the compiler worker probe the cache with
    // FAIL_ON_PIPELINE_COMPILE_REQUIRED instead of compiling blind.
    d->pipeline_cache_control = vulkan13_features.pipelineCreationCacheControl == VK_TRUE;
    if (d->pipeline_cache_control) {
        vulkan13_features.pipelineCreationCacheControl = VK_TRUE;
    }

#if !defined(IZ_VK_PROFILE_BINDLESS)
    vulkan14_features.maintenance5 = VK_TRUE;
    vulkan14_features.maintenance6 = VK_TRUE;

    desc_heap_features.descriptorHeap             = VK_TRUE;
    untyped_ptr_features.shaderUntypedPointers    = VK_TRUE;
    unified_layouts_features.unifiedImageLayouts  = VK_TRUE;
#endif

    const VkDeviceCreateInfo create_info{
        .sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext                   = &device_features,
        .flags                   = 0,
        .queueCreateInfoCount    = 1,
        .pQueueCreateInfos       = &queue_create_info,
        .enabledLayerCount       = 0,
        .ppEnabledLayerNames     = nullptr,
        .enabledExtensionCount   = static_cast<uint32_t>(enabled_extension_count(d)),
        .ppEnabledExtensionNames = kRequiredDeviceExtensions,
        .pEnabledFeatures        = nullptr, // Using pNext chain instead
    };

    VkResult result = vkCreateDevice(d->physical_device, &create_info, nullptr, &d->device);
    if (result != VK_SUCCESS) { return result; }

    volkLoadDevice(d->device);

    // Verify critical features were enabled
#if defined(IZ_VK_PROFILE_BINDLESS)
    if (!*dyn_rendering_out || !*sync2_out) {
        IZ_LOG(d, LogLevel::Error, "bindless profile: dynamic rendering / synchronization2 not enabled");
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }
    if (!vulkan12_features.bufferDeviceAddress || !vulkan12_features.timelineSemaphore ||
        !vulkan12_features.scalarBlockLayout || !vulkan12_features.drawIndirectCount ||
        !vulkan12_features.descriptorIndexing || !vulkan12_features.runtimeDescriptorArray ||
        !vulkan12_features.descriptorBindingPartiallyBound ||
        !vulkan12_features.descriptorBindingSampledImageUpdateAfterBind ||
        !vulkan12_features.descriptorBindingStorageImageUpdateAfterBind ||
        !vulkan12_features.descriptorBindingUpdateUnusedWhilePending ||
        !vulkan12_features.shaderSampledImageArrayNonUniformIndexing ||
        !vulkan12_features.shaderStorageImageArrayNonUniformIndexing ||
        !device_features.features.shaderInt64) {
        IZ_LOG(d, LogLevel::Error, "bindless profile: required feature not enabled");
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }
#else
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
#endif

#if defined(IZ_VK_PROFILE_BINDLESS)
    // Function-pointer validation for the selected dispatch path: every
    // command the path needs must resolve. The names are route-aware — a
    // device with effective API version >= 1.3 exports the core names; a 1.2
    // device (e.g. Mesa dzn) exports only the KHR/EXT names. The backend
    // dispatch helpers select the same way (route_is_core13).
    if (d->dispatch.effective_api_version >= VK_API_VERSION_1_3) {
        if (vkQueueSubmit2 == nullptr || vkCmdPipelineBarrier2 == nullptr) {
            IZ_LOG(d, LogLevel::Error,
                   "bindless: core 1.3 synchronization2 entry points missing");
            return VK_ERROR_FEATURE_NOT_PRESENT;
        }
    } else if (vkQueueSubmit2KHR == nullptr || vkCmdPipelineBarrier2KHR == nullptr) {
        IZ_LOG(d, LogLevel::Error, "bindless: VK_KHR_synchronization2 entry points missing");
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }
    if (!d->dispatch.use_legacy_copy_commands &&
        (d->dispatch.effective_api_version >= VK_API_VERSION_1_3 ? vkCmdCopyBuffer2 == nullptr
                                                                 : vkCmdCopyBuffer2KHR == nullptr)) {
        IZ_LOG(d, LogLevel::Error, "bindless: copy-commands2 entry points missing but modern path selected");
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }
    if (!d->dispatch.use_static_graphics_state &&
        (d->dispatch.effective_api_version >= VK_API_VERSION_1_3 ? vkCmdSetDepthTestEnable == nullptr
                                                                 : vkCmdSetDepthTestEnableEXT == nullptr)) {
        IZ_LOG(d, LogLevel::Error, "bindless: extended-dynamic-state entry points missing but dynamic path selected");
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }
#endif
    return VK_SUCCESS;
}

// --- VMA allocator ------------------------------------------------------------------------
static VkResult create_vma_allocator(DeviceImpl* d) {
    VmaVulkanFunctions vulkan_functions{};
    vulkan_functions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    vulkan_functions.vkGetDeviceProcAddr   = vkGetDeviceProcAddr;

    VmaAllocatorCreateInfo create_info{
#if defined(IZ_VK_PROFILE_BINDLESS)
        // Bindless slice: 1.3 device, no maintenance5 (VMA must not use
        // 1.4-only paths).
        .flags                       = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
#else
        .flags                       = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT |
                                        VMA_ALLOCATOR_CREATE_KHR_MAINTENANCE5_BIT,
#endif
        .physicalDevice              = d->physical_device,
        .device                      = d->device,
        .preferredLargeHeapBlockSize = 0,
        .pAllocationCallbacks        = nullptr,
        .pDeviceMemoryCallbacks      = nullptr,
        .pHeapSizeLimit              = nullptr,
        .pVulkanFunctions            = &vulkan_functions,
        .instance                    = d->instance,
#if defined(IZ_VK_PROFILE_BINDLESS)
        .vulkanApiVersion            = d->device_api_version,
#else
        .vulkanApiVersion            = VK_API_VERSION_1_4,
#endif
        .pTypeExternalMemoryHandleTypes = nullptr,
    };

    return vmaCreateAllocator(&create_info, &d->vma);
}

// --- Descriptor heap creation ---------------------------------------------------------------
#if defined(IZ_VK_PROFILE_BINDLESS)
// Bindless profile: one long-lived update-after-bind descriptor set holding
// the global arrays (binding 0 sampled, 1 storage, 2 samplers) + one shared
// pipeline layout (set 0 + 16-byte push constants). The slot allocators /
// generations / state machines are SHARED with the native profile — only the
// descriptor WRITE target differs.
static VkResult create_bindless_descriptor_sets(DeviceImpl* d) {
    VkPhysicalDeviceDescriptorIndexingProperties indexing_props{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_PROPERTIES,
        .pNext = nullptr,
    };
    VkPhysicalDeviceProperties2 props2{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        .pNext = &indexing_props,
    };
    vkGetPhysicalDeviceProperties2(d->physical_device, &props2);

    uint32_t combined = std::min(indexing_props.maxPerStageUpdateAfterBindResources,
                                 indexing_props.maxUpdateAfterBindDescriptorsInAllPools);
    const bool unlimited = combined == 0xFFFFFFFFu;
    auto cap = [&](uint32_t per_stage, uint32_t per_set, uint32_t max_public) {
        uint32_t c = std::min({per_stage, per_set, max_public});
        if (!unlimited && c > combined) { c = combined; }
        return c;
    };
    uint32_t sampled = cap(indexing_props.maxPerStageDescriptorUpdateAfterBindSampledImages,
                           indexing_props.maxDescriptorSetUpdateAfterBindSampledImages,
                           kMaxSampledTextures);
    uint32_t storage = cap(indexing_props.maxPerStageDescriptorUpdateAfterBindStorageImages,
                           indexing_props.maxDescriptorSetUpdateAfterBindStorageImages,
                           kMaxStorageTextures);
    uint32_t sampler = cap(indexing_props.maxPerStageDescriptorUpdateAfterBindSamplers,
                           indexing_props.maxDescriptorSetUpdateAfterBindSamplers,
                           kMaxSamplers);
    // The three arrays share the combined budget; scale proportionally when
    // they do not fit (keeping the profile floors where possible).
    uint64_t total = uint64_t(sampled) + storage + sampler;
    if (!unlimited && total > combined) {
        auto scale = [&](uint32_t& v, uint32_t floor) {
            uint64_t nv = uint64_t(v) * combined / total;
            if (nv < floor) { nv = floor; }
            v = static_cast<uint32_t>(nv);
        };
        scale(sampled, kMinBindlessSampledImages);
        scale(storage, kMinBindlessStorageImages);
        scale(sampler, kMinBindlessSamplers);
        if (uint64_t(sampled) + storage + sampler > combined) {
            IZ_LOG(d, LogLevel::Error, "bindless: combined descriptor budget too small for the profile floors");
            return VK_ERROR_INITIALIZATION_FAILED;
        }
    }
    d->bindless_sampled_capacity = sampled;
    d->bindless_storage_capacity = storage;
    d->bindless_sampler_capacity = sampler;

    // Slot allocators sized for the bindless capacities (shared with native).
    d->sampled_bitset = TwoLevelBitset(d->allocator, sampled);
    d->storage_bitset = TwoLevelBitset(d->allocator, storage);
    d->sampler_bitset = TwoLevelBitset(d->allocator, sampler);
    d->sampled_gen   = Vector<uint16_t>(d->allocator, 0, sampled);
    d->storage_gen   = Vector<uint16_t>(d->allocator, 0, storage);
    d->sampler_gen   = Vector<uint16_t>(d->allocator, 0, sampler);
    d->sampled_state = Vector<uint8_t>(d->allocator, 0, sampled);
    d->storage_state = Vector<uint8_t>(d->allocator, 0, storage);
    d->sampler_state = Vector<uint8_t>(d->allocator, 0, sampler);
    d->sampled_owner = Vector<TextureImpl*>(d->allocator, nullptr, sampled);
    d->storage_owner = Vector<TextureImpl*>(d->allocator, nullptr, storage);
    // Reserve descriptor index zero as the null descriptor (0 is never a
    // valid view/sampler handle) — same rule as the native heap.
    d->sampled_bitset.set_leading_zero();
    d->storage_bitset.set_leading_zero();
    d->sampler_bitset.set_leading_zero();

    const VkDescriptorSetLayoutBinding bindings[3] = {
        {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, sampled, VK_SHADER_STAGE_ALL, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, storage, VK_SHADER_STAGE_ALL, nullptr},
        {2, VK_DESCRIPTOR_TYPE_SAMPLER, sampler, VK_SHADER_STAGE_ALL, nullptr},
    };
    // UPDATE_UNUSED_WHILE_PENDING makes updating any slot of the one global
    // set legal while submissions using it are pending — the feature is
    // gated at device creation (no snapshot path in this slice yet).
    const VkDescriptorBindingFlags binding_flags[3] = {
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
            VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT,
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
            VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT,
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
            VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT |
            VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT,
    };
    VkDescriptorSetLayoutBindingFlagsCreateInfo flags_info{
        .sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
        .pNext         = nullptr,
        .bindingCount  = 3,
        .pBindingFlags = binding_flags,
    };
    VkDescriptorSetLayoutCreateInfo layout_info{
        .sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext        = &flags_info,
        .flags        = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
        .bindingCount = 3,
        .pBindings    = bindings,
    };
    if (!IZ_CHK(d, vkCreateDescriptorSetLayout(d->device, &layout_info, nullptr, &d->bindless_set_layout),
                "bindless: vkCreateDescriptorSetLayout failed")) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    const VkDescriptorPoolSize pool_sizes[3] = {
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, sampled},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, storage},
        {VK_DESCRIPTOR_TYPE_SAMPLER, sampler},
    };
    VkDescriptorPoolCreateInfo pool_info{
        .sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .pNext         = nullptr,
        .flags         = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT,
        .maxSets       = 1,
        .poolSizeCount = 3,
        .pPoolSizes    = pool_sizes,
    };
    if (!IZ_CHK(d, vkCreateDescriptorPool(d->device, &pool_info, nullptr, &d->bindless_pool),
                "bindless: vkCreateDescriptorPool failed")) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    // Binding 2 is a runtime (variable-count) descriptor array.
    uint32_t variable_counts[1] = {sampler};
    VkDescriptorSetVariableDescriptorCountAllocateInfo variable_info{
        .sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO,
        .pNext              = nullptr,
        .descriptorSetCount = 1,
        .pDescriptorCounts  = variable_counts,
    };
    VkDescriptorSetAllocateInfo set_info{
        .sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .pNext              = &variable_info,
        .descriptorPool     = d->bindless_pool,
        .descriptorSetCount = 1,
        .pSetLayouts        = &d->bindless_set_layout,
    };
    if (!IZ_CHK(d, vkAllocateDescriptorSets(d->device, &set_info, &d->bindless_set),
                "bindless: vkAllocateDescriptorSets failed")) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    const VkPushConstantRange push_range{
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT |
                      VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size   = 16,
    };
    VkPipelineLayoutCreateInfo pipeline_layout_info{
        .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext                  = nullptr,
        .flags                  = 0,
        .setLayoutCount         = 1,
        .pSetLayouts            = &d->bindless_set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges    = &push_range,
    };
    if (!IZ_CHK(d, vkCreatePipelineLayout(d->device, &pipeline_layout_info, nullptr, &d->bindless_pipeline_layout),
                "bindless: vkCreatePipelineLayout failed")) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    d->bindless_sampled_views   = Vector<VkImageView>(d->allocator, VK_NULL_HANDLE, sampled);
    d->bindless_storage_views   = Vector<VkImageView>(d->allocator, VK_NULL_HANDLE, storage);
    d->bindless_sampler_handles = Vector<VkSampler>(d->allocator, VK_NULL_HANDLE, sampler);

    return VK_SUCCESS;
}
#else
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
    // Reserve descriptor index zero as the null descriptor: 0 is never a
    // valid view/sampler handle.
    d->sampled_bitset.set_leading_zero();
    d->storage_bitset.set_leading_zero();
    d->sampler_bitset.set_leading_zero();
    // Per-slot CPU generations + Free/Allocated/Retiring state tables
    d->sampled_gen = Vector<uint16_t>(d->allocator, 0, d->heap.sampled_capacity);
    d->storage_gen = Vector<uint16_t>(d->allocator, 0, d->heap.storage_capacity);
    d->sampler_gen = Vector<uint16_t>(d->allocator, 0, d->heap.sampler_capacity);
    d->sampled_state = Vector<uint8_t>(d->allocator, 0, d->heap.sampled_capacity);
    d->storage_state = Vector<uint8_t>(d->allocator, 0, d->heap.storage_capacity);
    d->sampler_state = Vector<uint8_t>(d->allocator, 0, d->heap.sampler_capacity);
    d->sampled_owner = Vector<TextureImpl*>(d->allocator, nullptr, d->heap.sampled_capacity);
    d->storage_owner = Vector<TextureImpl*>(d->allocator, nullptr, d->heap.storage_capacity);

    return VK_SUCCESS;
}

// --- Debug messenger -------------------------------------------------------------------------
#endif  // !IZ_VK_PROFILE_BINDLESS (create_descriptor_heap)

// --- Debug messenger callback (profile-neutral) ------------------------------------------------
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

    // White-box force overrides (env vars; also settable via test hooks).
    if (std::getenv("IZANAGI_FORCE_LEGACY_COPY_COMMANDS") != nullptr) {
        d->force_legacy_copy = 1;
    }
    if (std::getenv("IZANAGI_FORCE_STATIC_GRAPHICS_STATE") != nullptr) {
        d->force_static_state = 1;
    }

    VkResult result = create_instance(d, desc);
    if (result != VK_SUCCESS) {
        log_vk_impl(d, result, "Failed to create Vulkan instance", __LINE__, "device.cpp"_sv);
        goto fail;
    }

    // Create surface if a window handle is provided and the WSI is surface-capable.
#if defined(IZ_WSI_WIN32)
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
#else
    // No surface-capable WSI in this build; surface APIs report clean errors.
    (void)desc.native_window_handle;
    (void)desc.native_instance_handle;
#endif

    result = select_physical_device(d, desc.gpu_preference);
    if (result != VK_SUCCESS) {
        log_vk_impl(d, result, "Failed to select physical device", __LINE__, "device.cpp"_sv);
        goto fail;
    }

    // Effective API version + dispatch capabilities BEFORE any feature
    // routing, VMA creation, or cache-identity construction.
    capture_device_capabilities(d);

#if defined(IZ_VK_PROFILE_BINDLESS)
    // Capability gate: the bindless profile is decided by exact feature bits
    // and limits, never by version or generation. Unsupported devices fail
    // with the complete missing-requirement list.
    {
        VulkanProfileReport report = evaluate_vulkan_bindless_profile(query_vulkan_profile_features(d));
        if (!report.supported) {
            for (uint32_t i = 0; i < report.missing_count; ++i) {
                const char* name = vulkan_requirement_name(report.missing[i]);
                IZ_LOG(d, LogLevel::Error, "bindless profile missing requirement:");
                log_impl(d, LogLevel::Error, Span<const char>(name, strlen(name)),
                         __LINE__, "device.cpp"_sv);
            }
            result = VK_ERROR_FEATURE_NOT_PRESENT;
            goto fail;
        }
        // Temporary initial-slice gates (removed when the fallbacks land):
        // one descriptor set (no snapshot path) requires
        // update-unused-while-pending; the private render-pass / legacy
        // barrier paths are not implemented yet, so dynamic rendering +
        // synchronization2 are required (Vulkan 1.3 cores).
        if (report.descriptor_snapshots != 1) {
            IZ_LOG(d, LogLevel::Error,
                   "bindless: initial implementation requires descriptorBindingUpdateUnusedWhilePending "
                   "(the snapshot path lands in the descriptor-update phase)");
            result = VK_ERROR_FEATURE_NOT_PRESENT;
            goto fail;
        }
        if (!report.dynamic_rendering || !report.synchronization2) {
            IZ_LOG(d, LogLevel::Error,
                   "bindless: initial implementation requires dynamic rendering + synchronization2 "
                   "(the private render-pass / legacy-barrier paths land in the command phase)");
            result = VK_ERROR_FEATURE_NOT_PRESENT;
            goto fail;
        }
        // Devices without VK_EXT_extended_dynamic_state select the private
        // static-graphics-state fallback (static pipeline variants compiled
        // on the compiler worker, resolved at draw time).
        if (d->dispatch.use_static_graphics_state) {
            IZ_LOG(d, LogLevel::Info,
                   "bindless: extended dynamic state unavailable; using the private "
                   "static graphics-state fallback");
        }
    }
#endif
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

#if defined(IZ_VK_PROFILE_BINDLESS)
    result = create_bindless_descriptor_sets(d);
#else
    result = create_descriptor_heap(d);
#endif
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
        d->cache_identity.profile   = device_backend_profile();
        d->cache_identity.vendor_id = props2.properties.vendorID;
        d->cache_identity.device_id = props2.properties.deviceID;
        memcpy(d->cache_identity.driver_uuid, id_props.driverUUID,
               sizeof(d->cache_identity.driver_uuid));
        d->non_coherent_atom_size = props2.properties.limits.nonCoherentAtomSize;
        d->min_uniform_alignment  = props2.properties.limits.minUniformBufferOffsetAlignment;
        d->min_storage_alignment  = props2.properties.limits.minStorageBufferOffsetAlignment;

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

        // pipelineCacheUUID is the primary compatibility identity: read it
        // from the cache blob header (VkPipelineCacheHeaderVersionOne layout:
        // headerSize, headerVersion, vendorID, deviceID, pipelineCacheUUID).
        // Fall back to driverUUID when the driver exposes no header yet.
        if (d->vk_pipeline_cache != VK_NULL_HANDLE) {
            size_t size = 0;
            vkGetPipelineCacheData(d->device, d->vk_pipeline_cache, &size, nullptr);
            if (size >= 32) {
                MemoryBlock tmp = d->allocator.alloc(size);
                if (tmp.ptr != nullptr) {
                    VkResult r = vkGetPipelineCacheData(d->device, d->vk_pipeline_cache, &size, tmp.ptr);
                    if (r == VK_SUCCESS && size >= 32) {
                        const uint8_t* data = static_cast<const uint8_t*>(tmp.ptr);
                        uint32_t header_size = 0;
                        uint32_t version     = 0;
                        memcpy(&header_size, data, 4);
                        memcpy(&version, data + 4, 4);
                        if (version == 1 && header_size >= 32) {
                            memcpy(d->cache_identity.cache_uuid, data + 16, 16);
                        }
                    }
                    d->allocator.free(tmp);
                }
            }
            bool any_uuid = false;
            for (int i = 0; i < 16; ++i) {
                if (d->cache_identity.cache_uuid[i] != 0) { any_uuid = true; break; }
            }
            if (!any_uuid) {
                memcpy(d->cache_identity.cache_uuid, d->cache_identity.driver_uuid, 16);
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
        for (auto& av : t->attachment_views) {
            vkDestroyImageView(dd->device, av.view, nullptr);
        }
        t->attachment_views.clear();
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
    d->uninitialized_textures = Vector<TextureImpl*>(d->allocator);
    d->pipeline_records       = Vector<PipelineRecord*>(d->allocator);
    d->compiler_queue         = Vector<PipelineRecord*>(d->allocator);
    d->static_variants        = Vector<StaticVariantRecord*>(d->allocator);
    d->variant_queue          = Vector<StaticVariantRecord*>(d->allocator);

#if defined(IZ_VK_PROFILE_BINDLESS)
    // Valid null-handle behavior: descriptor slot 0 is reserved as null. A
    // real 1x1 dummy texture + views + sampler occupy that slot so a shader
    // reading an uninitialized or zero handle gets a VALID, STABLE descriptor
    // (never undefined descriptor data under partial binding). The dummy's
    // texel content is driver-defined (it is never written) — the guarantee
    // is descriptor validity, not sample data. Runs AFTER the pools are
    // initialized (create_texture needs the texture pool).
    {
        const TextureDesc dummy_desc{
            .type = TextureType::Tex2D,
            .dimensions = {1, 1, 1},
            .format = Format::RGBA8Unorm,
            .usage  = UsageFlags::Sampled | UsageFlags::Storage,
        };
        d->bindless_dummy_texture = create_texture(reinterpret_cast<Device>(d), dummy_desc);
        if (d->bindless_dummy_texture.h == 0) {
            IZ_LOG(d, LogLevel::Error, "bindless: dummy null-texture creation failed");
            result = VK_ERROR_INITIALIZATION_FAILED;
            goto fail;
        }
        TextureImpl& dummy = d->texture_pool[handle_cast<TextureImpl>(d->bindless_dummy_texture)];
        const VkImageViewCreateInfo view_info{
            .sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .pNext    = nullptr,
            .flags    = 0,
            .image    = dummy.vk_image,
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format   = VK_FORMAT_R8G8B8A8_UNORM,
            .components = {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                           VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY},
            .subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                 .baseMipLevel = 0, .levelCount = 1,
                                 .baseArrayLayer = 0, .layerCount = 1},
        };
        if (!write_sampled_descriptor(d, 0, view_info) || !write_storage_descriptor(d, 0, view_info)) {
            IZ_LOG(d, LogLevel::Error, "bindless: dummy slot-0 descriptor write failed");
            result = VK_ERROR_INITIALIZATION_FAILED;
            goto fail;
        }
        const VkSamplerCreateInfo sampler_info{
            .sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            .pNext        = nullptr,
            .flags        = 0,
            .magFilter    = VK_FILTER_NEAREST,
            .minFilter    = VK_FILTER_NEAREST,
            .mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST,
            .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        };
        if (!write_sampler_descriptor(d, 0, sampler_info)) {
            IZ_LOG(d, LogLevel::Error, "bindless: dummy slot-0 sampler write failed");
            result = VK_ERROR_INITIALIZATION_FAILED;
            goto fail;
        }
    }
#endif

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
    // Remaining private static variants first (map-owned; the base records
    // below are destroyed after them).
    for (StaticVariantRecord* v : d->static_variants) {
        if (v->vk_pipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(d->device, v->vk_pipeline, nullptr);
        }
        condvar_destroy(&v->wait_cv);
        v->~StaticVariantRecord();
        d->allocator.free({.ptr = v, .len = sizeof(StaticVariantRecord)});
    }
    d->static_variants.clear();
    d->variant_queue.clear();
    for (PipelineRecord* rec : d->pipeline_records) {
        if (rec->vk_pipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(d->device, rec->vk_pipeline, nullptr);
        }
        free_key(d, rec);
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
#if defined(IZ_VK_PROFILE_BINDLESS)
    // The descriptor-sidecar VkImageViews reference the texture images: they
    // must be destroyed BEFORE the texture pool clears those images.
    for (VkImageView v : d->bindless_sampled_views) {
        if (v != VK_NULL_HANDLE) { vkDestroyImageView(d->device, v, nullptr); }
    }
    d->bindless_sampled_views.clear();
    for (VkImageView v : d->bindless_storage_views) {
        if (v != VK_NULL_HANDLE) { vkDestroyImageView(d->device, v, nullptr); }
    }
    d->bindless_storage_views.clear();
    for (VkSampler smp : d->bindless_sampler_handles) {
        if (smp != VK_NULL_HANDLE) { vkDestroySampler(d->device, smp, nullptr); }
    }
    d->bindless_sampler_handles.clear();
#endif
    d->texture_pool.clear();
    d->semaphore_pool.clear();
    d->pipeline_pool.clear();
    d->depth_stencil_pool.clear();

#if defined(IZ_VK_PROFILE_BINDLESS)
    if (d->bindless_pool != VK_NULL_HANDLE) { vkDestroyDescriptorPool(d->device, d->bindless_pool, nullptr); }
    if (d->bindless_set_layout != VK_NULL_HANDLE) { vkDestroyDescriptorSetLayout(d->device, d->bindless_set_layout, nullptr); }
    if (d->bindless_pipeline_layout != VK_NULL_HANDLE) { vkDestroyPipelineLayout(d->device, d->bindless_pipeline_layout, nullptr); }
#else
    // Destroy descriptor heap buffers
    if (d->heap.sampler_buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(d->vma, d->heap.sampler_buffer, d->heap.sampler_allocation);
    }
    if (d->heap.resource_buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(d->vma, d->heap.resource_buffer, d->heap.resource_allocation);
    }
#endif

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

BackendProfile device_backend_profile() {
#if defined(IZ_VK_PROFILE_BINDLESS)
    return BackendProfile::VulkanBindless;
#else
    return BackendProfile::VulkanNative;
#endif
}

DeviceLimits device_limits(Device dev) {
    auto* d = reinterpret_cast<DeviceImpl*>(dev);
#if defined(IZ_VK_PROFILE_BINDLESS)
    uint32_t sampled = d->bindless_sampled_capacity;
    uint32_t storage = d->bindless_storage_capacity;
    uint32_t sampler = d->bindless_sampler_capacity;
#else
    uint32_t sampled = d->heap.sampled_capacity;
    uint32_t storage = d->heap.storage_capacity;
    uint32_t sampler = d->heap.sampler_capacity;
#endif
    return DeviceLimits{
        .max_sampled_textures = sampled,
        .max_storage_textures = storage,
        .max_samplers         = sampler,
        .min_uniform_alignment = d->min_uniform_alignment,
        .min_storage_alignment = d->min_storage_alignment,
        .non_coherent_atom_size = d->non_coherent_atom_size,
    };
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

// Releases one buffer reference; the pool slot (and native buffer) is
// destroyed with the last (1 per live allocation + command-buffer retentions).
void release_buffer_ref(DeviceImpl* d, Handle<Buffer> buf) {
    auto& b = d->buffer_pool[handle_cast<Buffer>(buf)];
    if (atomic_fetch_add(&b.refs, -1) == 1) {
        d->buffer_pool.erase(buf);
    }
}

void free(Device dev, GpuPtr ptr) {
    auto* d = reinterpret_cast<DeviceImpl*>(dev);
    rwlock_lock_write(&d->ptr_map_lock);
    // Binary search for exact match
    uint32_t lo = 0, hi = d->ptr_map.size();
    while (lo < hi) {
        uint32_t mid = (lo + hi) / 2;
        if (d->ptr_map[mid].ptr == ptr) {
            const Handle<Buffer> h = d->ptr_map[mid].buffer;
            d->ptr_map.erase(d->ptr_map.begin() + mid, d->ptr_map.begin() + mid + 1);
            rwlock_unlock_write(&d->ptr_map_lock);
            release_buffer_ref(d, h);
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

    // Collect due completion events and retirement batches under the submit
    // lock (queue_process_events is thread-safe); fire callbacks and process
    // batches outside it so no application callback or native destruction runs
    // while holding queue locks.
    Vector<CompletionEvent> due_events(d->allocator);
    Vector<RetireBatch*>    due_batches(d->allocator);
    mutex_lock(&q->submit_lock);
    uint32_t i = 0;
    while (i < q->pending_events.size() && q->pending_events[i].completed_time <= current_time) { i++; }
    for (uint32_t k = 0; k < i; ++k) { due_events.push_back(q->pending_events[k]); }
    if (i != 0) {
        q->pending_events.erase(q->pending_events.begin(), q->pending_events.begin() + i);
    }
    uint32_t j = 0;
    while (j < q->retire_queue.size() && q->retire_queue[j]->value <= current_time) { j++; }
    for (uint32_t k = 0; k < j; ++k) { due_batches.push_back(q->retire_queue[k]); }
    if (j != 0) {
        q->retire_queue.erase(q->retire_queue.begin(), q->retire_queue.begin() + j);
    }
    mutex_unlock(&q->submit_lock);

    for (const CompletionEvent& e : due_events) { e.callback(e.userdata); }
    for (RetireBatch* batch : due_batches) { process_retire_batch(d, batch); }
}


}  // namespace gpu
