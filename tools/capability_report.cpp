// capability_report.cpp — prints a structured, machine-readable capability
// report (JSON) for the selected backend profile. Used by CI to archive the
// exact device/driver/profile combination a conformance run was certified on.
//
// White-box: reads the backend's physical-device identity directly. This is a
// build/test tool, not part of the public API.

#include <cstdio>

#include "izanagi/gpu.h"
#include "vk/internal.h"

using namespace gpu;

static void log_cb(LogLevel, Span<const char>, uint32_t, Span<const char>, void*) {}

int main() {
    DeviceDesc desc{
        .log_callback = log_cb,
        .log_level    = LogLevel::Error,
    };
    Device dev = create_device(desc);
    if (dev == nullptr) {
        printf("{\"backend\":\"VulkanNative\",\"profile\":\"%s\",\"profile_supported\":false,"
               "\"missing_features\":[\"device creation failed\"]}\n", IZ_PROFILE);
        return 1;
    }
    auto* d = reinterpret_cast<DeviceImpl*>(dev);
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(d->physical_device, &props);
    DeviceLimits lim = device_limits(dev);

    // Evaluate the compatibility (BINDLESS) profile against this device's
    // actual feature bits — the same decision the compatibility backend will
    // make at device creation. Reported regardless of the compiled profile.
    VulkanProfileReport bindless = evaluate_vulkan_bindless_profile(query_vulkan_profile_features(d));

    const char* platform = "Unknown";
#if defined(_WIN32)
    platform = "Windows";
#elif defined(__APPLE__)
    platform = "Apple";
#elif defined(__ANDROID__)
    platform = "Android";
#elif defined(__linux__)
    platform = "Linux";
#endif

    printf("{\n");
    printf("  \"backend\": \"VulkanNative\",\n");
    printf("  \"profile\": \"%s\",\n", IZ_PROFILE);
    printf("  \"profile_supported\": true,\n");
    printf("  \"platform\": \"%s\",\n", platform);
    printf("  \"device_name\": \"%s\",\n", props.deviceName);
    printf("  \"api_version\": \"%u.%u.%u\",\n",
           VK_API_VERSION_MAJOR(props.apiVersion), VK_API_VERSION_MINOR(props.apiVersion),
           VK_API_VERSION_PATCH(props.apiVersion));
    printf("  \"driver_version\": %u,\n", props.driverVersion);
    printf("  \"vendor_id\": %u,\n", props.vendorID);
    printf("  \"device_id\": %u,\n", props.deviceID);
    printf("  \"missing_features\": [],\n");
    printf("  \"descriptor_capacity\": {\n");
    printf("    \"sampled_textures\": %u,\n", lim.max_sampled_textures);
    printf("    \"storage_textures\": %u,\n", lim.max_storage_textures);
    printf("    \"samplers\": %u\n", lim.max_samplers);
    printf("  },\n");
    printf("  \"pipeline_cache_control\": %s,\n", d->pipeline_cache_control ? "true" : "false");
    printf("  \"bindless_profile\": {\n");
    printf("    \"supported\": %s,\n", bindless.supported ? "true" : "false");
    printf("    \"api_version\": \"%u.%u.%u\",\n",
           VK_API_VERSION_MAJOR(bindless.api_version), VK_API_VERSION_MINOR(bindless.api_version),
           VK_API_VERSION_PATCH(bindless.api_version));
    printf("    \"real_gpu_pointers\": %s,\n",
           !bindless.missing_has(VulkanProfileRequirement::BufferDeviceAddress) ? "true" : "false");
    printf("    \"dynamic_rendering\": %s,\n", bindless.dynamic_rendering ? "true" : "false");
    printf("    \"synchronization2\": %s,\n", bindless.synchronization2 ? "true" : "false");
    printf("    \"descriptor_set_snapshots\": %u,\n", bindless.descriptor_snapshots);
    printf("    \"sampled_image_capacity\": %u,\n", bindless.sampled_image_capacity);
    printf("    \"storage_image_capacity\": %u,\n", bindless.storage_image_capacity);
    printf("    \"sampler_capacity\": %u,\n", bindless.sampler_capacity);
    printf("    \"missing_requirements\": [");
    for (uint32_t i = 0; i < bindless.missing_count; ++i) {
        if (i != 0) { printf(", "); }
        printf("\"%s\"", vulkan_requirement_name(bindless.missing[i]));
    }
    printf("]\n");
    printf("  }\n");
    printf("}\n");

    destroy_device(dev);
    return 0;
}
