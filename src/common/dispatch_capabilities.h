#pragma once
// Vulkan 1.2 dispatch-capability derivation — pure logic, no Vulkan types
// (the effective API version is passed as a raw version word, e.g.
// VK_MAKE_API_VERSION values). Injectable for the capability matrix tests;
// the backend fills the inputs from the selected physical device.

#include <cstdint>

namespace gpu {

struct VulkanDispatchCapabilities {
    uint32_t effective_api_version = 0;

    bool dynamic_rendering      = false;
    bool synchronization2       = false;
    bool copy_commands2         = false;
    bool extended_dynamic_state = false;

    bool dynamic_rendering_is_core      = false;
    bool synchronization2_is_core       = false;
    bool copy_commands2_is_core         = false;
    bool extended_dynamic_state_is_core = false;

    bool use_legacy_copy_commands  = false;
    bool use_static_graphics_state = false;
};

VulkanDispatchCapabilities derive_dispatch_capabilities(uint32_t effective_api_version,
                                                        bool has_dynamic_rendering_ext,
                                                        bool has_sync2_ext,
                                                        bool has_copy2_ext,
                                                        bool has_extended_dynamic_state_ext,
                                                        bool force_legacy_copy,
                                                        bool force_static_state);

}  // namespace gpu
