#include "common/dispatch_capabilities.h"

namespace gpu {

// Vulkan 1.3 API version word (VK_MAKE_API_VERSION(0, 1, 3, 0)) — defined
// locally so this module stays Vulkan-header-free.
inline constexpr uint32_t kApiVersion13 = (1u << 22) | (3u << 12);

VulkanDispatchCapabilities derive_dispatch_capabilities(uint32_t effective_api_version,
                                                        bool has_dynamic_rendering_ext,
                                                        bool has_sync2_ext,
                                                        bool has_copy2_ext,
                                                        bool has_extended_dynamic_state_ext,
                                                        bool force_legacy_copy,
                                                        bool force_static_state) {
    VulkanDispatchCapabilities c;
    c.effective_api_version = effective_api_version;

    const bool core = effective_api_version >= kApiVersion13;

    c.dynamic_rendering_is_core      = core;
    c.synchronization2_is_core       = core;
    c.copy_commands2_is_core         = core;
    c.extended_dynamic_state_is_core = core;

    // A family is available when core OR its extension is exported.
    c.dynamic_rendering      = core || has_dynamic_rendering_ext;
    c.synchronization2       = core || has_sync2_ext;
    c.copy_commands2         = core || has_copy2_ext;
    c.extended_dynamic_state = core || has_extended_dynamic_state_ext;

    // Fallbacks: selected when the family is unavailable OR force-tested.
    // Force overrides a normally available modern path (white-box tests).
    c.use_legacy_copy_commands  = force_legacy_copy || !c.copy_commands2;
    c.use_static_graphics_state = force_static_state || !c.extended_dynamic_state;
    return c;
}

}  // namespace gpu
