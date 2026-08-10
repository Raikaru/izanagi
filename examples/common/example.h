#pragma once
// Example framework: registry, frame loop, screenshot.

#include "izanagi/gpu.h"

// The swapchain format negotiated by the framework before example init.
// Examples must use this (not a hardcoded sRGB format) for their color target,
// since some drivers/swapchains don't offer sRGB variants.
extern gpu::Format g_example_surface_format;

// Load a compiled SPIR-V shader from <exe_dir>/shaders/<name>.
// Returns empty span on failure. Example init must check size > 0 and
// fail cleanly (skip pipeline creation) if the shader is missing.
gpu::Span<const uint8_t> example_load_shader(const char* name, uint32_t* out_size);

struct ExampleVTable {
    void (*init)(gpu::Device, void* userdata);
    bool (*render)(gpu::Device, gpu::CommandBuffer, gpu::Handle<gpu::Texture> color_texture,
                   gpu::Dimension2D size, float time, void* userdata);
    void (*shutdown)(gpu::Device, void* userdata);
    void* userdata;
};

void register_example_vtables();
int get_example_count();
const char* get_example_name(int idx);
ExampleVTable* get_example_vtable(int idx);
int find_example_by_name(const char* name);
