// compute_texture.cpp — compute shader writes animated pattern to storage texture,
// fullscreen triangle samples it. Exercises heap storage+sampled slots.

#include "../common/example.h"
#include "../common/gpu_args.h"
#include <cstdio>
#include <string>
#include <cstring>

using namespace gpu;

struct ComputeTextureState {
    Handle<Pipeline>  compute_pipeline;
    Handle<Pipeline>  graphics_pipeline;
    Handle<Texture>   storage_texture;
    TextureView       sampled_view;
    TextureView       storage_view;
    SamplerId         sampler;
    GpuArgs           args;
    uint32_t          tex_width  = 256;
    uint32_t          tex_height = 256;
    bool              ready = false;
};

static void init(Device device, void* userdata) {
    auto* state = new ComputeTextureState();
    *(void**)userdata = state;

    uint32_t spv_size;
    auto spv = example_load_shader("compute_texture.spv", &spv_size);
    if (spv.size() == 0) {
        printf("compute_texture: shader missing, example disabled\n");
        return;
    }

    // Compute pipeline
    ShaderSource compute_src{
        .source      = spv,
        .entry_point = "compute_main"_sv,
    };
    state->compute_pipeline = create_compute_pipeline(device, compute_src);

    // Graphics pipeline (fullscreen triangle sampling)
    ShaderSource vertex_src{
        .source      = spv,
        .entry_point = "vertex_main"_sv,
    };
    ShaderSource fragment_src{
        .source      = spv,
        .entry_point = "fragment_main"_sv,
    };
    ColorTarget color_target{
        .format = g_example_surface_format,
    };
    RasterDesc raster_desc{
        .color_targets = Span<const ColorTarget>(&color_target, 1),
    };
    state->graphics_pipeline = create_graphics_pipeline(device, vertex_src, fragment_src, raster_desc);
    state->ready = state->compute_pipeline.h != 0 && state->graphics_pipeline.h != 0;

    // Storage texture
    TextureDesc tex_desc{
        .type       = TextureType::Tex2D,
        .dimensions = {state->tex_width, state->tex_height, 1},
        .format     = Format::RGBA8Unorm,
        .usage      = UsageFlags::Storage | UsageFlags::Sampled,
    };
    state->storage_texture = create_texture(device, tex_desc);

    // Heap views
    TextureViewDesc view_desc{
        .texture = state->storage_texture,
        .format  = Format::RGBA8Unorm,
    };
    state->sampled_view = create_texture_view(device, view_desc);
    state->storage_view = create_rw_texture_view(device, view_desc);

    // Sampler
    SamplerDesc sampler_desc{
        .coord       = SamplerCoords::Normalized,
        .min_filter  = SamplerFilter::Linear,
        .mag_filter  = SamplerFilter::Linear,
        .mip_filter  = SamplerFilter::Linear,
        .address     = SamplerAddressing::ClampToEdge,
    };
    state->sampler = create_sampler(device, sampler_desc);

    gpu_args_init(&state->args, device);
}

static bool render(Device device, CommandBuffer cmd, Handle<Texture> color_texture,
                   Dimension2D size, float time, void* userdata) {
    auto* state = *(ComputeTextureState**)userdata;
    if (!state->ready) { return false; }

    gpu_args_begin_frame(&state->args);

    // 1. Dispatch compute shader to write pattern
    struct ComputeData {
        uint64_t texture;
        float    time;
        uint32_t width;
        uint32_t height;
    };
    ComputeData compute_data{
        .texture = state->storage_view,
        .time    = time,
        .width   = state->tex_width,
        .height  = state->tex_height,
    };
    GpuPtr compute_args_ptr = gpu_args_append(&state->args, compute_data);

    cmd_set_pipeline(cmd, state->compute_pipeline);
    Dimension3D groups{(state->tex_width + 7) / 8, (state->tex_height + 7) / 8, 1};
    cmd_dispatch(cmd, compute_args_ptr, groups);

    // 2. Barrier: compute write -> fragment shader read
    cmd_barrier(cmd, StageFlags::Compute, StageFlags::PixelShader);

    // 3. Render fullscreen triangle
    struct FragmentData {
        uint64_t texture;
        uint64_t sampler;
    };
    FragmentData frag_data{
        .texture = state->sampled_view,
        .sampler = state->sampler,
    };
    GpuPtr frag_args_ptr = gpu_args_append(&state->args, frag_data);

    RenderAttachment color_att{
        .texture     = color_texture,
        .load_op     = LoadOp::Clear,
        .store_op    = StoreOp::Store,
        .clear_color = Color{0, 0, 0, 0},
    };
    RenderPassDesc pass_desc{
        .color_attachments = Span<const RenderAttachment>(&color_att, 1),
        .render_area       = Rect2D{.width = size.x, .height = size.y},
    };
    cmd_begin_render_pass(cmd, pass_desc);
    cmd_set_pipeline(cmd, state->graphics_pipeline);
    cmd_draw(cmd, 0, frag_args_ptr, 3, 1);
    cmd_end_render_pass(cmd);

    gpu_args_end_frame(&state->args);
    return true;
}

static void shutdown(Device device, void* userdata) {
    auto* state = *(ComputeTextureState**)userdata;
    gpu_args_shutdown(&state->args, device);
    free_texture_view(device, state->sampled_view);
    free_rw_texture_view(device, state->storage_view);
    free_sampler(device, state->sampler);
    free(device, state->storage_texture);
    free(device, state->compute_pipeline);
    free(device, state->graphics_pipeline);
    delete state;
}

ExampleVTable compute_texture_vtable = {
    .init     = init,
    .render   = render,
    .shutdown = shutdown,
    .userdata = nullptr,
};
