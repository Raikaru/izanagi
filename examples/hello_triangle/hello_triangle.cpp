// hello_triangle.cpp — basic triangle, no depth, no heap use.

#include "../common/example.h"
#include "../common/gpu_args.h"
#include <cstdio>
#include <cstring>

using namespace gpu;

struct HelloTriangleState {
    Handle<Pipeline> pipeline;
    GpuArgs          args;
    bool             ready = false;
};

static void init(Device device, void* userdata) {
    auto* state = new HelloTriangleState();
    *(void**)userdata = state;

    uint32_t spv_size;
    auto spv = example_load_shader("hello_triangle.spv", &spv_size);
    if (spv.size() == 0) {
        printf("hello_triangle: shader missing, example disabled\n");
        return;
    }

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
    state->pipeline = create_graphics_pipeline(device, vertex_src, fragment_src, raster_desc);
    state->ready    = state->pipeline.h != 0;

    gpu_args_init(&state->args, device);
}

static bool render(Device device, CommandBuffer cmd, Handle<Texture> color_texture,
                   Dimension2D size, float time, void* userdata) {
    auto* state = *(HelloTriangleState**)userdata;
    if (!state->ready) { return false; }

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
    cmd_set_pipeline(cmd, state->pipeline);
    cmd_draw(cmd, 0, 0, 3, 1);
    cmd_end_render_pass(cmd);

    return true;
}

static void shutdown(Device device, void* userdata) {
    auto* state = *(HelloTriangleState**)userdata;
    gpu_args_shutdown(&state->args, device);
    free(device, state->pipeline);
    delete state;
}

ExampleVTable hello_triangle_vtable = {
    .init     = init,
    .render   = render,
    .shutdown = shutdown,
    .userdata = nullptr,
};
