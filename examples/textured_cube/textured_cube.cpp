// textured_cube.cpp — rotating checkerboard cube with depth testing.
// Exercises: depth texture, index buffer, vertex buffers via BDA, heap views,
// MVP via gpu_args, depth-stencil state.

#include "../common/example.h"
#include "../common/gpu_args.h"
#include "../common/math.h"
#include <cstdio>
#include <string>
#include <cstring>

using namespace gpu;

struct TexturedCubeState {
    Handle<Pipeline>         pipeline;
    Handle<DepthStencilState> depth_state;
    Handle<Texture>          checker_texture;
    Handle<Texture>          depth_texture;
    TextureView              checker_view;
    SamplerId                sampler;
    GpuPtr                   vertex_buffer; // positions + uvs interleaved
    GpuPtr                   index_buffer;
    GpuArgs                  args;
    uint32_t                 depth_width  = 0;
    uint32_t                 depth_height = 0;
    bool                     ready        = false;
};

// Cube vertices: 8 corners, each with position + uv
static const float cube_positions[] = {
    // Front face
    -0.5f, -0.5f,  0.5f,   0.5f, -0.5f,  0.5f,   0.5f,  0.5f,  0.5f,  -0.5f,  0.5f,  0.5f,
    // Back face
    -0.5f, -0.5f, -0.5f,   0.5f, -0.5f, -0.5f,   0.5f,  0.5f, -0.5f,  -0.5f,  0.5f, -0.5f,
};

static const float cube_uvs[] = {
    0.0f, 0.0f,  1.0f, 0.0f,  1.0f, 1.0f,  0.0f, 1.0f,
    1.0f, 0.0f,  0.0f, 0.0f,  0.0f, 1.0f,  1.0f, 1.0f,
};

// 36 indices for 12 triangles (6 faces × 2 triangles)
static const uint16_t cube_indices[] = {
    // Front
    0, 1, 2,  0, 2, 3,
    // Back
    5, 4, 7,  5, 7, 6,
    // Top
    3, 2, 6,  3, 6, 7,
    // Bottom
    4, 5, 1,  4, 1, 0,
    // Right
    1, 5, 6,  1, 6, 2,
    // Left
    4, 0, 3,  4, 3, 7,
};

// 4x4 checkerboard pattern (RGBA8)
static uint8_t checker_pattern[4 * 4 * 4];
static void init_checker() {
    for (int y = 0; y < 4; ++y) {
        for (int x = 0; x < 4; ++x) {
            int idx = (y * 4 + x) * 4;
            bool white = ((x + y) % 2) == 0;
            checker_pattern[idx + 0] = white ? 255 : 30;
            checker_pattern[idx + 1] = white ? 255 : 30;
            checker_pattern[idx + 2] = white ? 255 : 30;
            checker_pattern[idx + 3] = 255;
        }
    }
}

static void init(Device device, void* userdata) {
    auto* state = new TexturedCubeState();
    *(void**)userdata = state;

    init_checker();

    uint32_t spv_size;
    auto spv = example_load_shader("textured_cube.spv", &spv_size);
    if (spv.size() == 0) {
        printf("textured_cube: shader missing, example disabled\n");
        return;
    }

    // Graphics pipeline with depth
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
        .depth_format  = Format::Depth32Float,
        .color_targets = Span<const ColorTarget>(&color_target, 1),
    };
    state->pipeline = create_graphics_pipeline(device, vertex_src, fragment_src, raster_desc);
    state->ready = state->pipeline.h != 0;

    // Depth-stencil state
    DepthStencilDesc ds_desc{
        .depth_mode = DepthFlags::Read | DepthFlags::Write,
        .depth_test = Op::LessEqual,
    };
    state->depth_state = create_depth_stencil_state(device, ds_desc);

    // Checkerboard texture
    TextureDesc checker_desc{
        .type       = TextureType::Tex2D,
        .dimensions = {4, 4, 1},
        .format     = Format::RGBA8Unorm,
        .usage      = UsageFlags::Sampled | UsageFlags::TransferDst,
    };
    state->checker_texture = create_texture(device, checker_desc);

    // Upload checkerboard
    GpuPtr staging = malloc(device, sizeof(checker_pattern), Memory::Default);
    memcpy(get_host_pointer(device, staging), checker_pattern, sizeof(checker_pattern));

    Queue q = get_queue(device);
    CommandBuffer upload_cmd = queue_start_command_recording(q);
    BufferTextureCopyInfo copy_info{
        .image_extent = {4, 4, 1},
    };
    cmd_copy_to_texture(upload_cmd, staging, state->checker_texture, copy_info);
    cmd_barrier(upload_cmd, StageFlags::Transfer, StageFlags::PixelShader);
    cmd_finalize(upload_cmd);
    queue_submit(q, {&upload_cmd, 1});
    device_wait_for_idle(device);
    free(device, staging);

    // Heap view + sampler
    TextureViewDesc view_desc{
        .texture = state->checker_texture,
        .format  = Format::RGBA8Unorm,
    };
    state->checker_view = create_texture_view(device, view_desc);

    SamplerDesc sampler_desc{
        .coord   = SamplerCoords::Normalized,
        .filter  = SamplerFilter::Nearest,
        .address = SamplerAddressing::Repeat,
    };
    state->sampler = create_sampler(device, sampler_desc);

    // Vertex + index buffers in GPU memory
    state->vertex_buffer = malloc(device, sizeof(cube_positions) + sizeof(cube_uvs), Memory::Gpu);
    state->index_buffer  = malloc(device, sizeof(cube_indices), Memory::Gpu);

    // Upload vertex data via staging
    GpuPtr vstaging = malloc(device, sizeof(cube_positions) + sizeof(cube_uvs), Memory::Default);
    auto* vptr = static_cast<uint8_t*>(get_host_pointer(device, vstaging));
    memcpy(vptr, cube_positions, sizeof(cube_positions));
    memcpy(vptr + sizeof(cube_positions), cube_uvs, sizeof(cube_uvs));

    GpuPtr istaging = malloc(device, sizeof(cube_indices), Memory::Default);
    memcpy(get_host_pointer(device, istaging), cube_indices, sizeof(cube_indices));

    upload_cmd = queue_start_command_recording(q);
    cmd_memcpy(upload_cmd, state->vertex_buffer, vstaging, sizeof(cube_positions) + sizeof(cube_uvs));
    cmd_memcpy(upload_cmd, state->index_buffer, istaging, sizeof(cube_indices));
    cmd_barrier(upload_cmd, StageFlags::Transfer, StageFlags::VertexShader);
    cmd_finalize(upload_cmd);
    queue_submit(q, {&upload_cmd, 1});
    device_wait_for_idle(device);
    free(device, vstaging);
    free(device, istaging);

    gpu_args_init(&state->args, device);
}

static bool render(Device device, CommandBuffer cmd, Handle<Texture> color_texture,
                   Dimension2D size, float time, void* userdata) {
    auto* state = *(TexturedCubeState**)userdata;
    if (!state->ready) { return false; }

    // Recreate depth texture if size changed
    if (size.x != state->depth_width || size.y != state->depth_height) {
        if (state->depth_texture.h != 0) {
            free(device, state->depth_texture);
        }
        TextureDesc depth_desc{
            .type       = TextureType::Tex2D,
            .dimensions = {size.x, size.y, 1},
            .format     = Format::Depth32Float,
            .usage      = UsageFlags::DepthStencilAttachment,
        };
        state->depth_texture = create_texture(device, depth_desc);
        state->depth_width   = size.x;
        state->depth_height  = size.y;
    }

    gpu_args_begin_frame(&state->args);

    // Compute MVP
    float aspect = (float)size.x / (float)size.y;
    float4x4 projection = float4x4::perspective(3.14159f / 4.0f, aspect, 0.1f, 100.0f);
    float4 eye{0.0f, 0.0f, 3.0f, 1.0f};
    float4 center{0.0f, 0.0f, 0.0f, 1.0f};
    float4 up{0.0f, 1.0f, 0.0f, 1.0f};
    float4x4 view = float4x4::lookAt(eye, center, up);
    float4x4 world = float4x4::rotate_y(time);

    // Vertex data struct (matches shader layout)
    struct CameraData {
        float4x4 projection;
        float4x4 cameraFromWorld;
    };
    struct MeshData {
        uint64_t position;
        uint64_t uvs;
        float4x4 worldFromMesh;
    };
    struct InputData {
        CameraData camera;
        MeshData   mesh;
    };

    InputData input_data{};
    input_data.camera.projection      = projection.transposed();
    input_data.camera.cameraFromWorld = view.transposed();
    input_data.mesh.position          = state->vertex_buffer;
    input_data.mesh.uvs               = state->vertex_buffer + sizeof(cube_positions);
    input_data.mesh.worldFromMesh     = world.transposed();

    GpuPtr vert_args = gpu_args_append(&state->args, input_data);

    struct FragmentData {
        uint64_t texture;
        uint64_t sampler;
    };
    FragmentData frag_data{
        .texture = state->checker_view,
        .sampler = state->sampler,
    };
    GpuPtr frag_args = gpu_args_append(&state->args, frag_data);

    // Render pass
    RenderAttachment color_att{
        .texture     = color_texture,
        .load_op     = LoadOp::Clear,
        .store_op    = StoreOp::Store,
        .clear_color = Color{20, 20, 40, 255},
    };
    RenderAttachment depth_att{
        .texture     = state->depth_texture,
        .load_op     = LoadOp::Clear,
        .store_op    = StoreOp::Store,
        .clear_color = Color{255, 0, 0, 0}, // depth clear: r/255 = 1.0 (far)
    };
    RenderPassDesc pass_desc{
        .color_attachments = Span<const RenderAttachment>(&color_att, 1),
        .depth_attachment  = depth_att,
        .render_area       = Rect2D{.width = size.x, .height = size.y},
    };
    cmd_begin_render_pass(cmd, pass_desc);
    cmd_set_pipeline(cmd, state->pipeline);
    cmd_set_depth_stencil_state(cmd, state->depth_state);
    // Viewport Y-flip mirrors winding: y-up CCW data is CW in framebuffer space.
    cmd_set_front_face(cmd, FrontFace::CW);
    cmd_set_cull_mode(cmd, Cull::Back);

    DrawIndexedInstancedInfo draw_info{
        .vertexDataGpu  = vert_args,
        .fragmentDataGpu = frag_args,
        .indicesGpu     = state->index_buffer,
        .indexCount     = 36,
        .instanceCount  = 1,
        .type           = IndexType::UInt16,
    };
    cmd_draw_indexed_instanced(cmd, draw_info);

    cmd_end_render_pass(cmd);

    gpu_args_end_frame(&state->args);
    return true;
}

static void shutdown(Device device, void* userdata) {
    auto* state = *(TexturedCubeState**)userdata;
    gpu_args_shutdown(&state->args, device);
    free_texture_view(device, state->checker_view);
    free_sampler(device, state->sampler);
    free(device, state->checker_texture);
    if (state->depth_texture.h != 0) { free(device, state->depth_texture); }
    free(device, state->vertex_buffer);
    free(device, state->index_buffer);
    free_depth_stencil_state(device, state->depth_state);
    free(device, state->pipeline);
    delete state;
}

ExampleVTable textured_cube_vtable = {
    .init     = init,
    .render   = render,
    .shutdown = shutdown,
    .userdata = nullptr,
};
