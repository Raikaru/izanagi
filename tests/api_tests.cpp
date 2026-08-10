// api_tests.cpp — headless API tests for the Izanagi Vulkan backend.
// Plain main() returning nonzero on failure; no test framework.

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#endif

#include "izanagi/gpu.h"

using namespace gpu;

static int g_failures = 0;

#define CHECK(cond, msg)                                                                        \
    do {                                                                                        \
        if (!(cond)) {                                                                          \
            printf("FAIL [%s:%d]: %s\n", __FILE__, __LINE__, msg);                              \
            g_failures++;                                                                       \
        }                                                                                       \
    } while (0)

static void test_log_callback(LogLevel lvl, Span<const char> msg, uint32_t line,
                               Span<const char> file, void*) {
    const char* lvl_str[] = {"OFF", "ERROR", "WARN", "INFO", "DEBUG"};
    printf("[%s] %.*s (%.*s:%u)\n", lvl_str[static_cast<int>(lvl)],
           (int)msg.size(), msg.data(), (int)file.size(), file.data(), line);
}

// Simple bump arena for test use
struct Arena {
    uint8_t* base;
    size_t   offset;
    size_t   capacity;
    void*    alloc(size_t size) {
        size = (size + 15) & ~size_t(15);
        if (offset + size > capacity) { return nullptr; }
        void* p = base + offset;
        offset += size;
        return p;
    }
};

// Load a SPIR-V file into memory
static Span<const uint8_t> load_spirv(const char* path, Arena* arena) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) { return {}; }
    auto size = file.tellg();
    file.seekg(0);
    auto* data = reinterpret_cast<uint8_t*>(arena->alloc(size));
    file.read(reinterpret_cast<char*>(data), size);
    return Span<const uint8_t>(data, static_cast<uint32_t>(size));
}

static char g_arena_mem[4 * 1024 * 1024];

// Find the shader directory relative to the executable
static std::string find_shader_path(const char* name) {
    // Resolve relative to the executable directory so tests run from any CWD.
#ifdef _WIN32
    char exe_path[2048];
    DWORD len = GetModuleFileNameA(nullptr, exe_path, sizeof(exe_path));
    if (len > 0 && len < sizeof(exe_path)) {
        char* slash = strrchr(exe_path, '\\');
        if (slash) {
            *(slash + 1) = '\0';
            std::string path = std::string(exe_path) + "shaders/" + name;
            if (std::filesystem::exists(path)) { return path; }
        }
    }
#endif
    // Fallback: try relative candidates
    const char* candidates[] = {
        "shaders/",
        "../shaders/",
        "../../shaders/",
        "../../../shaders/",
    };
    for (auto* prefix : candidates) {
        std::string path = std::string(prefix) + name;
        if (std::filesystem::exists(path)) { return path; }
    }
    return std::string("shaders/") + name;
}

// --- Test 1: Device create/destroy ------------------------------------------------
static void test_device_create_destroy() {
    printf("--- Test: device create/destroy ---\n");
    DeviceDesc desc{
        .log_callback = test_log_callback,
        .log_level    = LogLevel::Warning,
    };
    Device d = create_device(desc);
    CHECK(d != nullptr, "create_device returned null");
    CHECK(device_backend() == Backend::Vulkan, "backend should be Vulkan");
    destroy_device(d);
    printf("  PASS\n");
}

// --- Test 2: malloc/get_host_pointer roundtrip -------------------------------------
static void test_malloc_host_pointer() {
    printf("--- Test: malloc/get_host_pointer roundtrip ---\n");
    DeviceDesc desc{
        .log_callback = test_log_callback,
        .log_level    = LogLevel::Warning,
    };
    Device d = create_device(desc);
    CHECK(d != nullptr, "create_device returned null");

    // Allocate two Default (host-visible) buffers
    GpuPtr src = malloc(d, 256, Memory::Default);
    GpuPtr dst = malloc(d, 256, Memory::Default);
    CHECK(src != 0, "malloc src failed");
    CHECK(dst != 0, "malloc dst failed");

    // Write pattern via host pointer
    void* src_host = get_host_pointer(d, src);
    CHECK(src_host != nullptr, "get_host_pointer src failed");
    uint32_t* pattern = reinterpret_cast<uint32_t*>(src_host);
    for (int i = 0; i < 64; ++i) { pattern[i] = 0xDEAD0000 + i; }

    // Copy via GPU
    Queue q = get_queue(d);
    CHECK(q != nullptr, "get_queue failed");
    CommandBuffer cmd = queue_start_command_recording(q);
    CHECK(cmd != nullptr, "queue_start_command_recording failed");
    cmd_memcpy(cmd, dst, src, 256);
    cmd_finalize(cmd);
    queue_submit(q, {&cmd, 1});

    // Wait for completion
    device_wait_for_idle(d);

    // Verify
    void* dst_host = get_host_pointer(d, dst);
    CHECK(dst_host != nullptr, "get_host_pointer dst failed");
    uint32_t* result = reinterpret_cast<uint32_t*>(dst_host);
    bool match = true;
    for (int i = 0; i < 64; ++i) {
        if (result[i] != 0xDEAD0000 + (uint32_t)i) {
            printf("  Mismatch at index %d: expected 0x%X, got 0x%X\n", i, 0xDEAD0000 + i, result[i]);
            match = false;
        }
    }
    CHECK(match, "GPU memcpy verification failed");

    free(d, src);
    free(d, dst);
    destroy_device(d);
    printf("  PASS\n");
}

// --- Test 3: Compute end-to-end ------------------------------------------------------
static void test_compute() {
    printf("--- Test: compute end-to-end ---\n");
    DeviceDesc desc{
        .log_callback = test_log_callback,
        .log_level    = LogLevel::Warning,
    };
    Device d = create_device(desc);
    CHECK(d != nullptr, "create_device returned null");

    // Load compute shader
    Arena arena{reinterpret_cast<uint8_t*>(g_arena_mem), 0, sizeof(g_arena_mem)};
    std::string shader_path = find_shader_path("memcpy_kernel.spv");
    auto spirv = load_spirv(shader_path.c_str(), &arena);
    CHECK(spirv.size() > 0, "Failed to load memcpy_kernel.spv");
    if (spirv.size() == 0) {
        destroy_device(d);
        return;
    }

    // Create compute pipeline
    ShaderSource shader_src{
        .source      = spirv,
        .entry_point = "compute_main"_sv,
    };
    Handle<Pipeline> pipeline = create_compute_pipeline(d, shader_src);
    CHECK(pipeline.h != 0, "create_compute_pipeline failed");
    if (pipeline.h == 0) {
        destroy_device(d);
        return;
    }

    // Allocate buffers
    constexpr uint32_t kCount = 1024;
    GpuPtr src_buf = malloc(d, kCount * sizeof(uint32_t), Memory::Default);
    GpuPtr dst_buf = malloc(d, kCount * sizeof(uint32_t), Memory::Default);
    GpuPtr args_buf = malloc(d, 32, Memory::Default); // CopyData struct
    CHECK(src_buf != 0 && dst_buf != 0 && args_buf != 0, "malloc failed");

    // Write input data
    auto* src_host = reinterpret_cast<uint32_t*>(get_host_pointer(d, src_buf));
    for (uint32_t i = 0; i < kCount; ++i) { src_host[i] = i; }

    // Write args struct
    struct CopyData {
        uint64_t dst;
        uint64_t src;
        uint32_t count;
        uint32_t pad;
    };
    auto* args_host = reinterpret_cast<CopyData*>(get_host_pointer(d, args_buf));
    args_host->dst   = dst_buf;
    args_host->src   = src_buf;
    args_host->count = kCount;
    args_host->pad   = 0;

    // Dispatch
    Queue q = get_queue(d);
    CommandBuffer cmd = queue_start_command_recording(q);
    cmd_set_pipeline(cmd, pipeline);
    Dimension3D groups{(kCount + 63) / 64, 1, 1};
    cmd_dispatch(cmd, args_buf, groups);
    cmd_finalize(cmd);
    queue_submit(q, {&cmd, 1});

    device_wait_for_idle(d);

    // Verify
    auto* dst_host = reinterpret_cast<uint32_t*>(get_host_pointer(d, dst_buf));
    bool match = true;
    for (uint32_t i = 0; i < kCount; ++i) {
        uint32_t expected = i * 2 + 1;
        if (dst_host[i] != expected) {
            printf("  Mismatch at %u: expected %u, got %u\n", i, expected, dst_host[i]);
            match = false;
            break;
        }
    }
    CHECK(match, "compute dst[i] = src[i]*2+1 verification failed");

    free(d, src_buf);
    free(d, dst_buf);
    free(d, args_buf);
    free(d, pipeline);
    destroy_device(d);
    printf("  PASS\n");
}

// --- Test 4: Texture upload/readback ---------------------------------------------------
static void test_texture_copy() {
    printf("--- Test: texture upload/readback ---\n");
    DeviceDesc desc{
        .log_callback = test_log_callback,
        .log_level    = LogLevel::Warning,
    };
    Device d = create_device(desc);
    CHECK(d != nullptr, "create_device returned null");

    constexpr uint32_t kWidth  = 16;
    constexpr uint32_t kHeight = 16;
    constexpr uint32_t kSize   = kWidth * kHeight * 4;

    // Create texture
    TextureDesc tex_desc{
        .type       = TextureType::Tex2D,
        .dimensions = {kWidth, kHeight, 1},
        .format     = Format::RGBA8Unorm,
        .usage      = UsageFlags::Sampled | UsageFlags::TransferSrc | UsageFlags::TransferDst,
    };
    Handle<Texture> tex = create_texture(d, tex_desc);
    CHECK(tex.h != 0, "create_texture failed");
    if (tex.h == 0) {
        destroy_device(d);
        return;
    }

    // Write pattern to staging buffer
    GpuPtr staging = malloc(d, kSize, Memory::Default);
    auto* staging_host = reinterpret_cast<uint8_t*>(get_host_pointer(d, staging));
    for (uint32_t y = 0; y < kHeight; ++y) {
        for (uint32_t x = 0; x < kWidth; ++x) {
            uint32_t idx = (y * kWidth + x) * 4;
            staging_host[idx + 0] = x;         // R
            staging_host[idx + 1] = y;         // G
            staging_host[idx + 2] = x ^ y;     // B
            staging_host[idx + 3] = 0xFF;      // A
        }
    }

    // Upload
    Queue q = get_queue(d);
    CommandBuffer cmd = queue_start_command_recording(q);
    BufferTextureCopyInfo copy_info{
        .image_extent = {kWidth, kHeight, 1},
    };
    cmd_copy_to_texture(cmd, staging, tex, copy_info);
    cmd_barrier(cmd, StageFlags::Transfer, StageFlags::Transfer);
    cmd_finalize(cmd);
    queue_submit(q, {&cmd, 1});
    device_wait_for_idle(d);

    // Read back
    GpuPtr readback = malloc(d, kSize, Memory::Readback);
    cmd = queue_start_command_recording(q);
    cmd_copy_from_texture(cmd, tex, readback, copy_info);
    cmd_finalize(cmd);
    queue_submit(q, {&cmd, 1});
    device_wait_for_idle(d);

    // Verify
    auto* rb_host = reinterpret_cast<uint8_t*>(get_host_pointer(d, readback));
    bool match = true;
    for (uint32_t y = 0; y < kHeight && match; ++y) {
        for (uint32_t x = 0; x < kWidth && match; ++x) {
            uint32_t idx = (y * kWidth + x) * 4;
            if (rb_host[idx] != x || rb_host[idx + 1] != y || rb_host[idx + 2] != (x ^ y)) {
                printf("  Mismatch at (%u,%u): got (%u,%u,%u), expected (%u,%u,%u)\n",
                       x, y, rb_host[idx], rb_host[idx + 1], rb_host[idx + 2], x, y, x ^ y);
                match = false;
            }
        }
    }
    CHECK(match, "texture upload/readback verification failed");

    free(d, staging);
    free(d, readback);
    free(d, tex);
    destroy_device(d);
    printf("  PASS\n");
}

// --- Test 5: Heap slot recycling ---------------------------------------------------------
static void test_heap_slot_recycling() {
    printf("--- Test: heap slot recycling ---\n");
    DeviceDesc desc{
        .log_callback = test_log_callback,
        .log_level    = LogLevel::Warning,
    };
    Device d = create_device(desc);
    CHECK(d != nullptr, "create_device returned null");

    // Create a small texture to view
    TextureDesc tex_desc{
        .type       = TextureType::Tex2D,
        .dimensions = {4, 4, 1},
        .format     = Format::RGBA8Unorm,
        .usage      = UsageFlags::Sampled | UsageFlags::Storage,
    };
    Handle<Texture> tex = create_texture(d, tex_desc);
    CHECK(tex.h != 0, "create_texture failed");

    // Need a queue to advance timeline for deferred frees
    Queue q = get_queue(d);

    constexpr int kIterations = 1000;
    uint32_t first_reuse = ~0u;
    for (int i = 0; i < kIterations; ++i) {
        TextureViewDesc view_desc{
            .texture = tex,
            .format  = Format::RGBA8Unorm,
        };
        TextureView view = create_texture_view(d, view_desc);
        CHECK(view != ~0ull, "create_texture_view failed");
        uint32_t slot = static_cast<uint32_t>(view);

        free_texture_view(d, view);

        // Advance timeline to allow recycling
        CommandBuffer cmd = queue_start_command_recording(q);
        cmd_finalize(cmd);
        queue_submit(q, {&cmd, 1});
        queue_process_events(q);

        if (i > 0 && slot <= (uint32_t)i && first_reuse == ~0u) {
            first_reuse = slot;
        }
    }
    CHECK(first_reuse != ~0u, "Heap slots should be recycled");

    free(d, tex);
    device_wait_for_idle(d);
    destroy_device(d);
    printf("  PASS (first recycled slot: %u)\n", first_reuse);
}

int main() {
    printf("Izanagi API Tests\n");
    printf("=================\n\n");

    test_device_create_destroy();
    test_malloc_host_pointer();
    test_compute();
    test_texture_copy();
    test_heap_slot_recycling();

    printf("\n=================\n");
    if (g_failures == 0) {
        printf("ALL TESTS PASSED\n");
    } else {
        printf("%d FAILURE(S)\n", g_failures);
    }
    return g_failures;
}
