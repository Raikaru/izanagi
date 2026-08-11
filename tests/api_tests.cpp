// api_tests.cpp — headless API tests for the Izanagi Vulkan backend.
// Plain main() returning nonzero on failure; no test framework.

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <thread>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#endif

#include "izanagi/gpu.h"

using namespace gpu;

// White-box test hooks for the pipeline tests (declared here to avoid
// pulling the backend's internal header into the test TU; implemented in
// pipeline.cpp / platform_utils.cpp).
namespace gpu {
struct DeviceImpl;
uint32_t  debug_live_pipelines(DeviceImpl*);         // records in the dedup map
uintptr_t debug_last_compile_thread(DeviceImpl*);    // thread that last compiled natively
void      debug_set_compiler_paused(DeviceImpl*, bool);
void      debug_force_submit_failure(DeviceImpl*, bool);
uint64_t  debug_queue_timeline(DeviceImpl*);         // last successfully submitted value
int64_t   debug_pool_resets(DeviceImpl*);            // command-pool reuse resets
uintptr_t current_thread_id();                       // platform primitive (worker uses it too)
}

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
        CHECK(view != 0, "create_texture_view failed");
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

static void sleep_ms(uint32_t ms) {
#ifdef _WIN32
    Sleep(ms);
#else
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
#endif
}

// The compiler worker releases its reference shortly after publishing Ready;
// poll (bounded) for the dedup map to reach the expected size.
static void poll_live_pipelines(gpu::DeviceImpl* impl, uint32_t expected) {
    for (int i = 0; i < 200; ++i) {
        if (gpu::debug_live_pipelines(impl) == expected) { return; }
        sleep_ms(1);
    }
}

// --- Test 6: Semaphore signal/wait + deferred completion callback ------------------------
static void test_semaphore_and_callback() {
    printf("--- Test: semaphores ---\n");
    DeviceDesc desc{
        .log_callback = test_log_callback,
        .log_level    = LogLevel::Warning,
    };
    Device d = create_device(desc);
    CHECK(d != nullptr, "create_device returned null");

    Handle<Semaphore> sem = create_semaphore(d, 0);
    CHECK(sem.h != 0, "create_semaphore failed");

    GpuPtr src = malloc(d, 64, Memory::Default);
    GpuPtr dst = malloc(d, 64, Memory::Default);
    CHECK(src != 0 && dst != 0, "malloc failed");
    auto* src_host = reinterpret_cast<uint32_t*>(get_host_pointer(d, src));
    src_host[0] = 0x12345678u;
    src_host[1] = 0x9ABCDEF0u;

    Queue q = get_queue(d);
    CHECK(q != nullptr, "get_queue failed");

    // Register the completion callback before the submit.
    bool callback_fired = false;
    queue_on_submitted_work_completed(q,
                                      [](void* userdata) {
                                          *static_cast<bool*>(userdata) = true;
                                      },
                                      &callback_fired);

    CommandBuffer cmd = queue_start_command_recording(q);
    CHECK(cmd != nullptr, "queue_start_command_recording failed");
    cmd_memcpy(cmd, dst, src, 64);
    cmd_finalize(cmd);

    SemaphoreInfo signal{.semaphore = sem, .value = 1, .stage = StageFlags::Transfer};
    queue_submit(q, {&cmd, 1}, {}, Span<const SemaphoreInfo>(&signal, 1));

    // Blocking host wait on the signaled value; returning exercises the signal path.
    wait_semaphore(d, sem, 1);

    // Poll until the deferred completion callback fires.
    for (int i = 0; i < 200 && !callback_fired; ++i) {
        queue_process_events(q);
        sleep_ms(10);
    }
    CHECK(callback_fired, "queue_on_submitted_work_completed callback never fired");

    auto* dst_host = reinterpret_cast<uint32_t*>(get_host_pointer(d, dst));
    CHECK(dst_host[0] == 0x12345678u && dst_host[1] == 0x9ABCDEF0u,
          "memcpy result wrong after semaphore wait");

    free(d, sem);
    free(d, src);
    free(d, dst);
    destroy_device(d);
    printf("  PASS\n");
}

// --- Test 7: Indirect dispatch -------------------------------------------------------------
static void test_dispatch_indirect() {
    printf("--- Test: dispatch indirect ---\n");
    DeviceDesc desc{
        .log_callback = test_log_callback,
        .log_level    = LogLevel::Warning,
    };
    Device d = create_device(desc);
    CHECK(d != nullptr, "create_device returned null");

    Arena arena{reinterpret_cast<uint8_t*>(g_arena_mem), 0, sizeof(g_arena_mem)};
    std::string shader_path = find_shader_path("memcpy_kernel.spv");
    auto spirv = load_spirv(shader_path.c_str(), &arena);
    CHECK(spirv.size() > 0, "Failed to load memcpy_kernel.spv");
    if (spirv.size() == 0) {
        destroy_device(d);
        return;
    }
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

    constexpr uint32_t kCount = 1024;
    GpuPtr src_buf  = malloc(d, kCount * sizeof(uint32_t), Memory::Default);
    GpuPtr dst_buf  = malloc(d, kCount * sizeof(uint32_t), Memory::Default);
    GpuPtr args_buf = malloc(d, 32, Memory::Default); // CopyData struct
    GpuPtr groups_buf = malloc(d, 3 * sizeof(uint32_t), Memory::Default);
    CHECK(src_buf != 0 && dst_buf != 0 && args_buf != 0 && groups_buf != 0, "malloc failed");

    auto* src_host = reinterpret_cast<uint32_t*>(get_host_pointer(d, src_buf));
    for (uint32_t i = 0; i < kCount; ++i) { src_host[i] = i; }

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

    auto* groups = reinterpret_cast<uint32_t*>(get_host_pointer(d, groups_buf));
    groups[0] = (kCount + 63) / 64;
    groups[1] = 1;
    groups[2] = 1;

    Queue q = get_queue(d);
    CommandBuffer cmd = queue_start_command_recording(q);
    cmd_set_pipeline(cmd, pipeline);
    cmd_dispatch_indirect(cmd, args_buf, groups_buf);
    cmd_finalize(cmd);
    queue_submit(q, {&cmd, 1});
    device_wait_for_idle(d);

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
    CHECK(match, "indirect dispatch dst[i] = src[i]*2+1 verification failed");

    free(d, src_buf);
    free(d, dst_buf);
    free(d, args_buf);
    free(d, groups_buf);
    free(d, pipeline);
    destroy_device(d);
    printf("  PASS\n");
}

// --- Test 8: Specialization constants ------------------------------------------------------
static void test_spec_constants() {
    printf("--- Test: spec constants ---\n");
    DeviceDesc desc{
        .log_callback = test_log_callback,
        .log_level    = LogLevel::Warning,
    };
    Device d = create_device(desc);
    CHECK(d != nullptr, "create_device returned null");

    Arena arena{reinterpret_cast<uint8_t*>(g_arena_mem), 0, sizeof(g_arena_mem)};
    std::string shader_path = find_shader_path("spec_mul_kernel.spv");
    auto spirv = load_spirv(shader_path.c_str(), &arena);
    CHECK(spirv.size() > 0, "Failed to load spec_mul_kernel.spv");
    if (spirv.size() == 0) {
        destroy_device(d);
        return;
    }
    ShaderSource shader_src{
        .source      = spirv,
        .entry_point = "compute_main"_sv,
    };

    // Override kMul (spec constant id 0, default 1) with 5.
    SpecializationConstant sc{
        .constant_id = 0,
        .int_val     = 5,
        .type        = SpecializationConstantType::UInt32,
    };
    Handle<Pipeline> pipeline = create_compute_pipeline(
        d, shader_src, Span<const SpecializationConstant>(&sc, 1));
    CHECK(pipeline.h != 0, "create_compute_pipeline (spec constants) failed");
    if (pipeline.h == 0) {
        destroy_device(d);
        return;
    }

    constexpr uint32_t kCount = 256;
    GpuPtr src_buf  = malloc(d, kCount * sizeof(uint32_t), Memory::Default);
    GpuPtr dst_buf  = malloc(d, kCount * sizeof(uint32_t), Memory::Default);
    GpuPtr args_buf = malloc(d, 32, Memory::Default);
    CHECK(src_buf != 0 && dst_buf != 0 && args_buf != 0, "malloc failed");

    auto* src_host = reinterpret_cast<uint32_t*>(get_host_pointer(d, src_buf));
    for (uint32_t i = 0; i < kCount; ++i) { src_host[i] = i + 1; }

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

    Queue q = get_queue(d);
    CommandBuffer cmd = queue_start_command_recording(q);
    cmd_set_pipeline(cmd, pipeline);
    Dimension3D groups{(kCount + 63) / 64, 1, 1};
    cmd_dispatch(cmd, args_buf, groups);
    cmd_finalize(cmd);
    queue_submit(q, {&cmd, 1});
    device_wait_for_idle(d);

    auto* dst_host = reinterpret_cast<uint32_t*>(get_host_pointer(d, dst_buf));
    bool match = true;
    for (uint32_t i = 0; i < kCount; ++i) {
        uint32_t expected = (i + 1) * 5; // kMul overridden 1 -> 5
        if (dst_host[i] != expected) {
            printf("  Mismatch at %u: expected %u, got %u\n", i, expected, dst_host[i]);
            match = false;
            break;
        }
    }
    CHECK(match, "spec constant kMul=5 not applied (shader default is 1)");

    free(d, src_buf);
    free(d, dst_buf);
    free(d, args_buf);
    free(d, pipeline);
    destroy_device(d);
    printf("  PASS\n");
}

// --- Test 9: Headless indirect draws (single + multi) --------------------------------------
static void test_draw_indirect() {
    printf("--- Test: draw indirect ---\n");
    DeviceDesc desc{
        .log_callback = test_log_callback,
        .log_level    = LogLevel::Warning,
    };
    Device d = create_device(desc);
    CHECK(d != nullptr, "create_device returned null");

    constexpr uint32_t kSize    = 64; // 64x64 headless color target
    constexpr uint32_t kTexBytes = kSize * kSize * 4;

    TextureDesc tex_desc{
        .type       = TextureType::Tex2D,
        .dimensions = {kSize, kSize, 1},
        .format     = Format::BGRA8Unorm,
        .usage      = UsageFlags::ColorAttachment | UsageFlags::TransferSrc,
    };
    Handle<Texture> color_tex = create_texture(d, tex_desc);
    CHECK(color_tex.h != 0, "create_texture (color target) failed");
    if (color_tex.h == 0) {
        destroy_device(d);
        return;
    }

    Arena arena{reinterpret_cast<uint8_t*>(g_arena_mem), 0, sizeof(g_arena_mem)};
    std::string shader_path = find_shader_path("offscreen_triangle.spv");
    auto spirv = load_spirv(shader_path.c_str(), &arena);
    CHECK(spirv.size() > 0, "Failed to load offscreen_triangle.spv");
    if (spirv.size() == 0) {
        free(d, color_tex);
        destroy_device(d);
        return;
    }
    ShaderSource vertex_src{.source = spirv, .entry_point = "vertex_main"_sv};
    ShaderSource fragment_src{.source = spirv, .entry_point = "fragment_main"_sv};
    ColorTarget color_target{.format = Format::BGRA8Unorm};
    RasterDesc raster_desc{
        .color_targets = Span<const ColorTarget>(&color_target, 1),
    };
    Handle<Pipeline> pipeline = create_graphics_pipeline(d, vertex_src, fragment_src, raster_desc);
    CHECK(pipeline.h != 0, "create_graphics_pipeline failed");
    if (pipeline.h == 0) {
        free(d, color_tex);
        destroy_device(d);
        return;
    }

    // Index buffer + indirect args + draw-count buffer
    GpuPtr idx_buf   = malloc(d, sizeof(uint16_t) * 3, Memory::Default);
    GpuPtr args_buf  = malloc(d, sizeof(DrawIndexedIndirectGpuArgs), Memory::Default);
    GpuPtr count_buf = malloc(d, sizeof(uint32_t), Memory::Default);
    CHECK(idx_buf != 0 && args_buf != 0 && count_buf != 0, "malloc failed");
    auto* idx_host = reinterpret_cast<uint16_t*>(get_host_pointer(d, idx_buf));
    idx_host[0] = 0;
    idx_host[1] = 1;
    idx_host[2] = 2;
    auto* args_host = reinterpret_cast<DrawIndexedIndirectGpuArgs*>(get_host_pointer(d, args_buf));
    *args_host = DrawIndexedIndirectGpuArgs{
        .index_count    = 3,
        .instance_count = 1,
        .first_index    = 0,
        .vertex_offset  = 0,
        .first_instance = 0,
    };
    *reinterpret_cast<uint32_t*>(get_host_pointer(d, count_buf)) = 1;

    Queue q = get_queue(d);
    GpuPtr readback_a = malloc(d, kTexBytes, Memory::Readback);
    GpuPtr readback_b = malloc(d, kTexBytes, Memory::Readback);
    CHECK(readback_a != 0 && readback_b != 0, "readback malloc failed");

    RenderAttachment color_att{
        .texture     = color_tex,
        .load_op     = LoadOp::Clear,
        .store_op    = StoreOp::Store,
        .clear_color = Color{0, 0, 0, 255},
    };
    RenderPassDesc pass_desc{
        .color_attachments = Span<const RenderAttachment>(&color_att, 1),
        .render_area       = Rect2D{.width = kSize, .height = kSize},
    };
    BufferTextureCopyInfo copy_info{.image_extent = {kSize, kSize, 1}};

    // Pass A: single indirect indexed draw
    CommandBuffer cmd = queue_start_command_recording(q);
    cmd_begin_render_pass(cmd, pass_desc);
    cmd_set_pipeline(cmd, pipeline);
    cmd_draw_indexed_instanced_indirect(cmd, DrawIndexedIndirectInfo{
        .vertexDataGpu   = 0,
        .fragmentDataGpu = 0,
        .indicesGpu      = idx_buf,
        .argsGpu         = args_buf,
        .type            = IndexType::UInt16,
    });
    cmd_end_render_pass(cmd);
    cmd_barrier(cmd, StageFlags::RasterColorOut, StageFlags::Transfer);
    cmd_copy_from_texture(cmd, color_tex, readback_a, copy_info);
    cmd_finalize(cmd);
    queue_submit(q, {&cmd, 1});
    device_wait_for_idle(d);

    auto* rb_a = reinterpret_cast<uint8_t*>(get_host_pointer(d, readback_a));
    // BGRA bytes: [2]=R, [1]=G
    CHECK(rb_a[(32 * kSize + 32) * 4 + 2] == 255 && rb_a[(32 * kSize + 32) * 4 + 1] == 0,
          "pass A: center pixel not red");
    CHECK(rb_a[2] == 0 && rb_a[1] == 0 && rb_a[0] == 0, "pass A: corner pixel not black");

    // Pass B: multi indirect indexed draw (drawCountGpu = 1, maxDraws = 1)
    cmd = queue_start_command_recording(q);
    cmd_begin_render_pass(cmd, pass_desc);
    cmd_set_pipeline(cmd, pipeline);
    cmd_draw_indexed_instanced_indirect_multi(cmd, MultiDrawIndirectInfo{
        .vertexDataGpu = 0,
        .pixelDataGpu  = 0,
        .indicesGpu    = idx_buf,
        .argsGpu       = args_buf,
        .drawCountGpu  = count_buf,
        .maxDraws      = 1,
        .type          = IndexType::UInt16,
    });
    cmd_end_render_pass(cmd);
    cmd_barrier(cmd, StageFlags::RasterColorOut, StageFlags::Transfer);
    cmd_copy_from_texture(cmd, color_tex, readback_b, copy_info);
    cmd_finalize(cmd);
    queue_submit(q, {&cmd, 1});
    device_wait_for_idle(d);

    auto* rb_b = reinterpret_cast<uint8_t*>(get_host_pointer(d, readback_b));
    CHECK(rb_b[(32 * kSize + 32) * 4 + 2] == 255 && rb_b[(32 * kSize + 32) * 4 + 1] == 0,
          "pass B: center pixel not red");
    CHECK(rb_b[2] == 0 && rb_b[1] == 0 && rb_b[0] == 0, "pass B: corner pixel not black");

    free(d, readback_a);
    free(d, readback_b);
    free(d, idx_buf);
    free(d, args_buf);
    free(d, count_buf);
    free(d, pipeline);
    free(d, color_tex);
    destroy_device(d);
    printf("  PASS\n");
}

// --- Test 11: Mip-chain and cube subresource copies ---------------------------------------
static void test_mip_and_cube() {
    printf("--- Test: mip + cube ---\n");
    DeviceDesc desc{
        .log_callback = test_log_callback,
        .log_level    = LogLevel::Warning,
    };
    Device d = create_device(desc);
    CHECK(d != nullptr, "create_device returned null");

    // Mip chain: 8x8, 4 mips
    TextureDesc mip_desc{
        .type       = TextureType::Tex2D,
        .dimensions = {8, 8, 1},
        .mip_count  = 4,
        .format     = Format::RGBA8Unorm,
        .usage      = UsageFlags::Sampled | UsageFlags::TransferSrc | UsageFlags::TransferDst,
    };
    Handle<Texture> mip_tex = create_texture(d, mip_desc);
    CHECK(mip_tex.h != 0, "create_texture (mips) failed");
    if (mip_tex.h == 0) {
        destroy_device(d);
        return;
    }

    GpuPtr mip0_src = malloc(d, 8 * 8 * 4, Memory::Default);
    GpuPtr mip2_src = malloc(d, 2 * 2 * 4, Memory::Default);
    GpuPtr mip2_rb  = malloc(d, 2 * 2 * 4, Memory::Readback);
    CHECK(mip0_src != 0 && mip2_src != 0 && mip2_rb != 0, "malloc failed");
    auto* mip0_host = reinterpret_cast<uint8_t*>(get_host_pointer(d, mip0_src));
    for (int p = 0; p < 8 * 8; ++p) {
        mip0_host[p * 4 + 0] = 10;
        mip0_host[p * 4 + 1] = 20;
        mip0_host[p * 4 + 2] = 30;
        mip0_host[p * 4 + 3] = 255;
    }
    auto* mip2_host = reinterpret_cast<uint8_t*>(get_host_pointer(d, mip2_src));
    for (int p = 0; p < 2 * 2; ++p) {
        mip2_host[p * 4 + 0] = 200;
        mip2_host[p * 4 + 1] = 100;
        mip2_host[p * 4 + 2] = 50;
        mip2_host[p * 4 + 3] = 255;
    }

    Queue q = get_queue(d);
    CommandBuffer cmd = queue_start_command_recording(q);
    cmd_copy_to_texture(cmd, mip0_src, mip_tex, BufferTextureCopyInfo{.image_extent = {8, 8, 1}});
    cmd_copy_to_texture(cmd, mip2_src, mip_tex,
                        BufferTextureCopyInfo{.image_extent = {2, 2, 1}, .base_mip = 2});
    cmd_barrier(cmd, StageFlags::Transfer, StageFlags::Transfer);
    cmd_copy_from_texture(cmd, mip_tex, mip2_rb,
                          BufferTextureCopyInfo{.image_extent = {2, 2, 1}, .base_mip = 2});
    cmd_finalize(cmd);
    queue_submit(q, {&cmd, 1});
    device_wait_for_idle(d);

    auto* mip2_out = reinterpret_cast<uint8_t*>(get_host_pointer(d, mip2_rb));
    bool mip_ok = true;
    for (int p = 0; p < 2 * 2; ++p) {
        if (mip2_out[p * 4 + 0] != 200 || mip2_out[p * 4 + 1] != 100 ||
            mip2_out[p * 4 + 2] != 50 || mip2_out[p * 4 + 3] != 255) {
            mip_ok = false;
        }
    }
    CHECK(mip_ok, "mip 2 subresource readback mismatch");

    free(d, mip0_src);
    free(d, mip2_src);
    free(d, mip2_rb);
    free(d, mip_tex);

    // Cube: 16x16, 6 faces (array_count 1 -> backend creates 6 layers)
    TextureDesc cube_desc{
        .type       = TextureType::TexCube,
        .dimensions = {16, 16, 1},
        .array_count = 1,
        .format     = Format::RGBA8Unorm,
        .usage      = UsageFlags::Sampled | UsageFlags::TransferSrc | UsageFlags::TransferDst,
    };
    Handle<Texture> cube_tex = create_texture(d, cube_desc);
    CHECK(cube_tex.h != 0, "create_texture (cube) failed");
    if (cube_tex.h == 0) {
        destroy_device(d);
        return;
    }

    GpuPtr face_src = malloc(d, 16 * 16 * 4, Memory::Default);
    GpuPtr face_rb  = malloc(d, 16 * 16 * 4, Memory::Readback);
    CHECK(face_src != 0 && face_rb != 0, "malloc failed");
    auto* face_host = reinterpret_cast<uint8_t*>(get_host_pointer(d, face_src));
    for (int p = 0; p < 16 * 16; ++p) {
        face_host[p * 4 + 0] = 60;
        face_host[p * 4 + 1] = 70;
        face_host[p * 4 + 2] = 80;
        face_host[p * 4 + 3] = 255;
    }

    cmd = queue_start_command_recording(q);
    cmd_copy_to_texture(cmd, face_src, cube_tex,
                        BufferTextureCopyInfo{.image_extent = {16, 16, 1}, .base_layer = 3});
    cmd_barrier(cmd, StageFlags::Transfer, StageFlags::Transfer);
    cmd_copy_from_texture(cmd, cube_tex, face_rb,
                          BufferTextureCopyInfo{.image_extent = {16, 16, 1}, .base_layer = 3});
    cmd_finalize(cmd);
    queue_submit(q, {&cmd, 1});
    device_wait_for_idle(d);

    auto* face_out = reinterpret_cast<uint8_t*>(get_host_pointer(d, face_rb));
    bool cube_ok = true;
    for (int p = 0; p < 16 * 16; ++p) {
        if (face_out[p * 4 + 0] != 60 || face_out[p * 4 + 1] != 70 ||
            face_out[p * 4 + 2] != 80 || face_out[p * 4 + 3] != 255) {
            cube_ok = false;
        }
    }
    CHECK(cube_ok, "cube face 3 subresource readback mismatch");

    free(d, face_src);
    free(d, face_rb);
    free(d, cube_tex);
    destroy_device(d);
    printf("  PASS\n");
}

// --- Test 12: BC1 block-compressed copy roundtrip -----------------------------------------
static void test_bc1_roundtrip() {
    printf("--- Test: BC1 roundtrip ---\n");
    DeviceDesc desc{
        .log_callback = test_log_callback,
        .log_level    = LogLevel::Warning,
    };
    Device d = create_device(desc);
    CHECK(d != nullptr, "create_device returned null");

    TextureDesc tex_desc{
        .type       = TextureType::Tex2D,
        .dimensions = {64, 64, 1},
        .format     = Format::BC1RGBAUnorm,
        .usage      = UsageFlags::Sampled | UsageFlags::TransferSrc | UsageFlags::TransferDst,
    };
    Handle<Texture> tex = create_texture(d, tex_desc);
    if (tex.h == 0) {
        printf("  SKIP (BC1 unsupported)\n");
        destroy_device(d);
        return;
    }

    constexpr uint32_t kBytes = 2048; // 16x16 blocks x 8 bytes/block
    GpuPtr staging  = malloc(d, kBytes, Memory::Default);
    GpuPtr readback = malloc(d, kBytes, Memory::Readback);
    CHECK(staging != 0 && readback != 0, "malloc failed");
    auto* st = reinterpret_cast<uint8_t*>(get_host_pointer(d, staging));
    for (uint32_t i = 0; i < kBytes; ++i) { st[i] = static_cast<uint8_t>(i * 7 + 3); }

    Queue q = get_queue(d);
    CommandBuffer cmd = queue_start_command_recording(q);
    cmd_copy_to_texture(cmd, staging, tex, BufferTextureCopyInfo{.image_extent = {64, 64, 1}});
    cmd_barrier(cmd, StageFlags::Transfer, StageFlags::Transfer);
    cmd_copy_from_texture(cmd, tex, readback, BufferTextureCopyInfo{.image_extent = {64, 64, 1}});
    cmd_finalize(cmd);
    queue_submit(q, {&cmd, 1});
    device_wait_for_idle(d);

    auto* rb = reinterpret_cast<uint8_t*>(get_host_pointer(d, readback));
    CHECK(memcmp(rb, st, kBytes) == 0, "BC1 copy roundtrip mismatch");

    free(d, tex);
    free(d, staging);
    free(d, readback);
    destroy_device(d);
    printf("  PASS\n");
}

// --- Test 13: MSAA render + resolve -------------------------------------------------------
static void test_msaa_resolve() {
    printf("--- Test: MSAA resolve ---\n");
    DeviceDesc desc{
        .log_callback = test_log_callback,
        .log_level    = LogLevel::Warning,
    };
    Device d = create_device(desc);
    CHECK(d != nullptr, "create_device returned null");

    constexpr uint32_t kSize     = 64;
    constexpr uint32_t kTexBytes = kSize * kSize * 4;

    // 4x MSAA color target; resolve target is sample_count 1
    TextureDesc msaa_desc{
        .type         = TextureType::Tex2D,
        .dimensions   = {kSize, kSize, 1},
        .sample_count = 4,
        .format       = Format::BGRA8Unorm,
        .usage        = UsageFlags::ColorAttachment,
    };
    Handle<Texture> msaa_tex = create_texture(d, msaa_desc);
    CHECK(msaa_tex.h != 0, "create_texture (MSAA target) failed");
    TextureDesc resolve_desc{
        .type       = TextureType::Tex2D,
        .dimensions = {kSize, kSize, 1},
        .format     = Format::BGRA8Unorm,
        .usage      = UsageFlags::ColorAttachment | UsageFlags::TransferSrc,
    };
    Handle<Texture> resolve_tex = create_texture(d, resolve_desc);
    CHECK(resolve_tex.h != 0, "create_texture (resolve target) failed");
    if (msaa_tex.h == 0 || resolve_tex.h == 0) {
        if (msaa_tex.h != 0) { free(d, msaa_tex); }
        if (resolve_tex.h != 0) { free(d, resolve_tex); }
        destroy_device(d);
        return;
    }

    Arena arena{reinterpret_cast<uint8_t*>(g_arena_mem), 0, sizeof(g_arena_mem)};
    std::string shader_path = find_shader_path("offscreen_triangle.spv");
    auto spirv = load_spirv(shader_path.c_str(), &arena);
    CHECK(spirv.size() > 0, "Failed to load offscreen_triangle.spv");
    if (spirv.size() == 0) {
        free(d, msaa_tex);
        free(d, resolve_tex);
        destroy_device(d);
        return;
    }
    ShaderSource vertex_src{.source = spirv, .entry_point = "vertex_main"_sv};
    ShaderSource fragment_src{.source = spirv, .entry_point = "fragment_main"_sv};
    ColorTarget color_target{.format = Format::BGRA8Unorm};
    RasterDesc raster_desc{
        .sample_count  = 4,
        .color_targets = Span<const ColorTarget>(&color_target, 1),
    };
    Handle<Pipeline> pipeline = create_graphics_pipeline(d, vertex_src, fragment_src, raster_desc);
    CHECK(pipeline.h != 0, "create_graphics_pipeline (MSAA) failed");
    if (pipeline.h == 0) {
        free(d, msaa_tex);
        free(d, resolve_tex);
        destroy_device(d);
        return;
    }

    RenderAttachment color_att{
        .texture         = msaa_tex,
        .load_op         = LoadOp::Clear,
        .store_op        = StoreOp::Store,
        .clear_color     = Color{0, 0, 0, 255},
        .resolve_texture = resolve_tex,
    };
    RenderPassDesc pass_desc{
        .color_attachments = Span<const RenderAttachment>(&color_att, 1),
        .render_area       = Rect2D{.width = kSize, .height = kSize},
    };

    GpuPtr readback = malloc(d, kTexBytes, Memory::Readback);
    CHECK(readback != 0, "readback malloc failed");

    Queue q = get_queue(d);
    CommandBuffer cmd = queue_start_command_recording(q);
    cmd_begin_render_pass(cmd, pass_desc);
    cmd_set_pipeline(cmd, pipeline);
    cmd_draw(cmd, 0, 0, 3, 1);
    cmd_end_render_pass(cmd);
    cmd_barrier(cmd, StageFlags::RasterColorOut, StageFlags::Transfer);
    cmd_copy_from_texture(cmd, resolve_tex, readback,
                          BufferTextureCopyInfo{.image_extent = {kSize, kSize, 1}});
    cmd_finalize(cmd);
    queue_submit(q, {&cmd, 1});
    device_wait_for_idle(d);

    // Readback is from the RESOLVE target: center red, corner black.
    auto* rb = reinterpret_cast<uint8_t*>(get_host_pointer(d, readback));
    CHECK(rb[(32 * kSize + 32) * 4 + 2] == 255 && rb[(32 * kSize + 32) * 4 + 1] == 0,
          "center pixel not red after MSAA resolve");
    CHECK(rb[2] == 0 && rb[1] == 0 && rb[0] == 0, "corner pixel not black after MSAA resolve");

    free(d, readback);
    free(d, pipeline);
    free(d, msaa_tex);
    free(d, resolve_tex);
    destroy_device(d);
    printf("  PASS\n");
}

// --- Test 14: Generate mipmaps -------------------------------------------------------------
static void test_generate_mipmaps() {
    printf("--- Test: generate mipmaps ---\n");
    DeviceDesc desc{
        .log_callback = test_log_callback,
        .log_level    = LogLevel::Warning,
    };
    Device d = create_device(desc);
    CHECK(d != nullptr, "create_device returned null");

    TextureDesc tex_desc{
        .type       = TextureType::Tex2D,
        .dimensions = {8, 8, 1},
        .mip_count  = 4,
        .format     = Format::RGBA8Unorm,
        .usage      = UsageFlags::Sampled | UsageFlags::TransferSrc | UsageFlags::TransferDst,
    };
    Handle<Texture> tex = create_texture(d, tex_desc);
    CHECK(tex.h != 0, "create_texture failed");
    if (tex.h == 0) {
        destroy_device(d);
        return;
    }

    // Solid color in mip 0 only; mips 1..3 must be derived by the blit chain.
    const uint8_t kColor[4] = {200, 100, 50, 255};
    GpuPtr mip0_src = malloc(d, 8 * 8 * 4, Memory::Default);
    GpuPtr mip3_rb  = malloc(d, 4, Memory::Readback);
    CHECK(mip0_src != 0 && mip3_rb != 0, "malloc failed");
    auto* mip0_host = reinterpret_cast<uint8_t*>(get_host_pointer(d, mip0_src));
    for (int p = 0; p < 8 * 8; ++p) {
        mip0_host[p * 4 + 0] = kColor[0];
        mip0_host[p * 4 + 1] = kColor[1];
        mip0_host[p * 4 + 2] = kColor[2];
        mip0_host[p * 4 + 3] = kColor[3];
    }

    Queue q = get_queue(d);
    CommandBuffer cmd = queue_start_command_recording(q);
    cmd_copy_to_texture(cmd, mip0_src, tex, BufferTextureCopyInfo{.image_extent = {8, 8, 1}});
    cmd_barrier(cmd, StageFlags::Transfer, StageFlags::Transfer);
    cmd_generate_mipmaps(cmd, tex);
    cmd_barrier(cmd, StageFlags::Transfer, StageFlags::Transfer);
    cmd_copy_from_texture(cmd, tex, mip3_rb,
                          BufferTextureCopyInfo{.image_extent = {1, 1, 1}, .base_mip = 3});
    cmd_finalize(cmd);
    queue_submit(q, {&cmd, 1});
    device_wait_for_idle(d);

    auto* rb = reinterpret_cast<uint8_t*>(get_host_pointer(d, mip3_rb));
    bool ok  = true;
    for (int c = 0; c < 4; ++c) {
        int diff = static_cast<int>(rb[c]) - static_cast<int>(kColor[c]);
        if (diff < -2 || diff > 2) { ok = false; }
    }
    CHECK(ok, "mip 3 not generated from mip 0 (linear filter, tolerance +-2)");

    free(d, mip0_src);
    free(d, mip3_rb);
    free(d, tex);
    destroy_device(d);
    printf("  PASS\n");
}

// --- Test 15: dual-source blending ----------------------------------------------
// The shader emits source0 (rgb known, a = 0.5) and source1 (a = 1.0). With
// RGB factors Src1Alpha/OneMinusSrc1Alpha and the alpha equation One/Zero:
//   RGB   = src_rgb * 1.0 + dst * 0.0   -> reflects source1 alpha (1.0)
//   alpha = 0.5                          -> reflects source0 alpha (stored)
// A single-source implementation using SrcAlpha would blend RGB with 0.5 and
// fail the RGB check, so this can only pass via dual-source blending.
static void test_dual_source_blend() {
    printf("--- Test: dual-source blending ---\n");
    DeviceDesc desc{.log_callback = test_log_callback, .log_level = LogLevel::Warning};
    Device d = create_device(desc);
    CHECK(d != nullptr, "create_device failed");
    if (!d) return;

    constexpr uint32_t kSize = 64;
    constexpr uint32_t kTexBytes = kSize * kSize * 4;
    TextureDesc tex_desc{
        .type       = TextureType::Tex2D,
        .dimensions = {kSize, kSize, 1},
        .format     = Format::RGBA8Unorm,
        .usage      = UsageFlags::ColorAttachment | UsageFlags::TransferSrc,
    };
    Handle<Texture> color_tex = create_texture(d, tex_desc);
    CHECK(color_tex.h != 0, "create_texture failed");
    if (color_tex.h == 0) {
        destroy_device(d);
        return;
    }

    Arena arena{reinterpret_cast<uint8_t*>(g_arena_mem), 0, sizeof(g_arena_mem)};
    std::string shader_path = find_shader_path("dual_src.spv");
    auto spirv = load_spirv(shader_path.c_str(), &arena);
    CHECK(spirv.size() > 0, "Failed to load dual_src.spv");
    if (spirv.size() == 0) {
        free(d, color_tex);
        destroy_device(d);
        return;
    }
    ShaderSource vertex_src{.source = spirv, .entry_point = "vertex_main"_sv};
    ShaderSource fragment_src{.source = spirv, .entry_point = "fragment_main"_sv};

    const bool dual_supported = device_supports_dual_source_blend(d);

    ColorTarget ct{
        .format = Format::RGBA8Unorm,
        .blendstate = BlendDesc{
            .color_op          = Blend::Add,
            .src_color_factor  = Factor::Src1Alpha,
            .dst_color_factor  = Factor::OneMinusSrc1Alpha,
            .alpha_op          = Blend::Add,
            .src_alpha_factor  = Factor::One,
            .dst_alpha_factor  = Factor::Zero,
        },
    };
    RasterDesc raster_desc{
        .color_targets = Span<const ColorTarget>(&ct, 1),
    };
    Handle<Pipeline> pipeline = create_graphics_pipeline(d, vertex_src, fragment_src, raster_desc);

    if (!dual_supported) {
        // Deterministic rejection is the contract on unsupported devices.
        CHECK(pipeline.h == 0,
              "dual-source pipeline must fail to create when dualSrcBlend is unsupported");
        printf("  PASS (unsupported-device rejection)\n");
        free(d, color_tex);
        destroy_device(d);
        return;
    }
    CHECK(pipeline.h != 0, "create_graphics_pipeline (dual-source) failed");
    if (pipeline.h == 0) {
        free(d, color_tex);
        destroy_device(d);
        return;
    }

    Queue q = get_queue(d);
    GpuPtr readback = malloc(d, kTexBytes, Memory::Readback);
    CHECK(readback != 0, "readback malloc failed");

    RenderAttachment color_att{
        .texture     = color_tex,
        .load_op     = LoadOp::Clear,
        .store_op    = StoreOp::Store,
        .clear_color = Color{26, 51, 77, 204},   // dst rgb (0.1, 0.2, 0.3), a 0.8
    };
    RenderPassDesc pass_desc{
        .color_attachments = Span<const RenderAttachment>(&color_att, 1),
        .render_area       = Rect2D{.width = kSize, .height = kSize},
    };
    BufferTextureCopyInfo copy_info{.image_extent = {kSize, kSize, 1}};

    CommandBuffer cmd = queue_start_command_recording(q);
    cmd_begin_render_pass(cmd, pass_desc);
    cmd_set_pipeline(cmd, pipeline);
    cmd_draw(cmd, 0, 0, 3, 1);
    cmd_end_render_pass(cmd);
    cmd_barrier(cmd, StageFlags::RasterColorOut, StageFlags::Transfer);
    cmd_copy_from_texture(cmd, color_tex, readback, copy_info);
    cmd_finalize(cmd);
    queue_submit(q, {&cmd, 1});
    device_wait_for_idle(d);

    auto* rb = reinterpret_cast<uint8_t*>(get_host_pointer(d, readback));
    // RGBA bytes. Expect: rgb = src (0.25, 0.5, 0.75) * 1.0 + dst * 0.0
    // (factor = source1 alpha 1.0), alpha = source0 alpha 0.5.
    const int er = int(0.25f * 255.0f), eg = int(0.5f * 255.0f), eb = int(0.75f * 255.0f);
    const int ea = int(0.5f * 255.0f);
    for (uint32_t i = 0; i < kSize * kSize; i++) {
        const uint8_t* p = rb + i * 4;
        int dr = int(p[0]) - er, dg = int(p[1]) - eg, db = int(p[2]) - eb, da = int(p[3]) - ea;
        if (dr < -2 || dr > 2 || dg < -2 || dg > 2 || db < -2 || db > 2 || da < -2 || da > 2) {
            CHECK(false, "dual-source blend result wrong (tolerance +-2)");
            printf("  at %u: got (%u,%u,%u,%u) want (%d,%d,%d,%d)\n", i, p[0], p[1], p[2],
                   p[3], er, eg, eb, ea);
            break;
        }
    }

    free(d, readback);
    free(d, color_tex);
    destroy_device(d);
    printf("  PASS\n");
}

// --- Test 15: Pipeline dedup ----------------------------------------------------------------
static void test_pipeline_dedup() {
    printf("--- Test: pipeline dedup ---\n");
    DeviceDesc desc{
        .log_callback = test_log_callback,
        .log_level    = LogLevel::Warning,
    };
    Device d = create_device(desc);
    CHECK(d != nullptr, "create_device returned null");

    Arena arena{reinterpret_cast<uint8_t*>(g_arena_mem), 0, sizeof(g_arena_mem)};
    std::string shader_path = find_shader_path("offscreen_triangle.spv");
    auto spirv = load_spirv(shader_path.c_str(), &arena);
    CHECK(spirv.size() > 0, "Failed to load offscreen_triangle.spv");
    if (spirv.size() == 0) {
        destroy_device(d);
        return;
    }
    ShaderSource vertex_src{.source = spirv, .entry_point = "vertex_main"_sv};
    ShaderSource fragment_src{.source = spirv, .entry_point = "fragment_main"_sv};
    ColorTarget color_target{.format = Format::BGRA8Unorm};
    RasterDesc raster_desc{
        .color_targets = Span<const ColorTarget>(&color_target, 1),
    };

    auto* impl = reinterpret_cast<gpu::DeviceImpl*>(d);

    // Identical creates share one compiled pipeline but stay distinct handles.
    Handle<Pipeline> a = create_graphics_pipeline(d, vertex_src, fragment_src, raster_desc);
    Handle<Pipeline> b = create_graphics_pipeline(d, vertex_src, fragment_src, raster_desc);
    CHECK(a.h != 0 && b.h != 0, "create_graphics_pipeline failed");
    CHECK(a.h != b.h, "identical creates must return distinct handles");
    CHECK(debug_live_pipelines(impl) == 1, "identical creates must share one pipeline");

    // Refcount: survives the first free, dies with the last (worker reference
    // may linger briefly after Ready is published — poll for the settle).
    free(d, a);
    CHECK(debug_live_pipelines(impl) == 1, "pipeline must stay alive while a handle remains");
    free(d, b);
    poll_live_pipelines(impl, 0);
    CHECK(debug_live_pipelines(impl) == 0, "pipeline must die with the last handle");

    // No retention: recreate after free compiles fresh (dedup only).
    Handle<Pipeline> c = create_graphics_pipeline(d, vertex_src, fragment_src, raster_desc);
    CHECK(c.h != 0, "recreate failed");
    CHECK(debug_live_pipelines(impl) == 1, "recreate after free must compile a new pipeline");
    free(d, c);
    poll_live_pipelines(impl, 0);

    // Every key field must produce a distinct pipeline.
    std::vector<Handle<Pipeline>> variants;
    auto make = [&](RasterDesc rd, Span<const SpecializationConstant> sc = {}) {
        Handle<Pipeline> h = create_graphics_pipeline(d, vertex_src, fragment_src, rd, sc);
        CHECK(h.h != 0, "variant pipeline create failed");
        return h;
    };
    {
        RasterDesc rd = raster_desc;
        rd.topology = Topology::TriangleStrip;
        variants.push_back(make(rd));
    }
    {
        RasterDesc rd = raster_desc;
        rd.sample_count = 4;
        variants.push_back(make(rd));
    }
    {
        RasterDesc rd = raster_desc;
        rd.alpha_to_coverage = true;
        variants.push_back(make(rd));
    }
    {
        ColorTarget ct{.format = Format::RGBA8Unorm};
        RasterDesc rd = raster_desc;
        rd.color_targets = Span<const ColorTarget>(&ct, 1);
        variants.push_back(make(rd));
    }
    {
        ColorTarget ct{.format = Format::BGRA8Unorm,
                       .blendstate = BlendDesc{.src_color_factor = Factor::SrcAlpha}};
        RasterDesc rd = raster_desc;
        rd.color_targets = Span<const ColorTarget>(&ct, 1);
        variants.push_back(make(rd));
    }
    {
        RasterDesc rd = raster_desc;
        rd.depth_format = Format::Depth32Float;
        variants.push_back(make(rd));
    }
    {
        RasterDesc rd = raster_desc;
        rd.stencil_format = Format::Stencil8;
        variants.push_back(make(rd));
    }
    {
        SpecializationConstant sc{.constant_id = 0, .int_val = 5,
                                  .type = SpecializationConstantType::UInt32};
        variants.push_back(make(raster_desc, Span<const SpecializationConstant>(&sc, 1)));
    }
    CHECK(debug_live_pipelines(impl) == variants.size(),
          "each varied key field must yield a distinct pipeline");
    for (auto v : variants) { free(d, v); }
    poll_live_pipelines(impl, 0);
    CHECK(debug_live_pipelines(impl) == 0, "all variant pipelines must be freed");

    // Compute pipelines dedup too; specialization values are part of the key.
    std::string comp_path = find_shader_path("spec_mul_kernel.spv");
    auto comp_spirv = load_spirv(comp_path.c_str(), &arena);
    CHECK(comp_spirv.size() > 0, "Failed to load spec_mul_kernel.spv");
    if (comp_spirv.size() == 0) {
        destroy_device(d);
        return;
    }
    ShaderSource comp_src{.source = comp_spirv, .entry_point = "compute_main"_sv};
    Handle<Pipeline> p1 = create_compute_pipeline(d, comp_src);
    Handle<Pipeline> p2 = create_compute_pipeline(d, comp_src);
    CHECK(p1.h != 0 && p2.h != 0, "create_compute_pipeline failed");
    CHECK(p1.h != p2.h, "compute dedup: handles must stay distinct");
    CHECK(debug_live_pipelines(impl) == 1, "compute identical creates must share");

    SpecializationConstant sc5{.constant_id = 0, .int_val = 5,
                               .type = SpecializationConstantType::UInt32};
    SpecializationConstant sc7{.constant_id = 0, .int_val = 7,
                               .type = SpecializationConstantType::UInt32};
    Handle<Pipeline> p3 = create_compute_pipeline(d, comp_src, Span<const SpecializationConstant>(&sc5, 1));
    Handle<Pipeline> p4 = create_compute_pipeline(d, comp_src, Span<const SpecializationConstant>(&sc7, 1));
    CHECK(p3.h != 0 && p4.h != 0, "spec-constant compute create failed");
    CHECK(debug_live_pipelines(impl) == 3, "spec constant values must be part of the key");

    free(d, p1);
    free(d, p2);
    free(d, p3);
    free(d, p4);
    poll_live_pipelines(impl, 0);
    CHECK(debug_live_pipelines(impl) == 0, "compute pipelines must be freed");

    destroy_device(d);
    printf("  PASS\n");
}

// --- Test 16: Persistent pipeline cache -----------------------------------------------------
struct TestPipelineCache {
    CacheIdentity identity;
    std::vector<uint8_t> blob;
    int  store_count = 0;
    bool load_called = false;
};

static bool test_cache_load(const CacheIdentity& id, void* user, MemoryBlock* blob) {
    auto* c = static_cast<TestPipelineCache*>(user);
    c->identity = id;
    c->load_called = true;
    if (c->blob.empty()) { return false; }
    *blob = MemoryBlock{c->blob.data(), static_cast<uint32_t>(c->blob.size())};
    return true;
}

static void test_cache_store(const CacheIdentity& id, MemoryBlock blob, void* user) {
    auto* c = static_cast<TestPipelineCache*>(user);
    c->identity = id;
    c->store_count++;
    c->blob.assign(static_cast<uint8_t*>(blob.ptr), static_cast<uint8_t*>(blob.ptr) + blob.len);
}

static void test_pipeline_cache_persistence() {
    printf("--- Test: pipeline cache persistence ---\n");

    // Device A: cold cache; create pipelines so the blob is non-trivial.
    TestPipelineCache cache_a;
    DeviceDesc desc_a{
        .log_callback = test_log_callback,
        .log_level    = LogLevel::Warning,
        .pipeline_cache_callbacks = PipelineCacheCallbacks{
            .load  = test_cache_load,
            .store = test_cache_store,
            .user  = &cache_a,
        },
    };
    Device a = create_device(desc_a);
    CHECK(a != nullptr, "create_device (A) failed");
    if (a == nullptr) { return; }
    CHECK(cache_a.load_called, "load callback must fire during create_device");

    {
        Arena arena{reinterpret_cast<uint8_t*>(g_arena_mem), 0, sizeof(g_arena_mem)};
        std::string comp_path = find_shader_path("memcpy_kernel.spv");
        auto comp_spirv = load_spirv(comp_path.c_str(), &arena);
        std::string gfx_path = find_shader_path("offscreen_triangle.spv");
        auto gfx_spirv = load_spirv(gfx_path.c_str(), &arena);
        CHECK(comp_spirv.size() > 0 && gfx_spirv.size() > 0, "shader load failed");
        if (comp_spirv.size() && gfx_spirv.size()) {
            ShaderSource comp_src{.source = comp_spirv, .entry_point = "compute_main"_sv};
            Handle<Pipeline> cp = create_compute_pipeline(a, comp_src);
            CHECK(cp.h != 0, "compute pipeline (A) failed");
            ShaderSource vs{.source = gfx_spirv, .entry_point = "vertex_main"_sv};
            ShaderSource fs{.source = gfx_spirv, .entry_point = "fragment_main"_sv};
            ColorTarget ct{.format = Format::BGRA8Unorm};
            RasterDesc rd{.color_targets = Span<const ColorTarget>(&ct, 1)};
            Handle<Pipeline> gp = create_graphics_pipeline(a, vs, fs, rd);
            CHECK(gp.h != 0, "graphics pipeline (A) failed");
            if (cp.h) { free(a, cp); }
            if (gp.h) { free(a, gp); }
        }
    }
    // Explicit flush persists the cache before device destruction (blocking;
    // loading-screen / checkpoint use).
    flush_pipeline_cache(a);
    CHECK(cache_a.store_count >= 1, "flush_pipeline_cache must invoke the store callback");
    CHECK(cache_a.blob.size() > 0, "flush must store a non-empty blob");

    destroy_device(a);
    CHECK(cache_a.store_count >= 1, "store callback must fire at destroy_device");
    CHECK(cache_a.blob.size() > 0, "stored cache blob must be non-empty");
    CHECK(cache_a.identity.backend == Backend::Vulkan, "cache identity backend must be Vulkan");
    CHECK(cache_a.identity.vendor_id != 0, "cache identity vendor_id must be set");

    // Device B: warm cache seeded from A's blob; creates must still succeed.
    TestPipelineCache cache_b;
    cache_b.blob = cache_a.blob;
    DeviceDesc desc_b{
        .log_callback = test_log_callback,
        .log_level    = LogLevel::Warning,
        .pipeline_cache_callbacks = PipelineCacheCallbacks{
            .load  = test_cache_load,
            .store = test_cache_store,
            .user  = &cache_b,
        },
    };
    Device b = create_device(desc_b);
    CHECK(b != nullptr, "create_device (B) with seeded cache failed");
    CHECK(cache_b.identity.vendor_id == cache_a.identity.vendor_id,
          "cache identity must be stable across devices");
    if (b != nullptr) {
        Arena arena{reinterpret_cast<uint8_t*>(g_arena_mem), 0, sizeof(g_arena_mem)};
        std::string comp_path = find_shader_path("memcpy_kernel.spv");
        auto comp_spirv = load_spirv(comp_path.c_str(), &arena);
        if (comp_spirv.size() > 0) {
            ShaderSource comp_src{.source = comp_spirv, .entry_point = "compute_main"_sv};
            Handle<Pipeline> cp = create_compute_pipeline(b, comp_src);
            CHECK(cp.h != 0, "compute pipeline (B) failed with seeded cache");
            if (cp.h) { free(b, cp); }
        }
        destroy_device(b);
    }
    CHECK(cache_b.store_count >= 1, "store must fire for device B");
    CHECK(cache_b.blob.size() > 0, "device B blob must be non-empty");

    // Device C: an invalid blob must be tolerated (falls back to empty cache).
    TestPipelineCache cache_c;
    cache_c.blob = {0, 0, 0, 0}; // definitely not a valid pipeline cache blob
    DeviceDesc desc_c{
        .log_callback = test_log_callback,
        .log_level    = LogLevel::Error,
        .pipeline_cache_callbacks = PipelineCacheCallbacks{
            .load  = test_cache_load,
            .store = test_cache_store,
            .user  = &cache_c,
        },
    };
    Device c = create_device(desc_c);
    CHECK(c != nullptr, "create_device (C) must tolerate an invalid cache blob");
    if (c != nullptr) { destroy_device(c); }

    printf("  PASS\n");
}

// --- Test 18: Async pipeline request/compile (non-blocking) -------------------------------
static void test_async_pipeline_compile() {
    printf("--- Test: async pipeline compile ---\n");
    DeviceDesc desc{
        .log_callback = test_log_callback,
        .log_level    = LogLevel::Warning,
    };
    Device d = create_device(desc);
    CHECK(d != nullptr, "create_device returned null");
    auto* impl = reinterpret_cast<gpu::DeviceImpl*>(d);
    const uintptr_t main_tid = gpu::current_thread_id();

    Arena arena{reinterpret_cast<uint8_t*>(g_arena_mem), 0, sizeof(g_arena_mem)};
    std::string shader_path = find_shader_path("memcpy_kernel.spv");
    auto spirv = load_spirv(shader_path.c_str(), &arena);
    CHECK(spirv.size() > 0, "Failed to load memcpy_kernel.spv");
    if (spirv.size() == 0) {
        destroy_device(d);
        return;
    }
    ShaderSource shader_src{.source = spirv, .entry_point = "compute_main"_sv};

    Handle<Pipeline> p = request_compute_pipeline(d, shader_src);
    CHECK(p.h != 0, "request_compute_pipeline returned null");
    CHECK(get_pipeline_status(d, p) == PipelineStatus::Pending ||
              get_pipeline_status(d, p) == PipelineStatus::Ready,
          "fresh request must be Pending or Ready (never Failed)");

    CHECK(wait_pipeline(d, p), "wait_pipeline failed for a valid shader");
    CHECK(get_pipeline_status(d, p) == PipelineStatus::Ready, "status must be Ready after wait");
    // Non-blocking proof: the native compile ran on the worker thread, never
    // on the requesting thread.
    CHECK(gpu::debug_last_compile_thread(impl) != 0, "compiler worker must have run");
    CHECK(gpu::debug_last_compile_thread(impl) != main_tid,
          "request path must not call vkCreate*Pipelines on the calling thread");

    // End-to-end through the async path (dst[i] = src[i]*2+1).
    constexpr uint32_t kCount = 1024;
    GpuPtr src_buf  = malloc(d, kCount * sizeof(uint32_t), Memory::Default);
    GpuPtr dst_buf  = malloc(d, kCount * sizeof(uint32_t), Memory::Default);
    GpuPtr args_buf = malloc(d, 32, Memory::Default);
    CHECK(src_buf != 0 && dst_buf != 0 && args_buf != 0, "malloc failed");
    auto* src_host = reinterpret_cast<uint32_t*>(get_host_pointer(d, src_buf));
    for (uint32_t i = 0; i < kCount; ++i) { src_host[i] = i; }
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

    Queue q = get_queue(d);
    CommandBuffer cmd = queue_start_command_recording(q);
    CHECK(cmd_set_pipeline(cmd, p), "cmd_set_pipeline must succeed for a Ready pipeline");
    cmd_dispatch(cmd, args_buf, Dimension3D{(kCount + 63) / 64, 1, 1});
    cmd_finalize(cmd);

    // Free the user handle while the submission is in flight: the record must
    // survive until the submitted work completes.
    free(d, p);
    queue_submit(q, {&cmd, 1});
    device_wait_for_idle(d);

    auto* dst_host = reinterpret_cast<uint32_t*>(get_host_pointer(d, dst_buf));
    bool match = true;
    for (uint32_t i = 0; i < kCount && match; ++i) {
        if (dst_host[i] != i * 2 + 1) { match = false; }
    }
    CHECK(match, "async compute verification failed");

    // After the submission completes, draining events releases the in-flight
    // reference and destroys the record.
    for (int i = 0; i < 200; ++i) {
        queue_process_events(q);
        if (gpu::debug_live_pipelines(impl) == 0) { break; }
        sleep_ms(10);
    }
    CHECK(gpu::debug_live_pipelines(impl) == 0,
          "record must be destroyed after submit completes and all refs drop");

    free(d, src_buf);
    free(d, dst_buf);
    free(d, args_buf);
    destroy_device(d);
    printf("  PASS\n");
}

// --- Test 19: Async Pending/Failed transitions + binding --------------------------------
static void test_async_pipeline_pending_and_failed() {
    printf("--- Test: async pipeline pending/failed ---\n");
    DeviceDesc desc{
        .log_callback = test_log_callback,
        .log_level    = LogLevel::Warning,
    };
    Device d = create_device(desc);
    CHECK(d != nullptr, "create_device returned null");
    auto* impl = reinterpret_cast<gpu::DeviceImpl*>(d);

    // Deterministic Pending: park the compiler worker before requesting.
    gpu::debug_set_compiler_paused(impl, true);

    Arena arena{reinterpret_cast<uint8_t*>(g_arena_mem), 0, sizeof(g_arena_mem)};
    std::string shader_path = find_shader_path("offscreen_triangle.spv");
    auto spirv = load_spirv(shader_path.c_str(), &arena);
    CHECK(spirv.size() > 0, "Failed to load offscreen_triangle.spv");
    if (spirv.size() == 0) {
        gpu::debug_set_compiler_paused(impl, false);
        destroy_device(d);
        return;
    }
    ShaderSource vs{.source = spirv, .entry_point = "vertex_main"_sv};
    ShaderSource fs{.source = spirv, .entry_point = "fragment_main"_sv};
    ColorTarget ct{.format = Format::BGRA8Unorm};
    RasterDesc rd{.color_targets = Span<const ColorTarget>(&ct, 1)};

    Queue q = get_queue(d);
    Handle<Pipeline> p = request_graphics_pipeline(d, vs, fs, rd);
    CHECK(p.h != 0, "request_graphics_pipeline returned null");
    CHECK(get_pipeline_status(d, p) == PipelineStatus::Pending,
          "pipeline must be Pending while the compiler is parked");

    // Pending bind: records nothing, returns false (application fallback path).
    CommandBuffer cmd = queue_start_command_recording(q);
    CHECK(!cmd_set_pipeline(cmd, p), "cmd_set_pipeline must fail for a Pending pipeline");
    cmd_finalize(cmd);
    queue_submit(q, {&cmd, 1});
    device_wait_for_idle(d);

    // Free while pending: the worker reference keeps the record alive; the
    // worker finishes and retires the result.
    free(d, p);
    gpu::debug_set_compiler_paused(impl, false);
    poll_live_pipelines(impl, 0);
    CHECK(gpu::debug_live_pipelines(impl) == 0,
          "record must be retired after free-while-pending and compilation finishes");

    // Ready bind now succeeds.
    Handle<Pipeline> p2 = request_graphics_pipeline(d, vs, fs, rd);
    CHECK(p2.h != 0, "second request failed");
    CHECK(wait_pipeline(d, p2), "wait_pipeline failed after unpause");
    CHECK(get_pipeline_status(d, p2) == PipelineStatus::Ready, "status must be Ready");
    cmd = queue_start_command_recording(q);
    CHECK(cmd_set_pipeline(cmd, p2), "cmd_set_pipeline must succeed for Ready");
    cmd_finalize(cmd);
    queue_submit(q, {&cmd, 1});
    device_wait_for_idle(d);
    free(d, p2);
    poll_live_pipelines(impl, 0);

    // Failed: garbage SPIR-V must fail deterministically on the worker.
    uint32_t junk[8] = {0xDEADBEEFu, 1, 2, 3, 4, 5, 6, 7};
    ShaderSource bad{
        .source = Span<const uint8_t>(reinterpret_cast<const uint8_t*>(junk), sizeof(junk)),
        .entry_point = "compute_main"_sv,
    };
    Handle<Pipeline> bad_p = request_compute_pipeline(d, bad);
    CHECK(bad_p.h != 0, "request for an invalid shader must still return a handle");
    CHECK(!wait_pipeline(d, bad_p), "wait_pipeline must report failure for an invalid shader");
    CHECK(get_pipeline_status(d, bad_p) == PipelineStatus::Failed, "status must be Failed");
    cmd = queue_start_command_recording(q);
    CHECK(!cmd_set_pipeline(cmd, bad_p), "cmd_set_pipeline must fail for a Failed pipeline");
    cmd_finalize(cmd);
    queue_submit(q, {&cmd, 1});
    device_wait_for_idle(d);
    free(d, bad_p);
    poll_live_pipelines(impl, 0);

    destroy_device(d);
    printf("  PASS\n");
}

// --- Test 20: Async input ownership --------------------------------------------------------
static void test_async_input_ownership() {
    printf("--- Test: async input ownership ---\n");
    DeviceDesc desc{
        .log_callback = test_log_callback,
        .log_level    = LogLevel::Warning,
    };
    Device d = create_device(desc);
    CHECK(d != nullptr, "create_device returned null");

    // Load the shader into temporary storage, then copy it to stack buffers
    // that we will destroy immediately after requesting.
    Arena arena{reinterpret_cast<uint8_t*>(g_arena_mem), 0, sizeof(g_arena_mem)};
    std::string shader_path = find_shader_path("spec_mul_kernel.spv");
    auto spirv = load_spirv(shader_path.c_str(), &arena);
    CHECK(spirv.size() > 0 && spirv.size() <= 8192, "Failed to load spec_mul_kernel.spv");
    if (spirv.size() == 0 || spirv.size() > 4096) {
        destroy_device(d);
        return;
    }

    uint8_t temp_spv[8192];
    char    temp_entry[32];
    memcpy(temp_spv, spirv.data(), spirv.size());
    memcpy(temp_entry, "compute_main", 13);

    SpecializationConstant temp_sc{.constant_id = 0, .int_val = 5,
                                   .type = SpecializationConstantType::UInt32};

    ShaderSource temp_src{
        .source = Span<const uint8_t>(temp_spv, spirv.size()),
        .entry_point = Span<const char>(temp_entry, 12),
    };
    Handle<Pipeline> p = request_compute_pipeline(d, temp_src,
                                                  Span<const SpecializationConstant>(&temp_sc, 1));
    CHECK(p.h != 0, "request_compute_pipeline returned null");
    if (p.h == 0) {
        destroy_device(d);
        return;
    }

    // Destroy the caller storage immediately; the record owns its inputs.
    memset(temp_spv, 0xAA, sizeof(temp_spv));
    memset(temp_entry, 0xBB, sizeof(temp_entry));
    temp_sc.int_val = 0;

    CHECK(wait_pipeline(d, p), "wait_pipeline failed");
    CHECK(get_pipeline_status(d, p) == PipelineStatus::Ready, "status must be Ready");

    // kMul must be 5 (copied), not the destroyed value: dst[i] = src[i] * 5.
    constexpr uint32_t kCount = 256;
    GpuPtr src_buf  = malloc(d, kCount * sizeof(uint32_t), Memory::Default);
    GpuPtr dst_buf  = malloc(d, kCount * sizeof(uint32_t), Memory::Default);
    GpuPtr args_buf = malloc(d, 32, Memory::Default);
    CHECK(src_buf != 0 && dst_buf != 0 && args_buf != 0, "malloc failed");
    auto* src_host = reinterpret_cast<uint32_t*>(get_host_pointer(d, src_buf));
    for (uint32_t i = 0; i < kCount; ++i) { src_host[i] = i + 1; }
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

    Queue q = get_queue(d);
    CommandBuffer cmd = queue_start_command_recording(q);
    CHECK(cmd_set_pipeline(cmd, p), "cmd_set_pipeline failed");
    cmd_dispatch(cmd, args_buf, Dimension3D{(kCount + 63) / 64, 1, 1});
    cmd_finalize(cmd);
    queue_submit(q, {&cmd, 1});
    device_wait_for_idle(d);

    auto* dst_host = reinterpret_cast<uint32_t*>(get_host_pointer(d, dst_buf));
    bool match = true;
    for (uint32_t i = 0; i < kCount && match; ++i) {
        if (dst_host[i] != (i + 1) * 5) { match = false; }
    }
    CHECK(match, "owned specialization value not applied (kMul must be 5)");

    free(d, p);
    free(d, src_buf);
    free(d, dst_buf);
    free(d, args_buf);
    destroy_device(d);
    printf("  PASS\n");
}

// --- Test 21: Concurrent dedup --------------------------------------------------------------
static void test_async_dedup_concurrent() {
    printf("--- Test: async dedup concurrent ---\n");
    DeviceDesc desc{
        .log_callback = test_log_callback,
        .log_level    = LogLevel::Warning,
    };
    Device d = create_device(desc);
    CHECK(d != nullptr, "create_device returned null");
    auto* impl = reinterpret_cast<gpu::DeviceImpl*>(d);

    Arena arena{reinterpret_cast<uint8_t*>(g_arena_mem), 0, sizeof(g_arena_mem)};
    std::string shader_path = find_shader_path("memcpy_kernel.spv");
    auto spirv = load_spirv(shader_path.c_str(), &arena);
    CHECK(spirv.size() > 0, "Failed to load memcpy_kernel.spv");
    if (spirv.size() == 0) {
        destroy_device(d);
        return;
    }
    ShaderSource shader_src{.source = spirv, .entry_point = "compute_main"_sv};

    constexpr int kThreads = 4;
    std::vector<Handle<Pipeline>> handles(kThreads);
    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t]() { handles[t] = request_compute_pipeline(d, shader_src); });
    }
    for (auto& th : threads) { th.join(); }

    bool distinct = true;
    for (int t = 1; t < kThreads; ++t) {
        if (handles[t].h == handles[0].h) { distinct = false; }
    }
    CHECK(distinct, "concurrent requests must return distinct handles");
    for (int t = 0; t < kThreads; ++t) {
        CHECK(handles[t].h != 0, "concurrent request returned null");
        CHECK(wait_pipeline(d, handles[t]), "concurrent request failed to compile");
    }
    CHECK(gpu::debug_live_pipelines(impl) == 1,
          "concurrent identical requests must share one record");

    for (int t = 0; t < kThreads; ++t) { free(d, handles[t]); }
    poll_live_pipelines(impl, 0);
    CHECK(gpu::debug_live_pipelines(impl) == 0, "concurrent handles must all free");

    destroy_device(d);
    printf("  PASS\n");
}

// --- Test 22: Shutdown with queued/compiling work ------------------------------------------
static void test_async_shutdown_with_pending() {
    printf("--- Test: async shutdown with pending ---\n");

    // Case 1: queued work (worker parked) at destroy. The worker drains the
    // queue during shutdown, then joins; nothing touches destroyed state.
    {
        DeviceDesc desc{
            .log_callback = test_log_callback,
            .log_level    = LogLevel::Warning,
        };
        Device d = create_device(desc);
        CHECK(d != nullptr, "create_device failed");
        auto* impl = reinterpret_cast<gpu::DeviceImpl*>(d);
        gpu::debug_set_compiler_paused(impl, true);

        Arena arena{reinterpret_cast<uint8_t*>(g_arena_mem), 0, sizeof(g_arena_mem)};
        std::string shader_path = find_shader_path("memcpy_kernel.spv");
        auto spirv = load_spirv(shader_path.c_str(), &arena);
        if (spirv.size() > 0) {
            ShaderSource src{.source = spirv, .entry_point = "compute_main"_sv};
            for (int i = 0; i < 3; ++i) {
                Handle<Pipeline> p = request_compute_pipeline(d, src);
                CHECK(p.h != 0, "request before shutdown failed");
                free(d, p);   // user ref dropped; worker ref keeps records alive
            }
        }
        destroy_device(d);   // must not crash; workers joined before teardown
    }

    // Case 2: work actively compiling at destroy (no pause).
    {
        DeviceDesc desc{
            .log_callback = test_log_callback,
            .log_level    = LogLevel::Warning,
        };
        Device d = create_device(desc);
        CHECK(d != nullptr, "create_device failed");
        Arena arena{reinterpret_cast<uint8_t*>(g_arena_mem), 0, sizeof(g_arena_mem)};
        std::string shader_path = find_shader_path("memcpy_kernel.spv");
        auto spirv = load_spirv(shader_path.c_str(), &arena);
        if (spirv.size() > 0) {
            ShaderSource src{.source = spirv, .entry_point = "compute_main"_sv};
            Handle<Pipeline> p = request_compute_pipeline(d, src);
            CHECK(p.h != 0, "request failed");
            // destroy immediately while the worker may be mid-compile
            destroy_device(d);
        } else {
            destroy_device(d);
        }
    }
    printf("  PASS\n");
}

// --- Test 23: Submission tokens ---------------------------------------------------------------
static void test_submission_tokens() {
    printf("--- Test: submission tokens ---\n");
    DeviceDesc desc{
        .log_callback = test_log_callback,
        .log_level    = LogLevel::Warning,
    };
    Device d = create_device(desc);
    CHECK(d != nullptr, "create_device returned null");
    auto* impl = reinterpret_cast<gpu::DeviceImpl*>(d);

    Queue q = get_queue(d);
    CHECK(q != nullptr, "get_queue failed");

    CommandBuffer cmd = queue_start_command_recording(q);
    CHECK(cmd != nullptr, "recording failed");
    cmd_finalize(cmd);
    Submission s1 = queue_submit(q, {&cmd, 1});
    CHECK(s1.status == SubmitStatus::Success, "first submit must succeed");
    CHECK(s1.value == 1, "first submission value is 1");
    CHECK(s1.queue == q, "submission carries the queue");
    CHECK(debug_queue_timeline(impl) == 1, "logical timeline published on success");
    CHECK(wait_submission(s1), "wait_submission must complete");
    CHECK(submission_complete(s1), "submission_complete true after completion");

    // Injected failure: no native submit, no logical timeline advance.
    gpu::debug_force_submit_failure(impl, true);
    CommandBuffer cmd2 = queue_start_command_recording(q);
    CHECK(cmd2 != nullptr, "recording failed");
    cmd_finalize(cmd2);
    Submission s2 = queue_submit(q, {&cmd2, 1});
    CHECK(s2.status == SubmitStatus::Error, "forced failure reports Error");
    CHECK(s2.value == 2, "failed submit carries the prospective value");
    CHECK(debug_queue_timeline(impl) == 1, "logical timeline must not advance on failure");
    CHECK(!submission_complete(s2) && !wait_submission(s2),
          "failed submission never completes and never blocks");
    gpu::debug_force_submit_failure(impl, false);

    // The next successful submit continues from the published value.
    CommandBuffer cmd3 = queue_start_command_recording(q);
    CHECK(cmd3 != nullptr, "recording failed");
    cmd_finalize(cmd3);
    Submission s3 = queue_submit(q, {&cmd3, 1});
    CHECK(s3.status == SubmitStatus::Success && s3.value == 2,
          "timeline continues from the last published value");
    CHECK(wait_submission(s3), "third submit must complete");
    CHECK(debug_queue_timeline(impl) == 2, "timeline published for the third submit");

    destroy_device(d);
    printf("  PASS\n");
}

// --- Test 24: Headless command-pool retirement ------------------------------------------------
static void test_headless_pool_retirement() {
    printf("--- Test: headless pool retirement ---\n");
    DeviceDesc desc{
        .log_callback = test_log_callback,
        .log_level    = LogLevel::Warning,
    };
    Device d = create_device(desc);
    CHECK(d != nullptr, "create_device returned null");
    auto* impl = reinterpret_cast<gpu::DeviceImpl*>(d);

    Queue q = get_queue(d);
    const int64_t resets_before = gpu::debug_pool_resets(impl);
    Submission last{};
    for (int i = 0; i < 2000; ++i) {
        CommandBuffer cmd = queue_start_command_recording(q);
        if (cmd == nullptr) {
            CHECK(false, "recording failed (pool starvation)");
            break;
        }
        cmd_finalize(cmd);
        last = queue_submit(q, {&cmd, 1});
        CHECK(last.status == SubmitStatus::Success, "headless submit failed");
        CHECK(last.value == static_cast<uint64_t>(i + 1), "submission values are sequential");
    }
    CHECK(wait_submission(last), "last headless submission must complete");
    CHECK(gpu::debug_pool_resets(impl) > resets_before,
          "pools must be reused (reset) headless, independent of presentation");
    CHECK(submission_complete(last), "submission_complete true after wait");

    destroy_device(d);
    printf("  PASS\n");
}

// --- Test 25: free_after (deferred destruction against a submission) --------------------------
static void test_free_after() {
    printf("--- Test: free_after ---\n");
    DeviceDesc desc{
        .log_callback = test_log_callback,
        .log_level    = LogLevel::Warning,
    };
    Device d = create_device(desc);
    CHECK(d != nullptr, "create_device returned null");
    Queue q = get_queue(d);

    // Buffer: used by a submission, then retired against it.
    GpuPtr buf = malloc(d, 64, Memory::Default);
    CHECK(buf != 0, "malloc failed");
    CHECK(get_host_pointer(d, buf) != nullptr, "host pointer valid before free_after");
    CommandBuffer cmd = queue_start_command_recording(q);
    CHECK(cmd != nullptr, "recording failed");
    cmd_memcpy(cmd, buf, buf, 64);   // self-copy: the buffer is named by the submission
    cmd_finalize(cmd);
    Submission s = queue_submit(q, {&cmd, 1});
    CHECK(s.status == SubmitStatus::Success, "submit failed");
    free_after(d, buf, s);
    CHECK(get_host_pointer(d, buf) == nullptr, "free_after invalidates the pointer immediately");
    CHECK(wait_submission(s), "submission must complete");
    for (int i = 0; i < 100; ++i) {
        queue_process_events(q);   // fires the retire batch
        sleep_ms(5);
    }
    GpuPtr buf2 = malloc(d, 64, Memory::Default);
    CHECK(buf2 != 0, "malloc after retirement works");
    free(d, buf2);

    // Texture: uploaded, then retired against the upload submission.
    TextureDesc tex_desc{
        .type       = TextureType::Tex2D,
        .dimensions = {16, 16, 1},
        .format     = Format::RGBA8Unorm,
        .usage      = UsageFlags::Sampled | UsageFlags::TransferSrc | UsageFlags::TransferDst,
    };
    Handle<Texture> tex = create_texture(d, tex_desc);
    CHECK(tex.h != 0, "create_texture failed");
    GpuPtr staging = malloc(d, 16 * 16 * 4, Memory::Default);
    GpuPtr readback = malloc(d, 16 * 16 * 4, Memory::Readback);
    auto* st = reinterpret_cast<uint8_t*>(get_host_pointer(d, staging));
    memset(st, 0x42, 16 * 16 * 4);
    cmd = queue_start_command_recording(q);
    cmd_copy_to_texture(cmd, staging, tex, BufferTextureCopyInfo{.image_extent = {16, 16, 1}});
    cmd_barrier(cmd, StageFlags::Transfer, StageFlags::Transfer);
    cmd_copy_from_texture(cmd, tex, readback, BufferTextureCopyInfo{.image_extent = {16, 16, 1}});
    cmd_finalize(cmd);
    Submission s2 = queue_submit(q, {&cmd, 1});
    CHECK(s2.status == SubmitStatus::Success, "texture submit failed");
    free_after(d, tex, s2);
    CHECK(wait_submission(s2), "texture submission must complete");
    for (int i = 0; i < 100; ++i) { queue_process_events(q); sleep_ms(5); }
    auto* rb = reinterpret_cast<uint8_t*>(get_host_pointer(d, readback));
    CHECK(rb[0] == 0x42, "copy result intact after retirement");

    // Pipeline: bound in a submission, then retired against it.
    Arena arena{reinterpret_cast<uint8_t*>(g_arena_mem), 0, sizeof(g_arena_mem)};
    std::string shader_path = find_shader_path("memcpy_kernel.spv");
    auto spirv = load_spirv(shader_path.c_str(), &arena);
    CHECK(spirv.size() > 0, "shader load failed");
    if (spirv.size() > 0) {
        ShaderSource src{.source = spirv, .entry_point = "compute_main"_sv};
        Handle<Pipeline> p = request_compute_pipeline(d, src);
        CHECK(p.h != 0, "pipeline request failed");
        CHECK(wait_pipeline(d, p), "pipeline must become Ready");
        cmd = queue_start_command_recording(q);
        CHECK(cmd_set_pipeline(cmd, p), "bind must succeed for Ready");
        cmd_finalize(cmd);
        Submission s3 = queue_submit(q, {&cmd, 1});
        CHECK(s3.status == SubmitStatus::Success, "pipeline submit failed");
        free_after(d, p, s3);
        CHECK(wait_submission(s3), "pipeline submission must complete");
        for (int i = 0; i < 100; ++i) { queue_process_events(q); sleep_ms(5); }
    }

    free(d, staging);
    free(d, readback);
    destroy_device(d);
    printf("  PASS\n");
}

// --- Test 26: command-buffer texture retention -----------------------------------------------
static void test_texture_cb_retention() {
    printf("--- Test: texture cb retention ---\n");
    DeviceDesc desc{
        .log_callback = test_log_callback,
        .log_level    = LogLevel::Warning,
    };
    Device d = create_device(desc);
    CHECK(d != nullptr, "create_device returned null");

    constexpr uint32_t kSize = 64;
    TextureDesc tex_desc{
        .type       = TextureType::Tex2D,
        .dimensions = {kSize, kSize, 1},
        .format     = Format::BGRA8Unorm,
        .usage      = UsageFlags::ColorAttachment | UsageFlags::TransferSrc,
    };
    Handle<Texture> color_tex = create_texture(d, tex_desc);
    CHECK(color_tex.h != 0, "create_texture failed");
    if (color_tex.h == 0) {
        destroy_device(d);
        return;
    }

    Arena arena{reinterpret_cast<uint8_t*>(g_arena_mem), 0, sizeof(g_arena_mem)};
    std::string shader_path = find_shader_path("offscreen_triangle.spv");
    auto spirv = load_spirv(shader_path.c_str(), &arena);
    CHECK(spirv.size() > 0, "shader load failed");
    if (spirv.size() == 0) {
        destroy_device(d);
        return;
    }
    ShaderSource vs{.source = spirv, .entry_point = "vertex_main"_sv};
    ShaderSource fs{.source = spirv, .entry_point = "fragment_main"_sv};
    ColorTarget ct{.format = Format::BGRA8Unorm};
    RasterDesc rd{.color_targets = Span<const ColorTarget>(&ct, 1)};
    Handle<Pipeline> pipeline = create_graphics_pipeline(d, vs, fs, rd);
    CHECK(pipeline.h != 0, "pipeline creation failed");
    if (pipeline.h == 0) {
        destroy_device(d);
        return;
    }

    RenderAttachment color_att{
        .texture     = color_tex,
        .load_op     = LoadOp::Clear,
        .store_op    = StoreOp::Store,
        .clear_color = Color{0, 0, 0, 255},
    };
    RenderPassDesc pass_desc{
        .color_attachments = Span<const RenderAttachment>(&color_att, 1),
        .render_area       = Rect2D{.width = kSize, .height = kSize},
    };
    GpuPtr readback = malloc(d, kSize * kSize * 4, Memory::Readback);

    Queue q = get_queue(d);
    CommandBuffer cmd = queue_start_command_recording(q);
    cmd_begin_render_pass(cmd, pass_desc);
    cmd_set_pipeline(cmd, pipeline);
    cmd_draw(cmd, 0, 0, 3, 1);
    cmd_end_render_pass(cmd);
    cmd_barrier(cmd, StageFlags::RasterColorOut, StageFlags::Transfer);
    cmd_copy_from_texture(cmd, color_tex, readback, BufferTextureCopyInfo{.image_extent = {kSize, kSize, 1}});
    cmd_finalize(cmd);

    // Free the user handle while the command buffer retains the texture; the
    // recorded commands must still find a live native image.
    free(d, color_tex);
    free(d, pipeline);
    Submission s = queue_submit(q, {&cmd, 1});
    CHECK(s.status == SubmitStatus::Success, "submit failed");
    CHECK(wait_submission(s), "submission must complete");
    auto* rb = reinterpret_cast<uint8_t*>(get_host_pointer(d, readback));
    CHECK(rb[(32 * kSize + 32) * 4 + 2] == 255, "render used the retained texture correctly");
    for (int i = 0; i < 100; ++i) { queue_process_events(q); sleep_ms(5); }

    free(d, readback);
    destroy_device(d);
    printf("  PASS\n");
}

// --- Test 27: Memory alignment, bounds validation, coherence ops -----------------------------
static void test_memory_alignment_and_bounds() {
    printf("--- Test: memory alignment + bounds ---\n");
    DeviceDesc desc{
        .log_callback = test_log_callback,
        .log_level    = LogLevel::Warning,
    };
    Device d = create_device(desc);
    CHECK(d != nullptr, "create_device returned null");

    GpuPtr p256 = malloc(d, 100, 256, Memory::Default);
    CHECK(p256 != 0, "aligned malloc failed");
    CHECK((p256 % 256) == 0, "user address honors the requested alignment");

    void* hp = get_host_pointer(d, p256);
    CHECK(hp != nullptr, "host pointer for aligned allocation");
    void* hp16 = get_host_pointer(d, p256 + 16);
    CHECK(hp16 == static_cast<char*>(hp) + 16, "interior host pointer is offset correctly");

    // Bounds validation
    CHECK(get_host_pointer(d, p256 + 100) == nullptr, "one-past-end rejected");
    CHECK(get_host_pointer(d, p256 + 101) == nullptr, "past-end rejected");
    CHECK(get_host_pointer(d, p256 - 1) == nullptr, "before-start rejected");

    // 64-bit interior offsets: copy through aligned interior pointers
    GpuPtr src = malloc(d, 64, 256, Memory::Default);
    GpuPtr dst = malloc(d, 64, 256, Memory::Default);
    CHECK(src != 0 && dst != 0, "aligned malloc failed");
    memset(get_host_pointer(d, src), 0xAB, 64);
    Queue q = get_queue(d);
    CommandBuffer cmd = queue_start_command_recording(q);
    cmd_memcpy(cmd, dst + 8, src + 8, 48);
    cmd_finalize(cmd);
    Submission s = queue_submit(q, {&cmd, 1});
    CHECK(s.status == SubmitStatus::Success, "submit failed");
    CHECK(wait_submission(s), "wait failed");
    auto* d8 = reinterpret_cast<uint8_t*>(get_host_pointer(d, dst + 8));
    CHECK(d8[0] == 0xAB && d8[47] == 0xAB, "interior copy offset correct");
    CHECK(reinterpret_cast<uint8_t*>(get_host_pointer(d, dst))[0] == 0,
          "bytes before the interior offset untouched");

    // Invalid alignment is rejected, not silently ignored
    CHECK(malloc(d, 64, 3, Memory::Default) == 0, "non-power-of-two alignment rejected");

    // Host memory sync: coherent memory is a successful no-op; invalid ranges fail
    CHECK(flush_host_memory(d, src, 64), "flush on coherent memory succeeds");
    CHECK(invalidate_host_memory(d, dst, 64), "invalidate on coherent memory succeeds");
    CHECK(!flush_host_memory(d, src + 100000, 4), "flush on a bogus pointer fails");
    CHECK(!invalidate_host_memory(d, src, 65), "range beyond allocation bounds fails");

    GpuPtr a = malloc(d, 8, 8, Memory::Default);
    GpuPtr b = malloc(d, 8, 8, Memory::Default);
    CHECK(a != 0 && b != 0, "small aligned malloc failed");
    CHECK(get_host_pointer(d, a) != nullptr, "small allocation host pointer");
    CHECK(get_host_pointer(d, a + 8) == nullptr, "adjacent allocation end is exclusive");
    free(d, a);
    free(d, b);
    free(d, p256);
    free(d, src);
    free(d, dst);
    destroy_device(d);
    printf("  PASS\n");
}

// --- Test 28: Descriptor handles, device limits, depth bias --------------------------------
static void test_descriptor_handles_and_limits() {
    printf("--- Test: descriptor handles + limits ---\n");
    DeviceDesc desc{
        .log_callback = test_log_callback,
        .log_level    = LogLevel::Warning,
    };
    Device d = create_device(desc);
    CHECK(d != nullptr, "create_device returned null");

    DeviceLimits lim = device_limits(d);
    CHECK(lim.max_sampled_textures > 0 && lim.max_storage_textures > 0 && lim.max_samplers > 0,
          "device limits nonzero");
    CHECK(lim.non_coherent_atom_size >= 1, "non-coherent atom size valid");
    CHECK(lim.min_uniform_alignment >= 1 && lim.min_storage_alignment >= 1, "alignments valid");

    TextureDesc tex_desc{
        .type       = TextureType::Tex2D,
        .dimensions = {16, 16, 1},
        .format     = Format::RGBA8Unorm,
        .usage      = UsageFlags::Sampled,
    };
    Handle<Texture> tex = create_texture(d, tex_desc);
    CHECK(tex.h != 0, "create_texture failed");
    if (tex.h == 0) {
        destroy_device(d);
        return;
    }

    TextureView v1 = create_texture_view(d, TextureViewDesc{.texture = tex, .format = Format::RGBA8Unorm});
    TextureView v2 = create_texture_view(d, TextureViewDesc{.texture = tex, .format = Format::RGBA8Unorm});
    CHECK(v1 != 0 && v2 != 0, "valid views are nonzero (never the null descriptor)");
    CHECK(v1 != v2, "distinct views carry distinct handles");
    CHECK((v1 & 0xFFFFFFFFull) != 0 && (v2 & 0xFFFFFFFFull) != 0, "descriptor index 0 is reserved");
    CHECK(((v1 >> 48) & 0xFF) == 1, "sampled-view type metadata");
    SamplerId sampler = create_sampler(d, SamplerDesc{});
    CHECK(sampler != 0, "sampler nonzero");
    CHECK(((sampler >> 48) & 0xFF) == 3, "sampler type metadata");

    Queue q = get_queue(d);   // needed for deferred slot retirement below

    // Recycling bumps the generation; a stale handle is rejected on free.
    free_texture_view(d, v1);
    CommandBuffer cmd = queue_start_command_recording(q);
    cmd_finalize(cmd);
    queue_submit(q, {&cmd, 1});
    device_wait_for_idle(d);
    for (int i = 0; i < 100; ++i) { queue_process_events(q); sleep_ms(5); }
    TextureView v3 = create_texture_view(d, TextureViewDesc{.texture = tex, .format = Format::RGBA8Unorm});
    CHECK((v3 & 0xFFFFFFFFull) == (v1 & 0xFFFFFFFFull), "slot is recycled");
    CHECK(((v3 >> 32) & 0xFFFF) != ((v1 >> 32) & 0xFFFF), "generation bumped on reuse");
    free_texture_view(d, v1);   // stale: rejected (logged), slot stays live
    free_texture_view(d, v3);
    // Wrong descriptor type: rejected (sampled handle passed to free_sampler)
    TextureView v4 = create_texture_view(d, TextureViewDesc{.texture = tex, .format = Format::RGBA8Unorm});
    free_sampler(d, v4);   // type mismatch -> no-op; slot intentionally leaked to device teardown
    free_sampler(d, sampler);
    free(d, tex);

    // Depth bias is applied via dynamic state (no longer silently ignored).
    DepthStencilDesc dsd{.depth_bias = 1.0f};
    Handle<DepthStencilState> ds = create_depth_stencil_state(d, dsd);
    CHECK(ds.h != 0, "depth stencil state creation");
    cmd = queue_start_command_recording(q);
    cmd_set_depth_stencil_state(cmd, ds);
    cmd_finalize(cmd);
    Submission s = queue_submit(q, {&cmd, 1});
    CHECK(s.status == SubmitStatus::Success, "depth-bias state submit");
    CHECK(wait_submission(s), "depth-bias state submission completes");
    free_depth_stencil_state(d, ds);

    destroy_device(d);
    printf("  PASS\n");
}

// --- Test 28: Texture-init submission pool stress --------------------------------------------
static void test_texture_init_pool_stress() {
    printf("--- Test: texture init pool stress ---\n");
    DeviceDesc desc{
        .log_callback = test_log_callback,
        .log_level    = LogLevel::Warning,
    };
    Device d = create_device(desc);
    CHECK(d != nullptr, "create_device returned null");
    Queue q = get_queue(d);

    // More texture-initializing submissions than the command-pool cap (64):
    // the internal transition command buffer must participate in pool
    // retirement (regression: its checkout leaked and exhausted the pools).
    for (int i = 0; i < 200; ++i) {
        TextureDesc td{
            .type       = TextureType::Tex2D,
            .dimensions = {4, 4, 1},
            .format     = Format::RGBA8Unorm,
            .usage      = UsageFlags::Sampled | UsageFlags::TransferDst | UsageFlags::TransferSrc,
        };
        Handle<Texture> tex = create_texture(d, td);
        CHECK(tex.h != 0, "create_texture failed");
        GpuPtr staging = malloc(d, 4 * 4 * 4, Memory::Default);
        CommandBuffer cmd = queue_start_command_recording(q);
        CHECK(cmd != nullptr, "recording failed — command-pool exhaustion");
        if (cmd == nullptr) { destroy_device(d); return; }
        cmd_copy_to_texture(cmd, staging, tex, BufferTextureCopyInfo{.image_extent = {4, 4, 1}});
        cmd_finalize(cmd);
        Submission s = queue_submit(q, {&cmd, 1});
        CHECK(s.status == SubmitStatus::Success, "submit failed");
        CHECK(wait_submission(s), "submission must complete");
        for (int k = 0; k < 100; ++k) { queue_process_events(q); sleep_ms(1); }
        free(d, staging);
        free(d, tex);
        for (int k = 0; k < 100; ++k) { queue_process_events(q); sleep_ms(1); }
    }
    destroy_device(d);
    printf("  PASS\n");
}

// --- Test 29: Descriptor double-free / reuse safety -------------------------------------------
static void test_descriptor_double_free() {
    printf("--- Test: descriptor double free ---\n");
    DeviceDesc desc{
        .log_callback = test_log_callback,
        .log_level    = LogLevel::Warning,
    };
    Device d = create_device(desc);
    CHECK(d != nullptr, "create_device returned null");
    Queue q = get_queue(d);

    TextureDesc td{
        .type       = TextureType::Tex2D,
        .dimensions = {8, 8, 1},
        .format     = Format::RGBA8Unorm,
        .usage      = UsageFlags::Sampled,
    };
    Handle<Texture> tex = create_texture(d, td);
    CHECK(tex.h != 0, "create_texture failed");
    if (tex.h == 0) {
        destroy_device(d);
        return;
    }

    TextureView v1 = create_texture_view(d, TextureViewDesc{.texture = tex, .format = Format::RGBA8Unorm});
    CHECK(v1 != 0, "view creation");
    const uint32_t slot1 = static_cast<uint32_t>(v1 & 0xFFFFFFFFull);

    // Accepted free marks the slot Retiring; a second free is rejected.
    free_texture_view(d, v1);
    free_texture_view(d, v1);   // double free while Retiring -> rejected (logged)

    // The Retiring slot must not be reusable before its retirement completes.
    TextureView v2 = create_texture_view(d, TextureViewDesc{.texture = tex, .format = Format::RGBA8Unorm});
    CHECK(v2 != 0, "second view");
    CHECK(static_cast<uint32_t>(v2 & 0xFFFFFFFFull) != slot1,
          "Retiring slot is not reusable before retirement");

    // Retire v1's slot; reuse it; a stale free must not corrupt the new owner.
    CommandBuffer cmd = queue_start_command_recording(q);
    cmd_finalize(cmd);
    Submission s = queue_submit(q, {&cmd, 1});
    CHECK(s.status == SubmitStatus::Success, "submit failed");
    CHECK(wait_submission(s), "wait failed");
    for (int k = 0; k < 100; ++k) { queue_process_events(q); sleep_ms(1); }

    TextureView v3 = create_texture_view(d, TextureViewDesc{.texture = tex, .format = Format::RGBA8Unorm});
    CHECK(static_cast<uint32_t>(v3 & 0xFFFFFFFFull) == slot1, "slot reused after retirement");
    CHECK(((v3 >> 32) & 0xFFFF) != ((v1 >> 32) & 0xFFFF), "generation bumped on reuse");
    free_texture_view(d, v1);   // stale generation -> rejected; must not enqueue anything
    free_texture_view(d, v2);
    free_texture_view(d, v3);

    // Owner retention: the texture survives its public free while a view holds it.
    TextureView v4 = create_texture_view(d, TextureViewDesc{.texture = tex, .format = Format::RGBA8Unorm});
    free(d, tex);   // user ref released; the view's owner reference keeps the record alive
    cmd = queue_start_command_recording(q);
    cmd_finalize(cmd);
    s = queue_submit(q, {&cmd, 1});
    CHECK(s.status == SubmitStatus::Success, "submit after texture free");
    CHECK(wait_submission(s), "wait failed");
    free_texture_view(d, v4);   // retirement releases the owner reference
    for (int k = 0; k < 100; ++k) { queue_process_events(q); sleep_ms(1); }

    destroy_device(d);
    printf("  PASS\n");
}

// --- Test 30: Concurrent texture-initializing submissions --------------------------------------
static void test_concurrent_texture_submits() {
    printf("--- Test: concurrent texture submits ---\n");
    DeviceDesc desc{
        .log_callback = test_log_callback,
        .log_level    = LogLevel::Warning,
    };
    Device d = create_device(desc);
    CHECK(d != nullptr, "create_device returned null");
    Queue q = get_queue(d);

    std::atomic<int> failures{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < 2; ++t) {
        threads.emplace_back([&]() {
            for (int i = 0; i < 50; ++i) {
                TextureDesc td{
                    .type       = TextureType::Tex2D,
                    .dimensions = {4, 4, 1},
                    .format     = Format::RGBA8Unorm,
                    .usage      = UsageFlags::Sampled | UsageFlags::TransferDst | UsageFlags::TransferSrc,
                };
                Handle<Texture> tex = create_texture(d, td);
                GpuPtr staging = malloc(d, 64, Memory::Default);
                if (tex.h == 0 || staging == 0) { failures++; continue; }
                CommandBuffer cmd = queue_start_command_recording(q);
                if (cmd == nullptr) { failures++; continue; }
                cmd_copy_to_texture(cmd, staging, tex, BufferTextureCopyInfo{.image_extent = {4, 4, 1}});
                cmd_finalize(cmd);
                Submission s = queue_submit(q, {&cmd, 1});
                if (s.status != SubmitStatus::Success) { failures++; }
                wait_submission(s);
                for (int k = 0; k < 100; ++k) { queue_process_events(q); sleep_ms(1); }
                free(d, staging);
                free(d, tex);
                for (int k = 0; k < 100; ++k) { queue_process_events(q); sleep_ms(1); }
            }
        });
    }
    for (auto& th : threads) { th.join(); }
    CHECK(failures == 0, "concurrent texture-initializing submits must all succeed");
    destroy_device(d);
    printf("  PASS\n");
}

int main() {
    printf("Izanagi API Tests\n");
    printf("=================\n\n");

    // CI runners have no Vulkan driver; allow an explicit opt-in skip.
    if (std::getenv("IZANAGI_TESTS_ALLOW_SKIP")) {
        DeviceDesc probe_desc{.log_callback = test_log_callback, .log_level = LogLevel::Error};
        Device probe = create_device(probe_desc);
        if (!probe) {
            printf("SKIPPED: no Vulkan device available\n");
            return 0;
        }
        destroy_device(probe);
    }

    test_device_create_destroy();
    test_malloc_host_pointer();
    test_compute();
    test_texture_copy();
    test_heap_slot_recycling();
    test_semaphore_and_callback();
    test_dispatch_indirect();
    test_spec_constants();
    test_draw_indirect();
    test_mip_and_cube();
    test_bc1_roundtrip();
    test_msaa_resolve();
    test_generate_mipmaps();
    test_pipeline_dedup();
    test_pipeline_cache_persistence();
    test_async_pipeline_compile();
    test_async_pipeline_pending_and_failed();
    test_async_input_ownership();
    test_async_dedup_concurrent();
    test_async_shutdown_with_pending();
    test_submission_tokens();
    test_headless_pool_retirement();
    test_free_after();
    test_texture_cb_retention();
    test_memory_alignment_and_bounds();
    test_descriptor_handles_and_limits();
    test_texture_init_pool_stress();
    test_descriptor_double_free();
    test_concurrent_texture_submits();
    test_dual_source_blend();

    printf("\n=================\n");
    if (g_failures == 0) {
        printf("ALL TESTS PASSED\n");
    } else {
        printf("%d FAILURE(S)\n", g_failures);
    }
    return g_failures;
}
