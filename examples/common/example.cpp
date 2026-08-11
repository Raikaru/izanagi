// example.cpp — example registry and frame loop.
#include "example.h"
#include <cstdio>
#include <cstring>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#endif

// --- Shader loading -----------------------------------------------------------
// Resolves <exe_dir>/shaders/<profile>/<name> (falling back to the plain
// shaders dir for older build trees) so examples work regardless of CWD.
// The returned Span points at a static buffer valid until the next call;
// pipeline creation copies the bytes synchronously, so this is safe.

gpu::Span<const uint8_t> example_load_shader(const char* name, uint32_t* out_size) {
    static uint8_t buffer[1 << 20]; // 1 MiB max SPIR-V
    *out_size = 0;

    char exe_path[2048];
    DWORD len = GetModuleFileNameA(nullptr, exe_path, sizeof(exe_path));
    if (len == 0 || len >= sizeof(exe_path)) { return {}; }

    // Strip the executable file name, keep the directory.
    char* slash = strrchr(exe_path, '\\');
    if (!slash) { return {}; }
    *(slash + 1) = '\0';

    const char* profile_dir = "shaders/vk_native/";
#if defined(IZ_VK_PROFILE_BINDLESS)
    profile_dir = "shaders/vk_bindless/";
#endif

    char path[4096];
    snprintf(path, sizeof(path), "%s%s%s", exe_path, profile_dir, name);
    FILE* f = fopen(path, "rb");
    if (!f) {
        snprintf(path, sizeof(path), "%sshaders/%s", exe_path, name);
        f = fopen(path, "rb");
    }
    if (!f) {
        printf("Failed to open shader: %s\n", path);
        return {};
    }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (size <= 0 || (size_t)size >= sizeof(buffer)) {
        fclose(f);
        printf("Shader too large or empty: %s\n", path);
        return {};
    }
    size_t got = fread(buffer, 1, (size_t)size, f);
    fclose(f);
    if (got != (size_t)size) {
        printf("Failed to read shader: %s\n", path);
        return {};
    }
    *out_size = (uint32_t)got;
    return gpu::Span<const uint8_t>(buffer, (uint32_t)got);
}

// Example registry
struct ExampleEntry {
    const char* name;
    ExampleVTable vtable;
};

static ExampleEntry g_examples[] = {
    {"hello_triangle", {}},
    {"compute_texture", {}},
    {"textured_cube", {}},
};
static int g_example_count;

// Forward declarations of example vtables
extern ExampleVTable hello_triangle_vtable;
extern ExampleVTable compute_texture_vtable;
extern ExampleVTable textured_cube_vtable;

void register_example_vtables() {
    g_examples[0].vtable = hello_triangle_vtable;
    g_examples[1].vtable = compute_texture_vtable;
    g_examples[2].vtable = textured_cube_vtable;
    g_example_count = 3;
}

int get_example_count() { return g_example_count; }
const char* get_example_name(int idx) { return g_examples[idx].name; }
ExampleVTable* get_example_vtable(int idx) { return &g_examples[idx].vtable; }
int find_example_by_name(const char* name) {
    for (int i = 0; i < g_example_count; ++i) {
        if (strcmp(g_examples[i].name, name) == 0) return i;
    }
    return -1;
}
