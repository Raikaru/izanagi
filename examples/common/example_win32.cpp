// example_win32.cpp — Win32 window, message pump, frame loop, screenshot mode.

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#include <shellapi.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "common/example.h"
#include "common/gpu_args.h"
#include "common/math.h"

// stb_image_write for PNG screenshots
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "common/stb_image_write.h"

using namespace gpu;

static HWND  g_window = nullptr;
static HINSTANCE g_instance = nullptr;
static uint32_t g_width  = 800;
static uint32_t g_height = 600;
static bool g_quit = false;

// CLI options
static const char* g_example_name = nullptr;
static int         g_max_frames   = -1; // -1 = interactive
static const char* g_screenshot_path = nullptr;
static int         g_screenshot_frame = 0; // 0 = last frame

gpu::Format g_example_surface_format = gpu::Format::BGRA8Unorm;
static int  g_cycle_request = 0; // +1 next example, -1 previous (M/N)
static int  g_cycle_test    = 0; // N>0: auto-cycle N times, then exit 0
static bool g_resize_pending = false;

LRESULT WINAPI window_proc(HWND wnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
        case WM_SIZE: {
            g_width  = LOWORD(lParam);
            g_height = HIWORD(lParam);
            g_resize_pending = true;
        } break;
        case WM_KEYDOWN: {
            if (wParam == VK_ESCAPE) {
                g_quit = true;
            } else if (wParam == 'M') {
                g_cycle_request = -1; // previous example
            } else if (wParam == 'N') {
                g_cycle_request = +1; // next example
            }
        } break;
        case WM_DESTROY:
            PostQuitMessage(0);
            g_quit = true;
            break;
        default: return DefWindowProcA(wnd, msg, wParam, lParam);
    }
    return 0;
}

static void log_callback(LogLevel lvl, Span<const char> msg, uint32_t line,
                          Span<const char> file, void*) {
    const char* lvl_str[] = {"OFF", "ERROR", "WARN", "INFO", "DEBUG"};
    printf("[%s] %.*s\n", lvl_str[static_cast<int>(lvl)], (int)msg.size(), msg.data());
}

// Read a SPIR-V file
static std::string load_spirv_file(const char* dir, const char* name, uint32_t* out_size) {
    char path[512];
    snprintf(path, sizeof(path), "%s/%s", dir, name);
    FILE* f = fopen(path, "rb");
    if (!f) {
        printf("Failed to open shader: %s\n", path);
        *out_size = 0;
        return {};
    }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::string data(size, '\0');
    fread(data.data(), 1, size, f);
    fclose(f);
    *out_size = (uint32_t)size;
    return data;
}

// Write a PNG screenshot from readback buffer
static bool write_screenshot(const char* path, const void* pixels, uint32_t w, uint32_t h) {
    return stbi_write_png(path, w, h, 4, pixels, w * 4) != 0;
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int) {
    g_instance = hInstance;

    // Attach to parent console for printf output
    AttachConsole(ATTACH_PARENT_PROCESS);
    FILE* dummy;
    freopen_s(&dummy, "CONOUT$", "w", stdout);
    freopen_s(&dummy, "CONOUT$", "w", stderr);

    // Parse CLI
    LPWSTR* argv = CommandLineToArgvW(GetCommandLineW(), nullptr);
    int argc = 0;
    argv = CommandLineToArgvW(GetCommandLineW(), &argc);
    for (int i = 0; i < argc; ++i) {
        if (wcscmp(argv[i], L"--example") == 0 && i + 1 < argc) {
            static char name_buf[256];
            WideCharToMultiByte(CP_UTF8, 0, argv[i + 1], -1, name_buf, sizeof(name_buf), NULL, NULL);
            g_example_name = name_buf;
            i++;
        } else if (wcscmp(argv[i], L"--frames") == 0 && i + 1 < argc) {
            g_max_frames = _wtoi(argv[i + 1]);
            i++;
        } else if (wcscmp(argv[i], L"--screenshot") == 0 && i + 1 < argc) {
            static char path_buf[512];
            WideCharToMultiByte(CP_UTF8, 0, argv[i + 1], -1, path_buf, sizeof(path_buf), NULL, NULL);
            g_screenshot_path = path_buf;
            i++;
        } else if (wcscmp(argv[i], L"--screenshot-frame") == 0 && i + 1 < argc) {
            g_screenshot_frame = _wtoi(argv[i + 1]);
            i++;
        } else if (wcscmp(argv[i], L"--cycle-test") == 0 && i + 1 < argc) {
            g_cycle_test = _wtoi(argv[i + 1]);
            i++;
        }
    }
    LocalFree(argv);

    register_example_vtables();

    // Find example
    int example_idx = 0;
    if (g_example_name) {
        example_idx = find_example_by_name(g_example_name);
        if (example_idx < 0) {
            printf("Unknown example: %s\n", g_example_name);
            return 1;
        }
    }

    // Create window
    const char CLASS_NAME[] = "IzanagiExamples";
    WNDCLASSEXA wc = {
        .cbSize        = sizeof(WNDCLASSEXA),
        .style         = CS_VREDRAW | CS_HREDRAW,
        .lpfnWndProc   = window_proc,
        .hInstance     = hInstance,
        .hbrBackground = (HBRUSH)(COLOR_WINDOW + 1),
        .lpszClassName = CLASS_NAME,
    };
    RegisterClassExA(&wc);

    g_window = CreateWindowExA(0, CLASS_NAME, "Izanagi Examples",
                               WS_OVERLAPPEDWINDOW,
                               CW_USEDEFAULT, CW_USEDEFAULT,
                               g_width, g_height,
                               NULL, NULL, hInstance, NULL);

    ShowWindow(g_window, SW_NORMAL);
    UpdateWindow(g_window);

    // Create device
    DeviceDesc device_desc{
        .native_window_handle   = (uintptr_t)g_window,
        .native_instance_handle = (uintptr_t)hInstance,
        .log_callback           = log_callback,
        .log_level              = LogLevel::Warning,
    };
    Device device = create_device(device_desc);
    if (!device) {
        printf("Failed to create device\n");
        return 1;
    }

    // Get surface capabilities and configure
    SurfaceCapabilities caps = get_surface_capabilities(device);
    Format surface_format = Format::BGRA8UnormSrgb;
    // First pass: prefer sRGB variant (accurate color for screenshots).
    bool format_found = false;
    for (uint32_t i = 0; i < caps.formats.size(); ++i) {
        if (caps.formats[i] == Format::BGRA8UnormSrgb) {
            surface_format = caps.formats[i];
            format_found = true;
            break;
        }
    }
    // Second pass: non-sRGB BGRA fallback.
    if (!format_found) {
        for (uint32_t i = 0; i < caps.formats.size(); ++i) {
            if (caps.formats[i] == Format::BGRA8Unorm) {
                surface_format = caps.formats[i];
                format_found = true;
                break;
            }
        }
    }
    // Last resort: whatever the driver offers first.
    if (!format_found && caps.formats.size() > 0) {
        surface_format = caps.formats[0];
    }
    bool surface_is_bgra = surface_format == Format::BGRA8UnormSrgb ||
                           surface_format == Format::BGRA8Unorm;

    PresentMode present_mode = PresentMode::Fifo;
    for (uint32_t i = 0; i < caps.present_modes.size(); ++i) {
        if (caps.present_modes[i] == PresentMode::Mailbox) {
            present_mode = PresentMode::Mailbox;
            break;
        }
    }

    SurfaceConfiguration surface_config{
        .format       = surface_format,
        .usages       = UsageFlags::ColorAttachment | UsageFlags::TransferSrc,
        .width        = g_width,
        .height       = g_height,
        .present_mode = present_mode,
    };
    g_example_surface_format = surface_format;

    if (!configure_surface(device, surface_config)) {
        printf("Failed to configure surface\n");
        destroy_device(device);
        return 1;
    }

    // Initialize example — pass the address of the userdata slot so init
    // can store its state pointer into it (examples use State** convention).
    ExampleVTable* example = get_example_vtable(example_idx);
    example->init(device, &example->userdata);

    // Find shader directory
    const char* shader_dir = "shaders";

    // Frame loop
    Queue queue = get_queue(device);
    GpuArgs gpu_args;
    gpu_args_init(&gpu_args, device);

    float time = 0.0f;
    int frame = 0;
    bool running = true;

    while (running && !g_quit) {
        // Message pump
        MSG msg = {};
        while (PeekMessageA(&msg, NULL, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessageA(&msg);
            if (msg.message == WM_QUIT) { running = false; }
        }
        if (!running) break;

        // Handle M/N example cycling
        if (g_cycle_request != 0) {
            device_wait_for_idle(device);
            example->shutdown(device, &example->userdata);
            int count = get_example_count();
            example_idx = (example_idx + g_cycle_request + count) % count;
            g_cycle_request = 0;
            example = get_example_vtable(example_idx);
            example->userdata = nullptr;
            example->init(device, &example->userdata);
            printf("Example: %s\n", get_example_name(example_idx));
            frame = 0; // reset animation time base per example
        }

        // --cycle-test N: auto-press N every 30 frames, resize once midway,
        // exit 0 after N cycles. Exercises the same paths as M/N + resize.
        if (g_cycle_test > 0) {
            static int cycles_done = 0;
            static bool did_resize = false;
            if (frame == 15 && !did_resize && cycles_done >= 1) {
                did_resize = true;
                SetWindowPos(g_window, nullptr, 0, 0, g_width / 2, g_height / 2,
                             SWP_NOMOVE | SWP_NOZORDER);
                printf("cycle-test: resizing to %dx%d\n", g_width / 2, g_height / 2);
            }
            if (frame == 30) {
                frame = 0;
                cycles_done++;
                if (cycles_done >= g_cycle_test) {
                    printf("cycle-test: %d cycles OK, exiting\n", cycles_done);
                    running = false;
                } else {
                    g_cycle_request = +1; // same as pressing N
                }
            }
        }

        // Handle resize
        if (g_resize_pending) {
            g_resize_pending = false;
            surface_config.width  = g_width;
            surface_config.height = g_height;
            configure_surface(device, surface_config);
        }

        // Get current surface texture
        SurfaceTextureInfo surface_tex = get_current_texture(device);
        if (surface_tex.status == SurfaceStatus::OutOfDate) {
            configure_surface(device, surface_config);
            continue;
        }
        if (surface_tex.status != SurfaceStatus::Success) {
        }

        // Record commands
        CommandBuffer cmd = queue_start_command_recording(queue);
        cmd_wait_for_surface_texture(cmd);

        Dimension2D tex_size{g_width, g_height};
        bool should_continue = example->render(device, cmd, surface_tex.texture,
                                                tex_size, time, &example->userdata);

        // Screenshot: read back the swapchain image in THIS command buffer,
        // before the present transition — the image is still in GENERAL layout
        // and owned by us here.
        int shot_frame = g_screenshot_frame > 0 ? g_screenshot_frame
                         : (g_max_frames > 0   ? g_max_frames
                                               : 60);
        bool take_screenshot = g_screenshot_path != nullptr && (frame + 1 == shot_frame);
        GpuPtr readback = 0;
        if (take_screenshot) {
            readback = malloc(device, g_width * g_height * 4, Memory::Readback);
            cmd_barrier(cmd, StageFlags::RasterColorOut, StageFlags::Transfer);
            BufferTextureCopyInfo copy_info{
                .image_extent = {g_width, g_height, 1},
            };
            cmd_copy_from_texture(cmd, surface_tex.texture, readback, copy_info);
        }

        cmd_signal_surface_texture(cmd);
        cmd_finalize(cmd);
        queue_submit(queue, {&cmd, 1});
        present(device, queue);

        queue_process_events(queue);

        if (take_screenshot) {
            device_wait_for_idle(device);
            void* pixels = get_host_pointer(device, readback);
            if (pixels) {
                // Swapchain is BGRA8; PNG wants RGBA. Swap R/B (BGRA only).
                if (surface_is_bgra) {
                    auto* bytes = static_cast<uint8_t*>(pixels);
                    for (int i = 0; i < g_width * g_height * 4; i += 4) {
                        uint8_t t = bytes[i];
                        bytes[i] = bytes[i + 2];
                        bytes[i + 2] = t;
                    }
                }
                if (write_screenshot(g_screenshot_path, pixels, g_width, g_height)) {
                    printf("Screenshot written to %s\n", g_screenshot_path);
                } else {
                    printf("Failed to write screenshot\n");
                }
            }
            free(device, readback);
            running = false;
        }

        time += 1.0f / 60.0f;
        frame++;

        if (g_max_frames > 0 && frame >= g_max_frames) {
            running = false;
        }

        if (!should_continue) { running = false; }
    }

    // Cleanup
    device_wait_for_idle(device);
    example->shutdown(device, &example->userdata);
    gpu_args_shutdown(&gpu_args, device);
    unconfigure_surface(device);
    destroy_device(device);

    DestroyWindow(g_window);
    return 0;
}
