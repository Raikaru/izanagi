// android_driver_loader.h — optional replacement-driver loading for Android
// test hosts. When IZANAGI_ADRENOTOOLS_DRIVER_DIR is set, the named driver
// (adrenotools packaging, e.g. K11MCH1/AdrenoToolsDrivers zips) is loaded via
// libadrenotools and volk is pre-initialized with it; the backend then skips
// its own loader init. No-op when the env is absent or on non-Android hosts.
//
// Required env when active:
//   IZANAGI_ADRENOTOOLS_DRIVER_DIR   directory holding the driver package
//   IZANAGI_ADRENOTOOLS_DRIVER_NAME  soname of the driver (meta.json libraryName)
//   IZANAGI_ADRENOTOOLS_HOOK_DIR    directory holding libmain_hook.so etc.
//
// Include AFTER vk/internal.h (needs volk declarations).
#pragma once

#if defined(__ANDROID__) && defined(IZANAGI_ANDROID_ADRENOTOOLS)
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>

#include <adrenotools/driver.h>

inline bool load_custom_android_driver() {
    const char* dir = std::getenv("IZANAGI_ADRENOTOOLS_DRIVER_DIR");
    if (dir == nullptr) { return true; }
    const char* name  = std::getenv("IZANAGI_ADRENOTOOLS_DRIVER_NAME");
    const char* hooks = std::getenv("IZANAGI_ADRENOTOOLS_HOOK_DIR");
    if (name == nullptr || hooks == nullptr) {
        std::fprintf(stderr,
                     "adrenotools: IZANAGI_ADRENOTOOLS_DRIVER_NAME and "
                     "IZANAGI_ADRENOTOOLS_HOOK_DIR must be set\n");
        return false;
    }
    void* lib = adrenotools_open_libvulkan(RTLD_NOW, ADRENOTOOLS_DRIVER_CUSTOM,
                                           nullptr, hooks, dir, name, nullptr, nullptr);
    if (lib == nullptr) {
        std::fprintf(stderr, "adrenotools: open_libvulkan failed: %s\n", dlerror());
        return false;
    }
    auto gipa = reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(lib, "vkGetInstanceProcAddr"));
    if (gipa == nullptr) {
        std::fprintf(stderr, "adrenotools: vkGetInstanceProcAddr missing: %s\n", dlerror());
        return false;
    }
    volkInitializeCustom(gipa);
    std::fprintf(stderr, "adrenotools: loaded custom driver %s from %s\n", name, dir);
    return true;
}
#else
inline bool load_custom_android_driver() { return true; }
#endif
