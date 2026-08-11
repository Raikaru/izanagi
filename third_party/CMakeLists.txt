include(FetchContent)

# --- Vulkan-Headers ----------------------------------------------------------
FetchContent_Declare(
    VulkanHeaders
    GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers.git
    GIT_TAG        vulkan-sdk-1.4.341.0
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(VulkanHeaders)
# Vulkan-Headers exposes an interface target "Vulkan::Headers"

# --- volk --------------------------------------------------------------------
FetchContent_Declare(
    volk
    GIT_REPOSITORY https://github.com/zeux/volk.git
    GIT_TAG        vulkan-sdk-1.4.341.0
    GIT_SHALLOW    TRUE
)
set(VOLK_PULL_IN_VULKAN OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(volk)
# volk creates "volk" (compiles volk.c) and "volk_headers" (interface).
# We compile our own impl TU (src/vk/volk_usage.cpp) with VOLK_IMPLEMENTATION,
# so exclude the volk.c target from the build to avoid duplicate symbols.
set_target_properties(volk PROPERTIES EXCLUDE_FROM_ALL TRUE)
target_link_libraries(volk_headers INTERFACE Vulkan::Headers)

# --- VulkanMemoryAllocator ---------------------------------------------------
FetchContent_Declare(
    VulkanMemoryAllocator
    GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
    GIT_TAG        v3.3.0
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(VulkanMemoryAllocator)
# VMA exposes interface target "VulkanMemoryAllocator" / "GPUOpen::VulkanMemoryAllocator"

# --- Slang (prebuilt, build-host tool only) ------------------------------------
# Only needed when tests or examples are built (the library itself compiles
# without slangc). slangc is a BUILD-HOST executable: selection uses
# CMAKE_HOST_SYSTEM_*, never CMAKE_SYSTEM_* (cross-compiling for Android/iOS
# must use a host binary). Discovery order:
#   1. explicit IZANAGI_SLANGC path
#   2. a compatible slangc on PATH
#   3. a repository-managed downloaded host package
#   4. a clear configuration failure
# Pinned to 2026.5.2 for Texture2D.Handle()/spvDescriptorHeapEXT support.
if(IZANAGI_BUILD_TESTS OR IZANAGI_BUILD_EXAMPLES)
    set(SLANGC_VERSION "2026.5.2")

    if(IZANAGI_SLANGC)
        set(SLANGC_EXECUTABLE "${IZANAGI_SLANGC}")
        if(NOT EXISTS "${SLANGC_EXECUTABLE}")
            message(FATAL_ERROR "IZANAGI_SLANGC points at a missing file: ${IZANAGI_SLANGC}")
        endif()
    else()
        find_program(SLANGC_EXECUTABLE NAMES slangc slangc.exe)
        if(NOT SLANGC_EXECUTABLE)
            # Host-aware package selection.
            if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Windows")
                set(_slang_os "windows")
                set(_slang_arch "x86_64")
                set(_slang_ext ".exe")
                set(_slang_sha "8751a5c897dacc3d05bb5bb063abf4200b2bd91052e893abd48cd941219a1549")
            elseif(CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
                set(_slang_os "linux")
                if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
                    set(_slang_arch "aarch64")
                else()
                    set(_slang_arch "x86_64")
                endif()
                set(_slang_ext "")
                set(_slang_sha "")   # TODO: pin a SHA256 for the Linux package
            elseif(CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin")
                set(_slang_os "macos")
                if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
                    set(_slang_arch "arm64")
                else()
                    set(_slang_arch "x86_64")
                endif()
                set(_slang_ext "")
                set(_slang_sha "")   # TODO: pin a SHA256 for the macOS package
            else()
                message(FATAL_ERROR
                    "No slangc host package for ${CMAKE_HOST_SYSTEM_NAME}. Install Slang "
                    "${SLANGC_VERSION} and add slangc to PATH, or set IZANAGI_SLANGC.")
            endif()

            set(_slangc_dir "${CMAKE_BINARY_DIR}/_deps/slang-${SLANGC_VERSION}-${_slang_os}-${_slang_arch}")
            set(_slangc_zip "${_slangc_dir}.zip")
            set(SLANGC_URL "https://github.com/shader-slang/slang/releases/download/v${SLANGC_VERSION}/slang-${SLANGC_VERSION}-${_slang_os}-${_slang_arch}.zip")
            if(NOT EXISTS "${_slangc_dir}/bin/slangc${_slang_ext}")
                message(STATUS "Downloading slangc ${SLANGC_VERSION} (${SLANGC_URL})")
                if(_slang_sha)
                    file(DOWNLOAD "${SLANGC_URL}" "${_slangc_zip}"
                         EXPECTED_HASH "SHA256=${_slang_sha}"
                         SHOW_PROGRESS STATUS _dl_status)
                else()
                    file(DOWNLOAD "${SLANGC_URL}" "${_slangc_zip}"
                         SHOW_PROGRESS STATUS _dl_status)
                endif()
                list(GET _dl_status 0 _dl_code)
                if(NOT _dl_code EQUAL 0)
                    list(GET _dl_status 1 _dl_error)
                    message(FATAL_ERROR "Failed to download slangc: ${_dl_error}")
                endif()
                file(ARCHIVE_EXTRACT INPUT "${_slangc_zip}" DESTINATION "${_slangc_dir}")
            endif()
            find_program(SLANGC_EXECUTABLE NAMES "slangc${_slang_ext}"
                         PATHS "${_slangc_dir}/bin" NO_DEFAULT_PATH)
        endif()
    endif()
    if(NOT SLANGC_EXECUTABLE)
        message(FATAL_ERROR
            "slangc not found: install Slang ${SLANGC_VERSION} for ${CMAKE_HOST_SYSTEM_NAME} "
            "on PATH, or set IZANAGI_SLANGC to a host slangc executable")
    endif()
    message(STATUS "Found slangc (host): ${SLANGC_EXECUTABLE}")
endif()
