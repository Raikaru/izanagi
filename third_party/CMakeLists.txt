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

# --- Slang (prebuilt, build-time tool only) -----------------------------------
# Only needed when tests or examples are built (the library itself compiles
# without slangc). Pinned to 2026.5.2 for Texture2D.Handle()/spvDescriptorHeapEXT
# support. Downloaded on demand to ${CMAKE_BINARY_DIR}/_deps/slang (not
# committed); a pre-installed slangc on PATH takes precedence.
if(IZANAGI_BUILD_TESTS OR IZANAGI_BUILD_EXAMPLES)
    set(SLANGC_VERSION "2026.5.2")
    set(SLANGC_URL "https://github.com/shader-slang/slang/releases/download/v${SLANGC_VERSION}/slang-${SLANGC_VERSION}-windows-x86_64.zip")
    set(SLANGC_SHA256 "8751a5c897dacc3d05bb5bb063abf4200b2bd91052e893abd48cd941219a1549")

    find_program(SLANGC_EXECUTABLE NAMES slangc slangc.exe)
    if(NOT SLANGC_EXECUTABLE)
        set(_slangc_dir "${CMAKE_BINARY_DIR}/_deps/slang-${SLANGC_VERSION}")
        set(_slangc_zip "${_slangc_dir}.zip")
        if(NOT EXISTS "${_slangc_dir}/bin/slangc.exe")
            message(STATUS "Downloading slangc ${SLANGC_VERSION} (${SLANGC_URL})")
            file(DOWNLOAD "${SLANGC_URL}" "${_slangc_zip}"
                 EXPECTED_HASH "SHA256=${SLANGC_SHA256}"
                 SHOW_PROGRESS
                 STATUS _dl_status)
            list(GET _dl_status 0 _dl_code)
            if(NOT _dl_code EQUAL 0)
                list(GET _dl_status 1 _dl_error)
                message(FATAL_ERROR "Failed to download slangc: ${_dl_error}")
            endif()
            file(ARCHIVE_EXTRACT INPUT "${_slangc_zip}" DESTINATION "${_slangc_dir}")
        endif()
        find_program(SLANGC_EXECUTABLE NAMES slangc.exe
                     PATHS "${_slangc_dir}/bin" NO_DEFAULT_PATH)
    endif()
    if(NOT SLANGC_EXECUTABLE)
        message(FATAL_ERROR "slangc not found: install Slang ${SLANGC_VERSION} and add slangc.exe to PATH")
    endif()
    message(STATUS "Found slangc: ${SLANGC_EXECUTABLE}")
endif()
