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
# Pinned to 2026.5.2 for Texture2D.Handle()/spvDescriptorHeapEXT support.
# The zip is vendored at third_party/slang (see plan); slangc.exe runs at build time.
find_program(SLANGC_EXECUTABLE
    NAMES slangc slangc.exe
    PATHS "${CMAKE_SOURCE_DIR}/third_party/slang/bin"
    NO_DEFAULT_PATH
)
if(NOT SLANGC_EXECUTABLE)
    message(FATAL_ERROR "slangc not found at ${CMAKE_SOURCE_DIR}/third_party/slang/bin — extract slang-2026.5.2-windows-x86_64.zip there.")
endif()
message(STATUS "Found slangc: ${SLANGC_EXECUTABLE}")
