cmake_minimum_required(VERSION 3.28)

# Read version from project root, default to 0.1.0
project(Izanagi VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "" FORCE)
endif()

# Place all binaries in predictable locations
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

add_subdirectory(third_party)

# --- Izanagi static library ---------------------------------------------------
add_library(izanagi STATIC
    src/common/containers.cpp
    src/common/format_info.cpp
    src/common/platform_utils.cpp
    src/vk/volk_usage.cpp
    src/vk/vma_usage.cpp
    src/vk/gpu_to_vk.cpp
    src/vk/device.cpp
    src/vk/resources.cpp
    src/vk/pipeline.cpp
    src/vk/commands.cpp
    src/vk/surface.cpp
)

target_include_directories(izanagi
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(izanagi
    PUBLIC  volk_headers
    PRIVATE GPUOpen::VulkanMemoryAllocator
)

target_compile_definitions(izanagi PRIVATE
    VK_USE_PLATFORM_WIN32_KHR
    VK_NO_PROTOTYPES
    WIN32_LEAN_AND_MEAN
    NOMINMAX
)

if(MSVC)
    target_compile_options(izanagi PRIVATE /W4 /permissive- /Zc:preprocessor /wd4201 /wd4324)
endif()

# --- Shader toolchain ---------------------------------------------------------
include(cmake/CompileSlangShader.cmake)

# --- Tests --------------------------------------------------------------------
add_slang_shader(izanagi_test_shaders
    SOURCE ${CMAKE_SOURCE_DIR}/tests/shaders/memcpy_kernel.slang
    OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/shaders/memcpy_kernel.spv
)

add_executable(izanagi_tests tests/api_tests.cpp)
target_link_libraries(izanagi_tests PRIVATE izanagi)
add_dependencies(izanagi_tests izanagi_test_shaders)

# --- Examples ------------------------------------------------------------------
add_slang_shader(izanagi_example_shaders
    SOURCE ${CMAKE_SOURCE_DIR}/examples/shaders/hello_triangle.slang
    OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/shaders/hello_triangle.spv
)
add_slang_shader(izanagi_example_shaders2
    SOURCE ${CMAKE_SOURCE_DIR}/examples/shaders/compute_texture.slang
    OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/shaders/compute_texture.spv
)
add_slang_shader(izanagi_example_shaders3
    SOURCE ${CMAKE_SOURCE_DIR}/examples/shaders/textured_cube.slang
    OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/shaders/textured_cube.spv
)

add_executable(izanagi_examples WIN32
    examples/common/example.cpp
    examples/common/example_win32.cpp
    examples/common/gpu_args.cpp
    examples/hello_triangle/hello_triangle.cpp
    examples/compute_texture/compute_texture.cpp
    examples/textured_cube/textured_cube.cpp
)
target_include_directories(izanagi_examples PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/examples)
target_link_libraries(izanagi_examples PRIVATE izanagi)
add_dependencies(izanagi_examples izanagi_example_shaders izanagi_example_shaders2 izanagi_example_shaders3)
