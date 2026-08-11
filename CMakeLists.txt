cmake_minimum_required(VERSION 3.28)

# Read version from project root, default to 0.1.0
project(Izanagi VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "" FORCE)
endif()

# Tests/examples default ON when Izanagi is the top-level project, OFF when
# consumed via add_subdirectory/FetchContent.
option(IZANAGI_BUILD_TESTS    "Build the API test suite"    ${PROJECT_IS_TOP_LEVEL})
option(IZANAGI_BUILD_EXAMPLES "Build the example programs"  ${PROJECT_IS_TOP_LEVEL})
mark_as_advanced(IZANAGI_BUILD_TESTS IZANAGI_BUILD_EXAMPLES)

# Place all binaries in predictable locations
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

add_subdirectory(third_party)

include(GNUInstallDirs)

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

add_library(Izanagi::izanagi ALIAS izanagi)

target_include_directories(izanagi
    PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# volk_headers / VMA are compile-time-only deps of the .cpp files; their code
# is baked into the static lib and the public header (gpu.h) is Vulkan-free,
# so the installed export must not reference them. $<BUILD_INTERFACE:...>
# keeps them for in-tree builds and drops them from the install export.
target_link_libraries(izanagi
    PRIVATE $<BUILD_INTERFACE:volk_headers>
    PRIVATE $<BUILD_INTERFACE:GPUOpen::VulkanMemoryAllocator>
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

# --- Install / export ----------------------------------------------------------
include(CMakePackageConfigHelpers)

install(TARGETS izanagi
    EXPORT IzanagiTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/izanagi
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(EXPORT IzanagiTargets
    FILE IzanagiTargets.cmake
    NAMESPACE Izanagi::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Izanagi
)

configure_package_config_file(
    cmake/IzanagiConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/IzanagiConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Izanagi
)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/IzanagiConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/IzanagiConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/IzanagiConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Izanagi
)

# --- Shader toolchain ---------------------------------------------------------
if(IZANAGI_BUILD_TESTS OR IZANAGI_BUILD_EXAMPLES)
    include(cmake/CompileSlangShader.cmake)
endif()

# --- Tests --------------------------------------------------------------------
if(IZANAGI_BUILD_TESTS)
    add_slang_shader(izanagi_test_shaders
        SOURCE ${CMAKE_SOURCE_DIR}/tests/shaders/memcpy_kernel.slang
        OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/shaders/memcpy_kernel.spv
    )
    add_slang_shader(izanagi_test_shaders2
        SOURCE ${CMAKE_SOURCE_DIR}/tests/shaders/spec_mul_kernel.slang
        OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/shaders/spec_mul_kernel.spv
    )
    add_slang_shader(izanagi_test_shaders3
        SOURCE ${CMAKE_SOURCE_DIR}/tests/shaders/offscreen_triangle.slang
        OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/shaders/offscreen_triangle.spv
    )
    add_slang_shader(izanagi_test_shaders4
        SOURCE ${CMAKE_SOURCE_DIR}/tests/shaders/dual_src.slang
        OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/shaders/dual_src.spv
    )

    add_executable(izanagi_tests tests/api_tests.cpp)
    target_link_libraries(izanagi_tests PRIVATE izanagi)
    add_dependencies(izanagi_tests izanagi_test_shaders izanagi_test_shaders2 izanagi_test_shaders3
                     izanagi_test_shaders4)

    enable_testing()
    add_test(NAME api_tests COMMAND izanagi_tests)
endif()

# --- Examples ------------------------------------------------------------------
if(IZANAGI_BUILD_EXAMPLES)
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
endif()
