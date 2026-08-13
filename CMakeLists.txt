cmake_minimum_required(VERSION 3.28)

# Read version from project root, default to 0.2.0
project(Izanagi VERSION 0.2.0 LANGUAGES CXX)

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
option(IZANAGI_BUILD_CAPABILITY_REPORT "Build the capability-report tool" ${PROJECT_IS_TOP_LEVEL})
# Android test hosts: optional replacement-driver loading (adrenotools
# packaging). Point IZANAGI_ADRENOTOOLS_ROOT at a libadrenotools source
# checkout; drivers are selected at runtime via IZANAGI_ADRENOTOOLS_* env.
set(IZANAGI_ADRENOTOOLS_ROOT "" CACHE PATH "libadrenotools source dir (Android test hosts only)")
mark_as_advanced(IZANAGI_BUILD_TESTS IZANAGI_BUILD_EXAMPLES IZANAGI_BUILD_CAPABILITY_REPORT)

# --- Backend and WSI selection -------------------------------------------------
# One backend per compiled library. IZANAGI_BACKEND selects the private
# implementation; IZANAGI_WSI selects the window-system integration (HEADLESS
# requires no desktop window system). Unsupported combinations fail configure
# with a clear message — never silently pick a different backend/WSI.
set(IZANAGI_BACKEND "VULKAN_NATIVE" CACHE STRING
    "Private backend: VULKAN_NATIVE | VULKAN_COMPAT | METAL")
set(IZANAGI_WSI "AUTO" CACHE STRING
    "Window-system integration: AUTO | HEADLESS | WIN32 | XCB | WAYLAND | ANDROID | METAL")
set_property(CACHE IZANAGI_BACKEND PROPERTY STRINGS VULKAN_NATIVE VULKAN_COMPAT METAL)
set_property(CACHE IZANAGI_WSI PROPERTY STRINGS AUTO HEADLESS WIN32 XCB WAYLAND ANDROID METAL)

# Host shader tool: explicit path wins, then PATH, then a repository-managed
# host package (see third_party/CMakeLists.txt).
set(IZANAGI_SLANGC "" CACHE FILEPATH "Path to a host slangc executable (empty = auto-discover)")

# Backend profile names. The trailing version suffix is PART of the shader
# artifact key (shaders/vk_<backend>_<N>_spv<VV>/): bumping a profile version
# changes the artifact directory, so old artifacts can never be mistaken for
# new ones. IZANAGI_PROFILE is the selected profile's advertised name.
set(IZANAGI_VK_NATIVE_PROFILE_NAME   "IZANAGI_VK_NATIVE_1"   CACHE STRING "Native profile name (version suffix is part of the artifact key)")
set(IZANAGI_VK_BINDLESS_PROFILE_NAME "IZANAGI_VK_BINDLESS_1" CACHE STRING "Bindless profile name (version suffix is part of the artifact key)")

# Vulkan capability profile: which private mechanism implements the public
# model. NATIVE = the modern descriptor-heap path (Vulkan 1.4 + descriptor
# heap + untyped pointers). BINDLESS = the descriptor-indexing compatibility
# path (declared; its backend lands in the compatibility phases).
set(IZANAGI_VK_PROFILE "NATIVE" CACHE STRING "Vulkan capability profile: NATIVE | BINDLESS")
set_property(CACHE IZANAGI_VK_PROFILE PROPERTY STRINGS NATIVE BINDLESS)

# IZANAGI_PROFILE (the advertised profile name) is derived from the selected
# VK profile and self-heals stale cache values; the name MUST match the
# selected profile because it is part of the artifact identity.
if(IZANAGI_VK_PROFILE STREQUAL "BINDLESS")
    if(NOT IZANAGI_PROFILE MATCHES "^IZANAGI_VK_BINDLESS_")
        set(IZANAGI_PROFILE "${IZANAGI_VK_BINDLESS_PROFILE_NAME}" CACHE STRING "Capability profile name" FORCE)
    endif()
else()
    if(NOT IZANAGI_PROFILE MATCHES "^IZANAGI_VK_NATIVE_")
        set(IZANAGI_PROFILE "${IZANAGI_VK_NATIVE_PROFILE_NAME}" CACHE STRING "Capability profile name" FORCE)
    endif()
endif()

if(NOT IZANAGI_BACKEND STREQUAL "VULKAN_NATIVE")
    message(FATAL_ERROR
        "IZANAGI_BACKEND=${IZANAGI_BACKEND} is not implemented yet. This phase "
        "ships the portable build boundary; VULKAN_COMPAT and METAL backends land "
        "in later phases. Supported now: VULKAN_NATIVE.")
endif()
if(NOT IZANAGI_VK_PROFILE STREQUAL "NATIVE" AND NOT IZANAGI_VK_PROFILE STREQUAL "BINDLESS")
    message(FATAL_ERROR "IZANAGI_VK_PROFILE=${IZANAGI_VK_PROFILE} is not a valid profile")
endif()

# Resolve AUTO WSI per host.
if(IZANAGI_WSI STREQUAL "AUTO")
    if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
        set(IZANAGI_WSI "WIN32")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Android")
        set(IZANAGI_WSI "ANDROID")
    elseif(APPLE)
        set(IZANAGI_WSI "HEADLESS")   # METAL WSI lands with the Metal backend
    else()
        set(IZANAGI_WSI "HEADLESS")   # Linux AUTO stays headless; select XCB/WAYLAND explicitly
    endif()
endif()

# Validate backend/WSI/host combinations.
if(IZANAGI_WSI STREQUAL "WIN32" AND NOT WIN32)
    message(FATAL_ERROR "IZANAGI_WSI=WIN32 requires a Windows host")
endif()
if(IZANAGI_WSI STREQUAL "XCB" OR IZANAGI_WSI STREQUAL "WAYLAND")
    if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
        message(FATAL_ERROR "IZANAGI_WSI=${IZANAGI_WSI} requires a Linux host")
    endif()
endif()
if(IZANAGI_WSI STREQUAL "ANDROID" AND NOT ANDROID)
    message(FATAL_ERROR "IZANAGI_WSI=ANDROID requires an Android target")
endif()
if(IZANAGI_WSI STREQUAL "METAL" AND NOT APPLE)
    message(FATAL_ERROR "IZANAGI_WSI=METAL requires an Apple target")
endif()
if(IZANAGI_BACKEND STREQUAL "METAL" AND NOT APPLE)
    message(FATAL_ERROR "IZANAGI_BACKEND=METAL requires an Apple host")
endif()

message(STATUS "Izanagi backend: ${IZANAGI_BACKEND} (profile ${IZANAGI_PROFILE})")
message(STATUS "Izanagi WSI:     ${IZANAGI_WSI}")

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
    src/common/dispatch_capabilities.cpp
    src/common/profile_report.cpp
    src/vk/volk_usage.cpp
    src/vk/vma_usage.cpp
    src/vk/gpu_to_vk.cpp
    src/vk/device.cpp
    src/vk/profile.cpp
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
target_compile_features(izanagi PUBLIC cxx_std_20)
# volk_headers / VMA are compile-time-only deps of the .cpp files; their code
# is baked into the static lib and the public header (gpu.h) is Vulkan-free,
# so the installed export must not reference them. $<BUILD_INTERFACE:...>
# keeps them for in-tree builds and drops them from the install export.
target_link_libraries(izanagi
    PRIVATE $<BUILD_INTERFACE:volk_headers>
    PRIVATE $<BUILD_INTERFACE:GPUOpen::VulkanMemoryAllocator>
)

# WSI-conditional platform definitions: VK_USE_PLATFORM_* applies only to the
# selected WSI's source set. Windows.h hygiene macros stay Windows-wide.
if(WIN32)
    target_compile_definitions(izanagi PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
endif()
target_compile_definitions(izanagi PRIVATE
    VK_NO_PROTOTYPES
    IZ_WSI_${IZANAGI_WSI}
    IZ_PROFILE="${IZANAGI_PROFILE}"
    IZ_VK_PROFILE_${IZANAGI_VK_PROFILE}
)
if(IZANAGI_WSI STREQUAL "WIN32")
    target_compile_definitions(izanagi PRIVATE VK_USE_PLATFORM_WIN32_KHR)
endif()
if(IZANAGI_WSI STREQUAL "ANDROID")
    target_compile_definitions(izanagi PRIVATE VK_USE_PLATFORM_ANDROID_KHR)
endif()
if(IZANAGI_WSI STREQUAL "XCB")
    target_compile_definitions(izanagi PRIVATE VK_USE_PLATFORM_XCB_KHR)
endif()
if(IZANAGI_WSI STREQUAL "WAYLAND")
    target_compile_definitions(izanagi PRIVATE VK_USE_PLATFORM_WAYLAND_KHR)
endif()
if(MSVC)
    target_compile_options(izanagi PRIVATE /W4 /permissive- /Zc:preprocessor /wd4201 /wd4324)
endif()

# One stable consumer target for the selected implementation.
add_library(Izanagi::Izanagi ALIAS izanagi)

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
include(cmake/IzanagiCompileShader.cmake)
if(IZANAGI_BUILD_TESTS OR IZANAGI_BUILD_EXAMPLES)
    include(cmake/CompileSlangShader.cmake)
endif()

# Shader artifacts must land next to the executables: multi-config generators
# place binaries under bin/<CONFIG>/, single-config generators under bin/.
get_property(IZANAGI_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(IZANAGI_MULTI_CONFIG)
    set(IZANAGI_SHADER_OUTPUT_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/shaders")
else()
    set(IZANAGI_SHADER_OUTPUT_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/shaders")
endif()

# --- Tests --------------------------------------------------------------------
if(IZANAGI_BUILD_TESTS)
    add_slang_shader(izanagi_test_shaders
        SOURCE ${CMAKE_SOURCE_DIR}/tests/shaders/memcpy_kernel.slang
        OUTPUT ${IZANAGI_SHADER_OUTPUT_DIR}/memcpy_kernel.spv
    )
    add_slang_shader(izanagi_test_shaders2
        SOURCE ${CMAKE_SOURCE_DIR}/tests/shaders/spec_mul_kernel.slang
        OUTPUT ${IZANAGI_SHADER_OUTPUT_DIR}/spec_mul_kernel.spv
    )
    add_slang_shader(izanagi_test_shaders3
        SOURCE ${CMAKE_SOURCE_DIR}/tests/shaders/offscreen_triangle.slang
        OUTPUT ${IZANAGI_SHADER_OUTPUT_DIR}/offscreen_triangle.spv
    )
    add_slang_shader(izanagi_test_shaders4
        SOURCE ${CMAKE_SOURCE_DIR}/tests/shaders/dual_src.slang
        OUTPUT ${IZANAGI_SHADER_OUTPUT_DIR}/dual_src.spv
    )
    # Phase 2: typed-pointer battery + GPU ABI test compile under BOTH
    # profiles; the native-only shader compiles natively and must FAIL under
    # the bindless profile (asserted by the negative test below).
    add_slang_shader(izanagi_test_shaders5
        SOURCE ${CMAKE_SOURCE_DIR}/tests/shaders/ptr_types.slang
        OUTPUT ${IZANAGI_SHADER_OUTPUT_DIR}/ptr_types.spv
    )
    add_slang_shader(izanagi_test_shaders6
        SOURCE ${CMAKE_SOURCE_DIR}/tests/shaders/abi_test.slang
        OUTPUT ${IZANAGI_SHADER_OUTPUT_DIR}/abi_test.spv
    )
    add_slang_shader(izanagi_test_shaders7
        SOURCE ${CMAKE_SOURCE_DIR}/tests/shaders/abi_int8_16.slang
        OUTPUT ${IZANAGI_SHADER_OUTPUT_DIR}/abi_int8_16.spv
    )
    add_slang_shader(izanagi_test_shaders8
        SOURCE ${CMAKE_SOURCE_DIR}/tests/shaders/null_sample.slang
        OUTPUT ${IZANAGI_SHADER_OUTPUT_DIR}/null_sample.spv
    )
    add_slang_shader(izanagi_test_shaders_untyped NATIVE_ONLY
        SOURCE ${CMAKE_SOURCE_DIR}/tests/shaders/untyped_negative.slang
        OUTPUT ${IZANAGI_SHADER_OUTPUT_DIR}/untyped_negative.spv
    )
    add_slang_negative_test(izanagi_test_shaders_untyped_neg
        SOURCE ${CMAKE_SOURCE_DIR}/tests/shaders/untyped_negative.slang
        OUTPUT ${IZANAGI_SHADER_OUTPUT_DIR}/untyped_negative.negstamp
    )

    add_executable(izanagi_tests tests/api_tests.cpp)
    target_link_libraries(izanagi_tests PRIVATE izanagi volk_headers GPUOpen::VulkanMemoryAllocator)
    target_include_directories(izanagi_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_compile_definitions(izanagi_tests PRIVATE
        IZ_VK_PROFILE_${IZANAGI_VK_PROFILE}
        IZ_NATIVE_ARTIFACT_TAG="${IZANAGI_VK_NATIVE_TAG}"
        IZ_BINDLESS_ARTIFACT_TAG="${IZANAGI_VK_BINDLESS_TAG}"
        IZ_PROFILE_VERSION_NATIVE="${IZANAGI_VK_PROFILE_VERSION_NATIVE}"
        IZ_PROFILE_VERSION_BINDLESS="${IZANAGI_VK_PROFILE_VERSION_BINDLESS}")
    add_dependencies(izanagi_tests izanagi_test_shaders izanagi_test_shaders2 izanagi_test_shaders3
                     izanagi_test_shaders4 izanagi_test_shaders5 izanagi_test_shaders6
                     izanagi_test_shaders7 izanagi_test_shaders8
                     izanagi_test_shaders_untyped izanagi_test_shaders_untyped_neg)
    if(ANDROID AND IZANAGI_ADRENOTOOLS_ROOT)
        add_subdirectory(${IZANAGI_ADRENOTOOLS_ROOT} adrenotools EXCLUDE_FROM_ALL)
        target_link_libraries(izanagi_tests PRIVATE adrenotools)
        target_compile_definitions(izanagi_tests PRIVATE IZANAGI_ANDROID_ADRENOTOOLS)
    endif()

    # GPU-independent container/arena/profile tests (no Vulkan device required)
    add_executable(izanagi_common_tests
        tests/common_tests.cpp
        src/common/containers.cpp
        src/common/platform_utils.cpp
        src/common/dispatch_capabilities.cpp
        src/common/profile_report.cpp
    )
    target_include_directories(izanagi_common_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_compile_definitions(izanagi_common_tests PRIVATE
        IZ_NATIVE_ARTIFACT_TAG="${IZANAGI_VK_NATIVE_TAG}"
        IZ_BINDLESS_ARTIFACT_TAG="${IZANAGI_VK_BINDLESS_TAG}"
        IZ_PROFILE="${IZANAGI_PROFILE}"
        IZ_PROFILE_VERSION_NATIVE="${IZANAGI_VK_PROFILE_VERSION_NATIVE}"
        IZ_PROFILE_VERSION_BINDLESS="${IZANAGI_VK_PROFILE_VERSION_BINDLESS}")
    # The extracted shader-layout manifest test (common_tests) reads the
    # compiled artifacts, so the common tests depend on the ABI shaders too.
    add_dependencies(izanagi_common_tests
                     izanagi_test_shaders5 izanagi_test_shaders6 izanagi_test_shaders7)

    enable_testing()
    add_test(NAME api_tests COMMAND izanagi_tests)
    add_test(NAME common_tests COMMAND izanagi_common_tests)
endif()

# --- Capability report tool ----------------------------------------------------
if(IZANAGI_BUILD_CAPABILITY_REPORT)
    add_executable(izanagi_capability_report tools/capability_report.cpp)
    target_link_libraries(izanagi_capability_report PRIVATE izanagi volk_headers GPUOpen::VulkanMemoryAllocator)
    target_include_directories(izanagi_capability_report PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_compile_definitions(izanagi_capability_report PRIVATE IZ_PROFILE="${IZANAGI_PROFILE}")
    if(ANDROID AND IZANAGI_ADRENOTOOLS_ROOT)
        if(NOT TARGET adrenotools)
            add_subdirectory(${IZANAGI_ADRENOTOOLS_ROOT} adrenotools EXCLUDE_FROM_ALL)
        endif()
        target_link_libraries(izanagi_capability_report PRIVATE adrenotools)
        target_compile_definitions(izanagi_capability_report PRIVATE IZANAGI_ANDROID_ADRENOTOOLS)
    endif()
endif()

# --- Examples ------------------------------------------------------------------
# The example host currently exists for the WIN32 WSI only; other WSI hosts
# land with their platform phases.
if(IZANAGI_BUILD_EXAMPLES AND NOT IZANAGI_WSI STREQUAL "WIN32")
    message(STATUS "Examples: skipped (no ${IZANAGI_WSI} example host yet)")
endif()
if(IZANAGI_BUILD_EXAMPLES AND IZANAGI_WSI STREQUAL "WIN32")
    add_slang_shader(izanagi_example_shaders
        SOURCE ${CMAKE_SOURCE_DIR}/examples/shaders/hello_triangle.slang
        OUTPUT ${IZANAGI_SHADER_OUTPUT_DIR}/hello_triangle.spv
    )
    add_slang_shader(izanagi_example_shaders2
        SOURCE ${CMAKE_SOURCE_DIR}/examples/shaders/compute_texture.slang
        OUTPUT ${IZANAGI_SHADER_OUTPUT_DIR}/compute_texture.spv
    )
    add_slang_shader(izanagi_example_shaders3
        SOURCE ${CMAKE_SOURCE_DIR}/examples/shaders/textured_cube.slang
        OUTPUT ${IZANAGI_SHADER_OUTPUT_DIR}/textured_cube.spv
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
    target_compile_definitions(izanagi_examples PRIVATE
        IZ_VK_PROFILE_${IZANAGI_VK_PROFILE}
        IZ_NATIVE_ARTIFACT_TAG="${IZANAGI_VK_NATIVE_TAG}")
    add_dependencies(izanagi_examples izanagi_example_shaders izanagi_example_shaders2 izanagi_example_shaders3)
endif()
