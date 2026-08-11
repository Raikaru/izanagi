# add_slang_shader(TARGET <t> SOURCE <f.slang> OUTPUT <f.spv> [NATIVE_ONLY])
#
# Compiles a .slang file to SPIR-V at build time using slangc, for BOTH
# capability profiles: vk_native_spv16 (SPIR-V 1.6 + spvDescriptorHeapEXT)
# and vk_bindless_spv15 (SPIR-V 1.5, descriptor-indexing arrays, no
# descriptor-heap capability). One .spv per profile containing ALL
# [shader(...)] entry points; ShaderSource.entry_point selects at pipeline
# creation. Artifact identity = source name + profile/SPIR-V version
# directory (+ entry points + profile version, verified by the extracted
# manifest test), so Native and Bindless artifacts never collide.
#
# NATIVE_ONLY marks a shader that is legitimately Native-profile-only (e.g.
# the negative compile test): the Bindless variant is not built for it.
#
# Requires SLANGC_EXECUTABLE to be set (found in third_party/CMakeLists.txt).

# Directory of THIS file (function bodies see the CALLER's list dir, so the
# helper scripts must be referenced through this captured path).
set(IZ_COMPILE_SLANG_DIR "${CMAKE_CURRENT_LIST_DIR}")

function(add_slang_shader TARGET_NAME)
    cmake_parse_arguments(ARG "NATIVE_ONLY" "SOURCE;OUTPUT" "" ${ARGN})

    if(NOT ARG_SOURCE)
        message(FATAL_ERROR "add_slang_shader: SOURCE is required")
    endif()
    if(NOT ARG_OUTPUT)
        message(FATAL_ERROR "add_slang_shader: OUTPUT is required")
    endif()

    set(_base_output "${ARG_OUTPUT}")
    # Artifact identity scheme (ABI requirements): the artifact path encodes
    #   <source>.<ext>               -> source identity (the file name)
    #   vk_native_spv16|vk_bindless_spv15 -> backend profile + target SPIR-V
    #                                        version
    # and IZ_PROFILE (compile-time) identifies the profile version. The
    # extracted-manifest test verifies the SPIR-V header version matches the
    # directory and enumerates the entry points from the artifact. Native and
    # Bindless artifacts can therefore never collide, even across profile or
    # SPIR-V version changes.
    string(REGEX REPLACE "^(.*/shaders/)([^/]+)$" "\\1vk_native_spv16/\\2" _native "${_base_output}")
    string(REGEX REPLACE "^(.*/shaders/)([^/]+)$" "\\1vk_bindless_spv15/\\2" _bindless "${_base_output}")
    if(NOT _native MATCHES "vk_native_spv16/" OR NOT _bindless MATCHES "vk_bindless_spv15/")
        message(FATAL_ERROR "add_slang_shader: OUTPUT must be under a /shaders/ directory")
    endif()

    # Debug/Release flag swap — use plain variables, not generator expressions
    # (VS generator doesn't evaluate $<CONFIG> inside add_custom_command reliably)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR "${CMAKE_CONFIGURATION_TYPES}" MATCHES "Debug")
        set(_opt_flags "-O0" "-g2")
    else()
        set(_opt_flags "-O2" "-g1")
    endif()

    get_filename_component(_native_dirname "${_native}" DIRECTORY)
    get_filename_component(_bindless_dirname "${_bindless}" DIRECTORY)

    # --- Native profile artifact (SPIR-V 1.6 + descriptor-heap capability) ---
    add_custom_command(
        OUTPUT "${_native}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${_native_dirname}"
        COMMAND "${SLANGC_EXECUTABLE}"
            "${ARG_SOURCE}"
            -lang slang
            -target spirv
            -profile spirv_1_6
            -emit-spirv-directly
            -force-glsl-scalar-layout
            -matrix-layout-row-major
            -fvk-use-entrypoint-name
            -capability spvDescriptorHeapEXT
            -D IZ_VK_PROFILE_NATIVE
            -I "${CMAKE_SOURCE_DIR}/shaders"
            -depfile "${_native}.dep"
            ${_opt_flags}
            -o "${_native}"
        DEPENDS "${ARG_SOURCE}"
                "${CMAKE_SOURCE_DIR}/shaders/izanagi.slang"
                "${CMAKE_SOURCE_DIR}/shaders/izanagi_vk_native.slang"
        DEPFILE "${_native}.dep"
        COMMENT "Slang(native): ${ARG_SOURCE}"
        VERBATIM
    )

    set(_outputs "${_native}")

    # --- Bindless profile artifact (SPIR-V 1.5, no descriptor-heap capability;
    #     skipped for NATIVE_ONLY shaders) ---
    if(NOT ARG_NATIVE_ONLY)
        add_custom_command(
            OUTPUT "${_bindless}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${_bindless_dirname}"
            COMMAND "${SLANGC_EXECUTABLE}"
                "${ARG_SOURCE}"
                -lang slang
                -target spirv
                -profile spirv_1_5
                -emit-spirv-directly
                -force-glsl-scalar-layout
                -matrix-layout-row-major
                -fvk-use-entrypoint-name
                -D IZ_VK_PROFILE_BINDLESS
                -I "${CMAKE_SOURCE_DIR}/shaders"
                -depfile "${_bindless}.dep"
                ${_opt_flags}
                -o "${_bindless}"
            DEPENDS "${ARG_SOURCE}"
                    "${CMAKE_SOURCE_DIR}/shaders/izanagi.slang"
                    "${CMAKE_SOURCE_DIR}/shaders/izanagi_vk_bindless.slang"
            DEPFILE "${_bindless}.dep"
            COMMENT "Slang(bindless): ${ARG_SOURCE}"
            VERBATIM
        )
        list(APPEND _outputs "${_bindless}")
    endif()

    add_custom_target(${TARGET_NAME} DEPENDS ${_outputs})
endfunction()

# add_slang_negative_test(TARGET <t> SOURCE <f.slang>)
#
# Proves a Native-only shader FAILS to compile under the Bindless profile:
# runs slangc with the bindless flags on the source and asserts a non-zero
# exit. The same source is compiled natively via add_slang_shader(NATIVE_ONLY)
# to prove the operation is native-valid. Failure of this test (bindless
# compile succeeds) means a Native-only operation silently leaked into the
# compatibility profile.

function(add_slang_negative_test TARGET_NAME)
    cmake_parse_arguments(ARG "" "SOURCE;OUTPUT" "" ${ARGN})
    if(NOT ARG_SOURCE)
        message(FATAL_ERROR "add_slang_negative_test: SOURCE is required")
    endif()
    if(NOT ARG_OUTPUT)
        message(FATAL_ERROR "add_slang_negative_test: OUTPUT is required")
    endif()

    if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR "${CMAKE_CONFIGURATION_TYPES}" MATCHES "Debug")
        set(_opt_flags "-O0" "-g2")
    else()
        set(_opt_flags "-O2" "-g1")
    endif()

    add_custom_command(
        OUTPUT "${ARG_OUTPUT}"
        COMMAND ${CMAKE_COMMAND}
            "-DSLANGC_EXECUTABLE=${SLANGC_EXECUTABLE}"
            "-DSOURCE=${ARG_SOURCE}"
            "-DOUTPUT=${ARG_OUTPUT}"
            "-DINCLUDE_DIR=${CMAKE_SOURCE_DIR}/shaders"
            -P "${IZ_COMPILE_SLANG_DIR}/ExpectSlangFailure.cmake"
        DEPENDS "${ARG_SOURCE}"
                "${CMAKE_SOURCE_DIR}/shaders/izanagi.slang"
                "${CMAKE_SOURCE_DIR}/shaders/izanagi_vk_bindless.slang"
        COMMENT "Slang-negative(bindless must fail): ${ARG_SOURCE}"
        VERBATIM
    )
    add_custom_target(${TARGET_NAME} DEPENDS "${ARG_OUTPUT}")
endfunction()
