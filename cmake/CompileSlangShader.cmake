# add_slang_shader(TARGET <t> SOURCE <f.slang> OUTPUT <f.spv>)
#
# Compiles a .slang file to SPIR-V at build time using slangc.
# One .spv per .slang containing ALL [shader(...)] entry points;
# ShaderSource.entry_point selects at pipeline creation.
#
# Requires SLANGC_EXECUTABLE to be set (found in third_party/CMakeLists.txt).

function(add_slang_shader TARGET_NAME)
    cmake_parse_arguments(ARG "" "SOURCE;OUTPUT" "" ${ARGN})

    if(NOT ARG_SOURCE)
        message(FATAL_ERROR "add_slang_shader: SOURCE is required")
    endif()
    if(NOT ARG_OUTPUT)
        message(FATAL_ERROR "add_slang_shader: OUTPUT is required")
    endif()

    # Debug/Release flag swap — use plain variables, not generator expressions
    # (VS generator doesn't evaluate $<CONFIG> inside add_custom_command reliably)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR "${CMAKE_CONFIGURATION_TYPES}" MATCHES "Debug")
        set(_opt_flags "-O0" "-g2")
    else()
        set(_opt_flags "-O2" "-g1")
    endif()

    get_filename_component(_out_dir "${ARG_OUTPUT}" DIRECTORY)

    add_custom_command(
        OUTPUT "${ARG_OUTPUT}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${_out_dir}"
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
            -I "${CMAKE_SOURCE_DIR}/shaders"
            -depfile "${ARG_OUTPUT}.dep"
            ${_opt_flags}
            -o "${ARG_OUTPUT}"
        DEPENDS "${ARG_SOURCE}" "${CMAKE_SOURCE_DIR}/shaders/izanagi.slang"
        DEPFILE "${ARG_OUTPUT}.dep"
        COMMENT "Slang: ${ARG_SOURCE} -> ${ARG_OUTPUT}"
        VERBATIM
    )

    add_custom_target(${TARGET_NAME} DEPENDS "${ARG_OUTPUT}")
endfunction()
