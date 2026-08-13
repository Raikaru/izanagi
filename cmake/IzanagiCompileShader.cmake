# izanagi_compile_shader_for_selected_profile(
#     TARGET <target> SOURCE <shader.slang> OUTPUT <shader.spv>
#     COMPILER <slangc>)
#
# Compile an application shader against the shader ABI for the capability
# profile selected by this Izanagi build. Consumers must not reproduce the
# profile-specific defines, capabilities, include paths, or dependencies.
function(izanagi_compile_shader_for_selected_profile)
    cmake_parse_arguments(ARG "" "TARGET;SOURCE;OUTPUT;COMPILER" "" ${ARGN})

    foreach(_required TARGET SOURCE OUTPUT COMPILER)
        if(NOT ARG_${_required})
            message(FATAL_ERROR
                "izanagi_compile_shader_for_selected_profile: ${_required} is required")
        endif()
    endforeach()

    if(NOT EXISTS "${ARG_SOURCE}")
        message(FATAL_ERROR
            "izanagi_compile_shader_for_selected_profile: missing source ${ARG_SOURCE}")
    endif()
    if(NOT EXISTS "${ARG_COMPILER}")
        message(FATAL_ERROR
            "izanagi_compile_shader_for_selected_profile: missing compiler ${ARG_COMPILER}")
    endif()

    set(_shader_dir "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../shaders")
    if(IZANAGI_VK_PROFILE STREQUAL "NATIVE")
        set(_profile_args
            -profile spirv_1_6
            -capability spvDescriptorHeapEXT
            -D IZ_VK_PROFILE_NATIVE)
        set(_profile_dependency "${_shader_dir}/izanagi_vk_native.slang")
    elseif(IZANAGI_VK_PROFILE STREQUAL "BINDLESS")
        set(_profile_args
            -profile spirv_1_5
            -D IZ_VK_PROFILE_BINDLESS)
        set(_profile_dependency "${_shader_dir}/izanagi_vk_bindless.slang")
    else()
        message(FATAL_ERROR
            "izanagi_compile_shader_for_selected_profile: unsupported profile ${IZANAGI_VK_PROFILE}")
    endif()

    get_filename_component(_output_dir "${ARG_OUTPUT}" DIRECTORY)
    add_custom_command(
        OUTPUT "${ARG_OUTPUT}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${_output_dir}"
        COMMAND "${ARG_COMPILER}"
            "${ARG_SOURCE}"
            -lang slang
            -target spirv
            ${_profile_args}
            -emit-spirv-directly
            -force-glsl-scalar-layout
            -matrix-layout-row-major
            -fvk-use-entrypoint-name
            -I "${_shader_dir}"
            -O2
            -o "${ARG_OUTPUT}"
        DEPENDS "${ARG_SOURCE}"
                "${_shader_dir}/izanagi.slang"
                "${_profile_dependency}"
        COMMENT "Compiling ${ARG_SOURCE} for ${IZANAGI_PROFILE}"
        VERBATIM
    )
    add_custom_target(${ARG_TARGET} DEPENDS "${ARG_OUTPUT}")
endfunction()
