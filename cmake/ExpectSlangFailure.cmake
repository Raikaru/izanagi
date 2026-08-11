# ExpectSlangFailure.cmake — asserts that slangc FAILS to compile SOURCE under
# the Bindless profile flags. Used by add_slang_negative_test to prove that a
# Native-only operation is rejected in the compatibility profile instead of
# silently compiling into different semantics.
#
# Required -D variables: SLANGC_EXECUTABLE, SOURCE, OUTPUT, INCLUDE_DIR.

if(NOT DEFINED SLANGC_EXECUTABLE OR NOT DEFINED SOURCE OR NOT DEFINED OUTPUT OR NOT DEFINED INCLUDE_DIR)
    message(FATAL_ERROR "ExpectSlangFailure: SLANGC_EXECUTABLE/SOURCE/OUTPUT/INCLUDE_DIR required")
endif()

execute_process(
    COMMAND "${SLANGC_EXECUTABLE}"
            "${SOURCE}"
            -lang slang
            -target spirv
            -profile spirv_1_5
            -emit-spirv-directly
            -force-glsl-scalar-layout
            -matrix-layout-row-major
            -fvk-use-entrypoint-name
            -D IZ_VK_PROFILE_BINDLESS
            -I "${INCLUDE_DIR}"
            -o "${OUTPUT}"
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _stdout
    ERROR_VARIABLE  _stderr
)

if(_result EQUAL 0)
    message(FATAL_ERROR
        "ExpectSlangFailure: bindless compilation of ${SOURCE} SUCCEEDED — a "
        "Native-only operation silently compiled into the Bindless profile. "
        "Reject it in the prelude or mark the shader NATIVE_ONLY.")
endif()

# Expected failure: touch the output so the build graph sees the test ran.
file(WRITE "${OUTPUT}" "expected failure: slangc exited ${_result}\n")
