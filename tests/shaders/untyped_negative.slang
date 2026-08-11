// untyped_negative.slang — NATIVE-ONLY shader (see add_slang_shader
// NATIVE_ONLY): uses the descriptor-heap handle model, which exists only in
// the native prelude. The bindless prelude deliberately does NOT define
// unpack_heap_handle / Texture2D.Handle — so this file must FAIL to compile
// under the Bindless profile (asserted by add_slang_negative_test) while
// compiling fine natively. This is the "fail on accidental use of Native-only
// intrinsics in Bindless shaders" guarantee.
import izanagi;

[shader("compute")]
[numthreads(1, 1, 1)]
void main_cs()
{
    uint2 h = unpack_heap_handle(0);   // descriptor-heap handle model: native-only
}
