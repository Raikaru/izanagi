// izanagi.slang — profile dispatcher for the Izanagi shader prelude.
//
// Application shaders `import izanagi;` and use the shared helper surface
// (getTexture2D / getRWTexture2D / getSampler). The build selects which
// profile prelude implements that surface via the IZ_VK_PROFILE_NATIVE /
// IZ_VK_PROFILE_BINDLESS define passed to slangc (see
// cmake/CompileSlangShader.cmake). Only ONE profile is ever compiled into a
// given artifact — a Native-only helper simply does not exist in a Bindless
// build, so accidental use fails at shader compile time.
#if defined(IZ_VK_PROFILE_BINDLESS)
#    include "izanagi_vk_bindless.slang"
#else
#    include "izanagi_vk_native.slang"
#endif
