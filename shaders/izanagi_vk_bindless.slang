// izanagi_vk_bindless.slang — compatibility (descriptor-indexing) prelude.
// Compiled only for the IZANAGI_VK_BINDLESS_1 profile (SPIR-V 1.5, no
// descriptor-heap capability). Included textually by the dispatcher
// izanagi.slang; not a module itself.
//
// The global resource namespace is implemented with one backend-private
// descriptor set holding three runtime-sized descriptor arrays (bindings 0,
// 1, 2 in set 0). The public handle encoding is UNCHANGED: low 32 bits are
// the array index, upper bits are CPU-side generation/type metadata — so the
// same public data structures and the same handle values work on both
// profiles, and no per-draw CPU resource gathering ever happens.
//
// Root arguments are ordinary push constants (a backend-private pipeline
// layout maps set 0 + push constants); the shader-side Args structs and the
// 8/16-byte pointer payloads are identical to the native profile.
//
// NON-UNIFORM INDEXING: the public model requires lane-divergent handle
// selection, so every array lookup wraps its index in
// NonUniformResourceIndex. This makes Slang emit the SPIR-V NonUniform
// decoration (5300) + ShaderNonUniform capability on the descriptor-array
// accesses (verified with slangc 2026.5.2); WITHOUT the wrap no decoration is
// emitted. Keep the wraps on every lookup.

[[vk::binding(0, 0)]] Texture2D     g_sampled_images[];
[[vk::binding(1, 0)]] RWTexture2D   g_storage_images[];
[[vk::binding(2, 0)]] SamplerState  g_samplers[];

// Unpack the array index from a public handle (low 32 bits).
public uint32_t unpack_heap_index(uint64_t v)
{
    return uint32_t(v);
}

// Typed accessor for a sampled texture: direct index into the global array.
public Texture2D getTexture2D(uint64_t view)
{
    return g_sampled_images[NonUniformResourceIndex(unpack_heap_index(view))];
}

// Typed accessor for a storage texture.
public RWTexture2D getRWTexture2D(uint64_t view)
{
    return g_storage_images[NonUniformResourceIndex(unpack_heap_index(view))];
}

// Typed accessor for a sampler.
public SamplerState getSampler(uint64_t samplerId)
{
    return g_samplers[NonUniformResourceIndex(unpack_heap_index(samplerId))];
}

// Root-argument conventions (identical across profiles): see
// izanagi_vk_native.slang — the Args structs and pointer payloads are the
// same; the backend delivers them with ordinary push constants.
