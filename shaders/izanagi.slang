// izanagi.slang — shared prelude for all Izanagi shaders.
// Provides root-argument convention and descriptor-heap access helpers.
//
// TextureView/SamplerId encoding (must match include/izanagi/gpu.h):
//   uint64 handle: low 32 bits = descriptor index, upper bits = CPU-side
//   generation/type metadata (masked here; the driver only sees the index).
//   Unpack with uint2(uint(v), 0) for .Handle(uint2).

// Unpack a uint64 heap handle into the uint2 form Slang's .Handle() expects.
public uint2 unpack_heap_handle(uint64_t v)
{
    return uint2(uint(v), 0);
}

// Typed accessor for a sampled texture from the resource heap.
public Texture2D getTexture2D(uint64_t view)
{
    return Texture2D.Handle(unpack_heap_handle(view));
}

// Typed accessor for a storage texture from the resource heap.
public RWTexture2D getRWTexture2D(uint64_t view)
{
    return RWTexture2D.Handle(unpack_heap_handle(view));
}

// Typed accessor for a sampler from the sampler heap.
public SamplerState getSampler(uint64_t samplerId)
{
    return SamplerState.Handle(unpack_heap_handle(samplerId));
}

// Root-argument conventions:
//   Graphics: struct Args { VertexData* vert; FragmentData* frag; }
//             [[vk::push_constant]] uniform Args args;
//             vkCmdPushDataEXT writes 16 bytes: vert at offset 0, frag at offset 8.
//   Compute:  struct Args { Data* data; }
//             [[vk::push_constant]] uniform Args args;
//             vkCmdPushDataEXT writes 8 bytes: data at offset 0.
