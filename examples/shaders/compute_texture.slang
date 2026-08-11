// compute_texture.slang — compute writes an animated pattern to a storage texture,
// then a fullscreen triangle samples it.
import izanagi;

// --- Compute entry: write animated pattern ---
struct ComputeData {
    uint64_t texture;   // RWTexture2D handle (storage heap index)
    float    time;
    uint32_t width;
    uint32_t height;
};

struct ComputeArgs {
    ComputeData* data;
};

[[vk::push_constant]] uniform ComputeArgs computeArgs;

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint3 dtid : SV_DispatchThreadID)
{
    if (dtid.x >= computeArgs.data.width || dtid.y >= computeArgs.data.height) return;
    RWTexture2D tex = getRWTexture2D(computeArgs.data.texture);
    float2 uv = float2(dtid.xy) / float2(computeArgs.data.width, computeArgs.data.height);
    float t = computeArgs.data.time;
    float r = 0.5 + 0.5 * sin(uv.x * 6.28 + t);
    float g = 0.5 + 0.5 * sin(uv.y * 6.28 + t * 1.3);
    float b = 0.5 + 0.5 * sin((uv.x + uv.y) * 6.28 + t * 0.7);
    tex.Store(int2(dtid.xy), float4(r, g, b, 1.0));
}

// --- Graphics entries: fullscreen triangle sampling the texture ---
struct FragmentData {
    uint64_t texture;   // Texture2D handle (sampled heap index)
    uint64_t sampler;   // SamplerState handle (sampler heap index)
};

struct GraphicsArgs {
    // Typed root pointer (never dereferenced by this shader). The untyped
    // `void*` form was the only Native-only declaration in the corpus; typed
    // physical pointers are ABI-identical (64-bit device address) and work on
    // both profiles.
    uint8_t* vert;
    FragmentData* frag;
};

[[vk::push_constant]] uniform GraphicsArgs graphicsArgs;

struct VertexOutput {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

[shader("vertex")]
VertexOutput vertex_main(uint32_t vertexIdx : SV_VertexID)
{
    // Fullscreen triangle: covers the entire screen with 3 vertices
    VertexOutput output;
    output.uv = float2((vertexIdx << 1) & 2, vertexIdx & 2);
    output.position = float4(output.uv * 2.0 - 1.0, 0.0, 1.0);
    output.uv.y = 1.0 - output.uv.y; // Flip for texture coordinates
    return output;
}

[shader("fragment")]
float4 fragment_main(VertexOutput input) : SV_Target
{
    Texture2D tex = getTexture2D(graphicsArgs.frag.texture);
    SamplerState samp = getSampler(graphicsArgs.frag.sampler);
    return tex.Sample(samp, input.uv);
}
