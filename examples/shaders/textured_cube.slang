// textured_cube.slang — rotating textured cube with depth testing.
import izanagi;

struct CameraData {
    float4x4 projection;
    float4x4 cameraFromWorld;
};

struct Mesh {
    float3* position;
    float2* uvs;
    float4x4 worldFromMesh;
};

struct InputData {
    CameraData camera;
    Mesh mesh;
};

struct FragmentData {
    uint64_t texture;   // Texture2D handle
    uint64_t sampler;   // SamplerState handle
};

struct Args {
    InputData* vert;
    FragmentData* frag;
};

[[vk::push_constant]] uniform Args args;

struct VertexStageOutput {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

[shader("vertex")]
VertexStageOutput vertex_main(uint32_t vertexIdx : SV_VertexID)
{
    VertexStageOutput output;
    const Mesh mesh = args.vert->mesh;
    float4x4 mvp = mul(args.vert->camera.projection,
                       mul(args.vert->camera.cameraFromWorld, mesh.worldFromMesh));
    output.position = mul(mvp, float4(mesh.position[vertexIdx], 1.0));
    output.uv = mesh.uvs[vertexIdx];
    return output;
}

[shader("fragment")]
float4 fragment_main(VertexStageOutput vertex) : SV_Target
{
    Texture2D tex = getTexture2D(args.frag.texture);
    SamplerState samp = getSampler(args.frag.sampler);
    return tex.Sample(samp, vertex.uv);
}
