// offscreen_triangle.slang — solid red triangle for headless draw tests.
struct VertexStageOutput {
    float4 position : SV_Position;
};

[shader("vertex")]
VertexStageOutput vertex_main(uint32_t vertexIdx : SV_VertexID)
{
    const float2 verts[3] = {float2(0, 0.5), float2(0.5, -0.5), float2(-0.5, -0.5)};
    const float2 uv = verts[vertexIdx % 3];
    const float4 p = float4(uv, 0.5, 1);
    VertexStageOutput output = {p};
    return output;
}

[shader("fragment")]
float4 fragment_main() : SV_Target
{
    return float4(1, 0, 0, 1);
}
