// dual_src.slang — deterministic dual-source blend test shader.
//
// Fullscreen triangle emitting two fragment outputs:
//   source0: rgb = (0.25, 0.5, 0.75), a = 0.5   -> stored framebuffer alpha
//   source1: a = 1.0                             -> RGB blend factor (Src1Alpha)
// A single-source implementation (SrcAlpha) would blend RGB with 0.5 and
// fail the test's RGB expectation, so this can only pass via dual-source.

struct VsOut {
    float4 pos : SV_Position;
};

[shader("vertex")]
VsOut vertex_main(uint32_t vertexIdx : SV_VertexID)
{
    const float2 verts[3] = {float2(-1, -1), float2(3, -1), float2(-1, 3)};
    VsOut o;
    o.pos = float4(verts[vertexIdx % 3], 0.0, 1.0);
    return o;
}

struct PsOut {
    [[vk::location(0)]] float4 color  : SV_Target0;
    [[vk::location(0), vk::index(1)]] float4 factor;
};

[shader("fragment")]
PsOut fragment_main()
{
    PsOut o;
    o.color = float4(0.25, 0.5, 0.75, 0.5);   // rgb + stored alpha
    o.factor = float4(0.0, 0.0, 0.0, 1.0);    // RGB blend factor
    return o;
}
