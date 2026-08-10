//------------------------------------------------------------------------------
// This shader demonstrates key Slang features and syntax while implementing
// a textured triangle renderer with dynamic point overlay effects
//------------------------------------------------------------------------------

#include "shader_io.h"

//------------------------------------------------------------------------------
// Input Structures
//------------------------------------------------------------------------------

// Define the vertex attributes that will be provided by the vertex buffer
// Note: In Slang, we use semantic annotations (: POSITION, etc.) to specify
// how these fields map to vertex buffer data
struct VertexInput
{
    [[vk::location(LVPosition)]] // Vulkan layout(location = 0)
    float3 position : POSITION;  // Vertex position in model space
    [[vk::location(LVColor)]]    // Vulkan layout(location = 1)
    float3 color : COLOR;        // Vertex color
    [[vk::location(LVTexCoord)]] // Vulkan layout(location = 2)
    float2 uv : TEXCOORD0;       // Texture coordinates
};

// Define the data that will be interpolated and passed from vertex to fragment shader
// Note: SV_Position is a special semantic in Slang/HLSL for the clip-space position
struct VertexOutput
{
    float4 position : SV_Position; // Clip space position (required)
    float3 color : TEXCOORD0;      // Interpolated vertex color
    float2 uv : TEXCOORD1;         // Interpolated texture coordinates
};

// Define the final output of the fragment shader
// Note: SV_TARGET specifies this is the color output to the render target
struct PixelOutput
{
    float4 color : SV_TARGET; // Final color output (RGBA)
};

//------------------------------------------------------------------------------
// Resource Bindings
//------------------------------------------------------------------------------

// Push data (vkCmdPushDataEXT): carries only the scene buffer address and per-draw color.
// With descriptor heap (VK_EXT_descriptor_heap), the pipeline layout is VK_NULL_HANDLE,
// so traditional push constants and push descriptors (UBOs) cannot be used.
[[vk::push_constant]]
ConstantBuffer<GraphicsPushData> pushData;

// Descriptor heap: textures and samplers live in GPU heap buffers (VK_EXT_descriptor_heap).
// We access them by heap index using Texture2D.Handle(uint2(index, 0)) and
// SamplerState.Handle(uint2(index, 0)). This replaces the old [[vk::binding(...)]] Sampler2D array.

// Specialization constant - can be set at pipeline creation time
// Note: This is equivalent to GLSL's layout(constant_id = X)
[[vk::constant_id(0)]]
const bool useTexture = false; // Controls whether texture sampling is enabled

//------------------------------------------------------------------------------
// Vertex Shader
//------------------------------------------------------------------------------

// Note: [shader("vertex")] is Slang's way of marking the vertex shader entry point
[shader("vertex")]
VertexOutput vertexMain(VertexInput input)
{
    // Access the scene info buffer via buffer device address (updated once per frame)
    // Cast the uint64 address to a typed pointer -- Slang compiles this to SPIR-V untyped pointer access
    SceneInfo* scene = (SceneInfo*)pushData.sceneInfoAddress;

    // Start with the input position
    vec3 pos = input.position;

    // Correct for screen aspect ratio to prevent stretching
    float aspectRatio = scene.resolution.y / scene.resolution.x;
    pos.x *= aspectRatio;

    VertexOutput output;
    output.position = float4(pos, 1.0); // Convert to homogeneous coordinates
    output.color    = input.color;      // Pass through the vertex color
    output.uv       = input.uv;         // Pass through the texture coordinates

    return output;
}

//------------------------------------------------------------------------------
// Fragment/Pixel Shader
//------------------------------------------------------------------------------

// Note: [shader("pixel")] is Slang's way of marking the fragment shader entry point
[shader("pixel")]
PixelOutput fragmentMain(VertexOutput input)
{
    PixelOutput output;

    // Access the scene info buffer via buffer device address (same buffer, updated once per frame)
    SceneInfo* scene = (SceneInfo*)pushData.sceneInfoAddress;

    // Convert fragment position to normalized device coordinates (-1 to 1 range)
    float2 fragPos = (input.position.xy / scene.resolution) * 2.0 - 1.0;

    // Access the data buffer using buffer device address (stored in SceneInfo)
    vec2* datas = (vec2*)scene.dataBufferAddress;

    // Find the minimum distance to any point in our data buffer
    float minDist = 1e10; // Initialize to a large value
    for (int i = 0; i < scene.numData; i++)
    {
        vec2  pnt  = datas[i];
        float dist = distance(fragPos, pnt);
        minDist    = min(minDist, dist);
    }

    // Create smooth circular points using distance field
    float radius     = 0.02; // Size of the points
    float edgeSmooth = 0.01; // Amount of anti-aliasing smoothing
    // Create smooth transition from 0 to 1 based on distance
    float alpha = 1.0 - smoothstep(radius, radius - edgeSmooth, minDist);

    // Calculate colors for blending
    // Point color animated using scene data + per-draw color from push data
    float4 pointColor = float4(scene.animValue * pushData.color, 1.0);
    // Triangle color from interpolated vertex color
    float4 triangleColor = float4(input.color, 1.0);

    // Conditionally sample texture using descriptor heap handles.
    // Texture2D.Handle(uint2(index, 0)) creates a handle from the resource heap at the given index.
    // SamplerState.Handle(uint2(0, 0)) gets the linear sampler at index 0 in the sampler heap.
    // This is the Slang equivalent of GLSL's layout(descriptor_heap) access.
    if (useTexture)
    {
        Texture2D    tex  = Texture2D.Handle(uint2(scene.texId, 0));
        SamplerState samp = SamplerState.Handle(uint2(0, 0));
        triangleColor *= tex.Sample(samp, input.uv);
    }

    // Blend between point and triangle color based on distance field
    output.color = lerp(pointColor, triangleColor, alpha);

    return output;
}
