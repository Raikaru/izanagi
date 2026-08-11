// null_sample.slang — deterministic null-handle test.
//
// Samples descriptor slot 0 (the reserved null handle) through the public
// getTexture2D/getSampler helpers and writes the result to a readback buffer.
// Under the bindless profile the reserved slot is backed by a real dummy
// texture+sampler, so the result is a stable descriptor; two dispatches with
// no intervening writes must return identical values. (Native-profile only in
// the sense that the heap has no such guarantee — the api test runs this only
// under the bindless profile.)
import izanagi;

struct NSData {
    uint32_t* out;   // 4 words
};
struct NSArgs {
    NSData* data;
};
[[vk::push_constant]] uniform NSArgs args;

[shader("compute")]
[numthreads(1, 1, 1)]
void main_cs()
{
    float4 c = getTexture2D(0).SampleLevel(getSampler(0), float2(0.5, 0.5), 0.0);
    args.data.out[0] = asuint(c.x);
    args.data.out[1] = asuint(c.y);
    args.data.out[2] = asuint(c.z);
    args.data.out[3] = asuint(c.w);
}
