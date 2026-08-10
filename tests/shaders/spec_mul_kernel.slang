// spec_mul_kernel.slang - dst[i] = src[i] * kMul (kMul is spec constant id 0)
[[vk::constant_id(0)]] const uint kMul = 1;

struct CopyData {
    uint32_t* dst;
    uint32_t* src;
    uint32_t  count;
};
struct Args { CopyData* data; };
[[vk::push_constant]] uniform Args args;

[shader("compute")]
[numthreads(64, 1, 1)]
void compute_main(uint3 dtid : SV_DispatchThreadID)
{
    uint32_t i = dtid.x;
    if (i < args.data.count) {
        args.data.dst[i] = args.data.src[i] * kMul;
    }
}
