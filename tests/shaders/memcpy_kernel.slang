// memcpy_kernel.slang — compute test shader: dst[i] = src[i] * 2 + 1
import izanagi;

struct CopyData {
    uint32_t* dst;
    uint32_t* src;
    uint32_t  count;
};

struct Args {
    CopyData* data;
};

[[vk::push_constant]] uniform Args args;

[shader("compute")]
[numthreads(64, 1, 1)]
void compute_main(uint3 dtid : SV_DispatchThreadID)
{
    uint32_t i = dtid.x;
    if (i < args.data.count) {
        args.data.dst[i] = args.data.src[i] * 2 + 1;
    }
}
