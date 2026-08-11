// abi_test.slang — cross-profile GPU ABI test.
//
// Mirrors the C++ AbiRoot (tests/api_tests.cpp) EXACTLY under scalar block
// layout + row-major matrices (the same flags every Izanagi shader is
// compiled with). The shader reads every field of an irregular nested
// structure — scalars of different widths, padding-sensitive fields, arrays,
// nested structs, a GpuPtr member, TextureView/SamplerId members, and a
// second-level pointer — and writes the interpreted values to a readback
// buffer. The api test compares every value with the CPU expectations; any
// layout divergence between C++ and Slang lands a wrong value and fails.
//
// Compute root ABI: the push constant carries exactly ONE pointer to a
// GPU-resident argument struct.
//
// C++ layout contract (scalar layout):
//   AbiInner : uint32 a @0, float b @4, uint32 c @8        -> 12
//   AbiNested: AbiInner inner @0, uint32_t* ptr @16        -> 24
//   AbiRoot  : u @0, f @4, gpu_ptr @8, tex @16, samp @24,
//              nested @32, arr[3] @56, s @68, bytes[3] @70 -> 80

struct AbiInner {
    uint32_t a;
    float    b;
    uint32_t c;
};

struct AbiNested {
    AbiInner  inner;
    uint32_t* ptr;  // second-level pointer (typed; points at a scratch value)
};

struct AbiRoot {
    uint32_t  u;
    float     f;
    uint64_t  gpu_ptr;   // GpuPtr member (a real device address)
    uint64_t  tex;       // TextureView handle
    uint64_t  samp;      // SamplerId handle
    AbiNested nested;
    float     arr[3];
    uint16_t  s;
    uint8_t   bytes[3];
};

struct AbiArgData {
    AbiRoot*  root;
    uint32_t* out;      // 19 words
};

struct AbiArgs {
    AbiArgData* data;   // the ONE pointer in the push constant
};

[[vk::push_constant]] uniform AbiArgs args;

[shader("compute")]
[numthreads(1, 1, 1)]
void main_cs()
{
    AbiRoot* r = args.data.root;
    uint32_t o[19];
    o[0]  = r.u;
    o[1]  = asuint(r.f);
    o[2]  = uint32_t(r.gpu_ptr);
    o[3]  = uint32_t(r.gpu_ptr >> 32);
    o[4]  = uint32_t(r.tex);
    o[5]  = uint32_t(r.tex >> 32);
    o[6]  = uint32_t(r.samp);
    o[7]  = uint32_t(r.samp >> 32);
    o[8]  = r.nested.inner.a;
    o[9]  = asuint(r.nested.inner.b);
    o[10] = r.nested.inner.c;
    o[11] = *r.nested.ptr;          // second-level pointer dereference
    o[12] = asuint(r.arr[0]);
    o[13] = asuint(r.arr[1]);
    o[14] = asuint(r.arr[2]);
    o[15] = r.s;
    o[16] = r.bytes[0];
    o[17] = r.bytes[1];
    o[18] = r.bytes[2];
    for (int i = 0; i < 19; ++i) { args.data.out[i] = o[i]; }
}
