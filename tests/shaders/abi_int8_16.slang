// abi_int8_16.slang — capability-gated 8/16-bit storage ABI test.
//
// The MANDATORY cross-profile ABI test (abi_test.slang) deliberately avoids
// 8/16-bit storage because shaderInt8/shaderInt16 are optional device
// capabilities. This test covers them and only runs when the device exposes
// both features (reported by the capability profile as optional conveniences,
// per the compatibility profile contract).
//
// C++ layout contract (scalar layout):
//   Int8_16Root: uint16 s16 @0, uint8 b8[3] @2, uint32 tail @8 -> 12

struct Int8_16Root {
    uint16_t s16;
    uint8_t  b8[3];
    uint32_t tail;
};

struct Int8_16ArgData {
    Int8_16Root* root;
    uint32_t*    out;    // 5 words
};

struct Int8_16Args {
    Int8_16ArgData* data;   // the ONE pointer in the push constant
};

[[vk::push_constant]] uniform Int8_16Args args;

[shader("compute")]
[numthreads(1, 1, 1)]
void main_cs()
{
    Int8_16Root* r = args.data.root;
    args.data.out[0] = r.s16;
    args.data.out[1] = r.b8[0];
    args.data.out[2] = r.b8[1];
    args.data.out[3] = r.b8[2];
    args.data.out[4] = r.tail;
}
