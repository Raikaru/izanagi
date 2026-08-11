// ptr_types.slang — typed physical-storage-buffer pointer battery.
//
// Exercises every pointer operation the Izanagi shader corpus uses, all
// through TYPED pointers (no untyped reinterpretation): root pointer
// dereference, pointers to scalars, pointers to structs, second-level
// pointers stored inside GPU memory, arrays behind pointers, pointer
// arithmetic, pointer arrays in GPU memory, typed loads and stores.
//
// Compute root ABI: the push constant carries exactly ONE pointer to a
// GPU-resident argument struct (the same convention the examples use).
// Compiled under BOTH profiles (the Bindless profile's typed-pointer
// compatibility compile test); the api test binds it under the native
// backend to verify the semantics on hardware.

struct PtrNested {
    uint32_t  a;
    uint32_t* p;  // second-level pointer (typed; points into GPU memory)
    uint32_t  b;
};

struct PtrArray {
    uint32_t vals[4];
};

struct PtrRoot {
    uint32_t*  scalar_ptr;   // pointer to a scalar (also arithmetic base)
    PtrNested* nested_ptr;   // pointer to a struct containing a pointer
    PtrArray*  array_ptr;    // pointer to a struct with an array
    uint32_t*  arr_of_ptr[2]; // pointers stored in GPU memory
    uint32_t   direct;       // typed member load
    uint32_t   sum;          // output field (typed store)
};

// GPU-resident argument struct (pointed to by the single root pointer).
struct PtrArgData {
    PtrRoot*  root;
    uint32_t* out;           // typed store target
};

struct PtrArgs {
    PtrArgData* data;        // the ONE pointer in the push constant
};

[[vk::push_constant]] uniform PtrArgs args;

[shader("compute")]
[numthreads(1, 1, 1)]
void main_cs()
{
    PtrRoot* r = args.data.root;

    uint32_t sum = *r.scalar_ptr;            // pointer to scalar, typed load
    sum += *(r.scalar_ptr + 1);              // pointer arithmetic, typed load
    sum += r.nested_ptr.a;                   // pointer to struct, member load
    sum += *r.nested_ptr.p;                  // second-level pointer dereference
    sum += r.nested_ptr.b;                   // member after the pointer field
    sum += r.array_ptr.vals[0];              // array behind pointer
    sum += r.array_ptr.vals[1];
    sum += r.array_ptr.vals[2];
    sum += r.array_ptr.vals[3];
    sum += *r.arr_of_ptr[0];                 // pointer stored in GPU memory
    sum += *r.arr_of_ptr[1];
    sum += r.direct;                         // typed member load

    r.sum = sum;                             // typed store through root pointer
    *args.data.out = sum;                    // typed store through argument struct
}
