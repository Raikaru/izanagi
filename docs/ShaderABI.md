# Shader ABI

The C++/Slang ABI is a hard contract: the same public structures, handle
encodings, root-argument layouts, and pointer semantics must work unchanged on
every profile and every certified device.

## Pointer representation

- `GpuPtr` is a real 64-bit GPU virtual address (buffer device address).
  Shaders dereference it through physical-storage-buffer pointers.
- The bindless profile uses **typed** physical pointers (SPIR-V
  `PhysicalStorageBuffer` with typed pointees); the native profile enables
  untyped pointers but the shader corpus uses typed pointers only — the one
  untyped declaration (`void*` root arg in compute_texture) was removed.
- All pointer operations are typed: struct member access, arrays behind
  pointers, pointer arithmetic, second-level pointers stored in GPU memory,
  and pointer arrays in GPU memory. `tests/shaders/ptr_types.slang` exercises
  the full battery and runs on hardware (api test 30).
- Unsupported untyped reinterpretation is **native-only**: the bindless
  prelude deliberately omits the descriptor-heap handle model, so using it
  fails at shader compile time (asserted by `add_slang_negative_test`).

## Structure layout

- Shaders compile with `-force-glsl-scalar-layout -matrix-layout-row-major`
  (scalar block layout + row-major matrices), matching C++ natural layout for
  the shared structs.
- **Slang does not tail-pad structs to 8.** A C++ struct with 64-bit members
  has natural alignment 8 and would stride 80 while the shader sees 76 — so
  the shared ABI structs carry an explicit named tail member on both sides
  (`AbiRoot.tail_pad`). This was found by the extracted-manifest test.
- The extracted-manifest test (common_tests) parses the compiled SPIR-V
  (OpMemberDecorate offsets, OpDecorate ArrayStride, type widths, constant
  lengths) and compares against C++ `sizeof`/`alignof`/`offsetof` — the
  shader layout is the source of truth and any divergence fails.
- The GPU ABI test (api test 31) uploads an irregular nested structure
  (mixed-width scalars, padding, arrays, nested structs, a `GpuPtr` member, a
  second-level pointer) and verifies every interpreted value.

## Handle encoding

`TextureView` / `SamplerId` are uint64: bits 0..31 = descriptor index
(shader-visible), bits 32..47 = CPU generation (stale-handle detection), bits
48..55 = descriptor type. The encoding is identical across profiles — the
bindless profile's arrays are indexed by the same low 32 bits, so no data
structure changes and no per-draw CPU work exists on either path. Descriptor
index 0 is reserved as null.

## Root arguments

The push constant carries exactly ONE pointer to a GPU-resident argument
struct (compute: 8 B; graphics: two pointers, 16 B). The shader-side
`Args` structs and the pointer payloads are identical across profiles; only
the delivery command differs privately (vkCmdPushDataEXT vs
vkCmdPushConstants).

## 8/16-bit storage

8/16-bit integer storage through physical-storage-buffer pointers requires
`shaderInt8`/`shaderInt16` AND `storageBuffer8BitAccess`/`16BitAccess` — the
mandatory ABI test uses only 32/64-bit fields; the 8/16-bit ABI test is
capability-gated (api test 32) and runs only when all four features are
present (reported by the profile query).

## ABI conformance tests

- Static manifest: handle widths, root shapes, `AbiRoot` offsets/sizes
  (common_tests).
- Extracted shader manifest: SPIR-V offsets/sizes/strides vs C++ (common_tests).
- GPU round-trip: `abi_test.slang` + `abi_int8_16.slang` (api tests 31/32).
- Typed-pointer battery: `ptr_types.slang` (api test 30).
- Native-only boundary: `untyped_negative.slang` fails under bindless
  (negative compile test).
