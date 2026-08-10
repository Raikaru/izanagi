#pragma once
// Izanagi GPU API — backend-neutral public header.
// No Vulkan includes; the API is mechanism-agnostic (opaque handles + indices).

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <type_traits>
#include <utility>

#ifdef __cpp_concepts
#    define IZ_REQUIRES(x) requires x
#else
#    define IZ_REQUIRES(x)
#endif

// Flag-enum bitwise operators.
#define IZ_BITWISE_BINARY_OP(name, op)                                                             \
    inline constexpr name operator op(name lhs, name rhs) {                                        \
        using U = std::underlying_type_t<name>;                                                    \
        return name(static_cast<U>(lhs) op static_cast<U>(rhs));                                   \
    }
#define IZ_BITWISE_ASSIGNMENT_OP(name, op)                                                         \
    inline constexpr name operator op## = (name lhs, name rhs) {                                   \
        lhs = lhs op rhs;                                                                          \
        return lhs;                                                                                \
    }
#define IZ_BITWISE_BOOL_OP(name)                                                                   \
    inline constexpr bool any(name x) {                                                            \
        return static_cast<std::underlying_type_t<name>>(x) != 0;                                  \
    }
#define IZ_DEFINE_BITWISE_OPS(name)                                                                \
    IZ_BITWISE_BINARY_OP(name, |);                                                                 \
    IZ_BITWISE_BINARY_OP(name, &);                                                                 \
    IZ_BITWISE_BINARY_OP(name, ^);                                                                 \
    IZ_BITWISE_ASSIGNMENT_OP(name, |);                                                             \
    IZ_BITWISE_ASSIGNMENT_OP(name, &);                                                             \
    IZ_BITWISE_ASSIGNMENT_OP(name, ^);                                                             \
    IZ_BITWISE_BOOL_OP(name);

namespace gpu {

// --- Heap capacities (compile-time) -------------------------------------------
inline constexpr uint32_t kMaxSampledTextures = 65536;
inline constexpr uint32_t kMaxStorageTextures = 65536;
inline constexpr uint32_t kMaxSamplers        = 4096;
inline constexpr uint32_t kMaxFramesInFlight  = 2;

// --- Span ---------------------------------------------------------------------
template <class T>
class Span {
    template <class U>
    static constexpr bool is_const = false;
    template <class U>
    static constexpr bool is_const<const U> = true;
    template <class U, class V>
    static constexpr bool is_const_of = false;
    template <class U>
    static constexpr bool is_const_of<U, const U> = true;

   public:
    constexpr Span() noexcept = default;
    constexpr Span(T* ptr, size_t len) noexcept : m_ptr{ptr}, m_len{len} {}
    constexpr Span(T* begin, T* end) noexcept :
        m_ptr{begin}, m_len{static_cast<size_t>(end - begin)} {}
    template <size_t N>
    constexpr Span(T (&a)[N]) noexcept : Span(a, N) {}
    constexpr Span(std::initializer_list<T> v) noexcept IZ_REQUIRES(is_const<T>) :
        Span(v.begin(), v.size()) {}
    constexpr Span(const T& v) noexcept IZ_REQUIRES(is_const<T>) : Span(&v, 1) {}
    template <typename U>
    constexpr Span(const Span<U>& src) noexcept IZ_REQUIRES((is_const_of<U, T>)) :
        Span(src.data(), src.size()) {}
    constexpr Span(const Span<T>& src) noexcept = default;
    Span& operator=(const Span<T>& src)         = default;

    constexpr T*       data() const noexcept { return m_ptr; }
    constexpr size_t   size() const noexcept { return m_len; }
    constexpr bool     empty() const noexcept { return m_len == 0; }
    constexpr T&       operator[](size_t i) const noexcept { return m_ptr[i]; }
    constexpr T&       front() const noexcept { return *m_ptr; }
    constexpr T&       back() const noexcept { return *(m_ptr + m_len - 1); }
    constexpr T*       begin() const noexcept { return m_ptr; }
    constexpr T*       end() const noexcept { return m_ptr + m_len; }
    constexpr const T* cbegin() const noexcept { return begin(); }
    constexpr const T* cend() const noexcept { return end(); }

    Span<const uint8_t> as_bytes() const {
        return Span<const uint8_t>((const uint8_t*)m_ptr, m_len * sizeof(T));
    }
    template <typename U>
    Span<U> cast() const {
        return Span<U>((U*)m_ptr, m_len * sizeof(T) / sizeof(U));
    }

   private:
    T*     m_ptr{nullptr};
    size_t m_len{0};
};

inline constexpr Span<const char> operator""_sv(const char* val, size_t len) {
    return Span<const char>(val, len);
}

// --- Handles -------------------------------------------------------------------
struct MemoryBlock {
    void*    ptr;
    uint32_t len;
};

template <class T>
struct Handle {
    uint64_t       h = 0;
    constexpr      operator bool() const noexcept { return h != 0; }
    constexpr bool operator==(const Handle& other) const { return other.h == h; }
};

typedef struct DeviceImpl*        Device;
typedef struct QueueImpl*         Queue;
typedef struct CommandBufferImpl* CommandBuffer;
struct Pipeline;
struct Texture;
struct DepthStencilState;
struct Semaphore;

using GpuPtr      = uint64_t;   // Device address; 0 = null
using TextureView = uint64_t;   // Heap slot handle (low 32 = heap index)
using SamplerId   = uint64_t;   // Sampler heap slot handle

enum class LogLevel : uint8_t { Off, Error, Warning, Info, Debug };

using ProcLogCallback = void (*)(LogLevel         lvl,
                                 Span<const char> msg,
                                 uint32_t         line,
                                 Span<const char> file,
                                 void*            userdata);
using ProcAllocatorCallback = MemoryBlock (*)(void*    userdata,
                                              void*    ptr,
                                              uint32_t old_size,
                                              uint32_t new_size);

// --- Enums ---------------------------------------------------------------------

enum class GpuPreference : uint8_t { Discrete = 0, Integrated };

enum class Backend { Vulkan, Metal };

enum class Memory : uint8_t {
    Default,   // CPU visible, optimized for CPU write -> GPU read
    Gpu,       // GPU-only
    Readback,  // CPU visible, optimized for GPU write -> CPU read
};

enum class FrontFace : uint8_t { CCW = 0, CW };

enum class Cull : uint8_t { Front, Back, None };

enum class DepthFlags : uint8_t { None = 0, Read = 0x1, Write = 0x2 };
IZ_DEFINE_BITWISE_OPS(DepthFlags);

enum class Op : uint8_t {
    Never,
    Less,
    Equal,
    LessEqual,
    Greater,
    NotEqual,
    GreaterEqual,
    Always,
};

enum class StencilOp : uint8_t {
    Keep,
    Zero,
    Replace,
    IncrementClamp,
    DecrementClamp,
    Invert,
    IncrementWrap,
    DecrementWrap,
};

enum class Blend : uint8_t { Add, Subtract, RevSubtract, Min, Max };

enum class Factor : uint8_t { Zero, One, SrcColor, DstColor, SrcAlpha, OneMinusSrcAlpha };

enum class Topology : uint8_t { TriangleList, TriangleStrip };

enum class TextureType : uint8_t { Tex1D, Tex2D, Tex3D, TexCube, Tex2DArray, TexCubeArray };

enum class Format : uint32_t {
    None                 = 0x00000000,
    R8Unorm              = 0x00000001,
    R8Snorm              = 0x00000002,
    R8Uint               = 0x00000003,
    R8Sint               = 0x00000004,
    R16Unorm             = 0x00000005,
    R16Snorm             = 0x00000006,
    R16Uint              = 0x00000007,
    R16Sint              = 0x00000008,
    R16Float             = 0x00000009,
    RG8Unorm             = 0x0000000A,
    RG8Snorm             = 0x0000000B,
    RG8Uint              = 0x0000000C,
    RG8Sint              = 0x0000000D,
    R32Float             = 0x0000000E,
    R32Uint              = 0x0000000F,
    R32Sint              = 0x00000010,
    RG16Unorm            = 0x00000011,
    RG16Snorm            = 0x00000012,
    RG16Uint             = 0x00000013,
    RG16Sint             = 0x00000014,
    RG16Float            = 0x00000015,
    RGBA8Unorm           = 0x00000016,
    RGBA8UnormSrgb       = 0x00000017,
    RGBA8Snorm           = 0x00000018,
    RGBA8Uint            = 0x00000019,
    RGBA8Sint            = 0x0000001A,
    BGRA8Unorm           = 0x0000001B,
    BGRA8UnormSrgb       = 0x0000001C,
    RGB10A2Uint          = 0x0000001D,
    RGB10A2Unorm         = 0x0000001E,
    RG11B10Ufloat        = 0x0000001F,
    RGB9E5Ufloat         = 0x00000020,
    RG32Float            = 0x00000021,
    RG32Uint             = 0x00000022,
    RG32Sint             = 0x00000023,
    RGBA16Unorm          = 0x00000024,
    RGBA16Snorm          = 0x00000025,
    RGBA16Uint           = 0x00000026,
    RGBA16Sint           = 0x00000027,
    RGBA16Float          = 0x00000028,
    RGBA32Float          = 0x00000029,
    RGBA32Uint           = 0x0000002A,
    RGBA32Sint           = 0x0000002B,
    Stencil8             = 0x0000002C,
    Depth16Unorm         = 0x0000002D,
    Depth24Plus          = 0x0000002E,
    Depth24PlusStencil8  = 0x0000002F,
    Depth32Float         = 0x00000030,
    Depth32FloatStencil8 = 0x00000031,
    BC1RGBAUnorm         = 0x00000032,
    BC1RGBAUnormSrgb     = 0x00000033,
    BC2RGBAUnorm         = 0x00000034,
    BC2RGBAUnormSrgb     = 0x00000035,
    BC3RGBAUnorm         = 0x00000036,
    BC3RGBAUnormSrgb     = 0x00000037,
    BC4RUnorm            = 0x00000038,
    BC4RSnorm            = 0x00000039,
    BC5RGUnorm           = 0x0000003A,
    BC5RGSnorm           = 0x0000003B,
    BC6HRGBUfloat        = 0x0000003C,
    BC6HRGBFloat         = 0x0000003D,
    BC7RGBAUnorm         = 0x0000003E,
    BC7RGBAUnormSrgb     = 0x0000003F,
    ETC2RGB8Unorm        = 0x00000040,
    ETC2RGB8UnormSrgb    = 0x00000041,
    ETC2RGB8A1Unorm      = 0x00000042,
    ETC2RGB8A1UnormSrgb  = 0x00000043,
    ETC2RGBA8Unorm       = 0x00000044,
    ETC2RGBA8UnormSrgb   = 0x00000045,
    EACR11Unorm          = 0x00000046,
    EACR11Snorm          = 0x00000047,
    EACRG11Unorm         = 0x00000048,
    EACRG11Snorm         = 0x00000049,
    ASTC4x4Unorm         = 0x0000004A,
    ASTC4x4UnormSrgb     = 0x0000004B,
    ASTC5x4Unorm         = 0x0000004C,
    ASTC5x4UnormSrgb     = 0x0000004D,
    ASTC5x5Unorm         = 0x0000004E,
    ASTC5x5UnormSrgb     = 0x0000004F,
    ASTC6x5Unorm         = 0x00000050,
    ASTC6x5UnormSrgb     = 0x00000051,
    ASTC6x6Unorm         = 0x00000052,
    ASTC6x6UnormSrgb     = 0x00000053,
    ASTC8x5Unorm         = 0x00000054,
    ASTC8x5UnormSrgb     = 0x00000055,
    ASTC8x6Unorm         = 0x00000056,
    ASTC8x6UnormSrgb     = 0x00000057,
    ASTC8x8Unorm         = 0x00000058,
    ASTC8x8UnormSrgb     = 0x00000059,
    ASTC10x5Unorm        = 0x0000005A,
    ASTC10x5UnormSrgb    = 0x0000005B,
    ASTC10x6Unorm        = 0x0000005C,
    ASTC10x6UnormSrgb    = 0x0000005D,
    ASTC10x8Unorm        = 0x0000005E,
    ASTC10x8UnormSrgb    = 0x0000005F,
    ASTC10x10Unorm       = 0x00000060,
    ASTC10x10UnormSrgb   = 0x00000061,
    ASTC12x10Unorm       = 0x00000062,
    ASTC12x10UnormSrgb   = 0x00000063,
    ASTC12x12Unorm       = 0x00000064,
    ASTC12x12UnormSrgb   = 0x00000065,
    ValidCount,
};

enum class UsageFlags : uint16_t {
    None                   = 0,
    Sampled                = 0x01,
    Storage                = 0x02,
    ColorAttachment        = 0x04,
    DepthStencilAttachment = 0x08,
    TransferSrc            = 0x10,
    TransferDst            = 0x20,
};
IZ_DEFINE_BITWISE_OPS(UsageFlags);

enum class StageFlags : uint16_t {
    None              = 0,
    IndirectArguments = 0x01,
    Transfer          = 0x02,
    Compute           = 0x04,
    RasterColorOut    = 0x08,
    PixelShader       = 0x10,
    FragmentTests     = 0x20,
    VertexShader      = 0x40,
    Host              = 0x80,
};
IZ_DEFINE_BITWISE_OPS(StageFlags);

enum class LoadOp : uint8_t { Undefined, Load, Clear };

enum class StoreOp : uint8_t { Undefined, Store, Discard };

enum class QueueType : uint8_t { Default, ValidCount };

enum class PresentMode : uint8_t { Immediate, Mailbox, Fifo, FifoRelaxed, ValidCount };

enum class SurfaceStatus : uint8_t { Success, Suboptimal, OutOfDate, Error };

enum class SamplerCoords : uint8_t { Normalized, Pixel };

enum class SamplerFilter : uint8_t { Nearest, Linear };

enum class SamplerAddressing : uint8_t { ClampToEdge, Repeat, Mirrored };

enum class SpecializationConstantType : uint8_t {
    UInt8,
    UInt16,
    UInt32,
    Int8,
    Int16,
    Int32,
    Boolean,
    Float32,
};

enum class IndexType : uint8_t { UInt16, UInt32 };

// --- Structs -------------------------------------------------------------------
struct Dimension2D {
    uint32_t x, y;
};
struct Dimension3D {
    uint32_t x, y, z;
};
struct Rect2D {
    uint32_t offset_x = 0;
    uint32_t offset_y = 0;
    uint32_t width;
    uint32_t height;
};
struct Color {
    uint32_t r;
    uint32_t g;
    uint32_t b;
    uint32_t a;
};
struct Stencil {
    Op        test          = Op::Always;
    StencilOp fail_op       = StencilOp::Keep;
    StencilOp pass_op       = StencilOp::Keep;
    StencilOp depth_fail_op = StencilOp::Keep;
    uint8_t   reference     = 0;
};
struct SamplerDesc {
    SamplerCoords     coord          = SamplerCoords::Normalized;
    SamplerFilter     filter         = SamplerFilter::Nearest;
    SamplerAddressing address        = SamplerAddressing::ClampToEdge;
    float             max_anisotropy = 1.0f;
};
struct DeviceDesc {
    GpuPreference         gpu_preference         = GpuPreference::Discrete;
    uintptr_t             native_window_handle   = 0;
    uintptr_t             native_instance_handle = 0;
    ProcLogCallback       log_callback           = nullptr;
    void*                 log_userdata           = nullptr;
    LogLevel              log_level              = LogLevel::Off;
    ProcAllocatorCallback alloc_callback         = nullptr;
    void*                 alloc_userdata         = nullptr;
    bool                  enable_validation      = false;
};
struct DepthStencilDesc {
    DepthFlags depth_mode              = DepthFlags::None;
    Op         depth_test              = Op::Always;
    float      depth_bias              = 0.0f;
    float      depth_bias_slope_factor = 0.0f;
    float      depth_bias_clamp        = 0.0f;
    uint8_t    stencil_read_mask       = 0xff;
    uint8_t    stencil_write_mask      = 0xff;
    Stencil    stencil_front;
    Stencil    stencil_back;
};
struct BlendDesc {
    Blend   color_op         = Blend::Add;
    Factor  src_color_factor = Factor::One;
    Factor  dst_color_factor = Factor::Zero;
    Blend   alpha_op         = Blend::Add;
    Factor  src_alpha_factor = Factor::One;
    Factor  dst_alpha_factor = Factor::Zero;
    uint8_t color_write_mask = 0xf;
};
struct ColorTarget {
    Format    format     = Format::None;
    BlendDesc blendstate = {};
};
struct RasterDesc {
    Topology                topology          = Topology::TriangleList;
    bool                    alpha_to_coverage = false;
    uint8_t                 sample_count      = 1;
    Format                  depth_format      = Format::None;
    Format                  stencil_format    = Format::None;
    Span<const ColorTarget> color_targets     = {};
};
struct RenderAttachment {
    Handle<Texture> texture = {0};
    LoadOp          load_op;
    StoreOp         store_op;
    Color           clear_color;
};
struct RenderPassDesc {
    Span<const RenderAttachment> color_attachments;
    RenderAttachment             depth_attachment;
    RenderAttachment             stencil_attachment;
    Rect2D                       render_area;
};
struct TextureDesc {
    TextureType type = TextureType::Tex2D;
    Dimension3D dimensions;
    uint32_t    mip_count    = 1;
    uint32_t    array_count  = 1;
    uint32_t    sample_count = 1;
    Format      format       = Format::None;
    UsageFlags  usage        = UsageFlags::None;
};
struct TextureViewDesc {
    Handle<Texture> texture;
    TextureType     type        = TextureType::Tex2D;
    Format          format      = Format::None;
    uint8_t         base_mip    = 0;
    uint8_t         mip_count   = 1;
    uint16_t        base_layer  = 0;
    uint16_t        layer_count = 1;
};
struct TextureSizeAlign {
    size_t size;
    size_t align;
};
struct SpecializationConstant {
    uint32_t constant_id;
    union {
        uint64_t int_val;
        float    float_val;
        bool     bool_val;
    };
    SpecializationConstantType type;
};
struct ShaderSource {
    Span<const uint8_t> source;
    Span<const char>    entry_point;
};
struct SurfaceCapabilities {
    UsageFlags              usages;
    Span<const Format>      formats;
    Span<const PresentMode> present_modes;
};
struct SurfaceConfiguration {
    Format      format;
    UsageFlags  usages;
    uint32_t    width;
    uint32_t    height;
    PresentMode present_mode;
};
struct SurfaceTextureInfo {
    SurfaceStatus   status;
    Handle<Texture> texture;
};
struct SemaphoreInfo {
    Handle<Semaphore> semaphore;
    uint64_t          value;
    StageFlags        stage = StageFlags::None;
};
struct BufferTextureCopyInfo {
    Dimension3D image_extent;
    uint32_t    buffer_row_pixels_stride  = 0;
    uint32_t    buffer_plane_rows_stride  = 0;
    Dimension3D texture_image_offset{0, 0, 0};
    uint8_t     base_mip   = 0;
    uint8_t     base_layer = 0;
};
struct DrawIndexedInstancedInfo {
    GpuPtr    vertexDataGpu;
    GpuPtr    fragmentDataGpu;
    GpuPtr    indicesGpu;
    uint32_t  indexCount;
    uint32_t  instanceCount = 1;
    IndexType type          = IndexType::UInt16;
};
struct DrawIndexedIndirectInfo {
    GpuPtr    vertexDataGpu;
    GpuPtr    fragmentDataGpu;
    GpuPtr    indicesGpu;
    GpuPtr    argsGpu;
    IndexType type = IndexType::UInt16;
};
struct MultiDrawIndirectInfo {
    GpuPtr    vertexDataGpu;
    GpuPtr    pixelDataGpu;
    GpuPtr    indicesGpu;
    GpuPtr    argsGpu;
    GpuPtr    drawCountGpu;
    uint32_t  maxDraws;
    IndexType type = IndexType::UInt16;
};
struct alignas(8) DrawIndexedIndirectGpuArgs {
    uint32_t index_count;
    uint32_t instance_count;
    uint32_t first_index;
    int32_t  vertex_offset;
    uint32_t first_instance;
};

// --- API functions --------------------------------------------------------------

// Device
Device  create_device(const DeviceDesc&);
void    destroy_device(Device);
Backend device_backend();
void    device_wait_for_idle(Device);

// Surface
SurfaceCapabilities get_surface_capabilities(Device);
bool                configure_surface(Device, const SurfaceConfiguration&);
void                unconfigure_surface(Device);
SurfaceTextureInfo  get_current_texture(Device);
SurfaceStatus       present(Device, Queue);

// Memory
GpuPtr malloc(Device, size_t bytes, Memory = Memory::Default);
GpuPtr malloc(Device, size_t bytes, size_t align, Memory = Memory::Default);
void   free(Device, GpuPtr);
void*  get_host_pointer(Device, GpuPtr);

// Textures + global heap
TextureSizeAlign get_texture_size_align(Device, const TextureDesc&);
Handle<Texture>  create_texture(Device, const TextureDesc&, GpuPtr placement = 0);
void             free(Device, Handle<Texture>);
TextureView      create_texture_view(Device, const TextureViewDesc&);
TextureView      create_rw_texture_view(Device, const TextureViewDesc&);
SamplerId        create_sampler(Device, const SamplerDesc&);
void             free_texture_view(Device, TextureView);
void             free_rw_texture_view(Device, TextureView);
void             free_sampler(Device, SamplerId);

// Pipelines & state
Handle<Pipeline> create_compute_pipeline(Device,
                                         ShaderSource,
                                         Span<const SpecializationConstant> = {});
Handle<Pipeline> create_graphics_pipeline(Device,
                                          ShaderSource vertex,
                                          ShaderSource fragment,
                                          const RasterDesc&,
                                          Span<const SpecializationConstant> = {});
void             free(Device, Handle<Pipeline>);
Handle<DepthStencilState> create_depth_stencil_state(Device, const DepthStencilDesc&);
void                      free_depth_stencil_state(Device, Handle<DepthStencilState>);

// Sync + queue
Handle<Semaphore> create_semaphore(Device, uint64_t init);
void              wait_semaphore(Device, Handle<Semaphore>, uint64_t value);
void              free(Device, Handle<Semaphore>);
Queue             get_queue(Device, QueueType = QueueType::Default);
CommandBuffer     queue_start_command_recording(Queue);
void              queue_submit(Queue,
                               Span<const CommandBuffer>,
                               Span<const SemaphoreInfo> wait   = {},
                               Span<const SemaphoreInfo> signal = {});
void              queue_on_submitted_work_completed(Queue, void (*fn)(void*), void* userdata);
void              queue_process_events(Queue);

// Commands
void cmd_memcpy(CommandBuffer, GpuPtr dst, GpuPtr src, size_t size);
void cmd_copy_to_texture(CommandBuffer, GpuPtr src, Handle<Texture>, const BufferTextureCopyInfo&);
void cmd_copy_from_texture(CommandBuffer, Handle<Texture>, GpuPtr dst, const BufferTextureCopyInfo&);
void cmd_barrier(CommandBuffer, StageFlags before, StageFlags after);
void cmd_set_pipeline(CommandBuffer, Handle<Pipeline>);
void cmd_set_depth_stencil_state(CommandBuffer, Handle<DepthStencilState>);
void cmd_set_viewport(CommandBuffer, const Rect2D&);
void cmd_set_scissor_rect(CommandBuffer, const Rect2D&);
void cmd_set_front_face(CommandBuffer, FrontFace);
void cmd_set_cull_mode(CommandBuffer, Cull);
void cmd_dispatch(CommandBuffer, GpuPtr data, const Dimension3D& groups);
void cmd_dispatch_indirect(CommandBuffer, GpuPtr data, GpuPtr groupsGpu);
void cmd_begin_render_pass(CommandBuffer, const RenderPassDesc&);
void cmd_end_render_pass(CommandBuffer);
void cmd_draw(CommandBuffer, GpuPtr vertexData, GpuPtr fragmentData, uint32_t vertexCount, uint32_t instanceCount);
void cmd_draw_indexed_instanced(CommandBuffer, const DrawIndexedInstancedInfo&);
void cmd_draw_indexed_instanced_indirect(CommandBuffer, const DrawIndexedIndirectInfo&);
void cmd_draw_indexed_instanced_indirect_multi(CommandBuffer, const MultiDrawIndirectInfo&);
void cmd_wait_for_surface_texture(CommandBuffer);
void cmd_signal_surface_texture(CommandBuffer);
void cmd_push_debug_group(CommandBuffer, Span<const char>);
void cmd_pop_debug_group(CommandBuffer);
void cmd_finalize(CommandBuffer);

// Explicit template instantiation declarations
extern template class Span<const char>;
extern template class Span<uint8_t>;
extern template class Span<const ColorTarget>;
extern template class Span<const RenderAttachment>;
extern template class Span<const Format>;
extern template class Span<const PresentMode>;
extern template class Span<const CommandBuffer>;
extern template class Span<const SemaphoreInfo>;
extern template class Span<const SpecializationConstant>;

}  // namespace gpu
