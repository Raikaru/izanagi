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
    inline constexpr name& operator op##= (name& lhs, name rhs) {                                 \
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
// Cross-const conversion trait: Span<U> is convertible to Span<T> when
// T is `const U`. (Member variable-template partial specializations are not
// portable across compilers; use free helpers + std traits instead.)
template <class T, class U>
inline constexpr bool span_is_const_of = std::is_same_v<T, const U>;

template <class T>
class Span {
   public:
    constexpr Span() noexcept = default;
    constexpr Span(T* ptr, size_t len) noexcept : m_ptr{ptr}, m_len{len} {}
    constexpr Span(T* begin, T* end) noexcept :
        m_ptr{begin}, m_len{static_cast<size_t>(end - begin)} {}
    template <size_t N>
    constexpr Span(T (&a)[N]) noexcept : Span(a, N) {}
    constexpr Span(std::initializer_list<T> v) noexcept IZ_REQUIRES(std::is_const_v<T>) :
        Span(v.begin(), v.size()) {}
    constexpr Span(const T& v) noexcept IZ_REQUIRES(std::is_const_v<T>) : Span(&v, 1) {}
    template <typename U>
    constexpr Span(const Span<U>& src) noexcept IZ_REQUIRES((span_is_const_of<T, U>)) :
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

// Selected capability profile of the compiled backend. Profiles preserve the
// same public programming model (real GPU pointers + persistent GPU-indexed
// resource namespace) using different private mechanisms; they never degrade
// semantics. VulkanNative = the modern descriptor-heap path;
// VulkanBindless = the descriptor-indexing compatibility path; Metal = the
// future Apple backend.
enum class BackendProfile : uint8_t { VulkanNative, VulkanBindless, Metal };

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

enum class Factor : uint8_t {
    Zero,
    One,
    SrcColor,
    OneMinusSrcColor,
    DstColor,
    OneMinusDstColor,
    SrcAlpha,
    OneMinusSrcAlpha,
    DstAlpha,
    OneMinusDstAlpha,
    Src1Color,
    OneMinusSrc1Color,
    Src1Alpha,
    OneMinusSrc1Alpha,
    SrcAlphaSaturate,
};

enum class Topology : uint8_t {
    TriangleList  = 0,
    TriangleStrip = 1,
    LineList      = 2,
    LineStrip     = 3,
};
enum class PolygonMode : uint8_t { Fill, Line };


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

// Result of queue_submit: identifies the submitted GPU work by its queue
// timeline value. The logical timeline advances only on a successful submit.
enum class SubmitStatus : uint8_t { Success, DeviceLost, OutOfMemory, Error };
struct Submission {
    Queue        queue  = nullptr;
    uint64_t     value  = 0;
    SubmitStatus status = SubmitStatus::Error;

    explicit operator bool() const { return status == SubmitStatus::Success; }
};

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

// Asynchronous pipeline request lifecycle. Request -> Pending, then Ready or
// Failed (see request_compute_pipeline / request_graphics_pipeline).
enum class PipelineStatus : uint8_t { Pending, Ready, Failed };

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
// Minification, magnification, and mip-level interpolation are independent.
// max_anisotropy is clamped by the backend; 1 disables anisotropic filtering.
struct SamplerDesc {
    SamplerCoords     coord          = SamplerCoords::Normalized;
    SamplerFilter     min_filter     = SamplerFilter::Nearest;
    SamplerFilter     mag_filter     = SamplerFilter::Nearest;
    SamplerFilter     mip_filter     = SamplerFilter::Nearest;
    SamplerAddressing address        = SamplerAddressing::ClampToEdge;
    float             max_anisotropy = 1.0f;
    float             mip_lod_bias   = 0.0f;
};
// Opaque identity of the device's native pipeline cache. Use it to key
// persistent storage per backend/driver/GPU. The cache blob is NOT
// transferable across drivers or GPUs; the driver rejects incompatible
// blobs at load time (treat rejection as "no cache").
// cache_uuid is the backend's pipeline-cache compatibility identity (Vulkan:
// pipelineCacheUUID from the cache blob header); driver_uuid is the stable
// per-driver identity (VkPhysicalDeviceIDProperties.driverUUID), used as a
// fallback when the driver does not expose the cache UUID for an empty cache.
struct CacheIdentity {
    Backend        backend    = Backend::Vulkan;
    // The backend profile that produced the cache blob. Native and Bindless
    // blobs are never interchangeable, even for the same driver/GPU — the
    // profile is part of the persistent-cache key.
    BackendProfile profile    = BackendProfile::VulkanNative;
    uint32_t       vendor_id  = 0;
    uint32_t       device_id  = 0;
    uint8_t        driver_uuid[16] = {};
    uint8_t        cache_uuid[16]  = {};
};
// Optional persistent native pipeline cache (per device, whole-blob).
// load is called once during create_device to seed the cache; return false
// when no usable cache exists (first run, rejected blob, etc.). store is
// called once during destroy_device with the final blob. Both are optional;
// provide at least one to enable caching, provide neither to keep none.
// Blob memory in load is application-owned and only read during create_device;
// the blob in store is valid only for the duration of the call.
struct PipelineCacheCallbacks {
    bool (*load)(const CacheIdentity&, void* user, MemoryBlock* blob);
    void (*store)(const CacheIdentity&, MemoryBlock blob, void* user);
    void* user;
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
    PipelineCacheCallbacks pipeline_cache_callbacks = {};
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
    PolygonMode             polygon_mode      = PolygonMode::Fill;
    bool                    alpha_to_coverage = false;
    uint8_t                 sample_count      = 1;
    Format                  depth_format      = Format::None;
    Format                  stencil_format    = Format::None;
    Span<const ColorTarget> color_targets     = {};
};
// Color attachments may set resolve_texture (a sample_count 1 texture of the
// same format/size) to have the MSAA result resolved into it at pass end.
// Depth/stencil attachments do not support resolve; any resolve_texture set
// on them is ignored. mip/layer select the attachment subresource (no public
// image views; the backend caches native views internally).
struct RenderAttachment {
    Handle<Texture> texture = {0};
    uint16_t        mip     = 0;
    uint16_t        layer   = 0;
    LoadOp          load_op;
    StoreOp         store_op;
    Color           clear_color;
    Handle<Texture> resolve_texture = {0};
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
// Actual device limits and optional raster capabilities.
// framebuffer_sample_counts is a bit mask whose set bits are the supported
// sample counts (1, 2, 4, 8, ...). non_solid_fill gates PolygonMode::Line.
struct DeviceLimits {
    uint32_t max_sampled_textures;
    uint32_t max_storage_textures;
    uint32_t max_samplers;
    uint32_t framebuffer_sample_counts;
    bool     non_solid_fill;
    uint64_t min_uniform_alignment;
    uint64_t min_storage_alignment;
    uint64_t non_coherent_atom_size;
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
// Selected capability profile of the compiled backend (compile-time for the
// current per-build profile selection).
BackendProfile device_backend_profile();
DeviceLimits device_limits(Device);
void    device_wait_for_idle(Device);
// True when dual-source blending (Factor::Src1*/OneMinusSrc1*) is supported
// on this device. Pipeline creation with Src1 factors fails (returns null)
// when this is false.
bool    device_supports_dual_source_blend(Device);

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
// Host-memory synchronization for non-coherent allocations. flush makes CPU
// writes visible to the GPU (upload); invalidate makes GPU writes visible to
// the CPU (readback). Coherent memory is a successful no-op. Ranges are
// validated against the allocation and aligned to the non-coherent atom size.
// Call flush after writing an upload buffer before submitting; call invalidate
// after GPU work completes before reading a readback buffer.
bool   flush_host_memory(Device, GpuPtr, size_t size);
bool   invalidate_host_memory(Device, GpuPtr, size_t size);

// Textures + global heap
// Textures allocate their own GPU memory (VMA); there is no buffer-backed
// placement token — a shader-visible GpuPtr is not a texture-memory token.
Handle<Texture>  create_texture(Device, const TextureDesc&);
void             free(Device, Handle<Texture>);
TextureView      create_texture_view(Device, const TextureViewDesc&);
TextureView      create_rw_texture_view(Device, const TextureViewDesc&);
SamplerId        create_sampler(Device, const SamplerDesc&);
void             free_texture_view(Device, TextureView);
void             free_rw_texture_view(Device, TextureView);
void             free_sampler(Device, SamplerId);

// Pipelines & state
// Blocking convenience creators: request + wait for Ready. May perform
// expensive native compilation — intended for loading screens, tools, tests,
// and small known pipeline sets. Returns a null handle on failure.
Handle<Pipeline> create_compute_pipeline(Device,
                                         ShaderSource,
                                         Span<const SpecializationConstant> = {});
Handle<Pipeline> create_graphics_pipeline(Device,
                                          ShaderSource vertex,
                                          ShaderSource fragment,
                                          const RasterDesc&,
                                          Span<const SpecializationConstant> = {});
// Asynchronous pipeline requests: never block on native compilation (a
// device-owned compiler worker compiles in the background). The returned
// handle is immediately valid; poll get_pipeline_status or block with
// wait_pipeline. All inputs are deep-copied by the request. Identical
// descriptions share one compiled pipeline.
Handle<Pipeline> request_compute_pipeline(Device,
                                          ShaderSource,
                                          Span<const SpecializationConstant> = {});
Handle<Pipeline> request_graphics_pipeline(Device,
                                           ShaderSource vertex,
                                           ShaderSource fragment,
                                           const RasterDesc&,
                                           Span<const SpecializationConstant> = {});
PipelineStatus   get_pipeline_status(Device, Handle<Pipeline>);
// Block until the pipeline reaches Ready or Failed. Returns true only for
// Ready.
bool             wait_pipeline(Device, Handle<Pipeline>);
// Optional graphics state for the static-graphics-state fallback (private
// static pipeline variants). On devices without extended dynamic state the
// baked members (front face, cull, depth/stencil test + ops) are compiled
// into private pipeline variants; this is the prewarm API for that path.
// On the regular path it is a no-op wrapper over the base pipeline.
struct GraphicsStateDesc {
    Handle<DepthStencilState> depth_stencil;   // null: depth/stencil disabled
    FrontFace                 front_face       = FrontFace::CCW;
    Cull                      cull             = Cull::None;
};
// Ensures a private static variant for (pipeline, graphics state) is
// requested (compilation happens on the device compiler worker; never
// blocks). Returns its current status. Identity dedup: repeated calls with
// the same description share one variant.
PipelineStatus   request_graphics_state(Device, Handle<Pipeline>, const GraphicsStateDesc&);
// Blocks until the requested variant is Ready or Failed (timeout_ms == 0
// waits forever). Returns true only for Ready.
bool             wait_graphics_state(Device, Handle<Pipeline>, const GraphicsStateDesc&,
                                     uint64_t timeout_ms = 0);
// Explicitly persist the native pipeline cache now (blocking: waits for
// queued compilation to drain). For loading screens / shutdown, not frame
// recording. No-op when no cache callbacks were provided.
void             flush_pipeline_cache(Device);
// Explicitly retire a resource against a submission. The CPU handle is
// invalidated immediately; native destruction / descriptor recycling happens
// only after the target submission completes. This is the normal path for
// resources reachable through raw GPU pointers or descriptor indices stored
// in user GPU data (the backend cannot infer such usage). An invalid or
// failed Submission retires the resource conservatively after the latest
// successfully submitted work completes.
void free_after(Device, GpuPtr, Submission);
void free_after(Device, Handle<Texture>, Submission);
void free_after(Device, Handle<Pipeline>, Submission);
void free_texture_view_after(Device, TextureView, Submission);
void free_rw_texture_view_after(Device, TextureView, Submission);
void free_sampler_after(Device, SamplerId, Submission);
// Immediate destruction: valid only when the application guarantees no
// recorded, pending, in-flight, or future GPU access to the resource.
void             free(Device, Handle<Pipeline>);
Handle<DepthStencilState> create_depth_stencil_state(Device, const DepthStencilDesc&);
void                      free_depth_stencil_state(Device, Handle<DepthStencilState>);

// Sync + queue
Handle<Semaphore> create_semaphore(Device, uint64_t init);
void              wait_semaphore(Device, Handle<Semaphore>, uint64_t value);
void              free(Device, Handle<Semaphore>);
Queue             get_queue(Device, QueueType = QueueType::Default);
CommandBuffer     queue_start_command_recording(Queue);
// Submits command buffers, serialized per queue. Returns a Submission that
// identifies the work; the queue's logical timeline advances only when the
// native submit succeeds (a failed submit never publishes a value the GPU
// will not signal). Non-blocking with respect to GPU completion.
Submission        queue_submit(Queue,
                               Span<const CommandBuffer>,
                               Span<const SemaphoreInfo> wait   = {},
                               Span<const SemaphoreInfo> signal = {});
// True once the submission's GPU work has completed. False for failed
// submissions and for incomplete work.
bool              submission_complete(Submission);
// Blocks until the submission's GPU work completes; true only for successful
// submissions that complete. Intended for loading phases, not frame recording.
bool              wait_submission(Submission);
void              queue_on_submitted_work_completed(Queue, void (*fn)(void*), void* userdata);
void              queue_process_events(Queue);

// Commands
void cmd_memcpy(CommandBuffer, GpuPtr dst, GpuPtr src, size_t size);
void cmd_copy_to_texture(CommandBuffer, GpuPtr src, Handle<Texture>, const BufferTextureCopyInfo&);
void cmd_copy_from_texture(CommandBuffer, Handle<Texture>, GpuPtr dst, const BufferTextureCopyInfo&);
// Generate mips 1..N-1 from mip 0 by successive linear blits (layer 0 only;
// cube faces / array layers are not generated).
// Texture must have UsageFlags::TransferSrc | TransferDst.
// Caller barriers: writes to mip 0 -> StageFlags::Transfer before,
// StageFlags::Transfer -> consumer stage after.
void cmd_generate_mipmaps(CommandBuffer, Handle<Texture>);
void cmd_barrier(CommandBuffer, StageFlags before, StageFlags after);
// Bind a Ready pipeline; returns false and records nothing when the pipeline
// is Pending or Failed, so the application can explicitly bind a fallback or
// skip the operation. Ignoring the return value is valid for code that only
// binds blocking-created pipelines.
bool cmd_set_pipeline(CommandBuffer, Handle<Pipeline>);
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
