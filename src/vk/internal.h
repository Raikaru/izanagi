#pragma once
// Internal header shared by the Vulkan backend TUs.
// Contains DeviceImpl, QueueImpl, CommandBufferImpl, and all backend-internal structs.

#include "common/containers.h"
#include "common/platform_utils.h"
#include "gpu_to_vk.h"
#include "vma_usage.h"
#include "volk.h"

namespace gpu {

// --- Logging helpers -----------------------------------------------------------
#define IZ_SV_IMPL(x) #x##_sv
#define IZ_MAKE_SV(x) IZ_SV_IMPL(x)
#define IZ_LOG(d, lvl, msg) log_impl(d, lvl, IZ_MAKE_SV(msg), __LINE__, IZ_MAKE_SV(__FILE__))
#define IZ_CHK(d, expr, msg)                                                                     \
    [&]() -> bool {                                                                              \
        VkResult chk_result = (expr);                                                            \
        if (chk_result != VK_SUCCESS) {                                                          \
            log_vk_impl(d, chk_result, IZ_MAKE_SV(msg), __LINE__, IZ_MAKE_SV(__FILE__));        \
            return false;                                                                        \
        } else {                                                                                 \
            return true;                                                                         \
        }                                                                                        \
    }()

// --- Forward declarations -------------------------------------------------------
struct DeviceImpl;
struct QueueImpl;
struct CommandBufferImpl;

// --- Internal structs ------------------------------------------------------------

struct Buffer {
    VkBuffer       vk_buffer;
    VmaAllocation  vk_allocation;
    void*          host_ptr;
    GpuPtr         device_ptr;
};

struct TextureImpl {
    VkImage        vk_image           = VK_NULL_HANDLE;
    VkImageView    default_image_view = VK_NULL_HANDLE;
    VmaAllocation  vk_allocation      = VK_NULL_HANDLE;
    VkImageViewType vk_type           = VK_IMAGE_VIEW_TYPE_2D;
    Format         format             = Format::None;
    bool           is_swapchain_image = false;
    uint32_t       mip_count          = 1;
    Dimension3D    dimensions         = {};
};

struct DepthStencilState {
    DepthStencilDesc desc;
};

struct PipelineImpl {
    VkPipeline            vk_pipeline = VK_NULL_HANDLE;
    VkPipelineBindPoint   bind_point  = VK_PIPELINE_BIND_POINT_COMPUTE;
};

struct SemaphoreImpl {
    VkSemaphore vk_semaphore = VK_NULL_HANDLE;
};

// --- Surface (swapchain) state ----------------------------------------------------
struct Surface {
    static constexpr uint32_t kMaxSwapchainImages = 8;
    static constexpr uint32_t kMaxNumFormats      = static_cast<size_t>(Format::ValidCount);
    static constexpr uint32_t kMaxNumPresentModes = static_cast<size_t>(PresentMode::ValidCount);

    VkSurfaceKHR   surface    = VK_NULL_HANDLE;
    VkSwapchainKHR swapchain  = VK_NULL_HANDLE;

    Format       supported_formats[kMaxNumFormats];
    PresentMode  supported_present_modes[kMaxNumPresentModes];
    size_t       num_supported_formats       = 0;
    size_t       num_supported_present_modes = 0;

    VkFormat  swapchain_format = VK_FORMAT_UNDEFINED;
    VkExtent2D swapchain_extent{0, 0};

    uint32_t current_image_idx = 0;
    uint32_t image_count       = 0;

    Handle<Texture>  swapchain_images[kMaxSwapchainImages];
    Handle<Semaphore> frame_semaphore;
    Handle<Semaphore> acquire_semaphores[kMaxFramesInFlight];
    Handle<Semaphore> present_semaphores[kMaxSwapchainImages];

    uint64_t frame_idx = 0;

    VkCommandBuffer first_use_command[kMaxFramesInFlight]     = {VK_NULL_HANDLE};
    VkCommandBuffer transitioning_command[kMaxSwapchainImages] = {VK_NULL_HANDLE};
};

// --- Descriptor heap state ---------------------------------------------------------
struct DescriptorHeap {
    // Sampler heap
    VkBuffer       sampler_buffer       = VK_NULL_HANDLE;
    VmaAllocation  sampler_allocation   = VK_NULL_HANDLE;
    void*          sampler_host_ptr     = nullptr;
    GpuPtr         sampler_device_ptr   = 0;
    uint32_t       sampler_capacity     = 0;
    uint32_t       sampler_descriptor_size = 0;
    uint64_t       sampler_heap_size    = 0;
    uint64_t       sampler_reserved_offset = 0;
    uint64_t       sampler_reserved_size   = 0;

    // Resource heap (sampled + storage images share one buffer)
    VkBuffer       resource_buffer       = VK_NULL_HANDLE;
    VmaAllocation  resource_allocation   = VK_NULL_HANDLE;
    void*          resource_host_ptr     = nullptr;
    GpuPtr         resource_device_ptr   = 0;
    uint32_t       sampled_capacity      = 0;
    uint32_t       storage_capacity      = 0;
    uint32_t       sampled_descriptor_size  = 0;
    uint32_t       storage_descriptor_size  = 0;
    uint64_t       resource_heap_size    = 0;
    uint64_t       resource_reserved_offset = 0;
    uint64_t       resource_reserved_size   = 0;
    // Byte offset where storage descriptors begin within the resource heap buffer
    uint64_t       storage_region_offset = 0;
};

// --- Command pool management --------------------------------------------------------
struct CommandPool {
    VkCommandPool   command_pool = VK_NULL_HANDLE;
    SegmentArray<CommandBufferImpl> command_buffers;
    int64_t         buffer_free_idx = 0;
    uint64_t        frame_idx       = 0;
};

struct CommandSuperpool {
    static constexpr uint32_t kPoolsPerGroup          = kMaxFramesInFlight;
    static constexpr uint32_t kMaxSimultaneousCommands = 64;
    int64_t     available_pools = ~0ll;
    CommandPool pools[kMaxSimultaneousCommands * kPoolsPerGroup] = {};
};

// --- Event (deferred completion callback) -------------------------------------------
struct CompletionEvent {
    uint64_t completed_time;
    void     (*callback)(void*);
    void*     userdata;
};

// --- GPU pointer map (address -> buffer/offset) -------------------------------------
struct BufferAndOffset {
    VkBuffer      buffer;
    uint32_t      offset;
    VmaAllocation alloc;
};

struct GpuPtrMap {
    GpuPtr        ptr;
    Handle<Buffer> buffer;
};

// --- DeviceImpl ----------------------------------------------------------------------
// Cast between Handle<A> and Handle<B> when A and B are different tags
// for the same underlying slot. Safe: the uint64_t encoding is identical.
template <class To, class From>
Handle<To> handle_cast(Handle<From> h) {
    return Handle<To>{.h = h.h};
}

struct DeviceImpl {
    Allocator          allocator;
    ProcLogCallback    log_callback  = nullptr;
    void*              log_userdata  = nullptr;
    LogLevel           log_level     = LogLevel::Off;
    tls_key            thread_local_key;

    VkInstance         instance        = VK_NULL_HANDLE;
    VkPhysicalDevice   physical_device = VK_NULL_HANDLE;
    uint32_t           graphics_queue_family = 0;
    VkDevice           device          = VK_NULL_HANDLE;
    VmaAllocator       vma             = VK_NULL_HANDLE;

    // Surface
    Surface surface;

    // Debug
    bool                      has_debug_markers   = false;
    bool                      enable_validation   = false;
    VkDebugUtilsMessengerEXT  debug_messenger     = VK_NULL_HANDLE;

    // Memory tracking
    rwlock            ptr_map_lock = IZ_RWLOCK_INIT;
    Vector<GpuPtrMap> ptr_map;

    // Pools
    SlotMap<Buffer>            buffer_pool;
    SlotMap<TextureImpl>       texture_pool;
    SlotMap<SemaphoreImpl>     semaphore_pool;
    SlotMap<PipelineImpl>      pipeline_pool;
    SlotMap<DepthStencilState> depth_stencil_pool;

    // Queues (single Default queue in v1)
    QueueImpl* default_queue = nullptr;

    // Descriptor heap
    DescriptorHeap heap;
    TwoLevelBitset sampled_bitset;
    TwoLevelBitset storage_bitset;
    TwoLevelBitset sampler_bitset;

    // Deferred slot recycling (timeline value at free time)
    struct DeferredFree {
        uint32_t slot;
        uint64_t timeline_value;
        uint8_t  region; // 0=sampled, 1=storage, 2=sampler
    };
    mutex                 deferred_lock = IZ_MUTEX_INIT;
    Vector<DeferredFree>  deferred_frees;

    // Uninitialized textures (need UNDEFINED->GENERAL transition)
    mutex                  texture_init_lock = IZ_MUTEX_INIT;
    Vector<Handle<Texture>> uninitialized_textures;
};

// --- QueueImpl ----------------------------------------------------------------------
struct QueueImpl {
    DeviceImpl*      device        = nullptr;
    VkQueue          queue         = VK_NULL_HANDLE;
    CommandSuperpool command_superpool = {};
    Handle<Semaphore> timeline;
    uint32_t         queue_family  = 0;
    uint64_t         timeline_value = 0;
    Vector<CompletionEvent> pending_events;
};

// --- CommandBufferImpl ----------------------------------------------------------------
struct CommandBufferImpl {
    DeviceImpl* device = nullptr;
    QueueImpl*  queue  = nullptr;
    CommandPool* pool  = nullptr;
    VkCommandBuffer buffer = VK_NULL_HANDLE;

    GpuPtr current_idx_buffer = 0;
    bool   wait_for_surface_texture = false;
    bool   signal_surface_texture   = false;
};

// --- Thread-local arena state ----------------------------------------------------------
struct ThreadLocalState {
    static constexpr size_t kArenaSize = 256 * 1024;
    Allocator   allocator;
    MemoryBlock arena_memory;
    Arena       arena;
    ThreadLocalState(Allocator alloc, ProcLogCallback cb, void* userdata) :
        allocator(alloc),
        arena_memory(alloc.alloc(kArenaSize)),
        arena(arena_memory.ptr, arena_memory.len, cb, userdata) {}
    ~ThreadLocalState() { allocator.free(arena_memory); }
};

Arena* get_thread_local_arena(DeviceImpl* d);
void   log_impl(DeviceImpl* d, LogLevel lvl, Span<const char> msg, uint32_t line, Span<const char> file);
void   log_vk_impl(DeviceImpl* d, VkResult res, Span<const char> msg, uint32_t line, Span<const char> file);

// --- Internal helpers shared across TUs ------------------------------------------------
BufferAndOffset buffer_and_offset_from_ptr(DeviceImpl* d, GpuPtr ptr);

// Command pool management (commands.cpp)
CommandPool*  get_command_pool(QueueImpl* queue, uint64_t frame_idx);
void          release_command_pool(QueueImpl* q, CommandPool* pool);
CommandBuffer get_command_buffer(QueueImpl* q, CommandPool* pool);

// Surface helpers (surface.cpp)
Handle<Semaphore> create_semaphore_internal(DeviceImpl* d, uint64_t init_value);
void              drain_deferred_frees(DeviceImpl* d, uint64_t current_timeline);

// Descriptor heap write (resources.cpp)
void write_sampled_descriptor(DeviceImpl* d, uint32_t slot, const VkImageViewCreateInfo& view_info);
void write_storage_descriptor(DeviceImpl* d, uint32_t slot, const VkImageViewCreateInfo& view_info);
void write_sampler_descriptor(DeviceImpl* d, uint32_t slot, const VkSamplerCreateInfo& sampler_info);

// Descriptor heap bind (commands.cpp, called from queue_start_command_recording)
void cmd_bind_descriptor_heaps(DeviceImpl* d, VkCommandBuffer cmd);

}  // namespace gpu
