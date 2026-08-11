#pragma once
// Internal header shared by the Vulkan backend TUs.
// Contains DeviceImpl, QueueImpl, CommandBufferImpl, and all backend-internal structs.

#include <atomic>

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
    VkBuffer        vk_buffer       = VK_NULL_HANDLE;
    VmaAllocation   vk_allocation   = VK_NULL_HANDLE;
    void*           host_ptr        = nullptr;   // mapped base (backing)
    VkDeviceMemory  memory          = VK_NULL_HANDLE;   // for flush/invalidate ranges
    VkDeviceSize    memory_offset   = 0;
    VkDeviceAddress backing_address = 0;   // device address of the VkBuffer
    VkDeviceSize    backing_size    = 0;
    GpuPtr          user_address    = 0;   // aligned address handed to the application
    VkDeviceSize    user_size       = 0;   // requested size (not the padded backing)
    VkDeviceSize    user_offset     = 0;   // user_address - backing_address
    bool            coherent        = true;
};

// Buffer/offset pair validated against the allocation's user-visible bounds.
struct BufferAndOffset {
    VkBuffer      buffer;
    VkDeviceSize  offset;   // VkBuffer-relative (64-bit)
    VmaAllocation alloc;
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
    // 1 per public handle + 1 per command-buffer/in-flight retention;
    // the pool slot (and native image) is destroyed at zero.
    int64_t        refs               = 1;   // atomic via platform helpers
};

struct DepthStencilState {
    DepthStencilDesc desc;
};

// Pipeline state machine (monotonic; never Ready -> non-Ready).
enum class InternalPipelineState : uint8_t { Queued, ProbingCache, Compiling, Ready, Failed };

// A pipeline record shared by every handle created from an identical
// description. Owns one contiguous key copy (key_block) that fully
// determines the VkPipeline. Compiled on a device-owned worker thread;
// destroyed when the last reference (user handle, compiler worker, or
// command-buffer/in-flight) is released. Dedup only — no retention beyond
// live references.
struct PipelineRecord {
    std::atomic<InternalPipelineState> state{InternalPipelineState::Queued};
    VkPipeline          vk_pipeline    = VK_NULL_HANDLE;  // published with Ready
    VkPipelineBindPoint bind_point     = VK_PIPELINE_BIND_POINT_COMPUTE;
    VkResult            failure_result = VK_SUCCESS;
    std::atomic<uint32_t> refs{0};  // user + worker + command-buffer/in-flight

    mutex   wait_mutex = IZ_MUTEX_INIT;  // only wait_pipeline() sleeps on these
    condvar wait_cv;

    MemoryBlock key_block = {};  // owned copy of all create inputs

    // --- key data (pointers into key_block) ---
    uint8_t* vs_bytes   = nullptr;
    uint32_t vs_size    = 0;
    char*    vs_entry   = nullptr;   // null-terminated
    uint8_t* fs_bytes   = nullptr;
    uint32_t fs_size    = 0;
    char*    fs_entry   = nullptr;
    uint8_t*  spec_data  = nullptr;
    uint32_t  spec_size  = 0;
    uint32_t* spec_ids   = nullptr;
    uint32_t* spec_sizes = nullptr;
    uint32_t  spec_count = 0;
    // raster state (graphics pipelines)
    Topology topology           = Topology::TriangleList;
    uint8_t  sample_count       = 1;
    bool     alpha_to_coverage  = false;
    Format   depth_format       = Format::None;
    Format   stencil_format     = Format::None;
    uint8_t  color_target_count = 0;
    ColorTarget* color_targets  = nullptr;
};

struct PipelineImpl {
    PipelineRecord* record = nullptr;
};

// Unified queue-timeline retirement: everything retired at a submission's
// completion (buffers, textures, pipeline references, descriptor slots) is a
// RetireItem in a value-keyed batch, processed by queue_process_events.
enum class RetireKind : uint8_t { Buffer, Texture, PipelineRef, SampledSlot, StorageSlot, SamplerSlot };

struct RetireItem {
    RetireKind kind;
    uint64_t   a;   // Buffer: buffer handle; Texture: handle.h; PipelineRef: record ptr; slots: index
    uint64_t   b;   // reserved
};

struct RetireBatch {
    uint64_t          value;
    Vector<RetireItem> items;
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
// Pools are retired by queue timeline value, never by presentation frame
// counters: a pool is reusable only when the queue's completed timeline is at
// least its retire_value (the last submission that used a command buffer from
// it). Headless workloads therefore work without presentation.
struct CommandPool {
    VkCommandPool   command_pool = VK_NULL_HANDLE;
    SegmentArray<CommandBufferImpl> command_buffers;
    int64_t         buffer_free_idx = 0;
    uint64_t        retire_value = 0;   // reusable when completed >= this
    uint32_t        outstanding = 0;    // recorded, not yet submitted
};

struct CommandSuperpool {
    static constexpr uint32_t kMaxSimultaneousCommands = 64;
    CommandPool pools[kMaxSimultaneousCommands] = {};
};

// --- Event (deferred completion callback) -------------------------------------------
struct CompletionEvent {
    uint64_t completed_time;
    void     (*callback)(void*);
    void*     userdata;
};

// --- GPU pointer map (address -> buffer/offset) -------------------------------------
struct GpuPtrMap {
    GpuPtr        ptr;    // user (aligned) address, sorted ascending
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
    bool               dual_src_blend  = false;   // VkPhysicalDeviceFeatures.dualSrcBlend

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

    // Pipelines: dedup map + persistent native cache + async compiler worker.
    // pipeline_lock covers the map (and record destruction); compiler_lock
    // covers the worker queue. VkPipelineCache is externally synchronized:
    // only the worker creates pipelines, and flush/destroy wait for it to
    // drain before touching cache data.
    mutex                    pipeline_lock = IZ_MUTEX_INIT;
    Vector<PipelineRecord*>  pipeline_records;
    VkPipelineCache          vk_pipeline_cache = VK_NULL_HANDLE;
    PipelineCacheCallbacks   cache_callbacks   = {};
    CacheIdentity            cache_identity    = {};
    bool                     pipeline_cache_control = false;

    // Async compiler worker (single thread, FIFO)
    mutex                     compiler_lock = IZ_MUTEX_INIT;
    condvar                   compiler_cv;
    Vector<PipelineRecord*>   compiler_queue;
    thread_handle             compiler_thread = 0;
    int64_t                   compiler_shutdown = 0;  // atomic flags
    int64_t                   compiler_busy     = 0;  // jobs currently compiling
    int64_t                   compiler_paused   = 0;  // test hook (under compiler_lock)
    int64_t                   device_destroying = 0;  // atomic: stop accepting requests
    uintptr_t                 compiler_thread_id = 0;

    // Compiler diagnostics counters (atomic, best-effort)
    int64_t stat_requests        = 0;
    int64_t stat_dedup_hits      = 0;
    int64_t stat_probe_hits      = 0;
    int64_t stat_compile_required = 0;
    int64_t stat_full_compiles   = 0;
    int64_t stat_failures        = 0;
    int64_t stat_max_queue_depth = 0;
    // Queue/submission diagnostics
    int64_t stat_submissions     = 0;
    int64_t stat_failed_submits  = 0;
    int64_t stat_pool_resets     = 0;
    // Test hooks
    int64_t force_submit_failure = 0;   // atomic: make queue_submit fail without submitting

    // Descriptor heap
    DescriptorHeap heap;
    TwoLevelBitset sampled_bitset;
    TwoLevelBitset storage_bitset;
    TwoLevelBitset sampler_bitset;

    // Memory limits (for host flush/invalidate range alignment)
    VkDeviceSize non_coherent_atom_size = 1;

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
    uint64_t         timeline_value = 0;   // last successfully submitted value
    // Serializes queue submission and presentation on this queue.
    mutex            submit_lock   = IZ_MUTEX_INIT;
    Vector<CompletionEvent> pending_events;
    // Unified retirement queue: value-keyed batches (sorted ascending) of
    // resources/pipeline references/descriptor slots retired when the queue
    // timeline passes their value. Drained by queue_process_events.
    Vector<RetireBatch*> retire_queue;
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

    // Pipelines bound by cmd_set_pipeline (Ready only); each holds one
    // reference. Textures explicitly named by commands (render attachments,
    // copy sources/destinations); each holds one reference. Both are moved to
    // retire batches at queue_submit, released at pool reset if never
    // submitted.
    Vector<PipelineRecord*> retained_pipelines;
    Vector<Handle<Texture>> retained_textures;
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
CommandPool*  get_command_pool(QueueImpl* queue);
CommandBuffer get_command_buffer(QueueImpl* q, CommandPool* pool);

// Surface helpers (surface.cpp)
Handle<Semaphore> create_semaphore_internal(DeviceImpl* d, uint64_t init_value);

// Descriptor heap write (resources.cpp)
void write_sampled_descriptor(DeviceImpl* d, uint32_t slot, const VkImageViewCreateInfo& view_info);
void write_storage_descriptor(DeviceImpl* d, uint32_t slot, const VkImageViewCreateInfo& view_info);
void write_sampler_descriptor(DeviceImpl* d, uint32_t slot, const VkSamplerCreateInfo& sampler_info);
void release_texture_ref(DeviceImpl* d, Handle<Texture> tex);

// White-box test hooks (not part of the public API)
uint32_t   debug_live_pipelines(DeviceImpl* d);         // records in the dedup map
uintptr_t  debug_last_compile_thread(DeviceImpl* d);    // thread id of the last native compile (0 = none)
void       debug_set_compiler_paused(DeviceImpl* d, bool paused);
void       debug_force_submit_failure(DeviceImpl* d, bool force);
uint64_t   debug_queue_timeline(DeviceImpl* d);         // last successfully submitted value
int64_t    debug_pool_resets(DeviceImpl* d);            // command-pool reuse resets

// pipeline.cpp internals used by commands.cpp / device.cpp
void release_pipeline_ref(DeviceImpl* d, PipelineRecord* rec);
void free_record(DeviceImpl* d, PipelineRecord* rec);
void process_retire_batch(DeviceImpl* d, RetireBatch* batch);
void compiler_worker_main(void* arg);
void store_pipeline_cache(DeviceImpl* d);

// Queue retirement (commands.cpp)
void enqueue_retire(QueueImpl* q, uint64_t value, const RetireItem& item);

// Descriptor heap bind (commands.cpp, called from queue_start_command_recording)
void cmd_bind_descriptor_heaps(DeviceImpl* d, VkCommandBuffer cmd);

}  // namespace gpu
