// pipeline.cpp — compute & graphics pipelines, depth-stencil state, specialization
// constants, and the asynchronous compiler worker.
//
// request_compute_pipeline / request_graphics_pipeline never block on native
// compilation: they deep-copy the description into an owned PipelineRecord,
// deduplicate against live records, enqueue the record on a device-owned
// compiler worker, and return immediately. All vkCreate*Pipelines calls run on
// the worker. Blocking create_*_pipeline are request + wait_pipeline wrappers.

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <thread>

#include "internal.h"

namespace gpu {

// --- Specialization constants ----------------------------------------------------------

static VkSpecializationInfo construct_specialization_info(
    Span<const SpecializationConstant> constants,
    Arena*                             arena) {
    Span<uint8_t> data{};
    const auto    as_byte_span = []<class T>(const T& x) -> Span<const uint8_t> {
        return Span<const T>(x).as_bytes();
    };

    for (const auto& c : constants) {
        switch (c.type) {
            case SpecializationConstantType::UInt8:
            case SpecializationConstantType::Int8:
                data = concat(arena, data, as_byte_span(static_cast<uint8_t>(c.int_val)));
                break;
            case SpecializationConstantType::UInt16:
            case SpecializationConstantType::Int16:
                data = concat(arena, data, as_byte_span(static_cast<uint16_t>(c.int_val)));
                break;
            case SpecializationConstantType::UInt32:
            case SpecializationConstantType::Int32:
                data = concat(arena, data, as_byte_span(static_cast<uint32_t>(c.int_val)));
                break;
            case SpecializationConstantType::Boolean:
                data = concat(arena, data, as_byte_span(c.bool_val ? VK_TRUE : VK_FALSE));
                break;
            case SpecializationConstantType::Float32:
                data = concat(arena, data, as_byte_span(c.float_val));
                break;
        }
    }

    Span<VkSpecializationMapEntry> map_entries{};
    uint32_t                       offset = 0;
    for (const auto& c : constants) {
        size_t size = 0;
        switch (c.type) {
            case SpecializationConstantType::UInt8:
            case SpecializationConstantType::Int8: size = sizeof(uint8_t); break;
            case SpecializationConstantType::UInt16:
            case SpecializationConstantType::Int16: size = sizeof(uint16_t); break;
            case SpecializationConstantType::UInt32:
            case SpecializationConstantType::Int32:
            case SpecializationConstantType::Boolean:
            case SpecializationConstantType::Float32: size = sizeof(uint32_t); break;
        }
        map_entries = concat(arena,
                             map_entries,
                             VkSpecializationMapEntry{
                                 .constantID = c.constant_id,
                                 .offset     = offset,
                                 .size       = size,
                             });
        offset += size;
    }

    return VkSpecializationInfo{
        .mapEntryCount = static_cast<uint32_t>(constants.size()),
        .pMapEntries   = map_entries.data(),
        .dataSize      = data.size(),
        .pData         = data.data(),
    };
}

// --- Formatted logging (arena-backed) ---------------------------------------------------

static void log_fmt(DeviceImpl* d, LogLevel lvl, uint32_t line, const char* file, const char* fmt, ...) {
    Arena* arena = get_thread_local_arena(d);
    char*  buf   = static_cast<char*>(arena->alloc(256));
    if (buf == nullptr) { return; }
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, 256, fmt, args);
    va_end(args);
    log_impl(d, lvl, Span<const char>(buf, strlen(buf)), line, Span<const char>(file, strlen(file)));
}

// --- Owned canonical keys ----------------------------------------------------------------
// The key is a full copy of every input that reaches pipeline creation — shader
// bytes + entry points, specialization data, and the entire baked raster state.
// No retention beyond live references.

static bool entry_equal(const char* stored, Span<const char> app) {
    const size_t n = strlen(stored);
    return n == app.size() && (n == 0 || memcmp(stored, app.data(), n) == 0);
}

static bool spec_equal(const PipelineRecord& rec, const VkSpecializationInfo& spec) {
    if (rec.spec_count != spec.mapEntryCount || rec.spec_size != spec.dataSize) { return false; }
    if (rec.spec_size && memcmp(rec.spec_data, spec.pData, rec.spec_size) != 0) { return false; }
    for (uint32_t i = 0; i < rec.spec_count; ++i) {
        if (rec.spec_ids[i] != spec.pMapEntries[i].constantID ||
            rec.spec_sizes[i] != spec.pMapEntries[i].size) {
            return false;
        }
    }
    return true;
}

static bool graphics_key_equal(const PipelineRecord& rec, ShaderSource vertex, ShaderSource fragment,
                               const VkSpecializationInfo& spec, const RasterDesc& desc) {
    if (rec.bind_point != VK_PIPELINE_BIND_POINT_GRAPHICS) { return false; }
    if (rec.vs_size != vertex.source.size() ||
        (rec.vs_size && memcmp(rec.vs_bytes, vertex.source.data(), rec.vs_size) != 0)) {
        return false;
    }
    if (rec.fs_size != fragment.source.size() ||
        (rec.fs_size && memcmp(rec.fs_bytes, fragment.source.data(), rec.fs_size) != 0)) {
        return false;
    }
    if (!entry_equal(rec.vs_entry, vertex.entry_point) ||
        !entry_equal(rec.fs_entry, fragment.entry_point)) {
        return false;
    }
    if (!spec_equal(rec, spec)) { return false; }
    if (rec.topology != desc.topology || rec.polygon_mode != desc.polygon_mode ||
        rec.sample_count != desc.sample_count ||
        rec.alpha_to_coverage != desc.alpha_to_coverage ||
        rec.depth_format != desc.depth_format || rec.stencil_format != desc.stencil_format ||
        rec.color_target_count != desc.color_targets.size()) {
        return false;
    }
    for (uint32_t i = 0; i < rec.color_target_count; ++i) {
        const ColorTarget& a = rec.color_targets[i];
        const ColorTarget& b = desc.color_targets[i];
        if (a.format != b.format) { return false; }
        const BlendDesc& ba = a.blendstate;
        const BlendDesc& bb = b.blendstate;
        if (ba.color_op != bb.color_op || ba.src_color_factor != bb.src_color_factor ||
            ba.dst_color_factor != bb.dst_color_factor || ba.alpha_op != bb.alpha_op ||
            ba.src_alpha_factor != bb.src_alpha_factor || ba.dst_alpha_factor != bb.dst_alpha_factor ||
            ba.color_write_mask != bb.color_write_mask) {
            return false;
        }
    }
    return true;
}

static bool compute_key_equal(const PipelineRecord& rec, ShaderSource source,
                              const VkSpecializationInfo& spec) {
    if (rec.bind_point != VK_PIPELINE_BIND_POINT_COMPUTE) { return false; }
    if (rec.vs_size != source.source.size() ||
        (rec.vs_size && memcmp(rec.vs_bytes, source.source.data(), rec.vs_size) != 0)) {
        return false;
    }
    if (!entry_equal(rec.vs_entry, source.entry_point)) { return false; }
    return spec_equal(rec, spec);
}

// Frees every allocation owned by a record's canonical key.
void free_key(DeviceImpl* d, PipelineRecord* rec) {
    if (rec->key_block.ptr != nullptr) { d->allocator.free(rec->key_block); }
    if (rec->spec_ids_block.ptr != nullptr) { d->allocator.free(rec->spec_ids_block); }
    if (rec->spec_sizes_block.ptr != nullptr) { d->allocator.free(rec->spec_sizes_block); }
    if (rec->color_targets_block.ptr != nullptr) { d->allocator.free(rec->color_targets_block); }
    rec->key_block = {};
    rec->spec_ids_block = {};
    rec->spec_sizes_block = {};
    rec->color_targets_block = {};
}

// Copies every create input into owned storage inside `rec`. Byte/string data
// lives in one blob; typed arrays (spec ids/sizes, color targets) are
// allocated separately so they are always naturally aligned. Returns false on
// allocation failure (caller must destroy the record and bail).
static bool build_key(DeviceImpl* d, PipelineRecord* rec, ShaderSource vertex, ShaderSource fragment,
                      const VkSpecializationInfo& spec, const RasterDesc* desc) {
    const uint32_t ct_count = desc ? static_cast<uint32_t>(desc->color_targets.size()) : 0;
    size_t total = 0;
    total += vertex.source.size() + vertex.entry_point.size() + 1;
    total += fragment.source.size() + fragment.entry_point.size() + 1;
    total += spec.dataSize;
    if (total == 0) { total = 1; }

    MemoryBlock blk = d->allocator.alloc(total);
    if (blk.ptr == nullptr) { return false; }
    MemoryBlock ids_blk = d->allocator.alloc(spec.mapEntryCount * sizeof(uint32_t));
    MemoryBlock sizes_blk = d->allocator.alloc(spec.mapEntryCount * sizeof(uint32_t));
    MemoryBlock targets_blk = d->allocator.alloc(ct_count * sizeof(ColorTarget));
    if ((spec.mapEntryCount > 0 && ids_blk.ptr == nullptr) ||
        (spec.mapEntryCount > 0 && sizes_blk.ptr == nullptr) ||
        (ct_count > 0 && targets_blk.ptr == nullptr)) {
        d->allocator.free(blk);
        if (ids_blk.ptr != nullptr) { d->allocator.free(ids_blk); }
        if (sizes_blk.ptr != nullptr) { d->allocator.free(sizes_blk); }
        if (targets_blk.ptr != nullptr) { d->allocator.free(targets_blk); }
        return false;
    }
    rec->key_block = blk;
    rec->spec_ids_block = ids_blk;
    rec->spec_sizes_block = sizes_blk;
    rec->color_targets_block = targets_blk;

    uint8_t* p = static_cast<uint8_t*>(blk.ptr);
    auto take_bytes = [&p](const void* src, size_t n) -> uint8_t* {
        uint8_t* out = p;
        if (n) { memcpy(p, src, n); }
        p += n;
        return out;
    };
    auto take_str = [&p](Span<const char> str) -> char* {
        char* out = reinterpret_cast<char*>(p);
        if (str.size()) { memcpy(p, str.data(), str.size()); }
        p += str.size();
        *p = '\0';
        p += 1;
        return out;
    };

    rec->vs_size  = static_cast<uint32_t>(vertex.source.size());
    rec->vs_bytes = take_bytes(vertex.source.data(), vertex.source.size());
    rec->vs_entry = take_str(vertex.entry_point);
    rec->fs_size  = static_cast<uint32_t>(fragment.source.size());
    rec->fs_bytes = take_bytes(fragment.source.data(), fragment.source.size());
    rec->fs_entry = take_str(fragment.entry_point);

    rec->spec_size = static_cast<uint32_t>(spec.dataSize);
    rec->spec_data = take_bytes(spec.pData, spec.dataSize);
    rec->spec_count = spec.mapEntryCount;
    uint32_t* ids = static_cast<uint32_t*>(ids_blk.ptr);
    uint32_t* sizes = static_cast<uint32_t*>(sizes_blk.ptr);
    for (uint32_t i = 0; i < spec.mapEntryCount; ++i) {
        ids[i]   = spec.pMapEntries[i].constantID;
        sizes[i] = spec.pMapEntries[i].size;
    }
    rec->spec_ids   = ids;
    rec->spec_sizes = sizes;

    if (desc) {
        rec->topology          = desc->topology;
        rec->polygon_mode       = desc->polygon_mode;
        rec->sample_count      = desc->sample_count;
        rec->alpha_to_coverage = desc->alpha_to_coverage;
        rec->depth_format      = desc->depth_format;
        rec->stencil_format    = desc->stencil_format;
        rec->color_target_count = ct_count;
        ColorTarget* cts = static_cast<ColorTarget*>(targets_blk.ptr);
        for (uint32_t i = 0; i < ct_count; ++i) { cts[i] = desc->color_targets[i]; }
        rec->color_targets = cts;
    }
    return true;
}

static PipelineRecord* alloc_record(DeviceImpl* d) {
    MemoryBlock blk = d->allocator.alloc(sizeof(PipelineRecord));
    if (blk.ptr == nullptr) { return nullptr; }
    auto* rec = ::new (blk.ptr) PipelineRecord();
    condvar_init(&rec->wait_cv);
    rec->stable_id = static_cast<uint64_t>(atomic_fetch_add(&d->next_pipeline_stable_id, 1));
    return rec;
}

void free_record(DeviceImpl* d, PipelineRecord* rec) {
    condvar_destroy(&rec->wait_cv);
    rec->~PipelineRecord();
    d->allocator.free({.ptr = rec, .len = sizeof(PipelineRecord)});
}

// --- Worker-side compilation --------------------------------------------------------------

// Rebuilds VkSpecializationInfo from the record's owned data (offsets are the
// cumulative sizes, matching construct_specialization_info).
static VkSpecializationInfo rebuild_spec_info(Arena* arena, const PipelineRecord& rec) {
    Span<VkSpecializationMapEntry> entries{};
    uint32_t                       offset = 0;
    for (uint32_t i = 0; i < rec.spec_count; ++i) {
        entries = concat(arena, entries,
                         VkSpecializationMapEntry{
                             .constantID = rec.spec_ids[i],
                             .offset     = offset,
                             .size       = rec.spec_sizes[i],
                         });
        offset += rec.spec_sizes[i];
    }
    return VkSpecializationInfo{
        .mapEntryCount = rec.spec_count,
        .pMapEntries   = entries.data(),
        .dataSize      = rec.spec_size,
        .pData         = rec.spec_data,
    };
}

// SPIR-V pre-validation. Drivers are not required to survive modules that
// violate the pName VUIDs (Turnip faults inside its NIR lowering on a
// missing entry point; dzn traps on parse failures). Validate the header and
// the requested entry point on the worker before the driver ever sees the
// module, so invalid shaders retire deterministically as
// PipelineStatus::Failed on every driver.
static uint32_t spirv_word(const uint8_t* bytes, uint32_t index) {
    uint32_t w;
    memcpy(&w, bytes + index * 4u, 4);  // module bytes are not 4-aligned in key_block
    return w;
}

static bool spirv_has_entry_point(const uint8_t* bytes, uint32_t size, const char* entry) {
    if (bytes == nullptr || entry == nullptr) { return false; }
    if (size < 20 || (size % 4u) != 0) { return false; }
    const uint32_t count = size / 4u;
    if (spirv_word(bytes, 0) != 0x07230203u) { return false; }
    uint32_t i = 5;
    while (i < count) {
        const uint32_t inst = spirv_word(bytes, i);
        const uint32_t len  = inst >> 16;
        const uint32_t op   = inst & 0xFFFFu;
        if (len == 0 || i + len > count) { return false; }  // malformed stream
        if (op == 15u /*OpEntryPoint*/ && len >= 4) {
            // Literal name starts at operand word 3 and is nul-terminated
            // within the instruction.
            const uint8_t* name      = bytes + (i + 3u) * 4u;
            const uint32_t max_bytes = (len - 3u) * 4u;
            uint32_t j = 0;
            while (j < max_bytes && name[j] != '\0' && entry[j] != '\0' && name[j] == entry[j]) { ++j; }
            if (j < max_bytes && name[j] == '\0' && entry[j] == '\0') { return true; }
        }
        i += len;
    }
    return false;
}

static VkResult compile_compute(DeviceImpl* d, Arena* arena, PipelineRecord* rec,
                                bool fail_on_compile, VkPipeline* out) {
    if (!spirv_has_entry_point(rec->vs_bytes, rec->vs_size, rec->vs_entry)) {
        IZ_LOG(d, LogLevel::Error, "compile_compute: SPIR-V module invalid or entry point not found");
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    VkSpecializationInfo spec = rebuild_spec_info(arena, *rec);

    VkShaderModuleCreateInfo module_info{
        .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext    = nullptr,
        .flags    = 0,
        .codeSize = rec->vs_size,
        .pCode    = reinterpret_cast<const uint32_t*>(rec->vs_bytes),
    };
    // maintenance5 lets the stage reference shader code directly via pNext
    // (module = VK_NULL_HANDLE) — native profile only. The bindless profile
    // (1.3) creates a real VkShaderModule and destroys it after creation.
#if defined(IZ_VK_PROFILE_BINDLESS)
    VkShaderModule shader_module = VK_NULL_HANDLE;
    if (!IZ_CHK(d, vkCreateShaderModule(d->device, &module_info, nullptr, &shader_module),
                "compile_compute: vkCreateShaderModule failed")) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
#endif
    VkPipelineShaderStageCreateInfo stage{
        .sType               = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
#if defined(IZ_VK_PROFILE_BINDLESS)
        .pNext               = nullptr,
#else
        .pNext               = &module_info,
#endif
        .flags               = 0,
        .stage               = VK_SHADER_STAGE_COMPUTE_BIT,
#if defined(IZ_VK_PROFILE_BINDLESS)
        .module              = shader_module,
#else
        .module              = VK_NULL_HANDLE,
#endif
        .pName               = rec->vs_entry,
        .pSpecializationInfo = &spec,
    };

    // flags2 (VK_PIPELINE_CREATE_FLAGS_2_*) requires maintenance5 / Vulkan
    // 1.4 — native profile only. The bindless profile (1.3) uses the legacy
    // create flags + direct pNext.
#if !defined(IZ_VK_PROFILE_BINDLESS)
    VkPipelineCreateFlags2CreateInfo flags2{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT |
                 (fail_on_compile ? VK_PIPELINE_CREATE_2_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT : 0),
    };
#endif
    VkComputePipelineCreateInfo info{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
#if defined(IZ_VK_PROFILE_BINDLESS)
        .pNext = nullptr,
        .flags = static_cast<VkPipelineCreateFlags>(fail_on_compile
                                                       ? VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT
                                                       : 0),
#else
        .pNext = &flags2,
        .flags = 0,
#endif
        .stage = stage,
#if defined(IZ_VK_PROFILE_BINDLESS)
        .layout             = d->bindless_pipeline_layout,
#else
        .layout             = VK_NULL_HANDLE,
#endif
        .basePipelineHandle = VK_NULL_HANDLE,
        .basePipelineIndex  = 0,
    };
    if (arena->overflowed()) {
#if defined(IZ_VK_PROFILE_BINDLESS)
        vkDestroyShaderModule(d->device, shader_module, nullptr);
#endif
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    VkResult result = vkCreateComputePipelines(d->device, d->vk_pipeline_cache, 1, &info, nullptr, out);
#if defined(IZ_VK_PROFILE_BINDLESS)
    vkDestroyShaderModule(d->device, shader_module, nullptr);
#endif
    return result;
}

static VkResult compile_graphics(DeviceImpl* d, Arena* arena, PipelineRecord* rec,
                                 bool fail_on_compile, const LogicalGraphicsState* baked,
                                 VkPipeline* out) {
    // baked == nullptr  -> base pipeline (defaults on the static path; the
    // full extended-dynamic-state set on the dynamic path).
    // baked != nullptr  -> a private static variant (static path only).
    const bool static_state = d->dispatch.use_static_graphics_state;
    LogicalGraphicsState defaults;
    const LogicalGraphicsState& gs = (baked != nullptr) ? *baked : defaults;

    if (!spirv_has_entry_point(rec->vs_bytes, rec->vs_size, rec->vs_entry) ||
        !spirv_has_entry_point(rec->fs_bytes, rec->fs_size, rec->fs_entry)) {
        IZ_LOG(d, LogLevel::Error, "compile_graphics: SPIR-V module invalid or entry point not found");
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    // Depth and stencil formats must be a single combined format (or one
    // absent): dynamic rendering forbids two distinct depth/stencil images.
    if (rec->depth_format != Format::None && rec->stencil_format != Format::None &&
        rec->depth_format != rec->stencil_format) {
        IZ_LOG(d, LogLevel::Error, "compile_graphics: depth and stencil formats must match (combined format) or one must be None");
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    VkSpecializationInfo spec = rebuild_spec_info(arena, *rec);

    VkShaderModuleCreateInfo vert_module_info{
        .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext    = nullptr,
        .flags    = 0,
        .codeSize = rec->vs_size,
        .pCode    = reinterpret_cast<const uint32_t*>(rec->vs_bytes),
    };
    VkShaderModuleCreateInfo frag_module_info{
        .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext    = nullptr,
        .flags    = 0,
        .codeSize = rec->fs_size,
        .pCode    = reinterpret_cast<const uint32_t*>(rec->fs_bytes),
    };
#if defined(IZ_VK_PROFILE_BINDLESS)
    // 1.3/no maintenance5: real shader modules, destroyed after creation.
    VkShaderModule vert_module = VK_NULL_HANDLE;
    VkShaderModule frag_module = VK_NULL_HANDLE;
    if (!IZ_CHK(d, vkCreateShaderModule(d->device, &vert_module_info, nullptr, &vert_module),
                "compile_graphics: vkCreateShaderModule(vert) failed") ||
        !IZ_CHK(d, vkCreateShaderModule(d->device, &frag_module_info, nullptr, &frag_module),
                "compile_graphics: vkCreateShaderModule(frag) failed")) {
        if (vert_module != VK_NULL_HANDLE) { vkDestroyShaderModule(d->device, vert_module, nullptr); }
        return VK_ERROR_INITIALIZATION_FAILED;
    }
#endif
    VkPipelineShaderStageCreateInfo stages[] = {
        {
            .sType               = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
#if defined(IZ_VK_PROFILE_BINDLESS)
            .pNext               = nullptr,
#else
            .pNext               = &vert_module_info,
#endif
            .flags               = 0,
            .stage               = VK_SHADER_STAGE_VERTEX_BIT,
#if defined(IZ_VK_PROFILE_BINDLESS)
            .module              = vert_module,
#else
            .module              = VK_NULL_HANDLE,
#endif
            .pName               = rec->vs_entry,
            .pSpecializationInfo = &spec,
        },
        {
            .sType               = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
#if defined(IZ_VK_PROFILE_BINDLESS)
            .pNext               = nullptr,
#else
            .pNext               = &frag_module_info,
#endif
            .flags               = 0,
            .stage               = VK_SHADER_STAGE_FRAGMENT_BIT,
#if defined(IZ_VK_PROFILE_BINDLESS)
            .module              = frag_module,
#else
            .module              = VK_NULL_HANDLE,
#endif
            .pName               = rec->fs_entry,
            .pSpecializationInfo = &spec,
        },
    };

    VkPipelineVertexInputStateCreateInfo vertex_input_state{
        .sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .pNext                           = nullptr,
        .flags                           = 0,
        .vertexBindingDescriptionCount   = 0,
        .pVertexBindingDescriptions      = nullptr,
        .vertexAttributeDescriptionCount = 0,
        .pVertexAttributeDescriptions    = nullptr,
    };

    VkPipelineInputAssemblyStateCreateInfo input_assembly_state{
        .sType                  = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .pNext                  = nullptr,
        .flags                  = 0,
        .topology               = bridge(rec->topology),
        .primitiveRestartEnable = false,
    };

    if (rec->polygon_mode == PolygonMode::Line && !d->non_solid_fill) {
        IZ_LOG(d, LogLevel::Error,
               "create_graphics_pipeline: line polygon mode requested but the "
               "device does not support non-solid fill");
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }
    if ((d->framebuffer_sample_counts & rec->sample_count) == 0) {
        IZ_LOG(d, LogLevel::Error,
               "create_graphics_pipeline: unsupported framebuffer sample count");
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }

    // Color blend + attachment formats. Dual-source factors are optional:
    // reject deterministically when unsupported (no silent fallback).
    const bool needs_dual_src = [&]() {
        for (uint32_t i = 0; i < rec->color_target_count; ++i) {
            const auto& b = rec->color_targets[i].blendstate;
            if (b.src_color_factor == Factor::Src1Color || b.src_color_factor == Factor::OneMinusSrc1Color ||
                b.dst_color_factor == Factor::Src1Color || b.dst_color_factor == Factor::OneMinusSrc1Color ||
                b.src_alpha_factor == Factor::Src1Color || b.src_alpha_factor == Factor::OneMinusSrc1Color ||
                b.dst_alpha_factor == Factor::Src1Color || b.dst_alpha_factor == Factor::OneMinusSrc1Color ||
                b.src_color_factor == Factor::Src1Alpha || b.src_color_factor == Factor::OneMinusSrc1Alpha ||
                b.dst_color_factor == Factor::Src1Alpha || b.dst_color_factor == Factor::OneMinusSrc1Alpha ||
                b.src_alpha_factor == Factor::Src1Alpha || b.src_alpha_factor == Factor::OneMinusSrc1Alpha ||
                b.dst_alpha_factor == Factor::Src1Alpha || b.dst_alpha_factor == Factor::OneMinusSrc1Alpha) {
                return true;
            }
        }
        return false;
    }();
    if (needs_dual_src && !d->dual_src_blend) {
        IZ_LOG(d, LogLevel::Error,
               "create_graphics_pipeline: dual-source blend factors requested but the "
               "device does not support dualSrcBlend");
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }

    Span<VkPipelineColorBlendAttachmentState> color_blend_states{};
    Span<VkFormat>                            color_formats{};
    for (uint32_t i = 0; i < rec->color_target_count; ++i) {
        color_blend_states = concat(arena, color_blend_states, bridge(rec->color_targets[i].blendstate));
        color_formats      = concat(arena, color_formats, bridge(rec->color_targets[i].format));
    }

    VkPipelineColorBlendStateCreateInfo color_blend_state{
        .sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .pNext           = nullptr,
        .flags           = 0,
        .logicOpEnable   = false,
        .logicOp         = VK_LOGIC_OP_NO_OP,
        .attachmentCount = static_cast<uint32_t>(color_blend_states.size()),
        .pAttachments    = color_blend_states.data(),
        .blendConstants  = {1.f, 1.f, 1.f, 1.f},
    };

    VkPipelineRasterizationStateCreateInfo rasterization_state{
        .sType                   = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .pNext                   = nullptr,
        .flags                   = 0,
        .depthClampEnable        = false,
        .rasterizerDiscardEnable = false,
        .polygonMode             = rec->polygon_mode == PolygonMode::Line
                                       ? VK_POLYGON_MODE_LINE
                                       : VK_POLYGON_MODE_FILL,
        // Static path: front face + cull are baked (not dynamic).
        .cullMode                = static_state ? bridge(gs.cull) : VK_CULL_MODE_BACK_BIT,
        .frontFace               = static_state ? (gs.front_face == FrontFace::CCW
                                                       ? VK_FRONT_FACE_COUNTER_CLOCKWISE
                                                       : VK_FRONT_FACE_CLOCKWISE)
                                                : VK_FRONT_FACE_COUNTER_CLOCKWISE,
        .depthBiasEnable         = false,
        .depthBiasConstantFactor = 0,
        .depthBiasClamp          = 0,
        .depthBiasSlopeFactor    = 0,
        .lineWidth               = 1.f,
    };

    VkPipelineMultisampleStateCreateInfo multisample_state{
        .sType                 = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .pNext                 = nullptr,
        .flags                 = 0,
        .rasterizationSamples  = static_cast<VkSampleCountFlagBits>(rec->sample_count == 0 ? 1 : rec->sample_count),
        .sampleShadingEnable   = false,
        .minSampleShading      = 1.0f,
        .pSampleMask           = nullptr,
        .alphaToCoverageEnable = rec->alpha_to_coverage,
        .alphaToOneEnable      = false,
    };

    VkDynamicState dynamic_states[] = {
        VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE,
        VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE,
        VK_DYNAMIC_STATE_DEPTH_COMPARE_OP,
        VK_DYNAMIC_STATE_DEPTH_BOUNDS_TEST_ENABLE,
        VK_DYNAMIC_STATE_DEPTH_BIAS,
        VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE,
        VK_DYNAMIC_STATE_STENCIL_OP,
        VK_DYNAMIC_STATE_DEPTH_BOUNDS,
        VK_DYNAMIC_STATE_VIEWPORT_WITH_COUNT,
        VK_DYNAMIC_STATE_SCISSOR_WITH_COUNT,
        VK_DYNAMIC_STATE_STENCIL_REFERENCE,
        VK_DYNAMIC_STATE_CULL_MODE,
        VK_DYNAMIC_STATE_FRONT_FACE,
        VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK,
        VK_DYNAMIC_STATE_STENCIL_WRITE_MASK,
    };
    // Core-dynamic only: plain viewport/scissor + depth bias + stencil
    // reference/masks + depth bounds values. The baked members are static in
    // the pipeline state.
    VkDynamicState static_dynamic_states[] = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
        VK_DYNAMIC_STATE_DEPTH_BIAS,
        VK_DYNAMIC_STATE_DEPTH_BOUNDS,
        VK_DYNAMIC_STATE_STENCIL_REFERENCE,
        VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK,
        VK_DYNAMIC_STATE_STENCIL_WRITE_MASK,
    };
    const VkDynamicState* selected_dynamic_states = dynamic_states;
    uint32_t dynamic_state_count = static_cast<uint32_t>(sizeof(dynamic_states) / sizeof(VkDynamicState));
    if (static_state) {
        selected_dynamic_states = static_dynamic_states;
        dynamic_state_count     = static_cast<uint32_t>(sizeof(static_dynamic_states) / sizeof(VkDynamicState));
    }

    // With the plain (non-WithCount) dynamic viewport/scissor states on the
    // static path, the pipeline must declare the viewport/scissor count the
    // draw-time commands will set (one each; the WithCount path uses 0).
    VkPipelineViewportStateCreateInfo viewport_state{
        .sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .pNext         = nullptr,
        .flags         = 0,
        .viewportCount = static_state ? 1u : 0u,
        .scissorCount  = static_state ? 1u : 0u,
    };

    VkPipelineDynamicStateCreateInfo dynamic_state{
        .sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .pNext             = nullptr,
        .flags             = 0,
        .dynamicStateCount = dynamic_state_count,
        .pDynamicStates    = selected_dynamic_states,
    };

    // Static path: depth/stencil state is baked (compare ops + stencil ops),
    // with the reference/masks supplied dynamically. Defaults mirror the
    // public API: depth/stencil disabled, Always compare, Keep ops.
    VkPipelineDepthStencilStateCreateInfo depth_stencil_state{
        .sType                 = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .pNext                 = nullptr,
        .flags                 = 0,
        .depthTestEnable       = static_state && gs.depth_test_enable,
        .depthWriteEnable      = static_state && gs.depth_write_enable,
        .depthCompareOp        = bridge(gs.depth_compare),
        .depthBoundsTestEnable = static_state && gs.depth_bounds_test_enable,
        .stencilTestEnable     = static_state && gs.stencil_test_enable,
        .front                 = {.failOp      = bridge(gs.stencil_front.fail_op),
                                  .passOp      = bridge(gs.stencil_front.pass_op),
                                  .depthFailOp = bridge(gs.stencil_front.depth_fail_op),
                                  .compareOp   = bridge(gs.stencil_front.test),
                                  .compareMask = 0xff,
                                  .writeMask   = 0xff,
                                  .reference   = 0},
        .back                  = {.failOp      = bridge(gs.stencil_back.fail_op),
                                  .passOp      = bridge(gs.stencil_back.pass_op),
                                  .depthFailOp = bridge(gs.stencil_back.depth_fail_op),
                                  .compareOp   = bridge(gs.stencil_back.test),
                                  .compareMask = 0xff,
                                  .writeMask   = 0xff,
                                  .reference   = 0},
        .minDepthBounds         = 0.f,
        .maxDepthBounds         = 1.f,
    };

    VkFormat depth_format   = bridge(rec->depth_format);
    VkFormat stencil_format = bridge(rec->stencil_format);

    VkPipelineRenderingCreateInfo rendering_info{
        .sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .pNext                   = nullptr,
        .viewMask                = 0,
        .colorAttachmentCount    = static_cast<uint32_t>(color_formats.size()),
        .pColorAttachmentFormats = color_formats.data(),
        .depthAttachmentFormat   = depth_format,
        .stencilAttachmentFormat = stencil_format,
    };

#if !defined(IZ_VK_PROFILE_BINDLESS)
    VkPipelineCreateFlags2CreateInfo flags2{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO,
        .pNext = &rendering_info,
        .flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT |
                 (fail_on_compile ? VK_PIPELINE_CREATE_2_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT : 0),
    };
#endif

    VkGraphicsPipelineCreateInfo create_info{
        .sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
#if defined(IZ_VK_PROFILE_BINDLESS)
        .pNext               = &rendering_info,
        .flags               = static_cast<VkPipelineCreateFlags>(fail_on_compile
                                                                     ? VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT
                                                                     : 0),
#else
        .pNext               = &flags2,
        .flags               = 0,
#endif
        .stageCount          = 2,
        .pStages             = stages,
        .pVertexInputState   = &vertex_input_state,
        .pInputAssemblyState = &input_assembly_state,
        .pTessellationState  = nullptr,
        .pViewportState      = &viewport_state,
        .pRasterizationState = &rasterization_state,
        .pMultisampleState   = &multisample_state,
        .pDepthStencilState  = static_state ? &depth_stencil_state : nullptr,
        .pColorBlendState    = &color_blend_state,
        .pDynamicState       = &dynamic_state,
#if defined(IZ_VK_PROFILE_BINDLESS)
        .layout              = d->bindless_pipeline_layout,
#else
        .layout              = VK_NULL_HANDLE,
#endif
        .renderPass          = VK_NULL_HANDLE,
        .subpass             = 0,
        .basePipelineHandle  = VK_NULL_HANDLE,
        .basePipelineIndex   = 0,
    };
    if (arena->overflowed()) {
#if defined(IZ_VK_PROFILE_BINDLESS)
        vkDestroyShaderModule(d->device, vert_module, nullptr);
        vkDestroyShaderModule(d->device, frag_module, nullptr);
#endif
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    VkResult result = vkCreateGraphicsPipelines(d->device, d->vk_pipeline_cache, 1, &create_info, nullptr, out);
#if defined(IZ_VK_PROFILE_BINDLESS)
    vkDestroyShaderModule(d->device, vert_module, nullptr);
    vkDestroyShaderModule(d->device, frag_module, nullptr);
#endif
    return result;
}

static VkResult compile_record(DeviceImpl* d, Arena* arena, PipelineRecord* rec,
                               bool fail_on_compile, VkPipeline* out) {
    return rec->bind_point == VK_PIPELINE_BIND_POINT_GRAPHICS
               ? compile_graphics(d, arena, rec, fail_on_compile, /*baked=*/nullptr, out)
               : compile_compute(d, arena, rec, fail_on_compile, out);
}

// Compiles one record on the worker and publishes Ready or Failed (monotonic).
static void process_record(DeviceImpl* d, PipelineRecord* rec) {
    Arena*      arena = get_thread_local_arena(d);
    ScratchScope scope(*arena);
    const bool  probing = d->pipeline_cache_control;
    const double t0 = monotonic_seconds();
    VkPipeline  pipeline = VK_NULL_HANDLE;
    VkResult    result;

    rec->state.store(probing ? InternalPipelineState::ProbingCache
                             : InternalPipelineState::Compiling,
                     std::memory_order_relaxed);
    if (arena->overflowed()) {
        result = VK_ERROR_OUT_OF_HOST_MEMORY;   // scratch exhausted before compile
    } else if (probing) {
        // Cache-only probe. VK_PIPELINE_COMPILE_REQUIRED is expected control
        // flow, not an error: it means the driver would compile, so run the
        // real create (without the fail-on-compile bit).
        result = compile_record(d, arena, rec, /*fail_on_compile=*/true, &pipeline);
        if (result == VK_PIPELINE_COMPILE_REQUIRED) {
            atomic_fetch_add(&d->stat_compile_required, 1);
            rec->state.store(InternalPipelineState::Compiling, std::memory_order_relaxed);
            result = compile_record(d, arena, rec, /*fail_on_compile=*/false, &pipeline);
        } else if (result == VK_SUCCESS) {
            atomic_fetch_add(&d->stat_probe_hits, 1);
        }
    } else {
        result = compile_record(d, arena, rec, /*fail_on_compile=*/false, &pipeline);
    }

    const double elapsed_ms = (monotonic_seconds() - t0) * 1000.0;
    if (result == VK_SUCCESS) {
        atomic_fetch_add(&d->stat_full_compiles, 1);
        rec->vk_pipeline = pipeline;   // publish before the Ready store (release)
        rec->state.store(InternalPipelineState::Ready, std::memory_order_release);
        log_fmt(d, LogLevel::Info, __LINE__, "pipeline.cpp",
                "compiled %s pipeline in %.2f ms",
                rec->bind_point == VK_PIPELINE_BIND_POINT_GRAPHICS ? "graphics" : "compute",
                elapsed_ms);
    } else {
        atomic_fetch_add(&d->stat_failures, 1);
        rec->failure_result = result;
        rec->state.store(InternalPipelineState::Failed, std::memory_order_release);
        log_vk_impl(d, result, "pipeline compilation failed", __LINE__, "pipeline.cpp"_sv);
    }

    mutex_lock(&rec->wait_mutex);
    condvar_broadcast(&rec->wait_cv);
    mutex_unlock(&rec->wait_mutex);
}

void compiler_worker_main(void* arg) {
    auto* d = static_cast<DeviceImpl*>(arg);
    d->compiler_thread_id = current_thread_id();
    for (;;) {
        PipelineRecord*     rec  = nullptr;
        StaticVariantRecord* vrec = nullptr;
        mutex_lock(&d->compiler_lock);
        while (!atomic_load(&d->compiler_shutdown) &&
               ((d->compiler_queue.is_empty() && d->variant_queue.is_empty()) ||
                atomic_load(&d->compiler_paused))) {
            condvar_wait(&d->compiler_cv, &d->compiler_lock);
        }
        if (d->variant_queue.is_empty() && d->compiler_queue.is_empty() &&
            atomic_load(&d->compiler_shutdown)) {
            mutex_unlock(&d->compiler_lock);
            break;
        }
        // Static variants first (they gate the draw path), then base records.
        if (!d->variant_queue.is_empty()) {
            vrec = d->variant_queue[0];
            d->variant_queue.erase(d->variant_queue.begin(), d->variant_queue.begin() + 1);
        } else {
            rec = d->compiler_queue[0];
            d->compiler_queue.erase(d->compiler_queue.begin(), d->compiler_queue.begin() + 1);
        }
        atomic_fetch_add(&d->compiler_busy, 1);
        mutex_unlock(&d->compiler_lock);

        if (vrec != nullptr) {
            process_static_variant(d, vrec);
        } else {
            process_record(d, rec);
            release_pipeline_ref(d, rec);   // the worker's reference
        }

        mutex_lock(&d->compiler_lock);
        atomic_fetch_add(&d->compiler_busy, -1);
        condvar_broadcast(&d->compiler_cv);   // wake flush_pipeline_cache waiters
        mutex_unlock(&d->compiler_lock);
    }
}

// --- Private static graphics variants -------------------------------------------------

// Normalized key from the baked shadow members + the base pipeline's stable
// identity.
static StaticVariantKey make_variant_key(DeviceImpl* d, uint64_t base_stable_id,
                                          const LogicalGraphicsState& gs) {
    return StaticVariantKey{
        .base_stable_id = base_stable_id,
        .impl_version   = kStaticVariantImplVersion,
        .profile_id     = static_cast<uint32_t>(device_backend_profile()),
        .front_face     = static_cast<uint8_t>(gs.front_face == FrontFace::CCW ? 0u : 1u),
        .cull           = static_cast<uint8_t>(gs.cull),
        .depth_test_enable      = gs.depth_test_enable,
        .depth_write_enable     = gs.depth_write_enable,
        .depth_compare          = static_cast<uint8_t>(gs.depth_compare),
        .depth_bounds_test_enable = gs.depth_bounds_test_enable,
        .stencil_test_enable    = gs.stencil_test_enable,
        .stencil_front_test     = static_cast<uint8_t>(gs.stencil_front.test),
        .stencil_front_fail     = static_cast<uint8_t>(gs.stencil_front.fail_op),
        .stencil_front_pass     = static_cast<uint8_t>(gs.stencil_front.pass_op),
        .stencil_front_depth_fail = static_cast<uint8_t>(gs.stencil_front.depth_fail_op),
        .stencil_back_test      = static_cast<uint8_t>(gs.stencil_back.test),
        .stencil_back_fail      = static_cast<uint8_t>(gs.stencil_back.fail_op),
        .stencil_back_pass      = static_cast<uint8_t>(gs.stencil_back.pass_op),
        .stencil_back_depth_fail = static_cast<uint8_t>(gs.stencil_back.depth_fail_op),
    };
}

// Rebuilds a LogicalGraphicsState from a normalized key (variant worker side).
static LogicalGraphicsState state_from_key(const StaticVariantKey& k) {
    LogicalGraphicsState gs;
    gs.front_face = k.front_face == 0 ? FrontFace::CCW : FrontFace::CW;
    gs.cull       = static_cast<Cull>(k.cull);
    gs.depth_test_enable        = k.depth_test_enable;
    gs.depth_write_enable       = k.depth_write_enable;
    gs.depth_compare            = static_cast<Op>(k.depth_compare);
    gs.depth_bounds_test_enable = k.depth_bounds_test_enable;
    gs.stencil_test_enable      = k.stencil_test_enable;
    gs.stencil_front.test       = static_cast<Op>(k.stencil_front_test);
    gs.stencil_front.fail_op    = static_cast<StencilOp>(k.stencil_front_fail);
    gs.stencil_front.pass_op    = static_cast<StencilOp>(k.stencil_front_pass);
    gs.stencil_front.depth_fail_op = static_cast<StencilOp>(k.stencil_front_depth_fail);
    gs.stencil_back.test        = static_cast<Op>(k.stencil_back_test);
    gs.stencil_back.fail_op     = static_cast<StencilOp>(k.stencil_back_fail);
    gs.stencil_back.pass_op     = static_cast<StencilOp>(k.stencil_back_pass);
    gs.stencil_back.depth_fail_op = static_cast<StencilOp>(k.stencil_back_depth_fail);
    return gs;
}

// Fieldwise equality — NEVER memcmp: the struct has one trailing padding byte
// (align 8 on the uint64 + 15 uint8s) that would be nondeterministic.
static bool variant_key_eq(const StaticVariantKey& a, const StaticVariantKey& b) {
    return a.base_stable_id == b.base_stable_id && a.impl_version == b.impl_version &&
           a.profile_id == b.profile_id && a.front_face == b.front_face &&
           a.cull == b.cull && a.depth_test_enable == b.depth_test_enable &&
           a.depth_write_enable == b.depth_write_enable && a.depth_compare == b.depth_compare &&
           a.depth_bounds_test_enable == b.depth_bounds_test_enable &&
           a.stencil_test_enable == b.stencil_test_enable &&
           a.stencil_front_test == b.stencil_front_test && a.stencil_front_fail == b.stencil_front_fail &&
           a.stencil_front_pass == b.stencil_front_pass && a.stencil_front_depth_fail == b.stencil_front_depth_fail &&
           a.stencil_back_test == b.stencil_back_test && a.stencil_back_fail == b.stencil_back_fail &&
           a.stencil_back_pass == b.stencil_back_pass && a.stencil_back_depth_fail == b.stencil_back_depth_fail;
}

// Serialized key (header + body), explicit field-by-field — no padding bytes.
// Little-endian host; all supported targets (x86-64/arm64) are little-endian.
static void serialize_variant_key(const StaticVariantKey& k, uint8_t* out) {
    uint32_t header[3] = {kStaticVariantHeaderMagic, k.impl_version, k.profile_id};
    memcpy(out, header, sizeof(header));
    uint8_t* p = out + sizeof(header);
    memcpy(p, &k.base_stable_id, sizeof(k.base_stable_id));
    p += sizeof(k.base_stable_id);
    *p++ = k.front_face;
    *p++ = k.cull;
    *p++ = k.depth_test_enable;
    *p++ = k.depth_write_enable;
    *p++ = k.depth_compare;
    *p++ = k.depth_bounds_test_enable;
    *p++ = k.stencil_test_enable;
    *p++ = k.stencil_front_test;
    *p++ = k.stencil_front_fail;
    *p++ = k.stencil_front_pass;
    *p++ = k.stencil_front_depth_fail;
    *p++ = k.stencil_back_test;
    *p++ = k.stencil_back_fail;
    *p++ = k.stencil_back_pass;
    *p++ = k.stencil_back_depth_fail;
}

void process_static_variant(DeviceImpl* d, StaticVariantRecord* rec) {
    Arena*      arena = get_thread_local_arena(d);
    ScratchScope scope(*arena);
    VkPipeline  pipeline = VK_NULL_HANDLE;
    const LogicalGraphicsState gs = state_from_key(rec->key);
    VkResult    result   = arena->overflowed()
                               ? VK_ERROR_OUT_OF_HOST_MEMORY
                               : compile_graphics(d, arena, rec->base, /*fail_on_compile=*/false,
                                                  &gs, &pipeline);
    mutex_lock(&rec->wait_mutex);
    if (result == VK_SUCCESS) {
        rec->vk_pipeline = pipeline;
        rec->state.store(InternalPipelineState::Ready, std::memory_order_release);
    } else {
        rec->failure_result = result;
        rec->state.store(InternalPipelineState::Failed, std::memory_order_release);
        IZ_LOG(d, LogLevel::Error, "static graphics variant compile failed");
    }
    condvar_signal(&rec->wait_cv);
    mutex_unlock(&rec->wait_mutex);
    release_pipeline_ref(d, rec->base);   // the worker job's base reference
}

// Finds or enqueues the private static variant for (base, baked state). The
// device map owns the record for the base's lifetime; the caller receives a
// pure borrow (valid while the base is retained). The queued worker job
// retains the BASE so compilation can never race base eviction.
StaticVariantRecord* find_or_request_static_variant(DeviceImpl* d, PipelineRecord* base,
                                                    const LogicalGraphicsState& gs) {
    const StaticVariantKey key = make_variant_key(d, base->stable_id, gs);
    atomic_fetch_add(&d->stat_static_variant_lookups, 1);
    mutex_lock(&d->pipeline_lock);
    for (StaticVariantRecord* r : d->static_variants) {
        if (r->base == base && variant_key_eq(r->key, key)) {
            atomic_fetch_add(&d->stat_static_variant_hits, 1);
            mutex_unlock(&d->pipeline_lock);
            return r;
        }
    }
    atomic_fetch_add(&d->stat_static_variant_misses, 1);
    MemoryBlock blk = d->allocator.alloc(sizeof(StaticVariantRecord));
    if (blk.ptr == nullptr) {
        mutex_unlock(&d->pipeline_lock);
        return nullptr;
    }
    auto* rec = ::new (blk.ptr) StaticVariantRecord();
    rec->base = base;
    rec->key  = key;
    condvar_init(&rec->wait_cv);
    rec->next_variant   = base->first_variant;   // intrusive list on the base
    base->first_variant = rec;
    d->static_variants.push_back(rec);

    mutex_lock(&d->compiler_lock);
    d->variant_queue.push_back(rec);
    base->refs.fetch_add(1, std::memory_order_relaxed);   // the worker job's base reference
    condvar_signal(&d->compiler_cv);
    mutex_unlock(&d->compiler_lock);
    mutex_unlock(&d->pipeline_lock);

    atomic_fetch_add(&d->stat_static_variant_compilations, 1);
    return rec;
}

// --- Reference lifetime ------------------------------------------------------------------

// Releases one reference; destroys the record (and its VkPipeline) with the last.
void release_pipeline_ref(DeviceImpl* d, PipelineRecord* rec) {
    if (rec->refs.fetch_sub(1, std::memory_order_acq_rel) != 1) { return; }
    // Last reference: remove from the dedup map AND evict the base's private
    // static variants only if the locked recheck still sees zero references
    // (a concurrent lookup may have resurrected the record — then nothing is
    // destroyed). Destroy happens outside the lock.
    bool destroy = false;
    StaticVariantRecord* evict = nullptr;   // detached chain, destroyed after unlock
    mutex_lock(&d->pipeline_lock);
    if (rec->refs.load(std::memory_order_relaxed) == 0) {
        destroy = true;
        for (uint32_t i = 0; i < d->pipeline_records.size(); ++i) {
            if (d->pipeline_records[i] == rec) {
                d->pipeline_records[i] = d->pipeline_records[d->pipeline_records.size() - 1];
                d->pipeline_records.pop_back();
                break;
            }
        }
        // Evict the base's private static variants. Allocation-free: detach
        // the base's intrusive list under the lock, remove the records from
        // the map, destroy them one at a time after the unlock. Never orphan
        // a variant on OOM.
        evict         = rec->first_variant;
        rec->first_variant = nullptr;
        if (evict != nullptr) {
            for (uint32_t i = 0; i < d->static_variants.size();) {
                StaticVariantRecord* v = d->static_variants[i];
                if (v->base == rec) {
                    d->static_variants[i] = d->static_variants[d->static_variants.size() - 1];
                    d->static_variants.pop_back();
                } else {
                    ++i;
                }
            }
        }
    }
    mutex_unlock(&d->pipeline_lock);
    if (!destroy) { return; }   // resurrected by a concurrent lookup

    if (rec->vk_pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(d->device, rec->vk_pipeline, nullptr);
    }
    StaticVariantRecord* v = evict;
    while (v != nullptr) {
        StaticVariantRecord* next = v->next_variant;
        if (v->vk_pipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(d->device, v->vk_pipeline, nullptr);
        }
        condvar_destroy(&v->wait_cv);
        v->~StaticVariantRecord();
        d->allocator.free({.ptr = v, .len = sizeof(StaticVariantRecord)});
        v = next;
    }
    free_key(d, rec);
    free_record(d, rec);
}

// --- Request map helpers (pipeline_lock held by caller) ---------------------------------

static PipelineRecord* find_record_locked(DeviceImpl* d, ShaderSource vertex, ShaderSource fragment,
                                          const VkSpecializationInfo& spec, const RasterDesc* desc) {
    for (PipelineRecord* rec : d->pipeline_records) {
        if (rec->refs.load(std::memory_order_relaxed) == 0) { continue; }
        const bool eq = desc ? graphics_key_equal(*rec, vertex, fragment, spec, *desc)
                             : compute_key_equal(*rec, vertex, spec);
        if (eq) {
            rec->refs.fetch_add(1, std::memory_order_relaxed);
            return rec;
        }
    }
    return nullptr;
}

static void enqueue_record(DeviceImpl* d, PipelineRecord* rec) {
    mutex_lock(&d->compiler_lock);
    d->compiler_queue.push_back(rec);
    const int64_t depth = static_cast<int64_t>(d->compiler_queue.size());
    if (depth > atomic_load(&d->stat_max_queue_depth)) {
        atomic_exchange(&d->stat_max_queue_depth, depth);
    }
    condvar_signal(&d->compiler_cv);
    mutex_unlock(&d->compiler_lock);
}

// --- Async request API -------------------------------------------------------------------

Handle<Pipeline> request_compute_pipeline(Device                             dev,
                                          ShaderSource                       source,
                                          Span<const SpecializationConstant> constants) {
    auto* d = reinterpret_cast<DeviceImpl*>(dev);
    atomic_fetch_add(&d->stat_requests, 1);
    if (atomic_load(&d->device_destroying)) { return {}; }

    Arena* arena = get_thread_local_arena(d);
    if (arena == nullptr) { return {}; }
    const VkSpecializationInfo spec_info = construct_specialization_info(constants, arena);
    if (arena->overflowed()) {
        IZ_LOG(d, LogLevel::Error, "request_compute_pipeline: scratch arena overflow");
        return {};
    }

    mutex_lock(&d->pipeline_lock);
    if (PipelineRecord* hit = find_record_locked(d, source, {}, spec_info, nullptr)) {
        atomic_fetch_add(&d->stat_dedup_hits, 1);
        auto h = d->pipeline_pool.emplace(PipelineImpl{hit});
        mutex_unlock(&d->pipeline_lock);
        return handle_cast<Pipeline>(h);
    }
    mutex_unlock(&d->pipeline_lock);

    // Miss: build the owned key outside the lock (allocation), then re-check.
    PipelineRecord* rec = alloc_record(d);
    if (rec == nullptr) { return {}; }
    rec->bind_point = VK_PIPELINE_BIND_POINT_COMPUTE;
    if (!build_key(d, rec, source, {}, spec_info, nullptr)) {
        free_record(d, rec);
        return {};
    }

    mutex_lock(&d->pipeline_lock);
    if (PipelineRecord* other = find_record_locked(d, source, {}, spec_info, nullptr)) {
        atomic_fetch_add(&d->stat_dedup_hits, 1);
        auto h = d->pipeline_pool.emplace(PipelineImpl{other});
        mutex_unlock(&d->pipeline_lock);
        free_record(d, rec);
        return handle_cast<Pipeline>(h);
    }
    rec->refs.store(2, std::memory_order_relaxed);   // 1 user + 1 worker
    rec->state.store(InternalPipelineState::Queued, std::memory_order_relaxed);
    d->pipeline_records.push_back(rec);
    auto h = d->pipeline_pool.emplace(PipelineImpl{rec});
    mutex_unlock(&d->pipeline_lock);

    enqueue_record(d, rec);
    return handle_cast<Pipeline>(h);
}

Handle<Pipeline> request_graphics_pipeline(Device                             dev,
                                           ShaderSource                       vertex,
                                           ShaderSource                       fragment,
                                           const RasterDesc&                  desc,
                                           Span<const SpecializationConstant> constants) {
    auto* d = reinterpret_cast<DeviceImpl*>(dev);
    atomic_fetch_add(&d->stat_requests, 1);
    if (atomic_load(&d->device_destroying)) { return {}; }

    Arena* arena = get_thread_local_arena(d);
    if (arena == nullptr) { return {}; }
    const VkSpecializationInfo spec_info = construct_specialization_info(constants, arena);
    if (arena->overflowed()) {
        IZ_LOG(d, LogLevel::Error, "request_graphics_pipeline: scratch arena overflow");
        return {};
    }

    mutex_lock(&d->pipeline_lock);
    if (PipelineRecord* hit = find_record_locked(d, vertex, fragment, spec_info, &desc)) {
        atomic_fetch_add(&d->stat_dedup_hits, 1);
        auto h = d->pipeline_pool.emplace(PipelineImpl{hit});
        mutex_unlock(&d->pipeline_lock);
        return handle_cast<Pipeline>(h);
    }
    mutex_unlock(&d->pipeline_lock);

    PipelineRecord* rec = alloc_record(d);
    if (rec == nullptr) { return {}; }
    rec->bind_point = VK_PIPELINE_BIND_POINT_GRAPHICS;
    if (!build_key(d, rec, vertex, fragment, spec_info, &desc)) {
        free_record(d, rec);
        return {};
    }

    mutex_lock(&d->pipeline_lock);
    if (PipelineRecord* other = find_record_locked(d, vertex, fragment, spec_info, &desc)) {
        atomic_fetch_add(&d->stat_dedup_hits, 1);
        auto h = d->pipeline_pool.emplace(PipelineImpl{other});
        mutex_unlock(&d->pipeline_lock);
        free_record(d, rec);
        return handle_cast<Pipeline>(h);
    }
    rec->refs.store(2, std::memory_order_relaxed);
    rec->state.store(InternalPipelineState::Queued, std::memory_order_relaxed);
    d->pipeline_records.push_back(rec);
    auto h = d->pipeline_pool.emplace(PipelineImpl{rec});
    mutex_unlock(&d->pipeline_lock);

    enqueue_record(d, rec);
    return handle_cast<Pipeline>(h);
}

PipelineStatus get_pipeline_status(Device dev, Handle<Pipeline> pipeline) {
    auto* d = reinterpret_cast<DeviceImpl*>(dev);
    PipelineImpl* impl = d->pipeline_pool.try_get(handle_cast<PipelineImpl>(pipeline));
    if (impl == nullptr) { return PipelineStatus::Failed; }
    PipelineRecord* rec = impl->record;
    switch (rec->state.load(std::memory_order_acquire)) {
        case InternalPipelineState::Ready:  return PipelineStatus::Ready;
        case InternalPipelineState::Failed: return PipelineStatus::Failed;
        default:                            return PipelineStatus::Pending;
    }
}

// Blocks on a record's state machine until Ready/Failed or the timeout.
// timeout_ms == 0 waits forever (condvar); a nonzero timeout polls with a
// bounded sleep (at most 200 us or the remaining budget) so the deadline is
// honored even when no completion signal arrives in time. Works for base
// pipeline records AND static-variant records (same state/mutex/condvar
// members).
template <typename Rec>
static bool wait_record_state(Rec* rec, uint64_t timeout_ms) {
    mutex_lock(&rec->wait_mutex);
    const double t0 = monotonic_seconds();
    bool ready = false;
    for (;;) {
        const InternalPipelineState st = rec->state.load(std::memory_order_acquire);
        if (st == InternalPipelineState::Ready) { ready = true; break; }
        if (st == InternalPipelineState::Failed) { break; }
        if (timeout_ms != 0) {
            const double elapsed_ms = (monotonic_seconds() - t0) * 1000.0;
            if (elapsed_ms >= static_cast<double>(timeout_ms)) { break; }
            const double remaining_us =
                (static_cast<double>(timeout_ms) - elapsed_ms) * 1000.0;
            mutex_unlock(&rec->wait_mutex);
            std::this_thread::sleep_for(std::chrono::microseconds(
                static_cast<uint64_t>(remaining_us < 200.0 ? remaining_us : 200.0)));
            mutex_lock(&rec->wait_mutex);
            continue;
        }
        condvar_wait(&rec->wait_cv, &rec->wait_mutex);
    }
    mutex_unlock(&rec->wait_mutex);
    return ready;
}

bool wait_pipeline(Device dev, Handle<Pipeline> pipeline) {
    auto* d = reinterpret_cast<DeviceImpl*>(dev);
    PipelineImpl* impl = d->pipeline_pool.try_get(handle_cast<PipelineImpl>(pipeline));
    if (impl == nullptr) { return false; }
    PipelineRecord* rec = impl->record;
    const double t0 = monotonic_seconds();

    const bool ready = wait_record_state(rec, /*timeout_ms=*/0);

    const double waited_ms = (monotonic_seconds() - t0) * 1000.0;
    if (waited_ms > 0.5) {
        log_fmt(d, LogLevel::Info, __LINE__, "pipeline.cpp",
                "wait_pipeline blocked %.2f ms", waited_ms);
    }
    return ready;
}

// --- Static graphics-state prewarm ------------------------------------------------------

// Shared derivation of the baked shadow members from a public depth-stencil
// description. Used by the command-buffer setter AND the request_graphics_state
// prewarm, so both produce identical variant keys.
void apply_depth_stencil_to_shadow(const DepthStencilDesc& desc, LogicalGraphicsState& gs) {
    gs.depth_test_enable   = bool(desc.depth_mode & DepthFlags::Read);
    gs.depth_write_enable  = bool(desc.depth_mode & DepthFlags::Write);
    gs.depth_compare       = desc.depth_test;
    // Stencil testing is enabled only when the stencil state is non-default
    // (compare != Always or any op != Keep). Forcing it on for every state
    // made plain/depth-only pipelines run stencil testing without a stencil
    // attachment — undefined behavior (fragments rejected erratically).
    const auto& sf = desc.stencil_front;
    const auto& sb = desc.stencil_back;
    gs.stencil_test_enable = sf.test != Op::Always || sf.fail_op != StencilOp::Keep ||
                             sf.pass_op != StencilOp::Keep || sf.depth_fail_op != StencilOp::Keep ||
                             sb.test != Op::Always || sb.fail_op != StencilOp::Keep ||
                             sb.pass_op != StencilOp::Keep || sb.depth_fail_op != StencilOp::Keep;
    gs.stencil_front       = desc.stencil_front;
    gs.stencil_back        = desc.stencil_back;
    gs.stencil_read_mask   = desc.stencil_read_mask;
    gs.stencil_write_mask  = desc.stencil_write_mask;
}

// True when the baked shadow members equal the base pipeline's defaults (the
// base pipeline IS the default-state variant; no private record needed).
bool is_default_baked_state(const LogicalGraphicsState& gs) {
    return gs.front_face == FrontFace::CCW && gs.cull == Cull::None &&
           !gs.depth_test_enable && !gs.depth_write_enable && gs.depth_compare == Op::Always &&
           !gs.depth_bounds_test_enable && !gs.stencil_test_enable &&
           gs.stencil_front.test == Op::Always && gs.stencil_front.fail_op == StencilOp::Keep &&
           gs.stencil_front.pass_op == StencilOp::Keep &&
           gs.stencil_front.depth_fail_op == StencilOp::Keep &&
           gs.stencil_back.test == Op::Always && gs.stencil_back.fail_op == StencilOp::Keep &&
           gs.stencil_back.pass_op == StencilOp::Keep &&
           gs.stencil_back.depth_fail_op == StencilOp::Keep;
}

static PipelineStatus record_status(const PipelineRecord* rec) {
    const InternalPipelineState st = rec->state.load(std::memory_order_acquire);
    return st == InternalPipelineState::Ready   ? PipelineStatus::Ready
         : st == InternalPipelineState::Failed  ? PipelineStatus::Failed
                                                : PipelineStatus::Pending;
}

// Ensures a private static variant for (pipeline, graphics state) is queued
// on the compiler worker and returns its current status. On the dynamic path
// (or for compute pipelines) this is a no-op over the base pipeline.
PipelineStatus request_graphics_state(Device device, Handle<Pipeline> pipeline,
                                      const GraphicsStateDesc& desc) {
    auto* d = reinterpret_cast<DeviceImpl*>(device);
    PipelineImpl* impl = d->pipeline_pool.try_get(handle_cast<PipelineImpl>(pipeline));
    if (impl == nullptr) { return PipelineStatus::Failed; }
    PipelineRecord* rec = impl->record;
    if (!d->dispatch.use_static_graphics_state ||
        rec->bind_point != VK_PIPELINE_BIND_POINT_GRAPHICS) {
        return record_status(rec);
    }
    LogicalGraphicsState gs;
    gs.front_face = desc.front_face;
    gs.cull       = desc.cull;
    if (desc.depth_stencil.h != 0) {
        DepthStencilState* ds = d->depth_stencil_pool.try_get(desc.depth_stencil);
        if (ds == nullptr) { return PipelineStatus::Failed; }
        apply_depth_stencil_to_shadow(ds->desc, gs);
    }
    if (is_default_baked_state(gs)) { return record_status(rec); }   // base IS the variant
    StaticVariantRecord* v = find_or_request_static_variant(d, rec, gs);
    if (v == nullptr) { return PipelineStatus::Failed; }
    const InternalPipelineState st = v->state.load(std::memory_order_acquire);
    return st == InternalPipelineState::Ready   ? PipelineStatus::Ready
         : st == InternalPipelineState::Failed  ? PipelineStatus::Failed
                                                : PipelineStatus::Pending;
}

// Blocks until the requested variant is Ready or Failed (timeout_ms == 0
// waits forever). Returns true only for Ready.
bool wait_graphics_state(Device device, Handle<Pipeline> pipeline, const GraphicsStateDesc& desc,
                         uint64_t timeout_ms) {
    auto* d = reinterpret_cast<DeviceImpl*>(device);
    PipelineImpl* impl = d->pipeline_pool.try_get(handle_cast<PipelineImpl>(pipeline));
    if (impl == nullptr) { return false; }
    PipelineRecord* rec = impl->record;
    if (!d->dispatch.use_static_graphics_state ||
        rec->bind_point != VK_PIPELINE_BIND_POINT_GRAPHICS) {
        return wait_record_state(rec, timeout_ms);
    }
    LogicalGraphicsState gs;
    gs.front_face = desc.front_face;
    gs.cull       = desc.cull;
    if (desc.depth_stencil.h != 0) {
        DepthStencilState* ds = d->depth_stencil_pool.try_get(desc.depth_stencil);
        if (ds == nullptr) { return false; }
        apply_depth_stencil_to_shadow(ds->desc, gs);
    }
    if (is_default_baked_state(gs)) { return wait_record_state(rec, timeout_ms); }
    StaticVariantRecord* v = find_or_request_static_variant(d, rec, gs);
    if (v == nullptr) { return false; }
    return wait_record_state(v, timeout_ms);
}

// --- Persistent cache ----------------------------------------------------------------------

void store_pipeline_cache(DeviceImpl* d) {
    if (d->vk_pipeline_cache == VK_NULL_HANDLE || !d->cache_callbacks.store) { return; }
    size_t size = 0;
    vkGetPipelineCacheData(d->device, d->vk_pipeline_cache, &size, nullptr);
    if (size == 0) { return; }
    MemoryBlock blob = d->allocator.alloc(size);
    if (blob.ptr == nullptr) { return; }
    VkResult r = vkGetPipelineCacheData(d->device, d->vk_pipeline_cache, &size, blob.ptr);
    if (r == VK_SUCCESS && size > 0) {
        d->cache_callbacks.store(d->cache_identity, MemoryBlock{blob.ptr, static_cast<uint32_t>(size)},
                                 d->cache_callbacks.user);
    }
    d->allocator.free(blob);
}

void flush_pipeline_cache(Device dev) {
    auto* d = reinterpret_cast<DeviceImpl*>(dev);
    if (d->vk_pipeline_cache == VK_NULL_HANDLE) { return; }
    // VkPipelineCache is externally synchronized: wait for the compiler worker
    // to drain before reading cache data.
    mutex_lock(&d->compiler_lock);
    while (!d->compiler_queue.is_empty() || atomic_load(&d->compiler_busy) > 0) {
        condvar_wait(&d->compiler_cv, &d->compiler_lock);
    }
    mutex_unlock(&d->compiler_lock);
    store_pipeline_cache(d);
}

// --- Blocking convenience creators ---------------------------------------------------------

Handle<Pipeline> create_compute_pipeline(Device                             dev,
                                         ShaderSource                       source,
                                         Span<const SpecializationConstant> constants) {
    Handle<Pipeline> h = request_compute_pipeline(dev, source, constants);
    if (h.h == 0) { return {}; }
    if (!wait_pipeline(dev, h)) {
        free(dev, h);   // release the user reference on failure
        return {};
    }
    return h;
}

Handle<Pipeline> create_graphics_pipeline(Device                             dev,
                                          ShaderSource                       vertex,
                                          ShaderSource                       fragment,
                                          const RasterDesc&                  desc,
                                          Span<const SpecializationConstant> constants) {
    Handle<Pipeline> h = request_graphics_pipeline(dev, vertex, fragment, desc, constants);
    if (h.h == 0) { return {}; }
    if (!wait_pipeline(dev, h)) {
        free(dev, h);
        return {};
    }
    return h;
}

void free(Device dev, Handle<Pipeline> pipeline) {
    auto* d = reinterpret_cast<DeviceImpl*>(dev);
    PipelineImpl* impl = d->pipeline_pool.try_get(handle_cast<PipelineImpl>(pipeline));
    if (impl == nullptr) {
        IZ_LOG(d, LogLevel::Error, "free(pipeline): invalid or stale handle");
        return;
    }
    PipelineRecord* rec = impl->record;
    release_pipeline_ref(d, rec);   // user reference only
    d->pipeline_pool.erase(handle_cast<PipelineImpl>(pipeline));
}

void free_after(Device dev, Handle<Pipeline> pipeline, Submission s) {
    auto* d = reinterpret_cast<DeviceImpl*>(dev);
    QueueImpl* q = s.queue ? reinterpret_cast<QueueImpl*>(s.queue) : d->default_queue;
    if (q == nullptr) { return; }
    const uint64_t value = s.status == SubmitStatus::Success ? s.value : q->timeline_value;
    PipelineImpl* impl = d->pipeline_pool.try_get(handle_cast<PipelineImpl>(pipeline));
    if (impl == nullptr) {
        IZ_LOG(d, LogLevel::Error, "free_after(pipeline): invalid or stale handle");
        return;
    }
    PipelineRecord* rec = impl->record;
    // The user reference (and the native pipeline with it) is released when
    // the target submission completes. The public handle is invalidated
    // immediately by erasing its slot (the record survives via the deferred
    // reference).
    enqueue_retire(q, value, RetireItem{RetireKind::PipelineRef, reinterpret_cast<uint64_t>(rec), 0});
    d->pipeline_pool.erase(handle_cast<PipelineImpl>(pipeline));
}

// --- White-box test hooks -------------------------------------------------------------------

uint32_t debug_live_pipelines(DeviceImpl* d) {
    mutex_lock(&d->pipeline_lock);
    const uint32_t n = d->pipeline_records.size();
    mutex_unlock(&d->pipeline_lock);
    return n;
}

uintptr_t debug_last_compile_thread(DeviceImpl* d) {
    return d->compiler_thread_id;
}

void debug_set_compiler_paused(DeviceImpl* d, bool paused) {
    mutex_lock(&d->compiler_lock);
    atomic_exchange(&d->compiler_paused, paused ? 1 : 0);
    condvar_broadcast(&d->compiler_cv);
    mutex_unlock(&d->compiler_lock);
}

// --- Depth-stencil state ------------------------------------------------------------

Handle<DepthStencilState> create_depth_stencil_state(Device dev, const DepthStencilDesc& desc) {
    auto* d = reinterpret_cast<DeviceImpl*>(dev);
    return d->depth_stencil_pool.emplace(DepthStencilState{desc});
}

void free_depth_stencil_state(Device dev, Handle<DepthStencilState> state) {
    auto* d = reinterpret_cast<DeviceImpl*>(dev);
    if (d->depth_stencil_pool.try_get(state) == nullptr) {
        IZ_LOG(d, LogLevel::Error, "free_depth_stencil_state: invalid or stale handle");
        return;
    }
    d->depth_stencil_pool.erase(state);
}

}  // namespace gpu
