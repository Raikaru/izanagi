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
    if (rec.topology != desc.topology || rec.sample_count != desc.sample_count ||
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

// Copies every create input into one owned block inside `rec`. Returns false on
// allocation failure (caller must destroy the record and bail).
static bool build_key(DeviceImpl* d, PipelineRecord* rec, ShaderSource vertex, ShaderSource fragment,
                      const VkSpecializationInfo& spec, const RasterDesc* desc) {
    const uint32_t ct_count = desc ? static_cast<uint32_t>(desc->color_targets.size()) : 0;
    size_t total = 0;
    total += vertex.source.size() + vertex.entry_point.size() + 1;
    total += fragment.source.size() + fragment.entry_point.size() + 1;
    total += spec.dataSize + spec.mapEntryCount * 2 * sizeof(uint32_t);
    total += ct_count * sizeof(ColorTarget);
    if (total == 0) { total = 1; }

    MemoryBlock blk = d->allocator.alloc(total);
    if (blk.ptr == nullptr) { return false; }
    rec->key_block = blk;

    uint8_t* p = static_cast<uint8_t*>(blk.ptr);
    auto take_space = [&p](size_t n) -> void* {
        void* out = p;
        p += n;
        return out;
    };
    auto take_bytes = [&p](const void* src, size_t n) -> uint8_t* {
        uint8_t* out = p;
        if (n) { memcpy(p, src, n); }
        p += n;
        return out;
    };
    auto take_str = [&p](Span<const char> s) -> char* {
        char* out = reinterpret_cast<char*>(p);
        if (s.size()) { memcpy(p, s.data(), s.size()); }
        p += s.size();
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
    uint32_t* ids = static_cast<uint32_t*>(take_space(spec.mapEntryCount * sizeof(uint32_t)));
    uint32_t* sizes = static_cast<uint32_t*>(take_space(spec.mapEntryCount * sizeof(uint32_t)));
    for (uint32_t i = 0; i < spec.mapEntryCount; ++i) {
        ids[i]   = spec.pMapEntries[i].constantID;
        sizes[i] = spec.pMapEntries[i].size;
    }
    rec->spec_ids   = ids;
    rec->spec_sizes = sizes;

    if (desc) {
        rec->topology          = desc->topology;
        rec->sample_count      = desc->sample_count;
        rec->alpha_to_coverage = desc->alpha_to_coverage;
        rec->depth_format      = desc->depth_format;
        rec->stencil_format    = desc->stencil_format;
        rec->color_target_count = ct_count;
        ColorTarget* cts = static_cast<ColorTarget*>(take_space(ct_count * sizeof(ColorTarget)));
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

static VkResult compile_compute(DeviceImpl* d, Arena* arena, PipelineRecord* rec,
                                bool fail_on_compile, VkPipeline* out) {
    VkSpecializationInfo spec = rebuild_spec_info(arena, *rec);

    VkShaderModuleCreateInfo module_info{
        .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext    = nullptr,
        .flags    = 0,
        .codeSize = rec->vs_size,
        .pCode    = reinterpret_cast<const uint32_t*>(rec->vs_bytes),
    };
    VkPipelineShaderStageCreateInfo stage{
        .sType               = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext               = &module_info,
        .flags               = 0,
        .stage               = VK_SHADER_STAGE_COMPUTE_BIT,
        .module              = VK_NULL_HANDLE,
        .pName               = rec->vs_entry,
        .pSpecializationInfo = &spec,
    };

    VkPipelineCreateFlags2CreateInfo flags2{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT |
                 (fail_on_compile ? VK_PIPELINE_CREATE_2_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT : 0),
    };
    VkComputePipelineCreateInfo info{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .pNext = &flags2,
        .flags = 0,
        .stage = stage,
        .layout             = VK_NULL_HANDLE,
        .basePipelineHandle = VK_NULL_HANDLE,
        .basePipelineIndex  = 0,
    };
    if (arena->overflowed()) { return VK_ERROR_OUT_OF_HOST_MEMORY; }
    return vkCreateComputePipelines(d->device, d->vk_pipeline_cache, 1, &info, nullptr, out);
}

static VkResult compile_graphics(DeviceImpl* d, Arena* arena, PipelineRecord* rec,
                                 bool fail_on_compile, VkPipeline* out) {
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
    VkPipelineShaderStageCreateInfo stages[] = {
        {
            .sType               = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext               = &vert_module_info,
            .flags               = 0,
            .stage               = VK_SHADER_STAGE_VERTEX_BIT,
            .module              = VK_NULL_HANDLE,
            .pName               = rec->vs_entry,
            .pSpecializationInfo = &spec,
        },
        {
            .sType               = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext               = &frag_module_info,
            .flags               = 0,
            .stage               = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module              = VK_NULL_HANDLE,
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
        .polygonMode             = VK_POLYGON_MODE_FILL,
        .cullMode                = VK_CULL_MODE_BACK_BIT,
        .frontFace               = VK_FRONT_FACE_COUNTER_CLOCKWISE,
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

    VkPipelineViewportStateCreateInfo viewport_state{
        .sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .pNext         = nullptr,
        .flags         = 0,
        .viewportCount = 0,
        .scissorCount  = 0,
    };

    VkPipelineDynamicStateCreateInfo dynamic_state{
        .sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .pNext             = nullptr,
        .flags             = 0,
        .dynamicStateCount = sizeof(dynamic_states) / sizeof(VkDynamicState),
        .pDynamicStates    = dynamic_states,
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

    VkPipelineCreateFlags2CreateInfo flags2{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO,
        .pNext = &rendering_info,
        .flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT |
                 (fail_on_compile ? VK_PIPELINE_CREATE_2_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT : 0),
    };

    VkGraphicsPipelineCreateInfo create_info{
        .sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext               = &flags2,
        .flags               = 0,
        .stageCount          = 2,
        .pStages             = stages,
        .pVertexInputState   = &vertex_input_state,
        .pInputAssemblyState = &input_assembly_state,
        .pTessellationState  = nullptr,
        .pViewportState      = &viewport_state,
        .pRasterizationState = &rasterization_state,
        .pMultisampleState   = &multisample_state,
        .pDepthStencilState  = nullptr,
        .pColorBlendState    = &color_blend_state,
        .pDynamicState       = &dynamic_state,
        .layout              = VK_NULL_HANDLE,
        .renderPass          = VK_NULL_HANDLE,
        .subpass             = 0,
        .basePipelineHandle  = VK_NULL_HANDLE,
        .basePipelineIndex   = 0,
    };
    if (arena->overflowed()) { return VK_ERROR_OUT_OF_HOST_MEMORY; }
    return vkCreateGraphicsPipelines(d->device, d->vk_pipeline_cache, 1, &create_info, nullptr, out);
}

static VkResult compile_record(DeviceImpl* d, Arena* arena, PipelineRecord* rec,
                               bool fail_on_compile, VkPipeline* out) {
    return rec->bind_point == VK_PIPELINE_BIND_POINT_GRAPHICS
               ? compile_graphics(d, arena, rec, fail_on_compile, out)
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
        PipelineRecord* rec = nullptr;
        mutex_lock(&d->compiler_lock);
        while (!atomic_load(&d->compiler_shutdown) &&
               (d->compiler_queue.is_empty() || atomic_load(&d->compiler_paused))) {
            condvar_wait(&d->compiler_cv, &d->compiler_lock);
        }
        if (d->compiler_queue.is_empty() && atomic_load(&d->compiler_shutdown)) {
            mutex_unlock(&d->compiler_lock);
            break;
        }
        rec = d->compiler_queue[0];
        d->compiler_queue.erase(d->compiler_queue.begin(), d->compiler_queue.begin() + 1);
        atomic_fetch_add(&d->compiler_busy, 1);
        mutex_unlock(&d->compiler_lock);

        process_record(d, rec);
        release_pipeline_ref(d, rec);   // the worker's reference

        mutex_lock(&d->compiler_lock);
        atomic_fetch_add(&d->compiler_busy, -1);
        condvar_broadcast(&d->compiler_cv);   // wake flush_pipeline_cache waiters
        mutex_unlock(&d->compiler_lock);
    }
}

// --- Reference lifetime ------------------------------------------------------------------

// Releases one reference; destroys the record (and its VkPipeline) with the last.
void release_pipeline_ref(DeviceImpl* d, PipelineRecord* rec) {
    if (rec->refs.fetch_sub(1, std::memory_order_acq_rel) != 1) { return; }
    // Last reference: remove from the dedup map (under the lock, re-checking
    // that no lookup resurrected it), then destroy outside the lock.
    mutex_lock(&d->pipeline_lock);
    if (rec->refs.load(std::memory_order_relaxed) == 0) {
        for (uint32_t i = 0; i < d->pipeline_records.size(); ++i) {
            if (d->pipeline_records[i] == rec) {
                d->pipeline_records[i] = d->pipeline_records[d->pipeline_records.size() - 1];
                d->pipeline_records.pop_back();
                break;
            }
        }
    }
    mutex_unlock(&d->pipeline_lock);

    if (rec->vk_pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(d->device, rec->vk_pipeline, nullptr);
    }
    if (rec->key_block.ptr != nullptr) { d->allocator.free(rec->key_block); }
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
    ScratchScope scope(*arena);
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
    ScratchScope scope(*arena);
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
    PipelineRecord* rec = d->pipeline_pool[handle_cast<PipelineImpl>(pipeline)].record;
    switch (rec->state.load(std::memory_order_acquire)) {
        case InternalPipelineState::Ready:  return PipelineStatus::Ready;
        case InternalPipelineState::Failed: return PipelineStatus::Failed;
        default:                            return PipelineStatus::Pending;
    }
}

bool wait_pipeline(Device dev, Handle<Pipeline> pipeline) {
    auto* d = reinterpret_cast<DeviceImpl*>(dev);
    PipelineRecord* rec = d->pipeline_pool[handle_cast<PipelineImpl>(pipeline)].record;
    const double t0 = monotonic_seconds();

    mutex_lock(&rec->wait_mutex);
    for (;;) {
        const InternalPipelineState st = rec->state.load(std::memory_order_acquire);
        if (st == InternalPipelineState::Ready || st == InternalPipelineState::Failed) { break; }
        condvar_wait(&rec->wait_cv, &rec->wait_mutex);
    }
    const bool ready = rec->state.load(std::memory_order_acquire) == InternalPipelineState::Ready;
    mutex_unlock(&rec->wait_mutex);

    const double waited_ms = (monotonic_seconds() - t0) * 1000.0;
    if (waited_ms > 0.5) {
        log_fmt(d, LogLevel::Info, __LINE__, "pipeline.cpp",
                "wait_pipeline blocked %.2f ms", waited_ms);
    }
    return ready;
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
    PipelineRecord* rec = d->pipeline_pool[handle_cast<PipelineImpl>(pipeline)].record;
    release_pipeline_ref(d, rec);   // user reference only
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
    d->depth_stencil_pool.erase(state);
}

}  // namespace gpu
