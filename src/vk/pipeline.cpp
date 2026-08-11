// pipeline.cpp — compute & graphics pipelines, depth-stencil state, specialization constants.

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

// --- Pipeline dedup (SharedPipeline) --------------------------------------------------
// Identical create inputs share one compiled VkPipeline (refcounted; destroyed
// when the last handle is freed). The key is a full copy of every input that
// reaches pipeline creation — shader bytes + entry points, specialization
// data, and the entire baked raster state. No retention beyond live handles.

static bool entry_equal(const char* stored, Span<const char> app) {
    const size_t n = strlen(stored);
    return n == app.size() && (n == 0 || memcmp(stored, app.data(), n) == 0);
}

static bool spec_equal(const SharedPipeline& sp, const VkSpecializationInfo& spec) {
    if (sp.spec_count != spec.mapEntryCount || sp.spec_size != spec.dataSize) { return false; }
    if (sp.spec_size && memcmp(sp.spec_data, spec.pData, sp.spec_size) != 0) { return false; }
    for (uint32_t i = 0; i < sp.spec_count; ++i) {
        if (sp.spec_ids[i] != spec.pMapEntries[i].constantID ||
            sp.spec_sizes[i] != spec.pMapEntries[i].size) {
            return false;
        }
    }
    return true;
}

static bool graphics_key_equal(const SharedPipeline& sp, ShaderSource vertex, ShaderSource fragment,
                               const VkSpecializationInfo& spec, const RasterDesc& desc) {
    if (sp.bind_point != VK_PIPELINE_BIND_POINT_GRAPHICS) { return false; }
    if (sp.vs_size != vertex.source.size() ||
        (sp.vs_size && memcmp(sp.vs_bytes, vertex.source.data(), sp.vs_size) != 0)) {
        return false;
    }
    if (sp.fs_size != fragment.source.size() ||
        (sp.fs_size && memcmp(sp.fs_bytes, fragment.source.data(), sp.fs_size) != 0)) {
        return false;
    }
    if (!entry_equal(sp.vs_entry, vertex.entry_point) ||
        !entry_equal(sp.fs_entry, fragment.entry_point)) {
        return false;
    }
    if (!spec_equal(sp, spec)) { return false; }
    if (sp.topology != desc.topology || sp.sample_count != desc.sample_count ||
        sp.alpha_to_coverage != desc.alpha_to_coverage ||
        sp.depth_format != desc.depth_format || sp.stencil_format != desc.stencil_format ||
        sp.color_target_count != desc.color_targets.size()) {
        return false;
    }
    for (uint32_t i = 0; i < sp.color_target_count; ++i) {
        const ColorTarget& a = sp.color_targets[i];
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

static bool compute_key_equal(const SharedPipeline& sp, ShaderSource source,
                              const VkSpecializationInfo& spec) {
    if (sp.bind_point != VK_PIPELINE_BIND_POINT_COMPUTE) { return false; }
    if (sp.vs_size != source.source.size() ||
        (sp.vs_size && memcmp(sp.vs_bytes, source.source.data(), sp.vs_size) != 0)) {
        return false;
    }
    if (!entry_equal(sp.vs_entry, source.entry_point)) { return false; }
    return spec_equal(sp, spec);
}

// Copies every create input into one owned block inside `sp`. Returns false on
// allocation failure (caller must destroy the pipeline and bail).
static bool build_key(DeviceImpl* d, SharedPipeline* sp, ShaderSource vertex, ShaderSource fragment,
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
    sp->key_block = blk;

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

    sp->vs_size  = static_cast<uint32_t>(vertex.source.size());
    sp->vs_bytes = take_bytes(vertex.source.data(), vertex.source.size());
    sp->vs_entry = take_str(vertex.entry_point);
    sp->fs_size  = static_cast<uint32_t>(fragment.source.size());
    sp->fs_bytes = take_bytes(fragment.source.data(), fragment.source.size());
    sp->fs_entry = take_str(fragment.entry_point);

    sp->spec_size = static_cast<uint32_t>(spec.dataSize);
    sp->spec_data = take_bytes(spec.pData, spec.dataSize);
    sp->spec_count = spec.mapEntryCount;
    uint32_t* ids = static_cast<uint32_t*>(take_space(spec.mapEntryCount * sizeof(uint32_t)));
    uint32_t* sizes = static_cast<uint32_t*>(take_space(spec.mapEntryCount * sizeof(uint32_t)));
    for (uint32_t i = 0; i < spec.mapEntryCount; ++i) {
        ids[i]   = spec.pMapEntries[i].constantID;
        sizes[i] = spec.pMapEntries[i].size;
    }
    sp->spec_ids   = ids;
    sp->spec_sizes = sizes;

    if (desc) {
        sp->topology          = desc->topology;
        sp->sample_count      = desc->sample_count;
        sp->alpha_to_coverage = desc->alpha_to_coverage;
        sp->depth_format      = desc->depth_format;
        sp->stencil_format    = desc->stencil_format;
        sp->color_target_count = ct_count;
        ColorTarget* cts = static_cast<ColorTarget*>(take_space(ct_count * sizeof(ColorTarget)));
        for (uint32_t i = 0; i < ct_count; ++i) { cts[i] = desc->color_targets[i]; }
        sp->color_targets = cts;
    }
    return true;
}

// Finds a live shared pipeline with identical inputs and bumps its refcount.
// Returns nullptr on miss.
static SharedPipeline* dedup_find(DeviceImpl* d, ShaderSource vertex, ShaderSource fragment,
                                  const VkSpecializationInfo& spec, const RasterDesc* desc) {
    for (SharedPipeline* sp : d->shared_pipelines) {
        const bool eq = desc ? graphics_key_equal(*sp, vertex, fragment, spec, *desc)
                             : compute_key_equal(*sp, vertex, spec);
        if (eq) {
            sp->refcount++;
            return sp;
        }
    }
    return nullptr;
}

// Allocates a zero-initialized shared entry (stable address across vector
// growth). Returns nullptr on allocation failure.
static SharedPipeline* alloc_shared_pipeline(DeviceImpl* d) {
    MemoryBlock blk = d->allocator.alloc(sizeof(SharedPipeline));
    if (blk.ptr == nullptr) { return nullptr; }
    return ::new (blk.ptr) SharedPipeline();
}

static void free_shared_pipeline(DeviceImpl* d, SharedPipeline* sp) {
    sp->~SharedPipeline();
    d->allocator.free({.ptr = sp, .len = sizeof(SharedPipeline)});
}

// --- Compute pipeline ----------------------------------------------------------------

Handle<Pipeline> create_compute_pipeline(Device                             dev,
                                         ShaderSource                       source,
                                         Span<const SpecializationConstant> constants) {
    auto* d = reinterpret_cast<DeviceImpl*>(dev);

    Arena* arena = get_thread_local_arena(d);
    const VkSpecializationInfo specialization_info =
        construct_specialization_info(constants, arena);

    // maintenance5: chain VkShaderModuleCreateInfo into the stage create info
    VkShaderModuleCreateInfo module_info{
        .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext    = nullptr,
        .flags    = 0,
        .codeSize = source.source.size(),
        .pCode    = reinterpret_cast<const uint32_t*>(source.source.data()),
    };

    VkPipelineShaderStageCreateInfo stage{
        .sType               = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext               = &module_info,
        .flags               = 0,
        .stage               = VK_SHADER_STAGE_COMPUTE_BIT,
        .module              = VK_NULL_HANDLE,
        .pName               = make_null_terminated(arena, source.entry_point),
        .pSpecializationInfo = &specialization_info,
    };

    VkPipelineCreateFlags2CreateInfo flags2{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT,
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

    VkPipeline pipeline = VK_NULL_HANDLE;

    mutex_lock(&d->pipeline_lock);
    if (SharedPipeline* hit = dedup_find(d, source, {}, specialization_info, nullptr)) {
        auto h = d->pipeline_pool.emplace(PipelineImpl{hit});
        mutex_unlock(&d->pipeline_lock);
        return handle_cast<Pipeline>(h);
    }

    if (!IZ_CHK(d, vkCreateComputePipelines(d->device, d->vk_pipeline_cache, 1, &info, nullptr, &pipeline),
                "create_compute_pipeline failed")) {
        mutex_unlock(&d->pipeline_lock);
        return {};
    }

    SharedPipeline* sp = alloc_shared_pipeline(d);
    if (sp == nullptr) {
        vkDestroyPipeline(d->device, pipeline, nullptr);
        mutex_unlock(&d->pipeline_lock);
        return {};
    }
    sp->vk_pipeline = pipeline;
    sp->bind_point  = VK_PIPELINE_BIND_POINT_COMPUTE;
    sp->refcount    = 1;
    if (!build_key(d, sp, source, {}, specialization_info, nullptr)) {
        vkDestroyPipeline(d->device, pipeline, nullptr);
        free_shared_pipeline(d, sp);
        mutex_unlock(&d->pipeline_lock);
        return {};
    }
    d->shared_pipelines.push_back(sp);
    auto h = d->pipeline_pool.emplace(PipelineImpl{sp});
    mutex_unlock(&d->pipeline_lock);
    return handle_cast<Pipeline>(h);
}

// --- Graphics pipeline -----------------------------------------------------------------

Handle<Pipeline> create_graphics_pipeline(Device                             dev,
                                          ShaderSource                       vertex,
                                          ShaderSource                       fragment,
                                          const RasterDesc&                  desc,
                                          Span<const SpecializationConstant> constants) {
    auto* d = reinterpret_cast<DeviceImpl*>(dev);

    Arena* arena = get_thread_local_arena(d);
    const VkSpecializationInfo specialization_info =
        construct_specialization_info(constants, arena);

    // Shader module create infos chained into stages (maintenance5)
    VkShaderModuleCreateInfo vert_module_info{
        .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext    = nullptr,
        .flags    = 0,
        .codeSize = vertex.source.size(),
        .pCode    = reinterpret_cast<const uint32_t*>(vertex.source.data()),
    };
    VkShaderModuleCreateInfo frag_module_info{
        .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext    = nullptr,
        .flags    = 0,
        .codeSize = fragment.source.size(),
        .pCode    = reinterpret_cast<const uint32_t*>(fragment.source.data()),
    };

    VkPipelineShaderStageCreateInfo stages[] = {
        {
            .sType               = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext               = &vert_module_info,
            .flags               = 0,
            .stage               = VK_SHADER_STAGE_VERTEX_BIT,
            .module              = VK_NULL_HANDLE,
            .pName               = make_null_terminated(arena, vertex.entry_point),
            .pSpecializationInfo = &specialization_info,
        },
        {
            .sType               = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext               = &frag_module_info,
            .flags               = 0,
            .stage               = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module              = VK_NULL_HANDLE,
            .pName               = make_null_terminated(arena, fragment.entry_point),
            .pSpecializationInfo = &specialization_info,
        },
    };

    // Vertex input: none (vertex data via BDA pointers in push data)
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
        .topology               = bridge(desc.topology),
        .primitiveRestartEnable = false,
    };

    // Color blend + attachment formats
    Span<VkPipelineColorBlendAttachmentState> color_blend_states{};
    Span<VkFormat>                            color_formats{};
    // Dual-source factors are optional: reject deterministically when the
    // device does not support dualSrcBlend (no silent fallback).
    const bool needs_dual_src = [&]() {
        for (auto& t : desc.color_targets) {
            const auto& b = t.blendstate;
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
        return {};
    }
    for (auto& t : desc.color_targets) {
        color_blend_states = concat(arena, color_blend_states, bridge(t.blendstate));
        color_formats      = concat(arena, color_formats, bridge(t.format));
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
        .rasterizationSamples  = static_cast<VkSampleCountFlagBits>(desc.sample_count == 0 ? 1 : desc.sample_count),
        .sampleShadingEnable   = false,
        .minSampleShading      = 1.0f,
        .pSampleMask           = nullptr,
        .alphaToCoverageEnable = desc.alpha_to_coverage,
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

    VkFormat depth_format   = bridge(desc.depth_format);
    VkFormat stencil_format = bridge(desc.stencil_format);

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
        .flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT,
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

    VkPipeline pipeline = VK_NULL_HANDLE;

    mutex_lock(&d->pipeline_lock);
    if (SharedPipeline* hit = dedup_find(d, vertex, fragment, specialization_info, &desc)) {
        auto h = d->pipeline_pool.emplace(PipelineImpl{hit});
        mutex_unlock(&d->pipeline_lock);
        return handle_cast<Pipeline>(h);
    }

    if (!IZ_CHK(d, vkCreateGraphicsPipelines(d->device, d->vk_pipeline_cache, 1, &create_info, nullptr, &pipeline),
                "create_graphics_pipeline failed")) {
        mutex_unlock(&d->pipeline_lock);
        return {};
    }

    SharedPipeline* sp = alloc_shared_pipeline(d);
    if (sp == nullptr) {
        vkDestroyPipeline(d->device, pipeline, nullptr);
        mutex_unlock(&d->pipeline_lock);
        return {};
    }
    sp->vk_pipeline = pipeline;
    sp->bind_point  = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sp->refcount    = 1;
    if (!build_key(d, sp, vertex, fragment, specialization_info, &desc)) {
        vkDestroyPipeline(d->device, pipeline, nullptr);
        free_shared_pipeline(d, sp);
        mutex_unlock(&d->pipeline_lock);
        return {};
    }
    d->shared_pipelines.push_back(sp);
    auto h = d->pipeline_pool.emplace(PipelineImpl{sp});
    mutex_unlock(&d->pipeline_lock);
    return handle_cast<Pipeline>(h);
}

void free(Device dev, Handle<Pipeline> pipeline) {
    auto* d = reinterpret_cast<DeviceImpl*>(dev);

    // Decrement the shared pipeline's refcount; destroy it with the last handle.
    SharedPipeline* sp = d->pipeline_pool[handle_cast<PipelineImpl>(pipeline)].shared;
    mutex_lock(&d->pipeline_lock);
    if (--sp->refcount == 0) {
        vkDestroyPipeline(d->device, sp->vk_pipeline, nullptr);
        if (sp->key_block.ptr != nullptr) { d->allocator.free(sp->key_block); }
        for (uint32_t i = 0; i < d->shared_pipelines.size(); ++i) {
            if (d->shared_pipelines[i] == sp) {
                d->shared_pipelines[i] = d->shared_pipelines[d->shared_pipelines.size() - 1];
                d->shared_pipelines.pop_back();
                break;
            }
        }
        free_shared_pipeline(d, sp);
    }
    mutex_unlock(&d->pipeline_lock);

    d->pipeline_pool.erase(handle_cast<PipelineImpl>(pipeline));
}

uint32_t debug_live_pipelines(DeviceImpl* d) {
    mutex_lock(&d->pipeline_lock);
    const uint32_t n = d->shared_pipelines.size();
    mutex_unlock(&d->pipeline_lock);
    return n;
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
