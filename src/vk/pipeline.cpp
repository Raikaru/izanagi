// pipeline.cpp — compute & graphics pipelines, depth-stencil state, specialization constants.

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
    if (!IZ_CHK(d, vkCreateComputePipelines(d->device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline),
                "create_compute_pipeline failed")) {
        return {};
    }

    return handle_cast<Pipeline>(d->pipeline_pool.emplace(PipelineImpl{
        .vk_pipeline = pipeline,
        .bind_point  = VK_PIPELINE_BIND_POINT_COMPUTE,
    }));
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
        .rasterizationSamples  = VK_SAMPLE_COUNT_1_BIT,
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
    if (!IZ_CHK(d, vkCreateGraphicsPipelines(d->device, VK_NULL_HANDLE, 1, &create_info, nullptr, &pipeline),
                "create_graphics_pipeline failed")) {
        return {};
    }

    return handle_cast<Pipeline>(d->pipeline_pool.emplace(PipelineImpl{
        .vk_pipeline = pipeline,
        .bind_point  = VK_PIPELINE_BIND_POINT_GRAPHICS,
    }));
}

void free(Device dev, Handle<Pipeline> pipeline) {
    auto* d = reinterpret_cast<DeviceImpl*>(dev);
    d->pipeline_pool.erase(handle_cast<PipelineImpl>(pipeline));
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
