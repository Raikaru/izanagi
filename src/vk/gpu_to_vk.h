#pragma once
// Enum/flag bridging between the public gpu:: API and Vulkan types.

#include "izanagi/gpu.h"
#include "volk.h"

namespace gpu {

VkFormat                            bridge(Format format);
VkImageAspectFlags                  aspects_for_format(Format format);
Format                              bridge(VkFormat fmt);
VkPrimitiveTopology                 bridge(Topology topo);
PresentMode                         bridge(VkPresentModeKHR mode);
VkPresentModeKHR                    bridge(PresentMode mode);
VkPipelineStageFlags2               bridge_pipeline_stage(StageFlags stage);
UsageFlags                          bridge_usage_flags(VkImageUsageFlags flags);
VkImageUsageFlags                   bridge_usage_flags(UsageFlags usage);
VkImageType                         bridge(TextureType tex);
VkImageViewType                     bridge_view_type(TextureType tex);
VkPipelineColorBlendAttachmentState bridge(const BlendDesc& state);
VkAttachmentLoadOp                  bridge(LoadOp op);
VkAttachmentStoreOp                 bridge(StoreOp op);
VkCompareOp                         bridge(Op op);
VkIndexType                         bridge(IndexType t);
VkCullModeFlags                     bridge(Cull c);
VkStencilOp                         bridge(StencilOp op);
VkFilter                            bridge(SamplerFilter f);
VkSamplerMipmapMode                 bridge_mip_mode(SamplerFilter f);
VkSamplerAddressMode                bridge(SamplerAddressing a);

Span<const char> string_from_result(VkResult result);

}  // namespace gpu
