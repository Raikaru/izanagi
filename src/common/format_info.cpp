#include "format_info.h"

namespace gpu {

// Table indexed by (Format - 1). Order matches the Format enum exactly.
// clang-format off
static constexpr FormatInfo kFormatInfo[] = {
    {Format::R8Unorm, 1}, {Format::R8Snorm, 1}, {Format::R8Uint, 1}, {Format::R8Sint, 1},
    {Format::R16Unorm, 2}, {Format::R16Snorm, 2}, {Format::R16Uint, 2}, {Format::R16Sint, 2},
    {Format::R16Float, 2},
    {Format::RG8Unorm, 2}, {Format::RG8Snorm, 2}, {Format::RG8Uint, 2}, {Format::RG8Sint, 2},
    {Format::R32Float, 4}, {Format::R32Uint, 4}, {Format::R32Sint, 4},
    {Format::RG16Unorm, 4}, {Format::RG16Snorm, 4}, {Format::RG16Uint, 4}, {Format::RG16Sint, 4},
    {Format::RG16Float, 4},
    {Format::RGBA8Unorm, 4}, {Format::RGBA8UnormSrgb, 4}, {Format::RGBA8Snorm, 4},
    {Format::RGBA8Uint, 4}, {Format::RGBA8Sint, 4},
    {Format::BGRA8Unorm, 4}, {Format::BGRA8UnormSrgb, 4},
    {Format::RGB10A2Uint, 4}, {Format::RGB10A2Unorm, 4},
    {Format::RG11B10Ufloat, 4}, {Format::RGB9E5Ufloat, 4},
    {Format::RG32Float, 8}, {Format::RG32Uint, 8}, {Format::RG32Sint, 8},
    {Format::RGBA16Unorm, 8}, {Format::RGBA16Snorm, 8}, {Format::RGBA16Uint, 8},
    {Format::RGBA16Sint, 8}, {Format::RGBA16Float, 8},
    {Format::RGBA32Float, 16}, {Format::RGBA32Uint, 16}, {Format::RGBA32Sint, 16},
    {Format::Stencil8, 1}, {Format::Depth16Unorm, 2}, {Format::Depth24Plus, 4},
    {Format::Depth24PlusStencil8, 4}, {Format::Depth32Float, 4}, {Format::Depth32FloatStencil8, 5},
    {Format::BC1RGBAUnorm, 8, 4, 4}, {Format::BC1RGBAUnormSrgb, 8, 4, 4},
    {Format::BC2RGBAUnorm, 16, 4, 4}, {Format::BC2RGBAUnormSrgb, 16, 4, 4},
    {Format::BC3RGBAUnorm, 16, 4, 4}, {Format::BC3RGBAUnormSrgb, 16, 4, 4},
    {Format::BC4RUnorm, 8, 4, 4}, {Format::BC4RSnorm, 8, 4, 4},
    {Format::BC5RGUnorm, 16, 4, 4}, {Format::BC5RGSnorm, 16, 4, 4},
    {Format::BC6HRGBUfloat, 16, 4, 4}, {Format::BC6HRGBFloat, 16, 4, 4},
    {Format::BC7RGBAUnorm, 16, 4, 4}, {Format::BC7RGBAUnormSrgb, 16, 4, 4},
    {Format::ETC2RGB8Unorm, 8, 4, 4}, {Format::ETC2RGB8UnormSrgb, 8, 4, 4},
    {Format::ETC2RGB8A1Unorm, 8, 4, 4}, {Format::ETC2RGB8A1UnormSrgb, 8, 4, 4},
    {Format::ETC2RGBA8Unorm, 16, 4, 4}, {Format::ETC2RGBA8UnormSrgb, 16, 4, 4},
    {Format::EACR11Unorm, 8, 4, 4}, {Format::EACR11Snorm, 8, 4, 4},
    {Format::EACRG11Unorm, 16, 4, 4}, {Format::EACRG11Snorm, 16, 4, 4},
    {Format::ASTC4x4Unorm, 16, 4, 4}, {Format::ASTC4x4UnormSrgb, 16, 4, 4},
    {Format::ASTC5x4Unorm, 16, 5, 4}, {Format::ASTC5x4UnormSrgb, 16, 5, 4},
    {Format::ASTC5x5Unorm, 16, 5, 5}, {Format::ASTC5x5UnormSrgb, 16, 5, 5},
    {Format::ASTC6x5Unorm, 16, 6, 5}, {Format::ASTC6x5UnormSrgb, 16, 6, 5},
    {Format::ASTC6x6Unorm, 16, 6, 6}, {Format::ASTC6x6UnormSrgb, 16, 6, 6},
    {Format::ASTC8x5Unorm, 16, 8, 5}, {Format::ASTC8x5UnormSrgb, 16, 8, 5},
    {Format::ASTC8x6Unorm, 16, 8, 6}, {Format::ASTC8x6UnormSrgb, 16, 8, 6},
    {Format::ASTC8x8Unorm, 16, 8, 8}, {Format::ASTC8x8UnormSrgb, 16, 8, 8},
    {Format::ASTC10x5Unorm, 16, 10, 5}, {Format::ASTC10x5UnormSrgb, 16, 10, 5},
    {Format::ASTC10x6Unorm, 16, 10, 6}, {Format::ASTC10x6UnormSrgb, 16, 10, 6},
    {Format::ASTC10x8Unorm, 16, 10, 8}, {Format::ASTC10x8UnormSrgb, 16, 10, 8},
    {Format::ASTC10x10Unorm, 16, 10, 10}, {Format::ASTC10x10UnormSrgb, 16, 10, 10},
    {Format::ASTC12x10Unorm, 16, 12, 10}, {Format::ASTC12x10UnormSrgb, 16, 12, 10},
    {Format::ASTC12x12Unorm, 16, 12, 12}, {Format::ASTC12x12UnormSrgb, 16, 12, 12},
};
// clang-format on

static constexpr auto check_table = []() {
    bool result = true;
    for (uint32_t i = 1; i < static_cast<uint32_t>(Format::ValidCount); ++i) {
        result &= (static_cast<uint32_t>(kFormatInfo[i - 1].format) == i);
    }
    return result;
};
static_assert(check_table());

FormatInfo get_format_info(Format f) {
    const uint32_t idx = static_cast<uint32_t>(f);
    if (idx == 0 || idx >= static_cast<uint32_t>(Format::ValidCount)) {
        return {};   // Format::None or out-of-range: no block info
    }
    return kFormatInfo[idx - 1];
}

}  // namespace gpu
