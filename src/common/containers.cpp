#include "containers.h"

#include <cstdlib>
#include <cstring>

namespace gpu {

// Template instantiations for the public header's extern declarations
// (common code, so every consumer — including GPU-independent tests — links).
template class Span<const char>;
template class Span<uint8_t>;
template class Span<const ColorTarget>;
template class Span<const RenderAttachment>;
template class Span<const Format>;
template class Span<const PresentMode>;
template class Span<const CommandBuffer>;
template class Span<const SemaphoreInfo>;
template class Span<const SubmissionWait>;
template class Span<const SpecializationConstant>;

Allocator::Allocator() {
    m_alloc = [](void*, void* ptr, uint32_t, uint32_t new_size) -> MemoryBlock {
        if (new_size == 0) {
            ::free(ptr);
            return {.ptr = nullptr, .len = 0};
        } else {
            void* new_ptr = ::realloc(ptr, new_size);
            return {.ptr = new_ptr, .len = new_size};
        }
    };
}

static constexpr uint32_t div_round_up(uint32_t n, uint32_t div) {
    return n > 0 ? (n - 1) / div + 1 : 0;
}

static constexpr uint32_t num_entries(uint32_t n) {
    const uint32_t num_leaf_entries   = div_round_up(n, 64);
    const uint32_t num_header_entries = div_round_up(n, 64 * 64);
    return num_leaf_entries + num_header_entries;
}

static constexpr uint32_t num_header_entries(uint32_t n) {
    return div_round_up(n, 64 * 64);
}

TwoLevelBitset::TwoLevelBitset(Allocator alloc, uint32_t size) :
    m_data(alloc, 0ull, num_entries(size)), m_size{size} {
    // Reserve out-of-range entries so allocation never returns a slot >= size.
    const uint32_t header_size  = num_header_entries(m_size);
    const uint32_t leaf_entries = div_round_up(m_size, 64);
    if (leaf_entries == 0) { return; }

    // Leaf bits at/above size in the last leaf word.
    const uint32_t tail = m_size & 63;
    if (tail != 0) {
        m_data[header_size + leaf_entries - 1] |= ~((1ull << tail) - 1);
    }
    // Chunk entries at/above the chunk count in the last header word.
    const uint32_t tail_chunks = leaf_entries & 63;
    if (tail_chunks != 0) {
        const uint32_t last_header = (leaf_entries - 1) / 64;
        m_data[last_header] |= ~((1ull << tail_chunks) - 1);
    }
}

uint32_t TwoLevelBitset::set_leading_zero() {
    const uint32_t header_size = num_header_entries(m_size);
    uint32_t       header_idx  = 0;
    while (header_idx < header_size) {
        const uint64_t header = m_data[header_idx];
        if (header == UINT64_MAX) { header_idx++; continue; }
        const uint32_t chunk_pos_in_word = count_trailing_zeros(~header);
        const uint32_t leaf_idx          = header_size + header_idx * 64 + chunk_pos_in_word;
        uint64_t       leaf              = m_data[leaf_idx];
        if (leaf == UINT64_MAX) {
            // This chunk is full: mark it in the header and re-scan.
            m_data[header_idx] |= 1ull << chunk_pos_in_word;
            continue;
        }
        const uint32_t bit = count_trailing_zeros(~leaf);
        m_data[leaf_idx] |= 1ull << bit;
        if (m_data[leaf_idx] == UINT64_MAX) { m_data[header_idx] |= 1ull << chunk_pos_in_word; }
        return 64 * (header_idx * 64 + chunk_pos_in_word) + bit;
    }
    return ~0u;
}

void TwoLevelBitset::clear_bit(uint32_t idx) {
    const uint32_t header_size = num_header_entries(m_size);
    const uint32_t chunk_idx   = idx / 64;
    const uint32_t header_idx  = chunk_idx / 64;
    const uint64_t mask        = ~(1ull << (idx & 63));
    const uint64_t header_mask = ~(1ull << (chunk_idx & 63));
    m_data[header_size + chunk_idx] &= mask;
    m_data[header_idx] &= header_mask;
}

bool TwoLevelBitset::is_set(uint32_t idx) const {
    const uint32_t header_size = num_header_entries(m_size);
    const uint32_t chunk_idx   = idx / 64;
    return (m_data[header_size + chunk_idx] & (1ull << (idx & 63))) != 0;
}

}  // namespace gpu
