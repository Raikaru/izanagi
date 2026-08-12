// common_tests.cpp — GPU-independent tests for the common infrastructure:
// Arena (alignment, mark/rewind, overflow), ScratchScope, Vector (insert,
// growth, allocation failure), SlotMap (reuse, generations), TwoLevelBitset,
// and the enum bitwise operators. No Vulkan device required.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

#include "common/containers.h"
#include "common/dispatch_capabilities.h"
#include "common/profile_report.h"

using namespace gpu;

static int g_failures = 0;

#define CHECK(cond, msg)                                                                        \
    do {                                                                                        \
        if (!(cond)) {                                                                          \
            printf("FAIL [%s:%d]: %s\n", __FILE__, __LINE__, msg);                              \
            g_failures++;                                                                       \
        }                                                                                       \
    } while (0)

// --- Forced-failure allocator ---------------------------------------------------------------
static int g_alloc_budget = 0;   // new allocations left before failure

static MemoryBlock test_allocator(void*, void* ptr, uint32_t old_size, uint32_t new_size) {
    if (ptr == nullptr && new_size > 0) {              // alloc
        if (g_alloc_budget <= 0) { return {}; }
        g_alloc_budget--;
        return {std::malloc(new_size), new_size};
    }
    if (ptr != nullptr && new_size == 0) {             // free
        std::free(ptr);
        return {};
    }
    if (ptr != nullptr && new_size > 0) {              // realloc
        if (g_alloc_budget <= 0) { return {}; }
        g_alloc_budget--;
        return {std::realloc(ptr, new_size), new_size};
    }
    return {};
}

static void noop_log(LogLevel, Span<const char>, uint32_t, Span<const char>, void*) {}

// --- Arena -----------------------------------------------------------------------------------
static void test_arena_alignment_and_rewind() {
    printf("--- Test: arena alignment + rewind ---\n");
    alignas(64) uint8_t backing[4096];
    Arena arena(backing, sizeof(backing), noop_log, nullptr);

    // Over-aligned allocations
    void* p = arena.alloc(8, 64);
    CHECK(p != nullptr && (reinterpret_cast<uintptr_t>(p) % 64) == 0, "64-byte aligned alloc");
    p = arena.alloc(3, 32);
    CHECK(p != nullptr && (reinterpret_cast<uintptr_t>(p) % 32) == 0, "32-byte aligned alloc");
    p = arena.alloc(1, 1);
    CHECK(p != nullptr, "alignment-1 alloc");
    p = arena.alloc(16, 128);
    CHECK(p != nullptr && (reinterpret_cast<uintptr_t>(p) % 128) == 0, "128-byte aligned alloc");
    // Invalid alignment clamps to 16
    p = arena.alloc(4, 24);
    CHECK(p != nullptr && (reinterpret_cast<uintptr_t>(p) % 16) == 0, "invalid alignment clamps");

    // Mark/rewind
    auto m = arena.mark();
    void* q1 = arena.alloc(100, 8);
    void* q2 = arena.alloc(100, 8);
    arena.rewind(m);
    void* q3 = arena.alloc(100, 8);
    CHECK(q3 == q1, "rewind returns to the mark position");
    (void)q2;

    // Exact boundary: the last byte fits
    arena.rewind(arena.mark());
    size_t remaining = sizeof(backing) - (arena.mark().offset);
    void* exact = arena.alloc(remaining - 1, 1);
    CHECK(exact != nullptr, "exact-boundary alloc fits");
    void* one_over = arena.alloc(2, 1);
    CHECK(one_over == nullptr, "one byte over the boundary fails");
    CHECK(arena.overflowed(), "overflow flag set");

    // Millions of mark/rewind cycles never exhaust the arena
    arena.clear_overflow();
    arena.rewind(Arena::Marker{0});
    for (int i = 0; i < 1000000; ++i) {
        auto m2 = arena.mark();
        void* t = arena.alloc(64, 64);
        CHECK(t != nullptr, "cycle alloc fits");
        arena.rewind(m2);
    }
    CHECK(!arena.overflowed(), "mark/rewind cycles must not exhaust the arena");

    printf("  PASS\n");
}

struct ScopeProbe {
    Arena* arena;
    bool   saw_inner;
    bool   saw_outer;
};

static void nested_scope_worker(Arena* arena, int depth) {
    if (depth == 0) { return; }
    ScratchScope scope(*arena);
    void* p = arena->alloc(16, 16);
    CHECK(p != nullptr, "nested alloc fits");
    nested_scope_worker(arena, depth - 1);
}

static void test_arena_scopes() {
    printf("--- Test: arena scopes ---\n");
    alignas(64) uint8_t backing[1024];
    Arena arena(backing, sizeof(backing), noop_log, nullptr);

    const auto base = arena.mark();
    {
        ScratchScope outer(arena);
        void* a = arena.alloc(100, 8);
        CHECK(a != nullptr, "outer alloc");
        {
            ScratchScope inner(arena);
            void* b = arena.alloc(200, 8);
            CHECK(b != nullptr, "inner alloc");
            // After inner scope, arena is back at inner's mark
        }
        // allocate again after inner rewind — must reuse the same space
        void* c = arena.alloc(200, 8);
        (void)c;
    }
    // After the outer scope, everything is rewound to base
    void* d = arena.alloc(16, 8);
    CHECK(arena.mark().offset <= base.offset + 16 + 15, "scope rewind released space");

    // Nested recursion: LIFO discipline
    arena.rewind(base);
    nested_scope_worker(&arena, 64);
    CHECK(arena.mark().offset <= base.offset, "nested scopes rewind fully");

    printf("  PASS\n");
}

static void test_arena_concurrent() {
    printf("--- Test: arena concurrent (per-thread) ---\n");
    // Arenas are not thread-safe by design; each thread gets its own.
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([]() {
            alignas(64) uint8_t backing[2048];
            Arena arena(backing, sizeof(backing), noop_log, nullptr);
            for (int i = 0; i < 100000; ++i) {
                auto m = arena.mark();
                void* p = arena.alloc(32, 32);
                CHECK(p != nullptr && (reinterpret_cast<uintptr_t>(p) % 32) == 0,
                      "thread-local aligned alloc");
                arena.rewind(m);
            }
        });
    }
    for (auto& th : threads) { th.join(); }
    printf("  PASS\n");
}

// --- Vector -----------------------------------------------------------------------------------
static void test_vector_insert_growth() {
    printf("--- Test: vector insert/growth ---\n");
    Allocator alloc(test_allocator, nullptr);
    g_alloc_budget = 1000000;

    // Insert into empty vector (regression: used to read before the buffer)
    Vector<uint32_t> v(alloc);
    uint32_t val = 7;
    v.insert(v.begin(), val);
    CHECK(v.size() == 1 && v[0] == 7, "insert into empty vector");

    // Front / middle / back
    val = 1;
    v.insert(v.begin(), val);
    val = 5;
    v.insert(v.begin() + 1, val);
    val = 9;
    v.push_back(val);
    CHECK(v.size() == 4 && v[0] == 1 && v[1] == 5 && v[2] == 7 && v[3] == 9,
          "front/middle/back insertion order");

    // Capacity growth
    Vector<uint32_t> g(alloc);
    for (uint32_t i = 0; i < 1000; ++i) { g.push_back(i); }
    CHECK(g.size() == 1000, "growth count");
    bool ok = true;
    for (uint32_t i = 0; i < 1000; ++i) { if (g[i] != i) { ok = false; } }
    CHECK(ok, "growth contents");

    // Move
    Vector<uint32_t> moved = std::move(g);
    CHECK(moved.size() == 1000 && moved[500] == 500, "move preserves contents");

    // Clear + reuse
    moved.clear();
    CHECK(moved.is_empty(), "clear empties");
    moved.push_back(3);
    CHECK(moved.size() == 1 && moved[0] == 3, "reuse after clear");

    printf("  PASS\n");
}

static void test_vector_alloc_failure() {
    printf("--- Test: vector allocation failure ---\n");
    Allocator alloc(test_allocator, nullptr);
    g_alloc_budget = 1;   // one allocation total

    Vector<uint32_t> v(alloc);
    for (int i = 0; i < 8; ++i) { v.push_back(static_cast<uint32_t>(i)); }   // first push grows (budget -> 0)
    CHECK(v.size() == 8 && !v.alloc_failed(), "within capacity succeeds");
    v.push_back(8);   // needs growth -> fails
    CHECK(v.alloc_failed(), "allocation failure observable");
    CHECK(v.size() == 8 && v[0] == 0 && v[7] == 7, "no corruption after failed growth");
    // Further pushes keep failing without corruption
    v.push_back(9);
    CHECK(v.size() == 8 && v[0] == 0, "still consistent after repeated failure");

    // Insert failure: no write occurs
    g_alloc_budget = 0;
    Vector<uint32_t> w(alloc);
    uint32_t x = 42;
    w.insert(w.begin(), x);
    CHECK(w.alloc_failed() && w.is_empty(), "failed insert writes nothing");

    printf("  PASS\n");
}

// --- SlotMap ---------------------------------------------------------------------------------
struct Payload {
    uint32_t value = 0;
};

static void test_slotmap_generations() {
    printf("--- Test: slot map generations ---\n");
    Allocator alloc(test_allocator, nullptr);
    g_alloc_budget = 1000000;

    SlotMap<Payload> map(alloc, [](Payload*, void*) {});

    Handle<Payload> a = map.emplace(Payload{11});
    Handle<Payload> b = map.emplace(Payload{22});
    CHECK(a.h != 0 && b.h != 0 && a.h != b.h, "distinct handles");

    map.erase(a);
    Handle<Payload> c = map.emplace(Payload{33});
    // Slot is reused but the generation must differ -> different handle
    CHECK(c.h != a.h, "generation bump on slot reuse");
    CHECK(map[c].value == 33, "reused slot holds new payload");
    // The stale handle must not resolve to the new payload: decompose and
    // check generation mismatch at the erase level (asserts in debug).
    // Here we verify the handle encoding differs, which is the observable part.
    const uint64_t stale_gen = (a.h >> 32) & 0x7FFFFFFFull;
    const uint64_t new_gen   = (c.h >> 32) & 0x7FFFFFFFull;
    CHECK(stale_gen != new_gen, "stale handle carries an old generation");

    map.erase(b);
    map.erase(c);
    printf("  PASS\n");
}

// --- TwoLevelBitset ---------------------------------------------------------------------------
static void test_bitset() {
    printf("--- Test: two-level bitset ---\n");
    Allocator alloc(test_allocator, nullptr);
    g_alloc_budget = 1000000;
    TwoLevelBitset bits(alloc, 1024);
    CHECK(!bits.is_set(0), "fresh bitset is empty");

    uint32_t s0 = bits.set_leading_zero();
    CHECK(s0 == 0 && bits.is_set(0), "first slot is 0");
    uint32_t s1 = bits.set_leading_zero();
    CHECK(s1 == 1, "second slot is 1");
    bits.clear_bit(0);
    CHECK(!bits.is_set(0) && bits.is_set(1), "clear frees the slot");
    uint32_t s2 = bits.set_leading_zero();
    CHECK(s2 == 0, "freed slot is recycled");

    // Exhaustion: fill a small bitset and verify ~0u
    TwoLevelBitset small(alloc, 4);
    for (int i = 0; i < 4; ++i) { CHECK(small.set_leading_zero() != ~0u, "slot fits"); }
    CHECK(small.set_leading_zero() == ~0u, "exhausted bitset reports ~0u");
    printf("  PASS\n");
}

// --- Enum bitwise operators ---------------------------------------------------------------------
enum class TestFlags : uint16_t { None = 0, A = 0x01, B = 0x02, C = 0x04 };
IZ_DEFINE_BITWISE_OPS(TestFlags);

static void test_enum_ops() {
    printf("--- Test: enum bitwise ops ---\n");
    TestFlags f = TestFlags::A;
    f |= TestFlags::B;
    CHECK(any(f & TestFlags::A) && any(f & TestFlags::B), "compound OR modifies lhs");
    f &= TestFlags::B;
    CHECK(!any(f & TestFlags::A) && any(f & TestFlags::B), "compound AND modifies lhs");
    f ^= TestFlags::C;
    CHECK(any(f & TestFlags::B) && any(f & TestFlags::C), "compound XOR modifies lhs");
    CHECK(any(TestFlags::A | TestFlags::C), "operator|");
    CHECK(!any(TestFlags::A & TestFlags::B), "operator&");
    CHECK(any(TestFlags::A ^ TestFlags::A) == false, "operator^ self cancels");
    CHECK(any(TestFlags::None) == false, "any(None) is false");
    printf("  PASS\n");
}

// --- Span/extern templates ----------------------------------------------------------------------
static void test_span() {
    printf("--- Test: span ---\n");
    uint32_t arr[4] = {1, 2, 3, 4};
    Span<uint32_t> s(arr, 4);
    CHECK(s.size() == 4 && s[2] == 3, "span basics");
    Span<const uint32_t> cs = s;
    CHECK(cs.size() == 4, "span const conversion");
    Span<const uint8_t> bytes = s.as_bytes();
    CHECK(bytes.size() == 16, "span as_bytes");
    printf("  PASS\n");
}

// --- SlotMap stress: segments, clear/reuse, invalidate, failure -------------------------------
static void test_slotmap_stress() {
    printf("--- Test: slot map stress ---\n");
    Allocator alloc(test_allocator, nullptr);
    g_alloc_budget = 10000000;

    SlotMap<Payload> map(alloc, [](Payload*, void*) {});

    // Thousands of entries force multiple segment expansions; every slot
    // must be reachable exactly once from the free list.
    std::vector<Handle<Payload>> handles;
    handles.reserve(10000);
    for (uint32_t i = 0; i < 10000; ++i) {
        Handle<Payload> h = map.emplace(Payload{i});
        CHECK(h.h != 0, "emplace failed during segment expansion");
        handles.push_back(h);
    }
    bool ok = true;
    for (uint32_t i = 0; i < 10000; ++i) {
        if (map[handles[i]].value != i) { ok = false; }
    }
    CHECK(ok, "all segment entries hold their payload");

    // Erase + reuse: generations advance and old handles are stale.
    Handle<Payload> first = handles[0];
    map.erase(first);
    Handle<Payload> reused = map.emplace(Payload{9999});
    CHECK(reused.h != first.h, "reuse bumps generation");
    CHECK(map[reused].value == 9999, "reused slot payload");
    map.erase(reused);

    // invalidate: the record stays, the handle becomes stale.
    Handle<Payload> keep = map.emplace(Payload{7});
    Payload* kept_record = &map[keep];
    Handle<Payload> stale = keep;
    map.invalidate(keep);
    Handle<Payload> found = map.find_handle(kept_record);
    CHECK(found.h != 0 && found.h != stale.h, "invalidate makes the old handle stale");
    CHECK(kept_record->value == 7, "record survives invalidation");
    map.erase(found);

    // Clear + reuse: the free-list head must reset (regression: it dangled
    // into freed segments).
    map.clear();
    Handle<Payload> after_clear = map.emplace(Payload{42});
    CHECK(after_clear.h != 0, "emplace after clear works");
    CHECK(map[after_clear].value == 42, "payload after clear");
    map.erase(after_clear);

    // Allocation failure during segment expansion: emplace returns {} and the
    // map stays usable.
    g_alloc_budget = 0;
    SlotMap<Payload> failmap(alloc, [](Payload*, void*) {});
    Handle<Payload> f = failmap.emplace(Payload{1});
    CHECK(f.h == 0, "emplace reports allocation failure");
    g_alloc_budget = 10000000;
    Handle<Payload> f2 = failmap.emplace(Payload{2});
    CHECK(f2.h != 0, "map usable after failed expansion");

    printf("  PASS\n");
}

// --- Vector::insert with nontrivial T ----------------------------------------------------------
static void test_vector_nontrivial_insert() {
    printf("--- Test: vector nontrivial insert ---\n");
    Allocator alloc(test_allocator, nullptr);
    g_alloc_budget = 10000000;

    Vector<std::string> v(alloc);
    // Append via insert at end
    std::string s0 = "zero";
    v.insert(v.end(), s0);
    // Middle insert
    std::string s1 = "one";
    v.insert(v.end(), s1);
    std::string sm = "mid";
    v.insert(v.begin() + 1, sm);
    // Begin insert (empty vector case covered above; here at begin with data)
    std::string sb = "first";
    v.insert(v.begin(), sb);
    CHECK(v.size() == 4, "insert count");
    CHECK(v[0] == "first" && v[1] == "zero" && v[2] == "mid" && v[3] == "one",
          "insert order with nontrivial T");
    // Growth during insert (force realloc while inserting in the middle)
    Vector<std::string> g(alloc);
    for (int i = 0; i < 20; ++i) { g.push_back(std::string("v") + std::to_string(i)); }
    std::string ins = "inserted";
    g.insert(g.begin() + 10, ins);
    CHECK(g.size() == 21 && g[10] == "inserted" && g[9] == "v9" && g[11] == "v10",
          "middle insert after growth preserves order");

    printf("  PASS\n");
}

// --- Bindless profile capability evaluation -----------------------------------
// Negative capability tests: the evaluator must reject a device lacking ANY
// mandatory requirement with the exact requirement listed, must accept
// optional-feature absence (selecting fallbacks), and must never gate on the
// Vulkan version or generation name.

static VulkanProfileFeatures full_features() {
    VulkanProfileFeatures f;
    f.api_version                    = (1u << 22) | (2u << 12);  // 1.2
    f.buffer_device_address          = true;
    f.shader_int64                   = true;
    f.scalar_block_layout            = true;
    f.sampled_image_nonuniform_indexing = true;
    f.storage_image_nonuniform_indexing = true;
    f.sampler_array_indexing            = true;
    f.sampled_image_update_after_bind   = true;
    f.storage_image_update_after_bind   = true;
    f.descriptor_binding_partially_bound              = true;
    f.runtime_descriptor_array                        = true;
    f.descriptor_binding_variable_count               = true;
    f.descriptor_binding_update_unused_while_pending  = true;
    f.draw_indirect_count = true;
    f.timeline_semaphore  = true;
    f.dynamic_rendering   = true;
    f.synchronization2    = true;
    f.max_sampled_descriptors = 1'000'000;
    f.max_storage_descriptors = 1'000'000;
    f.max_samplers            = 1'000'000;
    f.combined_descriptor_budget = 1'000'000;
    return f;
}

static bool report_has(const VulkanProfileReport& r, VulkanProfileRequirement req) {
    return r.missing_has(req);
}

static void test_profile_report() {
    printf("--- Test: bindless profile evaluation ---\n");

    // Full feature set -> supported, direct descriptor path, capacities
    // clamped to the profile floor (not the raw limit).
    {
        VulkanProfileReport r = evaluate_vulkan_bindless_profile(full_features());
        CHECK(r.supported, "full feature set must be supported");
        CHECK(r.missing_count == 0, "full feature set must have no missing requirements");
        CHECK(r.descriptor_snapshots == 1, "update-unused-while-pending -> direct path (1 snapshot)");
        CHECK(r.sampled_image_capacity == kMinBindlessSampledImages, "capacity clamps to profile floor");
        CHECK(r.storage_image_capacity == kMinBindlessStorageImages, "storage capacity clamps to floor");
        CHECK(r.sampler_capacity == kMinBindlessSamplers, "sampler capacity clamps to floor");
        CHECK(r.dynamic_rendering && r.synchronization2, "optional conveniences echoed");
        CHECK(!r.missing_has(VulkanProfileRequirement::BufferDeviceAddress), "missing_has false when nothing missing");
    }

    // Every mandatory requirement, individually absent -> unsupported + listed.
    struct ReqCase {
        VulkanProfileRequirement req;
        void (*apply)(VulkanProfileFeatures&);
    };
    ReqCase cases[] = {
        {VulkanProfileRequirement::BufferDeviceAddress, [](VulkanProfileFeatures& f) { f.buffer_device_address = false; }},
        {VulkanProfileRequirement::ShaderInt64, [](VulkanProfileFeatures& f) { f.shader_int64 = false; }},
        {VulkanProfileRequirement::ScalarBlockLayout, [](VulkanProfileFeatures& f) { f.scalar_block_layout = false; }},
        {VulkanProfileRequirement::SampledNonUniformIndexing, [](VulkanProfileFeatures& f) { f.sampled_image_nonuniform_indexing = false; }},
        {VulkanProfileRequirement::StorageNonUniformIndexing, [](VulkanProfileFeatures& f) { f.storage_image_nonuniform_indexing = false; }},
        {VulkanProfileRequirement::SamplerArrayIndexing, [](VulkanProfileFeatures& f) { f.sampler_array_indexing = false; }},
        {VulkanProfileRequirement::SampledUpdateAfterBind, [](VulkanProfileFeatures& f) { f.sampled_image_update_after_bind = false; }},
        {VulkanProfileRequirement::StorageUpdateAfterBind, [](VulkanProfileFeatures& f) { f.storage_image_update_after_bind = false; }},
        {VulkanProfileRequirement::PartiallyBound, [](VulkanProfileFeatures& f) { f.descriptor_binding_partially_bound = false; }},
        {VulkanProfileRequirement::RuntimeDescriptorArray, [](VulkanProfileFeatures& f) { f.runtime_descriptor_array = false; }},
        {VulkanProfileRequirement::DescriptorVariableCount, [](VulkanProfileFeatures& f) { f.descriptor_binding_variable_count = false; }},
        {VulkanProfileRequirement::DrawIndirectCount, [](VulkanProfileFeatures& f) { f.draw_indirect_count = false; }},
        {VulkanProfileRequirement::TimelineSemaphore, [](VulkanProfileFeatures& f) { f.timeline_semaphore = false; }},
    };
    for (auto& c : cases) {
        VulkanProfileFeatures f = full_features();
        c.apply(f);
        VulkanProfileReport r = evaluate_vulkan_bindless_profile(f);
        CHECK(!r.supported, "missing requirement must reject the profile");
        CHECK(report_has(r, c.req), "the exact missing requirement must be listed");
        CHECK(r.missing_count == 1, "exactly one missing requirement for a single-feature absence");
    }

    // Capacity floors are mandatory; sub-floor limits reject with the
    // capacity requirement listed and the raw (sub-floor) value reported.
    {
        VulkanProfileFeatures f = full_features();
        f.max_sampled_descriptors = 64;
        VulkanProfileReport r = evaluate_vulkan_bindless_profile(f);
        CHECK(!r.supported, "sub-floor sampled capacity must reject");
        CHECK(report_has(r, VulkanProfileRequirement::SampledImageCapacity), "capacity requirement listed");
        CHECK(r.sampled_image_capacity == 64, "sub-floor capacity reported as-is");

        f = full_features();
        f.max_samplers = 0;
        r = evaluate_vulkan_bindless_profile(f);
        CHECK(!r.supported, "zero sampler capacity must reject");
        CHECK(report_has(r, VulkanProfileRequirement::SamplerCapacity), "sampler capacity listed");
    }

    // The shared combined update-after-bind budget is mandatory: per-type
    // ceilings that individually pass cannot fit the arrays together.
    {
        VulkanProfileFeatures f = full_features();
        f.combined_descriptor_budget = 2000;   // < 1024 + 1024 + 256
        VulkanProfileReport r = evaluate_vulkan_bindless_profile(f);
        CHECK(!r.supported, "combined budget below the floors' sum must reject");
        CHECK(report_has(r, VulkanProfileRequirement::CombinedDescriptorBudget),
              "combined budget requirement listed");
        CHECK(r.combined_descriptor_budget == 2000, "combined budget echoed");

        f = full_features();
        f.combined_descriptor_budget = 2304;   // exactly the floors' sum
        CHECK(evaluate_vulkan_bindless_profile(f).supported,
              "combined budget equal to the floors' sum passes");

        f = full_features();
        f.combined_descriptor_budget = 0;      // zero budget (no UAB pools)
        CHECK(!evaluate_vulkan_bindless_profile(f).supported,
              "zero combined budget must reject");
    }

    // Optional features: absence selects fallbacks, never rejection.
    {
        VulkanProfileFeatures f = full_features();
        f.descriptor_binding_update_unused_while_pending = false;
        VulkanProfileReport r = evaluate_vulkan_bindless_profile(f);
        CHECK(r.supported, "missing update-unused-while-pending must not reject (snapshot path)");
        CHECK(r.descriptor_snapshots == 2, "snapshot path selected (2 backing sets)");

        f = full_features();
        f.dynamic_rendering = false;
        r = evaluate_vulkan_bindless_profile(f);
        CHECK(r.supported, "missing dynamic rendering must not reject (private render pass)");
        CHECK(!r.dynamic_rendering, "dynamic rendering absence recorded");

        f = full_features();
        f.synchronization2 = false;
        r = evaluate_vulkan_bindless_profile(f);
        CHECK(r.supported, "missing synchronization2 must not reject (legacy barriers)");
        CHECK(!r.synchronization2, "synchronization2 absence recorded");
    }

    // Version is never the gate: all features on a 1.1-era report still pass;
    // no features on a 1.4 report fail.
    {
        VulkanProfileFeatures f = full_features();
        f.api_version = (1u << 22) | (1u << 12);
        CHECK(evaluate_vulkan_bindless_profile(f).supported, "version must not gate support");

        f = full_features();
        f.api_version = (1u << 22) | (4u << 12);
        f.buffer_device_address = false;
        f.timeline_semaphore    = false;
        f.sampled_image_nonuniform_indexing = false;
        CHECK(!evaluate_vulkan_bindless_profile(f).supported, "high version cannot paper over missing features");
    }

    // All-false snapshot: every requirement listed, none omitted.
    {
        VulkanProfileReport r = evaluate_vulkan_bindless_profile({});
        CHECK(!r.supported, "empty feature snapshot must reject");
        CHECK(r.missing_count == 17, "all 17 requirements listed when nothing is supported");
        CHECK(r.descriptor_snapshots == 2, "snapshot path recorded for empty snapshot");
    }

    // Name table sanity.
    {
        for (int i = 0; i < static_cast<int>(VulkanProfileRequirement::ValidCount); ++i) {
            const char* n = vulkan_requirement_name(static_cast<VulkanProfileRequirement>(i));
            CHECK(n != nullptr && n[0] != '\0', "requirement names must be non-empty");
        }
        CHECK(std::string(vulkan_requirement_name(VulkanProfileRequirement::ValidCount)) == "unknown",
              "out-of-range requirement names report unknown");
    }
}

// --- Static shader ABI manifest ----------------------------------------------
// Cross-profile ABI contract: public handle widths, push-constant root
// shapes, and the shared irregular data structures (mirrored by
// tests/shaders/abi_test.slang under scalar block layout + row-major). If
// any C++ layout diverges from the shader's, the GPU ABI test (api tests)
// lands a wrong value — these static checks pin the C++ side.

// Natural alignment (4): the shader's AbiInner has no alignas; forcing
// alignment 8 would pad the size from 12 to 16 and break the ABI contract.
struct AbiInner {
    uint32_t a;
    float    b;
    uint32_t c;
};
struct alignas(8) AbiNested {
    AbiInner inner;
    uint64_t ptr;
};
// Natural alignment (8) with EXPLICIT tail padding: Slang's scalar layout
// does not tail-pad structs, so the shader mirrors the C++ tail member by
// name (abi_test.slang tail_pad @76). C++ sizeof is 80; the extracted shader
// manifest test proves the artifact matches exactly (a mismatch would shift
// every element of a C++ struct array).
struct AbiRoot {
    uint32_t  u;
    float     f;
    uint64_t  gpu_ptr;   // GpuPtr member
    uint64_t  tex;       // TextureView handle
    uint64_t  samp;      // SamplerId handle
    AbiNested nested;
    float     arr[3];
    uint32_t  s;
    uint32_t  pad;
    uint32_t  tail_pad;  // explicit tail padding -> size 80
};
// Push-constant root shapes (1 or 2 pointers = 8/16 bytes).
struct alignas(8) AbiRoot1 {
    GpuPtr data;
};
struct alignas(8) AbiRoot2 {
    GpuPtr vert;
    GpuPtr frag;
};

static_assert(sizeof(GpuPtr) == 8, "GpuPtr must be a 64-bit device address");
static_assert(sizeof(TextureView) == 8, "TextureView must be 64-bit");
static_assert(sizeof(SamplerId) == 8, "SamplerId must be 64-bit");
static_assert(sizeof(AbiInner) == 12, "AbiInner size (shader: uint32,float,uint32)");
static_assert(sizeof(AbiNested) == 24, "AbiNested size (inner @0, ptr @16)");
static_assert(sizeof(AbiRoot) == 80, "AbiRoot size (natural alignment + explicit tail)");
static_assert(alignof(AbiRoot) == 8, "AbiRoot alignment 8");
static_assert(offsetof(AbiRoot, u) == 0, "AbiRoot.u @0");
static_assert(offsetof(AbiRoot, f) == 4, "AbiRoot.f @4");
static_assert(offsetof(AbiRoot, gpu_ptr) == 8, "AbiRoot.gpu_ptr @8");
static_assert(offsetof(AbiRoot, tex) == 16, "AbiRoot.tex @16");
static_assert(offsetof(AbiRoot, samp) == 24, "AbiRoot.samp @24");
static_assert(offsetof(AbiRoot, nested) == 32, "AbiRoot.nested @32");
static_assert(offsetof(AbiRoot, arr) == 56, "AbiRoot.arr @56");
static_assert(offsetof(AbiRoot, s) == 68, "AbiRoot.s @68");
static_assert(offsetof(AbiRoot, pad) == 72, "AbiRoot.pad @72");
static_assert(offsetof(AbiRoot, tail_pad) == 76, "AbiRoot.tail_pad @76");
// 8/16-bit storage ABI (capability-gated GPU test): natural alignment 4,
// size 12 — Slang emits the same scalar layout.
struct Int8_16Root {
    uint16_t s16;
    uint8_t  b8[3];
    uint32_t tail;
};
static_assert(sizeof(Int8_16Root) == 12, "Int8_16Root size (scalar layout)");
static_assert(offsetof(Int8_16Root, s16) == 0, "Int8_16Root.s16 @0");
static_assert(offsetof(Int8_16Root, b8) == 2, "Int8_16Root.b8 @2");
static_assert(offsetof(Int8_16Root, tail) == 8, "Int8_16Root.tail @8");
static_assert(sizeof(AbiRoot1) == 8, "compute root: one 64-bit pointer");
static_assert(sizeof(AbiRoot2) == 16, "graphics root: two 64-bit pointers");

static void test_abi_manifest() {
    printf("--- Test: shader ABI manifest (static) ---\n");
    // The static_asserts above pin the layout; runtime re-checks keep the
    // manifest visible in the test output even where asserts are disabled.
    CHECK(sizeof(AbiRoot) == 80 && offsetof(AbiRoot, nested) == 32,
          "AbiRoot layout matches the shader contract");
    CHECK(sizeof(AbiRoot1) == 8 && sizeof(AbiRoot2) == 16,
          "push-constant root shapes are 8/16 bytes");
}

// --- Dispatch capability matrix (mock, no device) --------------------------------
// The Vulkan 1.2 route's selection logic: required = dynamic rendering + sync2
// (1.3 cores or KHR extensions); copy_commands2 + extended_dynamic_state are
// optional with private fallbacks; force overrides choose the fallback even
// when the modern path exists; 1.3+ never needs the promoted names.
static void test_dispatch_capabilities() {
    printf("--- Test: dispatch capability matrix ---\n");
    const uint32_t v12 = (1u << 22) | (2u << 12);
    const uint32_t v13 = (1u << 22) | (3u << 12);
    const uint32_t v14 = (1u << 22) | (4u << 12);

    // 1.3+ core path: all families core, no promoted names required, no fallback.
    {
        auto c = derive_dispatch_capabilities(v14, false, false, false, false, false, false);
        CHECK(c.dynamic_rendering && c.synchronization2 && c.copy_commands2 && c.extended_dynamic_state,
              "1.4 core path provides all families");
        CHECK(c.dynamic_rendering_is_core && c.synchronization2_is_core &&
                  c.copy_commands2_is_core && c.extended_dynamic_state_is_core,
              "1.4 core path marks every family core");
        CHECK(!c.use_legacy_copy_commands && !c.use_static_graphics_state,
              "1.4 core path needs no fallbacks");
    }
    // 1.2 with both optional families.
    {
        auto c = derive_dispatch_capabilities(v12, true, true, true, true, false, false);
        CHECK(c.dynamic_rendering && c.synchronization2 && c.copy_commands2 && c.extended_dynamic_state,
              "1.2 + all four extensions provides every family");
        CHECK(!c.dynamic_rendering_is_core, "1.2 families are not core");
        CHECK(!c.use_legacy_copy_commands && !c.use_static_graphics_state,
              "no fallbacks when all extensions are present");
    }
    // 1.2 with only dynamic rendering + sync2 (the dzn case): copy + static
    // state fallbacks selected, device is NOT rejected for the missing pairs.
    {
        auto c = derive_dispatch_capabilities(v12, true, true, false, false, false, false);
        CHECK(c.dynamic_rendering && c.synchronization2, "required families present");
        CHECK(!c.copy_commands2 && !c.extended_dynamic_state, "optional families absent");
        CHECK(c.use_legacy_copy_commands, "legacy copy selected");
        CHECK(c.use_static_graphics_state, "static graphics state selected");
    }
    // 1.2 with copy2 but no extended dynamic state.
    {
        auto c = derive_dispatch_capabilities(v12, true, true, true, false, false, false);
        CHECK(!c.use_legacy_copy_commands, "copy2 present -> modern copy path");
        CHECK(c.use_static_graphics_state, "no extended dynamic state -> static variants");
    }
    // 1.2 missing dynamic rendering or sync2: NOT selectable (the evaluator
    // gate rejects separately) — but the derivation still reports the absence.
    {
        auto c = derive_dispatch_capabilities(v12, false, true, true, true, false, false);
        CHECK(!c.dynamic_rendering, "dynamic rendering absent");
        CHECK(!c.synchronization2_is_core && !c.dynamic_rendering_is_core, "not core on 1.2");
    }
    // Force overrides win over an available modern path (white-box testing).
    {
        auto c = derive_dispatch_capabilities(v14, false, false, false, false, true, true);
        CHECK(c.use_legacy_copy_commands && c.use_static_graphics_state,
              "force overrides select the fallbacks on a modern device");
        auto c2 = derive_dispatch_capabilities(v14, false, false, false, false, true, false);
        CHECK(c2.use_legacy_copy_commands && !c2.use_static_graphics_state,
              "force flags are independent");
    }
    // 1.3 core path with no promoted names.
    {
        auto c = derive_dispatch_capabilities(v13, false, false, false, false, false, false);
        CHECK(c.dynamic_rendering && c.synchronization2 && c.copy_commands2 && c.extended_dynamic_state,
              "1.3 cores provide all families without any extension name");
        CHECK(c.dynamic_rendering_is_core, "1.3 families are core");
        CHECK(!c.use_legacy_copy_commands && !c.use_static_graphics_state, "no fallbacks on 1.3");
    }

    // --- Bindless enabled-extension list (route-aware) --------------------------
    // Explicit 1.2/1.3 boundary cases: the version word comparison must treat
    // exactly >= 1.3 as core (a wrong word would misclassify 1.2).
    {
        auto expect = [](uint32_t version, bool copy2, bool eds, bool wsi,
                         std::initializer_list<const char*> want) {
            const char* out[8] = {};
            const uint32_t n = gpu::build_bindless_enabled_extensions(wsi, version, copy2, eds,
                                                                      out, 8);
            bool ok = n == want.size();
            uint32_t i = 0;
            for (const char* w : want) {
                if (i >= 8 || out[i] == nullptr || strcmp(out[i], w) != 0) { ok = false; }
                ++i;
            }
            return ok;
        };
        CHECK(expect(v12, false, false, true, {"VK_KHR_swapchain", "VK_KHR_dynamic_rendering",
                                               "VK_KHR_synchronization2"}),
              "1.2: swapchain + the two required promoted KHR names");
        CHECK(expect(v12, true, false, true, {"VK_KHR_swapchain", "VK_KHR_dynamic_rendering",
                                              "VK_KHR_synchronization2", "VK_KHR_copy_commands2"}),
              "1.2 + copy2: copy-commands2 included");
        CHECK(expect(v12, true, true, true, {"VK_KHR_swapchain", "VK_KHR_dynamic_rendering",
                                             "VK_KHR_synchronization2", "VK_KHR_copy_commands2",
                                             "VK_EXT_extended_dynamic_state"}),
              "1.2 + copy2 + EDS: both optionals included");
        CHECK(expect(v13, true, true, true, {"VK_KHR_swapchain"}),
              "1.3: swapchain only — even with copy2 + EDS advertised, NO promoted/EXT name");
        CHECK(expect(v13, true, true, false, {}),
              "1.3 headless: no extensions at all");
        CHECK(expect(v12, true, true, false, {"VK_KHR_dynamic_rendering",
                                              "VK_KHR_synchronization2",
                                              "VK_KHR_copy_commands2",
                                              "VK_EXT_extended_dynamic_state"}),
              "1.2 headless: promoted + optional names only");
        // Capacity contract: the count still reflects every entry on truncation.
        {
            const char* out[2] = {};
            const uint32_t n = gpu::build_bindless_enabled_extensions(
                true, v12, true, true, out, 2);
            CHECK(n == 5 && strcmp(out[0], "VK_KHR_swapchain") == 0 &&
                      strcmp(out[1], "VK_KHR_dynamic_rendering") == 0,
                  "truncated builder reports the full count");
        }
    }
}

// --- Extracted shader-layout manifest (ABI §11.1) ------------------------------
// Parses the COMPILED .spv artifact and extracts struct member offsets, sizes,
// and array strides (OpMemberDecorate Offset, OpDecorate ArrayStride, type
// widths from OpTypeInt/Float/Struct/Array/RuntimeArray/Pointer, lengths from
// OpConstant). The comparison against C++ sizeof/alignof/offsetof fails on any
// divergence. The shader layout is the source of truth for what the driver
// sees: a C++ struct array must stride identically, so C++ and the shader are
// kept in agreement explicitly (e.g. AbiRoot's named tail_pad).

namespace spirv_layout {

enum class Kind : uint8_t { Unknown = 0, Int, Float, Struct, Array, RuntimeArray, Ptr };

struct Type {
    Kind                  kind = Kind::Int;     // default; set explicitly per id
    uint32_t              width = 0;            // Int/Float: bit width
    std::vector<uint32_t> members;              // Struct: member type ids
    uint32_t              elem_type_id = 0;     // Array: element type id
    uint32_t              length_id    = 0;     // Array: length constant id
};

struct Module {
    std::vector<Type>                    types;   // id-indexed
    std::vector<std::vector<uint32_t>>   member_offsets;   // per struct id
    std::vector<uint32_t>                array_strides;
    std::vector<uint32_t>                constants;
};

bool parse(const std::vector<uint8_t>& spv, Module& m) {
    if (spv.size() < 20) { return false; }
    auto rd = [&](size_t off) -> uint32_t {
        return uint32_t(spv[off]) | (uint32_t(spv[off + 1]) << 8) |
               (uint32_t(spv[off + 2]) << 16) | (uint32_t(spv[off + 3]) << 24);
    };
    if (rd(0) != 0x07230203u) { return false; }
    m.types.resize(rd(12) + 1);
    m.member_offsets.resize(m.types.size());
    m.array_strides.assign(m.types.size(), 0);
    m.constants.assign(m.types.size(), 0);
    size_t i = 20;
    while (i + 4 <= spv.size()) {
        uint32_t word = rd(i);
        uint32_t op   = word & 0xFFFFu;
        uint32_t n    = word >> 16;
        if (n == 0 || i + 4 * n > spv.size()) { return false; }
        auto opid = [&](uint32_t k) -> uint32_t { return rd(i + 4 + 4 * k); };
        switch (op) {
            case 21: {  // OpTypeInt <result> <width> <signed>
                uint32_t id = opid(0);
                if (id < m.types.size()) { m.types[id] = {Kind::Int, opid(1), {}, 0, 0}; }
                break;
            }
            case 22: {  // OpTypeFloat <result> <width>
                uint32_t id = opid(0);
                if (id < m.types.size()) { m.types[id] = {Kind::Float, opid(1), {}, 0, 0}; }
                break;
            }
            case 30: {  // OpTypeStruct <result> <member types...>
                uint32_t id = opid(0);
                if (id < m.types.size()) {
                    Type t{Kind::Struct, 0, {}, 0, 0};
                    for (uint32_t k = 1; k + 1 < n; ++k) { t.members.push_back(opid(k)); }
                    m.types[id] = std::move(t);
                }
                break;
            }
            case 28: {  // OpTypeArray <result> <elementType> <lengthId>
                uint32_t id = opid(0);
                if (id < m.types.size()) { m.types[id] = {Kind::Array, 0, {}, opid(1), opid(2)}; }
                break;
            }
            case 29: {  // OpTypeRuntimeArray <result> <elementType>
                uint32_t id = opid(0);
                if (id < m.types.size()) { m.types[id] = {Kind::RuntimeArray, 0, {}, opid(1), 0}; }
                break;
            }
            case 32: {  // OpTypePointer <result> <storageClass> <type>
                uint32_t id = opid(0);
                if (id < m.types.size()) { m.types[id] = {Kind::Ptr, 0, {}, 0, 0}; }
                break;
            }
            case 43:  // OpConstant <resultType> <resultId> <value>
                if (n >= 4) {
                    uint32_t id = opid(1);   // result ID
                    if (id < m.constants.size()) { m.constants[id] = opid(2); }
                }
                break;
            case 72: {  // OpMemberDecorate <struct> <member> <decoration> [operands]
                if (n >= 4) {
                    uint32_t sid = opid(0), member = opid(1), dec = opid(2);
                    if (dec == 35 && sid < m.member_offsets.size()) {
                        auto& offs = m.member_offsets[sid];
                        if (offs.size() <= member) { offs.resize(member + 1, 0); }
                        offs[member] = opid(3);
                    }
                }
                break;
            }
            case 71:  // OpDecorate <target> <decoration> [operands]
                if (n >= 3) {
                    uint32_t tid = opid(0), dec = opid(1);
                    if (dec == 6 && tid < m.array_strides.size()) { m.array_strides[tid] = opid(2); }
                }
                break;
            default: break;
        }
        i += 4 * n;
    }
    return true;
}

uint32_t type_size(const Module& m, uint32_t id, uint32_t depth = 0) {
    if (depth > 8 || id >= m.types.size()) { return 0; }
    const Type& t = m.types[id];
    switch (t.kind) {
        case Kind::Int:
        case Kind::Float: return t.width / 8;
        case Kind::Ptr:   return 8;
        case Kind::Array: {
            uint32_t stride = id < m.array_strides.size() ? m.array_strides[id] : 4;
            uint32_t len    = t.length_id < m.constants.size() ? m.constants[t.length_id] : 0;
            return stride * len;
        }
        case Kind::Struct: {
            const auto& offs = m.member_offsets[id];
            uint32_t size = 0;
            for (uint32_t k = 0; k < t.members.size(); ++k) {
                uint32_t off = k < offs.size() ? offs[k] : 0;
                size = std::max(size, off + type_size(m, t.members[k], depth + 1));
            }
            return size;
        }
        case Kind::Unknown:
        default: return 0;
    }
}

}  // namespace spirv_layout

// Finds a compiled shader artifact (exe-relative then cwd-relative candidates).
static std::string find_shader_artifact(const char* name) {
    // Multi-config (Debug/Release subdir) and single-config layouts. The tag
    // encodes profile + profile version + SPIR-V version, so artifacts from a
    // different profile version can never be loaded by mistake.
    const std::string native_dir = "shaders/" IZ_NATIVE_ARTIFACT_TAG "/";
    const std::string candidates[] = {
        native_dir,
        "bin/shaders/" IZ_NATIVE_ARTIFACT_TAG "/",
        "../bin/shaders/" IZ_NATIVE_ARTIFACT_TAG "/",
        "bin/Debug/shaders/" IZ_NATIVE_ARTIFACT_TAG "/",
        "../bin/Debug/shaders/" IZ_NATIVE_ARTIFACT_TAG "/",
        "bin/Release/shaders/" IZ_NATIVE_ARTIFACT_TAG "/",
        "../bin/Release/shaders/" IZ_NATIVE_ARTIFACT_TAG "/",
        "build/bin/shaders/" IZ_NATIVE_ARTIFACT_TAG "/",
        "build/bin/Debug/shaders/" IZ_NATIVE_ARTIFACT_TAG "/",
        "build/bin/Release/shaders/" IZ_NATIVE_ARTIFACT_TAG "/",
    };
    for (auto& c : candidates) {
        std::string p = c + name;
        if (std::filesystem::exists(p)) { return p; }
    }
    return std::string("shaders/") + IZ_NATIVE_ARTIFACT_TAG + "/" + name;
}

// Bindless variant of find_shader_artifact (SPIR-V 1.5 directory).
static std::string find_shader_artifact_bindless(const char* name) {
    const std::string candidates[] = {
        "shaders/" IZ_BINDLESS_ARTIFACT_TAG "/",
        "bin/shaders/" IZ_BINDLESS_ARTIFACT_TAG "/",
        "../bin/shaders/" IZ_BINDLESS_ARTIFACT_TAG "/",
        "bin/Debug/shaders/" IZ_BINDLESS_ARTIFACT_TAG "/",
        "../bin/Debug/shaders/" IZ_BINDLESS_ARTIFACT_TAG "/",
        "bin/Release/shaders/" IZ_BINDLESS_ARTIFACT_TAG "/",
        "../bin/Release/shaders/" IZ_BINDLESS_ARTIFACT_TAG "/",
        "build/bin/shaders/" IZ_BINDLESS_ARTIFACT_TAG "/",
        "build/bin/Debug/shaders/" IZ_BINDLESS_ARTIFACT_TAG "/",
        "build/bin/Release/shaders/" IZ_BINDLESS_ARTIFACT_TAG "/",
    };
    for (auto& c : candidates) {
        std::string p = c + name;
        if (std::filesystem::exists(p)) { return p; }
    }
    return std::string("shaders/") + IZ_BINDLESS_ARTIFACT_TAG + "/" + name;
}

static std::vector<uint8_t> read_file(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    std::vector<uint8_t> out;
    if (!f) { return out; }
    std::fseek(f, 0, SEEK_END);
    long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (sz > 0) {
        out.resize(static_cast<size_t>(sz));
        if (std::fread(out.data(), 1, out.size(), f) != out.size()) { out.clear(); }
    }
    std::fclose(f);
    return out;
}

// Asserts that some struct in the artifact has exactly the expected member
// offsets and size (matched by signature — Slang emits no struct names).
static bool manifest_matches(const std::vector<uint8_t>& spv,
                             const std::vector<uint32_t>& expected_offsets,
                             uint32_t expected_size,
                             const char* label) {
    spirv_layout::Module m;
    if (!spirv_layout::parse(spv, m)) {
        CHECK(false, "SPIR-V parse failed");
        return false;
    }
    bool found = false;
    for (uint32_t id = 0; id < m.types.size(); ++id) {
        if (m.types[id].kind != spirv_layout::Kind::Struct) { continue; }
        const auto& offs = m.member_offsets[id];
        std::vector<uint32_t> got(offs.begin(), offs.end());
        got.resize(m.types[id].members.size(), 0);
        if (got == expected_offsets) {
            CHECK(spirv_layout::type_size(m, id) == expected_size,
                  "shader struct size matches the manifest");
            found = true;
        }
    }
    CHECK(found, label ? "shader struct layout matches the C++ manifest" : "");
    return found;
}

// Explicit array member check: AbiRoot member 6 (arr) must be a float[3]
// with array stride 4 (the C++ float arr[3] strides 4).
static void check_abi_root_array(const std::vector<uint8_t>& spv) {
    spirv_layout::Module m;
    if (!spirv_layout::parse(spv, m)) {
        CHECK(false, "SPIR-V parse failed (array check)");
        return;
    }
    bool checked = false;
    for (uint32_t id = 0; id < m.types.size() && !checked; ++id) {
        const auto& t = m.types[id];
        if (t.kind != spirv_layout::Kind::Struct) { continue; }
        const auto& offs = m.member_offsets[id];
        std::vector<uint32_t> got(offs.begin(), offs.end());
        got.resize(t.members.size(), 0);
        if (got != std::vector<uint32_t>{0, 4, 8, 16, 24, 32, 56, 68, 72, 76}) { continue; }
        // AbiRoot matched: member 6 must be an array of 3 floats, stride 4.
        // Slang wraps fixed-array struct members in a single-member struct,
        // so unwrap any such wrapper before the kind check.
        uint32_t arr_id = t.members.size() > 6 ? t.members[6] : 0;
        while (arr_id < m.types.size() &&
               m.types[arr_id].kind == spirv_layout::Kind::Struct &&
               m.types[arr_id].members.size() == 1) {
            arr_id = m.types[arr_id].members[0];
        }
        if (arr_id < m.types.size() && m.types[arr_id].kind == spirv_layout::Kind::Array) {
            const auto& at = m.types[arr_id];
            uint32_t elem = at.elem_type_id;
            uint32_t len  = at.length_id < m.constants.size() ? m.constants[at.length_id] : 0;
            CHECK(m.types[elem].kind == spirv_layout::Kind::Float && m.types[elem].width == 32,
                  "AbiRoot.arr element type is float");
            CHECK(len == 3, "AbiRoot.arr count is 3");
            uint32_t stride = arr_id < m.array_strides.size() ? m.array_strides[arr_id] : 0;
            CHECK(stride == 4, "AbiRoot.arr stride is 4");
            checked = true;
        }
    }
    CHECK(checked, "AbiRoot array member found and checked");
}

// Artifact identity verification (ABI): the SPIR-V header version must match
// the profile directory (vk_native_spv16 -> 1.6, vk_bindless_spv15 -> 1.5)
// and the artifact's entry points must be the expected set — the artifact-key
// scheme: source (file name) + profile + SPIR-V version (directory) + entry
// points (extracted here) + profile version (IZ_PROFILE, compile-time).
static uint32_t spirv_header_version(const std::vector<uint8_t>& spv) {
    if (spv.size() < 8) { return 0; }
    return uint32_t(spv[4]) | (uint32_t(spv[5]) << 8) |
           (uint32_t(spv[6]) << 16) | (uint32_t(spv[7]) << 24);
}

static std::vector<std::string> spirv_entry_points(const std::vector<uint8_t>& spv) {
    std::vector<std::string> out;
    if (spv.size() < 20) { return out; }
    auto rd = [&](size_t off) -> uint32_t {
        return uint32_t(spv[off]) | (uint32_t(spv[off + 1]) << 8) |
               (uint32_t(spv[off + 2]) << 16) | (uint32_t(spv[off + 3]) << 24);
    };
    size_t i = 20;
    while (i + 4 <= spv.size()) {
        uint32_t word = rd(i);
        uint32_t op   = word & 0xFFFFu;
        uint32_t n    = word >> 16;
        if (n == 0 || i + 4 * n > spv.size()) { return out; }
        if (op == 15) {  // OpEntryPoint <execModel> <entryId> <name...>
            std::string name;
            bool done = false;
            for (uint32_t k = 2; k < n && !done; ++k) {
                uint32_t w = rd(i + 4 + 4 * k);
                for (int b = 0; b < 4; ++b) {
                    char c = char((w >> (8 * b)) & 0xFF);
                    if (c == '\0') { done = true; break; }
                    name.push_back(c);
                }
            }
            out.push_back(name);
        }
        i += 4 * n;
    }
    return out;
}

static void check_artifact_identity(const std::vector<uint8_t>& spv,
                                    uint32_t expected_spv_version,
                                    const std::vector<std::string>& expected_entries,
                                    const char* label) {
    // Profile-version identity: the compile-time profile name's version
    // suffix must equal the version encoded in the artifact directory tag.
    CHECK(strcmp(IZ_PROFILE, "IZANAGI_VK_NATIVE_" IZ_PROFILE_VERSION_NATIVE) == 0 ||
              strcmp(IZ_PROFILE, "IZANAGI_VK_BINDLESS_" IZ_PROFILE_VERSION_BINDLESS) == 0,
          "profile name version matches the artifact tag version");
    CHECK(spirv_header_version(spv) == expected_spv_version,
          "artifact SPIR-V version matches the profile directory");
    auto entries = spirv_entry_points(spv);
    bool same = entries.size() == expected_entries.size();
    for (size_t k = 0; same && k < entries.size(); ++k) {
        same = entries[k] == expected_entries[k];
    }
    CHECK(same, "artifact entry points match the expected set");
}

static void test_shader_layout_manifest() {
    printf("--- Test: extracted shader layout manifest ---\n");
    auto abi = read_file(find_shader_artifact("abi_test.spv"));
    CHECK(!abi.empty(), "abi_test.spv artifact present");
    if (!abi.empty()) {
        check_artifact_identity(abi, 0x00010600u, {"main_cs"}, "abi_test native");
        manifest_matches(abi, {0, 4, 8, 16, 24, 32, 56, 68, 72, 76}, 80, "AbiRoot");
        check_abi_root_array(abi);
        manifest_matches(abi, {0, 4, 8}, 12, "AbiInner");
        manifest_matches(abi, {0, 16}, 24, "AbiNested");
        manifest_matches(abi, {0, 8}, 16, "AbiArgData");
    }
    // Bindless artifacts must carry the 1.5 header (version encoded in the
    // directory is real, not cosmetic).
    auto abi_b = read_file(find_shader_artifact_bindless("abi_test.spv"));
    CHECK(!abi_b.empty(), "bindless abi_test.spv artifact present");
    if (!abi_b.empty()) {
        check_artifact_identity(abi_b, 0x00010500u, {"main_cs"}, "abi_test bindless");
    }

    auto i816 = read_file(find_shader_artifact("abi_int8_16.spv"));
    CHECK(!i816.empty(), "abi_int8_16.spv artifact present");
    if (!i816.empty()) {
        manifest_matches(i816, {0, 2, 8}, 12, "Int8_16Root");
        manifest_matches(i816, {0, 8}, 16, "Int8_16ArgData");
    }
}

int main() {
    printf("Izanagi Common Tests\n");
    printf("====================\n\n");

    test_arena_alignment_and_rewind();
    test_arena_scopes();
    test_arena_concurrent();
    test_vector_insert_growth();
    test_vector_alloc_failure();
    test_slotmap_generations();
    test_slotmap_stress();
    test_vector_nontrivial_insert();
    test_bitset();
    test_enum_ops();
    test_span();
    test_profile_report();
    test_dispatch_capabilities();
    test_abi_manifest();
    test_shader_layout_manifest();

    printf("\n=================\n");
    if (g_failures == 0) {
        printf("ALL TESTS PASSED\n");
    } else {
        printf("%d FAILURE(S)\n", g_failures);
    }
    return g_failures;
}
