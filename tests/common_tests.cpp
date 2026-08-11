// common_tests.cpp — GPU-independent tests for the common infrastructure:
// Arena (alignment, mark/rewind, overflow), ScratchScope, Vector (insert,
// growth, allocation failure), SlotMap (reuse, generations), TwoLevelBitset,
// and the enum bitwise operators. No Vulkan device required.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "common/containers.h"

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

    printf("\n=================\n");
    if (g_failures == 0) {
        printf("ALL TESTS PASSED\n");
    } else {
        printf("%d FAILURE(S)\n", g_failures);
    }
    return g_failures;
}
