#include "platform_utils.h"

#include <cstdlib>
#include <cstring>

#if _WIN32

#    include <intrin.h>
extern "C" {
typedef void(__stdcall* PFLS_CALLBACK_FUNCTION)(void* lpFlsData);
__declspec(dllimport) unsigned long __stdcall FlsAlloc(PFLS_CALLBACK_FUNCTION lpCallback);
__declspec(dllimport) void* __stdcall         FlsGetValue(unsigned long dwFlsIndex);
__declspec(dllimport) int __stdcall           FlsSetValue(unsigned long dwFlsIndex, void* lpFlsData);
__declspec(dllimport) int __stdcall           FlsFree(unsigned long dwFlsIndex);
__declspec(dllimport) void __stdcall          AcquireSRWLockExclusive(void* SRWLock);
__declspec(dllimport) void __stdcall          AcquireSRWLockShared(void* SRWLock);
__declspec(dllimport) int __stdcall           TryAcquireSRWLockExclusive(void* SRWLock);
__declspec(dllimport) void __stdcall          ReleaseSRWLockExclusive(void* SRWLock);
__declspec(dllimport) void __stdcall          ReleaseSRWLockShared(void* SWRLock);
__declspec(dllimport) void __stdcall          InitializeConditionVariable(void* ConditionVariable);
__declspec(dllimport) int __stdcall           SleepConditionVariableSRW(void* ConditionVariable, void* SRWLock, unsigned long dwMilliseconds, unsigned long dwFlags);
__declspec(dllimport) void __stdcall          WakeAllConditionVariable(void* ConditionVariable);
__declspec(dllimport) void __stdcall          WakeConditionVariable(void* ConditionVariable);
typedef unsigned long(__stdcall* IZ_THREAD_START_ROUTINE)(void*);
__declspec(dllimport) void* __stdcall CreateThread(void* lpThreadAttributes, unsigned long dwStackSize,
                                                   IZ_THREAD_START_ROUTINE lpStartAddress, void* lpParameter,
                                                   unsigned long dwCreationFlags, unsigned long* lpThreadId);
__declspec(dllimport) unsigned long __stdcall WaitForSingleObject(void* hHandle, unsigned long dwMilliseconds);
__declspec(dllimport) int __stdcall           CloseHandle(void* hObject);
__declspec(dllimport) int __stdcall           SetThreadPriority(void* hThread, int nPriority);
__declspec(dllimport) unsigned long __stdcall GetCurrentThreadId(void);
__declspec(dllimport) int __stdcall           QueryPerformanceCounter(long long* lpPerformanceCount);
__declspec(dllimport) int __stdcall           QueryPerformanceFrequency(long long* lpFrequency);
}

#elif __linux__ || __APPLE__
#    include <pthread.h>
#    include <sched.h>
#    include <time.h>
#else
#    error "Unimplemented platform"
#endif

namespace gpu {
#if _WIN32
tls_key tls_alloc(tls_destructor d) {
    return FlsAlloc(d);
}

void tls_free(tls_key k) {
    FlsFree(k);
}

void tls_set_data(tls_key key, void* data) {
    FlsSetValue(key, data);
}

void* tls_get_data(tls_key key) {
    return FlsGetValue(key);
}

int64_t atomic_exchange(int64_t* x, int64_t val) {
    return _InterlockedExchange64(x, val);
}

int64_t atomic_load(int64_t* x) {
    return *x;
}

int64_t atomic_fetch_add(int64_t* x, int64_t val) {
    return _InterlockedExchangeAdd64(x, val);
}

bool atomic_compare_exchange(int64_t* dst, int64_t* expected, int64_t desired) {
    int64_t original = *expected;
    *expected        = _InterlockedCompareExchange64(dst, desired, *expected);
    return original == *expected;
}

void mutex_lock(mutex* mtx) {
    AcquireSRWLockExclusive(mtx);
}

void mutex_unlock(mutex* mtx) {
    ReleaseSRWLockExclusive(mtx);
}
bool mutex_try_lock(mutex* mtx) {
    return TryAcquireSRWLockExclusive(mtx) != 0;
}

void rwlock_lock_read(rwlock* l) {
    AcquireSRWLockShared(l);
}

void rwlock_unlock_read(rwlock* l) {
    ReleaseSRWLockShared(l);
}

void rwlock_lock_write(rwlock* l) {
    AcquireSRWLockExclusive(l);
}

void rwlock_unlock_write(rwlock* l) {
    ReleaseSRWLockExclusive(l);
}

void condvar_init(condvar* cv) {
    InitializeConditionVariable(&cv->impl);
}

void condvar_destroy(condvar* cv) {
    (void)cv; // CONDITION_VARIABLE needs no teardown
}

void condvar_wait(condvar* cv, mutex* mtx) {
    SleepConditionVariableSRW(&cv->impl, mtx, 0xFFFFFFFFul, 0);
}

void condvar_signal(condvar* cv) {
    WakeConditionVariable(&cv->impl);
}

void condvar_broadcast(condvar* cv) {
    WakeAllConditionVariable(&cv->impl);
}

struct thread_start {
    void (*fn)(void*);
    void* arg;
};

static unsigned long __stdcall thread_trampoline(void* p) {
    auto* start = static_cast<thread_start*>(p);
    void (*fn)(void*) = start->fn;
    void* arg         = start->arg;
    std::free(start);
    fn(arg);
    return 0;
}

bool thread_create(thread_handle* out, void (*fn)(void*), void* arg) {
    auto* start = static_cast<thread_start*>(std::malloc(sizeof(thread_start)));
    if (start == nullptr) { return false; }
    start->fn  = fn;
    start->arg = arg;
    unsigned long tid = 0;
    void* h = CreateThread(nullptr, 0, &thread_trampoline, start, 0, &tid);
    if (h == nullptr) {
        std::free(start);
        return false;
    }
    *out = reinterpret_cast<thread_handle>(h);
    return true;
}

void thread_join(thread_handle t) {
    if (t != 0) {
        WaitForSingleObject(reinterpret_cast<void*>(t), 0xFFFFFFFFul);
        CloseHandle(reinterpret_cast<void*>(t));
    }
}

void thread_set_low_priority(thread_handle t) {
    if (t != 0) { SetThreadPriority(reinterpret_cast<void*>(t), -1 /* THREAD_PRIORITY_BELOW_NORMAL */); }
}

uintptr_t current_thread_id() {
    return GetCurrentThreadId();
}

double monotonic_seconds() {
    static long long freq = 0;
    if (freq == 0) { QueryPerformanceFrequency(&freq); }
    long long now = 0;
    QueryPerformanceCounter(&now);
    return static_cast<double>(now) / static_cast<double>(freq);
}

#elif __linux__ || __APPLE__
tls_key tls_alloc(tls_destructor d) {
    pthread_key_t key;
    pthread_key_create(&key, d);
    return static_cast<tls_key>(key);
}

void tls_free(tls_key k) {
    pthread_key_delete(static_cast<pthread_key_t>(k));
}

void tls_set_data(tls_key key, void* data) {
    pthread_setspecific(static_cast<pthread_key_t>(key), data);
}

void* tls_get_data(tls_key key) {
    return pthread_getspecific(static_cast<pthread_key_t>(key));
}

int64_t atomic_exchange(int64_t* x, int64_t val) {
    return __atomic_exchange_n(x, val, __ATOMIC_ACQ_REL);
}

int64_t atomic_load(int64_t* x) {
    return __atomic_load_n(x, __ATOMIC_ACQUIRE);
}

int64_t atomic_fetch_add(int64_t* x, int64_t val) {
    return __atomic_fetch_add(x, val, __ATOMIC_ACQ_REL);
}

bool atomic_compare_exchange(int64_t* dst, int64_t* expected, int64_t desired) {
    return __atomic_compare_exchange_n(dst, expected, desired, true, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED);
}

void mutex_lock(mutex* mtx) {
    pthread_mutex_lock(mtx);
}

void mutex_unlock(mutex* mtx) {
    pthread_mutex_unlock(mtx);
}

bool mutex_try_lock(mutex* mtx) {
    return pthread_mutex_trylock(mtx) == 0;
}

void rwlock_lock_read(rwlock* l) {
    pthread_rwlock_rdlock(l);
}

void rwlock_unlock_read(rwlock* l) {
    pthread_rwlock_unlock(l);
}

void rwlock_lock_write(rwlock* l) {
    pthread_rwlock_wrlock(l);
}

void rwlock_unlock_write(rwlock* l) {
    pthread_rwlock_unlock(l);
}

void condvar_init(condvar* cv) {
    pthread_cond_init(&cv->impl, nullptr);
}

void condvar_destroy(condvar* cv) {
    pthread_cond_destroy(&cv->impl);
}

void condvar_wait(condvar* cv, mutex* mtx) {
    pthread_cond_wait(&cv->impl, mtx);
}

void condvar_signal(condvar* cv) {
    pthread_cond_signal(&cv->impl);
}

void condvar_broadcast(condvar* cv) {
    pthread_cond_broadcast(&cv->impl);
}

struct thread_start {
    void (*fn)(void*);
    void* arg;
};

static void* thread_trampoline(void* p) {
    auto* start = static_cast<thread_start*>(p);
    void (*fn)(void*) = start->fn;
    void* arg         = start->arg;
    std::free(start);
    fn(arg);
    return nullptr;
}

bool thread_create(thread_handle* out, void (*fn)(void*), void* arg) {
    auto* start = static_cast<thread_start*>(std::malloc(sizeof(thread_start)));
    if (start == nullptr) { return false; }
    start->fn  = fn;
    start->arg = arg;
    pthread_t t;
    if (pthread_create(&t, nullptr, &thread_trampoline, start) != 0) {
        std::free(start);
        return false;
    }
    *out = t;
    return true;
}

void thread_join(thread_handle t) {
    pthread_join(t, nullptr);
}

void thread_set_low_priority(thread_handle t) {
    if (t == 0) { return; }
    // Best-effort: SCHED_OTHER priority tweak; may fail without privileges.
    sched_param param{};
    param.sched_priority = 0;
    pthread_setschedparam(t, SCHED_OTHER, &param);
}

uintptr_t current_thread_id() {
    // pthread_t is integral on Linux but a pointer on macOS; copy
    // representation-agnostically instead of casting.
    pthread_t t = pthread_self();
    uintptr_t id = 0;
    static_assert(sizeof(t) <= sizeof(id), "pthread_t does not fit uintptr_t");
    std::memcpy(&id, &t, sizeof(t));
    return id;
}

double monotonic_seconds() {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) * 1e-9;
}

#endif

}  // namespace gpu
