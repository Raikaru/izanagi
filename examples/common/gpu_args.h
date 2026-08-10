#pragma once
// gpu_args.h — frame-ring bump allocator over gpu::malloc.
// The buffer is partitioned into kMaxFramesInFlight disjoint slices; frame N
// writes only into slice N % kMaxFramesInFlight. The surface's per-slot
// timeline wait guarantees slice reclamation is safe.

#include "izanagi/gpu.h"

struct GpuArgs {
    gpu::GpuPtr buffer       = 0;
    void*       host_ptr     = nullptr;
    size_t      slice_cap    = 0; // per-slot capacity
    size_t      offsets[gpu::kMaxFramesInFlight] = {};
    uint64_t    frame_idx    = 0;
};

void gpu_args_init(GpuArgs* args, gpu::Device device, size_t capacity = 1024 * 1024);
void gpu_args_shutdown(GpuArgs* args, gpu::Device device);

// Begin a new frame; resets the current frame's slice offset.
void gpu_args_begin_frame(GpuArgs* args);

// Append data for the current frame; returns the GPU address.
template <class T>
gpu::GpuPtr gpu_args_append(GpuArgs* args, const T& data) {
    size_t align = alignof(T) > 8 ? alignof(T) : 8;
    uint32_t slot = args->frame_idx % gpu::kMaxFramesInFlight;
    size_t  offset = (args->offsets[slot] + align - 1) & ~(align - 1);
    if (offset + sizeof(T) > args->slice_cap) { return 0; }

    size_t slice_base = (size_t)slot * args->slice_cap;
    uint8_t* dst = static_cast<uint8_t*>(args->host_ptr) + slice_base + offset;
    memcpy(dst, &data, sizeof(T));
    args->offsets[slot] = offset + sizeof(T);

    return args->buffer + slice_base + offset;
}

// Advance to the next frame (call once per frame).
void gpu_args_end_frame(GpuArgs* args);
