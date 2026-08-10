#include "gpu_args.h"
#include <cstring>

void gpu_args_init(GpuArgs* args, gpu::Device device, size_t capacity) {
    args->slice_cap = capacity;
    // Total allocation covers kMaxFramesInFlight disjoint slices.
    args->buffer   = gpu::malloc(device, capacity * gpu::kMaxFramesInFlight,
                                 gpu::Memory::Default);
    args->host_ptr = gpu::get_host_pointer(device, args->buffer);
    args->frame_idx = 0;
    memset(args->offsets, 0, sizeof(args->offsets));
}

void gpu_args_shutdown(GpuArgs* args, gpu::Device device) {
    if (args->buffer) {
        gpu::free(device, args->buffer);
        args->buffer   = 0;
        args->host_ptr = nullptr;
    }
}

void gpu_args_begin_frame(GpuArgs* args) {
    uint32_t slot = args->frame_idx % gpu::kMaxFramesInFlight;
    args->offsets[slot] = 0;
}

void gpu_args_end_frame(GpuArgs* args) {
    args->frame_idx++;
}
